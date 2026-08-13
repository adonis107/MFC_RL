from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, Optional

import torch

from mfc.algorithms import (
    AdaptiveSimplexControllerConfig,
    ConsistentAdaptiveSimplexMFREINFORCE,
    ContinuousTransportMFREINFORCE,
    FiniteBudgetAdaptiveSimplexMFREINFORCE,
    LogitsPerturbedMFREINFORCE,
    SimplexPerturbedMFREINFORCE,
)

from .registry import EXACT_ALGORITHMS, PATHWISE_ALGORITHMS, EnvironmentSpec
from .runtime import _aux_batch, _baseline, _eta_value, _lambda_value, _main_batch, _training_horizon


def make_algorithm(algorithm_name: str, env: Any, algorithm_config: Optional[Mapping[str, Any]] = None) -> Any:
    algorithm_config = dict(algorithm_config or {})
    if algorithm_name == "simplex":
        return SimplexPerturbedMFREINFORCE(env, algorithm_config)
    if algorithm_name == "logits":
        return LogitsPerturbedMFREINFORCE(env, algorithm_config)
    if algorithm_name == "finite-adaptive-simplex":
        return FiniteBudgetAdaptiveSimplexMFREINFORCE(env, _controller_config(algorithm_config))
    if algorithm_name == "consistent-adaptive-simplex":
        return ConsistentAdaptiveSimplexMFREINFORCE(env, _controller_config(algorithm_config))
    if algorithm_name == "continuous-mfreinforce":
        return ContinuousTransportMFREINFORCE(env, algorithm_config)
    if algorithm_name in EXACT_ALGORITHMS | PATHWISE_ALGORITHMS:
        return None
    raise ValueError(f"Unknown algorithm {algorithm_name!r}.")


def finite_gradient(
    algorithm_name: str,
    algorithm: Any,
    control: torch.Tensor | torch.nn.Module,
    mu0: torch.Tensor,
    mu_flow: torch.Tensor,
    iteration: int,
    B: int,
    n_aux: int,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    horizon = mu_flow.shape[0] - 1
    if algorithm_name == "simplex":
        value = _lambda_value(algorithm_config, train_config)
        return algorithm.complete_gradient_estimate(
            control,
            mu_flow,
            value,
            B,
            n_aux,
            eta=_eta_value(algorithm_config, value),
            baseline=_baseline(algorithm_config),
        )
    if algorithm_name == "logits":
        value = _lambda_value(algorithm_config, train_config)
        flow_particles = int(train_config.get("flow_particles", algorithm_config.get("flow_particles", 1)))
        return algorithm.gradient_estimate(
            control,
            mu0,
            value,
            B,
            n_aux,
            max(1, flow_particles),
            horizon=horizon,
            mu_flow=mu_flow,
        )
    if algorithm_name in {"finite-adaptive-simplex", "consistent-adaptive-simplex"}:
        return algorithm.gradient_estimate(
            control,
            mu_flow,
            iteration,
            B,
            n_aux,
            baseline=_baseline(algorithm_config),
        )
    raise ValueError(f"Unsupported finite algorithm {algorithm_name!r}.")


def exact_gradient_step(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor,
    algorithm_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if spec.name == "lq":
        objective, grad = env.exact_gradient(control)
        return objective, grad, {"objective": objective, "grad_norm": torch.linalg.norm(grad)}
    if spec.name == "portfolio":
        lambda_value = float(algorithm_config.get("lambda", algorithm_config.get("lambda_", 0.0)))
        objective, grad = env.exact_gradient(control, lambda_=lambda_value)
        return objective, grad, {"objective": objective, "grad_norm": torch.linalg.norm(grad), "lambda": lambda_value}
    raise ValueError(f"{spec.name!r} does not support exact-gradient training.")


def pathwise_gradient_step(
    env: Any,
    control: torch.nn.Module,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    n_particles = int(train_config.get("particles", algorithm_config.get("particles", getattr(env.config, "N_pop", 32))))
    replications = int(train_config.get("replications", algorithm_config.get("replications", 1)))
    lambda_value = float(algorithm_config.get("lambda", algorithm_config.get("lambda_", 0.0)))
    horizon = train_config.get("horizon")
    seed = int(train_config.get("seed", 0)) + iteration
    objective, grad = env.pathwise_gradient(
        control,
        n_particles=n_particles,
        replications=replications,
        seed=seed,
        lambda_=lambda_value,
        horizon=None if horizon is None else int(horizon),
        exploration=bool(train_config.get("exploration", True)),
    )
    return objective, grad, {
        "objective": objective,
        "grad_norm": torch.linalg.norm(grad),
        "particles": n_particles,
        "replications": replications,
        "lambda": lambda_value,
    }


def continuous_mfreinforce_gradient_step(
    env: Any,
    algorithm: ContinuousTransportMFREINFORCE,
    control: torch.Tensor | torch.nn.Module,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    B = _main_batch(env, train_config, algorithm_config)
    n_aux = _aux_batch(env, train_config, algorithm_config)
    lambda_value = _lambda_value(algorithm_config, train_config)
    eta_value = _eta_value(algorithm_config, lambda_value)
    horizon = _training_horizon(env, train_config)
    seed = int(train_config.get("seed", 0)) + iteration
    population_particles = int(
        train_config.get(
            "population_particles",
            algorithm_config.get("population_particles", train_config.get("particles", getattr(env.config, "N_pop", B))),
        )
    )
    grad, diag = algorithm.complete_gradient_estimate(
        control,
        lambda_value,
        B,
        n_aux,
        eta=eta_value,
        horizon=horizon,
        population_particles=population_particles,
        seed=seed,
        baseline=_baseline(algorithm_config),
        sensitivity_baseline=algorithm_config.get("sensitivity_baseline", "nominal"),
        keep_score_diagnostics=algorithm_config.get("keep_score_diagnostics"),
    )
    objective = diag["mean_return"]
    diag = dict(diag)
    diag["objective"] = objective
    diag["population_particles"] = torch.tensor(population_particles, device=env.config.device)
    return objective, grad, diag


def _controller_config(algorithm_config: Mapping[str, Any]) -> AdaptiveSimplexControllerConfig:
    raw = dict(algorithm_config.get("controller_config", algorithm_config))
    allowed = {field.name for field in fields(AdaptiveSimplexControllerConfig)}
    return AdaptiveSimplexControllerConfig(**{key: value for key, value in raw.items() if key in allowed})


__all__ = [
    "continuous_mfreinforce_gradient_step",
    "exact_gradient_step",
    "finite_gradient",
    "make_algorithm",
    "pathwise_gradient_step",
]
