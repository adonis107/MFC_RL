from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .registry import EnvironmentSpec
from .runtime import _validation_horizon, validation_laws


def finite_population_flow(
    env: Any,
    algorithm: Any,
    control: torch.Tensor | torch.nn.Module,
    mu0: torch.Tensor,
    horizon: int,
    flow_mode: str,
    flow_particles: int,
) -> torch.Tensor:
    if flow_mode == "exact":
        if not hasattr(env, "exact_population_flow"):
            raise ValueError(f"{type(env).__name__} does not expose exact_population_flow.")
        with torch.no_grad():
            return env.exact_population_flow(control, mu0, horizon).detach()
    if flow_mode == "particle":
        return algorithm.estimate_population_flow(control, mu0, flow_particles, horizon=horizon).detach()
    raise ValueError(f"Unknown flow_mode={flow_mode!r}.")


def evaluate_control(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    if spec.family == "finite":
        return evaluate_finite(spec, env, control, train_config, evaluation_config)
    if spec.name == "lq":
        lambda_value = float(evaluation_config.get("lambda", train_config.get("lambda", 0.0)))
        with torch.no_grad():
            cost = env.exact_cost(control, lambda_=lambda_value)
            metrics: Dict[str, Any] = {"objective": float(cost.item()), "cost": float(cost.item())}
            if lambda_value:
                metrics["lambda"] = lambda_value
            if hasattr(env, "riccati_policy"):
                optimal = env.riccati_policy()
                optimal_cost = env.exact_cost(optimal, lambda_=lambda_value)
                metrics["optimal_cost"] = float(optimal_cost.item())
                metrics["objective_gap"] = float((cost - optimal_cost).item())
            return metrics
    if spec.name == "portfolio":
        lambda_value = float(evaluation_config.get("lambda", train_config.get("lambda", 0.0)))
        with torch.no_grad():
            objective = env.exact_objective(control, lambda_=lambda_value)
            metrics = {"objective": float(objective.item()), "value": float(objective.item())}
            if hasattr(env, "optimal_policy"):
                optimal = env.optimal_policy()
                optimal_objective = env.exact_objective(optimal, lambda_=lambda_value)
                metrics["optimal_objective"] = float(optimal_objective.item())
                metrics["objective_gap"] = float((optimal_objective - objective).item())
            return metrics
    particles = int(evaluation_config.get("particles", getattr(env.config, "N_val", getattr(env.config, "N_pop", 32))))
    lambda_value = float(evaluation_config.get("lambda", 0.0))
    horizon = evaluation_config.get("horizon")
    with torch.no_grad():
        rollout = env.sample_trajectories(
            control,
            particles,
            seed=seed,
            lambda_=lambda_value,
            horizon=None if horizon is None else int(horizon),
            exploration=False,
        )
    metrics = {"objective": scalar(rollout["objective"]), "particles": particles}
    for key in ("cumulative_control_energy", "alignment_time", "synchronization_time"):
        if key in rollout:
            metrics[key] = scalar(rollout[key])
    return metrics


def evaluate_finite(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> Dict[str, Any]:
    laws = validation_laws(spec, env, evaluation_config)
    if laws is None:
        raise ValueError("Finite evaluation requires validation laws.")
    horizon = _validation_horizon(env, train_config, evaluation_config)
    was_training = control.training if isinstance(control, torch.nn.Module) else False
    if isinstance(control, torch.nn.Module):
        control.eval()
    try:
        with torch.no_grad():
            law_batch = laws.unsqueeze(0) if laws.ndim == 1 else laws
            values = torch.stack([env.exact_value(control, mu0, horizon) for mu0 in law_batch])
            flows = torch.stack([env.exact_population_flow(control, mu0, horizon) for mu0 in law_batch])
            metrics = {
                "value": float(values.mean().item()),
                "value_std": float(values.std(unbiased=law_batch.shape[0] > 1).item()) if law_batch.shape[0] > 1 else 0.0,
                "final_law": flows[:, -1].mean(dim=0).detach().cpu(),
            }
            if spec.name == "twostate":
                policy = env.policy_probs(control).detach()
                optimal = env.optimal_policy().detach()
                policy_error = (policy - optimal).abs().mean()
                metrics["policy_error"] = float(policy_error.item())
                metrics["policy"] = policy.cpu()
                metrics["optimal_policy"] = optimal.cpu()
            return metrics
    finally:
        if isinstance(control, torch.nn.Module):
            control.train(was_training)


def scalar(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.detach().reshape(-1)[0].cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_scalar = scalar


__all__ = ["evaluate_control", "evaluate_finite", "finite_population_flow", "scalar", "_scalar"]
