from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import torch

from ..core.controls import control_vector, initialize_control
from .common import (
    _float_list,
    _matrix_rows,
    _normal_ci_z,
    _safe_covariance,
    _save_diagnostic_result,
    _sign_match,
)
from ..core.evaluation import finite_population_flow
from ..core.gradient_steps import (
    continuous_mfreinforce_gradient_step,
    exact_gradient_step,
    finite_gradient,
    make_algorithm,
    pathwise_gradient_step,
)
from ..core.memory import release_memory
from ..core.registry import EnvironmentSpec, build_environment, require_algorithm_name, require_env_name, validate_compatibility
from ..core.runtime import _aux_batch, _lambda_value, _main_batch, _training_horizon, sample_initial_laws, validation_laws
from ..core.session import RunResult, normalize_experiment_config, set_seed


def run_gradient_diagnostic(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    env_name = require_env_name(config)
    algorithm_name = require_algorithm_name(config)
    validate_compatibility(env_name, algorithm_name)
    spec, env = build_environment(config)
    algorithm_config = dict(config.get("algorithm_config", {}))
    train_config = dict(config.get("train", {}))
    diagnostic = dict(config.get("diagnostic", {}))
    seed = int(train_config.get("seed", 0))
    set_seed(seed, env.config.device)
    control = initialize_control(spec, env)
    algorithm = make_algorithm(algorithm_name, env, algorithm_config)
    replications = int(diagnostic.get("replications", 16))
    lambdas = _float_list(diagnostic.get("lambdas", [_lambda_value(algorithm_config, train_config, default=0.1)]))
    rows = []
    sample_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    covariance_rows: List[Dict[str, Any]] = []
    oracle, oracle_metadata = oracle_gradient(spec, env, control, train_config, config.get("evaluation", {}), diagnostic, seed)
    for lambda_value in lambdas:
        local_algorithm_config = dict(algorithm_config)
        if algorithm_name == "logits":
            local_algorithm_config["epsilon"] = lambda_value
        else:
            local_algorithm_config["lambda"] = lambda_value
        estimates = []
        for rep in range(replications):
            set_seed(seed + rep, env.config.device)
            estimates.append(
                single_gradient_estimate(
                    spec,
                    env,
                    control,
                    algorithm_name,
                    algorithm,
                    local_algorithm_config,
                    train_config,
                    iteration=rep,
                )
            )
        stacked = torch.stack(estimates)
        row = gradient_summary_row(lambda_value, stacked, oracle)
        row.update(oracle_metadata)
        rows.append(row)
        artifacts = gradient_artifact_rows(
            lambda_value,
            stacked,
            oracle,
            ci_level=float(diagnostic.get("ci_level", 0.95)),
            max_sample_rows=int(diagnostic.get("max_gradient_sample_rows", 1_000_000)),
        )
        sample_rows.extend(artifacts["samples"])
        coordinate_rows.extend(artifacts["coordinates"])
        covariance_rows.extend(artifacts["covariance"])
        del estimates, stacked, artifacts
        release_memory()
    return _save_diagnostic_result(
        "diagnose-gradient",
        config,
        rows,
        {
            "rows": len(rows),
            "sample_rows": len(sample_rows),
            "coordinate_rows": len(coordinate_rows),
            "covariance_rows": len(covariance_rows),
        },
        extra_tables={
            "gradient_samples.csv": sample_rows,
            "gradient_coordinates.csv": coordinate_rows,
            "gradient_covariance.csv": covariance_rows,
        },
    )


def single_gradient_estimate(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    algorithm_name: str,
    algorithm: Any,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
) -> torch.Tensor:
    if spec.family == "finite":
        B = _main_batch(env, train_config, algorithm_config)
        n_aux = _aux_batch(env, train_config, algorithm_config)
        horizon = _training_horizon(env, train_config)
        flow_mode = str(train_config.get("flow_mode", "exact"))
        flow_particles = int(train_config.get("flow_particles", max(1, B)))
        mu0 = sample_initial_laws(spec, env, 1, train_config)[0]
        mu_flow = finite_population_flow(env, algorithm, control, mu0, horizon, flow_mode, flow_particles)
        grad, _ = finite_gradient(
            algorithm_name,
            algorithm,
            control,
            mu0,
            mu_flow,
            iteration,
            B,
            n_aux,
            algorithm_config,
            train_config,
        )
        return grad.detach().reshape(-1)
    if algorithm_name == "exact-gradient":
        _, grad, _ = exact_gradient_step(spec, env, control, algorithm_config)  # type: ignore[arg-type]
        return grad.detach().reshape(-1)
    if algorithm_name == "pathwise-gradient":
        _, grad, _ = pathwise_gradient_step(env, control, algorithm_config, train_config, iteration)  # type: ignore[arg-type]
        return grad.detach().reshape(-1)
    if algorithm_name == "continuous-mfreinforce":
        _, grad, _ = continuous_mfreinforce_gradient_step(
            env,
            algorithm,
            control,
            algorithm_config,
            train_config,
            iteration,
        )
        return grad.detach().reshape(-1)
    raise ValueError(f"Unsupported gradient diagnostic for {algorithm_name!r}.")


def oracle_gradient(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    diagnostic_config: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
    diagnostic_config = {} if diagnostic_config is None else diagnostic_config
    if spec.family == "finite":
        mu0 = validation_laws(spec, env, evaluation_config)
        if mu0 is not None and mu0.ndim != 1:
            mu0 = mu0[0]
        horizon = _training_horizon(env, train_config)
        if isinstance(control, torch.nn.Module):
            value = env.exact_value(control, mu0, horizon)
            grads = torch.autograd.grad(value, tuple(control.parameters()), allow_unused=False)
            return torch.cat([grad.detach().reshape(-1) for grad in grads]), {"oracle_kind": "exact_population_ad"}
        theta = control.detach().clone().requires_grad_(True)
        value = env.exact_value(theta, mu0, horizon)
        return torch.autograd.grad(value, theta)[0].detach().reshape(-1), {"oracle_kind": "exact_population_ad"}
    if spec.name == "lq":
        _, grad = env.exact_gradient(control)
        return grad.detach().reshape(-1), {"oracle_kind": "analytic_exact"}
    if spec.name == "portfolio":
        lambda_value = float(evaluation_config.get("lambda", train_config.get("lambda", 0.0)))
        _, grad = env.exact_gradient(control, lambda_=lambda_value)
        return grad.detach().reshape(-1), {"oracle_kind": "analytic_exact", "oracle_lambda": lambda_value}
    if spec.family == "pathwise" and isinstance(control, torch.nn.Module) and hasattr(env, "pathwise_gradient"):
        horizon = _training_horizon(env, train_config)
        default_particles = int(
            train_config.get(
                "population_particles",
                train_config.get("particles", getattr(env.config, "N_val", getattr(env.config, "N_pop", 32))),
            )
        )
        particles = int(
            diagnostic_config.get(
                "oracle_particles",
                evaluation_config.get("oracle_particles", max(1, default_particles)),
            )
        )
        replications = int(diagnostic_config.get("oracle_replications", evaluation_config.get("oracle_replications", 1)))
        lambda_value = float(diagnostic_config.get("oracle_lambda", evaluation_config.get("oracle_lambda", train_config.get("lambda", 0.0))))
        exploration = bool(diagnostic_config.get("oracle_exploration", train_config.get("exploration", True)))
        oracle_seed = int(diagnostic_config.get("oracle_seed", seed + 900_000))
        _, grad = env.pathwise_gradient(
            control,
            n_particles=max(1, particles),
            replications=max(1, replications),
            seed=oracle_seed,
            lambda_=lambda_value,
            horizon=horizon,
            exploration=exploration,
        )
        return grad.detach().reshape(-1), {
            "oracle_kind": "pathwise_ad_reference",
            "oracle_particles": max(1, particles),
            "oracle_replications": max(1, replications),
            "oracle_lambda": lambda_value,
            "oracle_exploration": exploration,
        }
    return None, {"oracle_kind": "unavailable"}


def gradient_summary_row(lambda_value: float, estimates: torch.Tensor, oracle: Optional[torch.Tensor]) -> Dict[str, Any]:
    mean = estimates.mean(dim=0)
    variance = estimates.var(dim=0, unbiased=estimates.shape[0] > 1) if estimates.shape[0] > 1 else torch.zeros_like(mean)
    covariance = _safe_covariance(estimates)
    if covariance.numel() > 0:
        eigvals = torch.linalg.eigvalsh(covariance)
        largest_eigenvalue = eigvals[-1]
    else:
        largest_eigenvalue = torch.tensor(0.0, dtype=estimates.dtype, device=estimates.device)
    row = {
        "lambda": lambda_value,
        "replications": int(estimates.shape[0]),
        "estimate_norm": float(torch.linalg.norm(mean).item()),
        "variance_trace": float(variance.sum().item()),
        "covariance_largest_eigenvalue": float(largest_eigenvalue.item()),
    }
    if oracle is not None:
        diff = mean - oracle
        sample_errors = estimates - oracle.unsqueeze(0)
        denom = torch.linalg.norm(mean) * torch.linalg.norm(oracle)
        error_norms = torch.linalg.norm(sample_errors, dim=1)
        outlier_cutoff = error_norms.mean() + 3.0 * error_norms.std(unbiased=error_norms.numel() > 1)
        cosine = mean @ oracle / denom if float(denom.item()) > 0 else torch.tensor(float("nan"), device=mean.device)
        row.update(
            {
                "oracle_norm": float(torch.linalg.norm(oracle).item()),
                "bias_norm": float(torch.linalg.norm(diff).item()),
                "relative_bias": float(torch.linalg.norm(diff).item() / max(float(torch.linalg.norm(oracle).item()), 1e-12)),
                "mse": float(sample_errors.square().sum(dim=1).mean().item()),
                "cosine_similarity": float(cosine.item()),
                "angular_error": float(torch.arccos(cosine.clamp(-1.0, 1.0)).item()) if torch.isfinite(cosine) else float("nan"),
                "norm_ratio": float(torch.linalg.norm(mean).item() / max(float(torch.linalg.norm(oracle).item()), 1e-12)),
                "outlier_rate": float((error_norms > outlier_cutoff).to(dtype=estimates.dtype).mean().item()) if error_norms.numel() > 1 else 0.0,
            }
        )
    return row


def gradient_artifact_rows(
    lambda_value: float,
    estimates: torch.Tensor,
    oracle: Optional[torch.Tensor],
    *,
    ci_level: float,
    max_sample_rows: int,
) -> Dict[str, List[Dict[str, Any]]]:
    estimates = estimates.reshape(estimates.shape[0], -1)
    mean = estimates.mean(dim=0)
    variance = estimates.var(dim=0, unbiased=estimates.shape[0] > 1) if estimates.shape[0] > 1 else torch.zeros_like(mean)
    std = torch.sqrt(variance.clamp_min(0.0))
    covariance = _safe_covariance(estimates)
    z = _normal_ci_z(ci_level)
    stderr = std / math.sqrt(max(1, estimates.shape[0]))
    ci_low = mean - z * stderr
    ci_high = mean + z * stderr

    sample_rows: List[Dict[str, Any]] = []
    written = 0
    for replication in range(estimates.shape[0]):
        for coordinate in range(estimates.shape[1]):
            if written >= max_sample_rows:
                break
            value = float(estimates[replication, coordinate].item())
            row = {
                "lambda": lambda_value,
                "replication": replication,
                "coordinate": coordinate,
                "estimate": value,
            }
            if oracle is not None:
                oracle_value = float(oracle[coordinate].item())
                row["oracle"] = oracle_value
                row["error"] = value - oracle_value
                row["sign_match"] = _sign_match(value, oracle_value)
            sample_rows.append(row)
            written += 1
        if written >= max_sample_rows:
            break

    coordinate_rows: List[Dict[str, Any]] = []
    for coordinate in range(estimates.shape[1]):
        row = {
            "lambda": lambda_value,
            "coordinate": coordinate,
            "mean": float(mean[coordinate].item()),
            "std": float(std[coordinate].item()),
            "variance": float(variance[coordinate].item()),
            "ci_level": ci_level,
            "ci_low": float(ci_low[coordinate].item()),
            "ci_high": float(ci_high[coordinate].item()),
        }
        if oracle is not None:
            oracle_value = float(oracle[coordinate].item())
            errors = estimates[:, coordinate] - oracle[coordinate]
            row.update(
                {
                    "oracle": oracle_value,
                    "bias": float((mean[coordinate] - oracle[coordinate]).item()),
                    "mse": float(errors.square().mean().item()),
                    "ci_covers_oracle": bool(ci_low[coordinate] <= oracle[coordinate] <= ci_high[coordinate]),
                    "sign_accuracy": float(
                        torch.as_tensor(
                            [_sign_match(float(value.item()), oracle_value) for value in estimates[:, coordinate]],
                            dtype=estimates.dtype,
                            device=estimates.device,
                        )
                        .mean()
                        .item()
                    ),
                }
            )
        coordinate_rows.append(row)

    covariance_rows = _matrix_rows(lambda_value, covariance, "covariance")
    return {"samples": sample_rows, "coordinates": coordinate_rows, "covariance": covariance_rows}


_gradient_artifact_rows = gradient_artifact_rows
_gradient_summary_row = gradient_summary_row
_oracle_gradient = oracle_gradient
_single_gradient_estimate = single_gradient_estimate


__all__ = [
    "_gradient_artifact_rows",
    "_gradient_summary_row",
    "_oracle_gradient",
    "_single_gradient_estimate",
    "gradient_artifact_rows",
    "gradient_summary_row",
    "oracle_gradient",
    "run_gradient_diagnostic",
    "single_gradient_estimate",
]
