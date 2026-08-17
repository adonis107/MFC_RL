from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from .common import _terminal_sample_rows


def _exact_application_outputs(
    env_name: str,
    env: Any,
    control: Any,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    if env_name == "lq":
        lambda_value = float(evaluation_config.get("lambda", train_config.get("lambda", 0.0)))
        means, variances = env.exact_moments(control, lambda_=lambda_value)
        optimal = env.riccati_policy()
        opt_means, opt_variances = env.exact_moments(optimal, lambda_=lambda_value)
        value = -env.exact_cost(control, lambda_=lambda_value)
        opt_value = -env.exact_cost(optimal, lambda_=lambda_value)
        sample = env.sample_trajectories(control, int(evaluation_config.get("particles", 256)), seed=seed, lambda_=lambda_value)
        return {
            "time_metrics": _moment_rows(env_name, means, variances, opt_means, opt_variances),
            "policy": _gain_rows(env_name, control, optimal),
            "terminal_samples": _terminal_sample_rows(sample["states"][:, -1], "terminal_state"),
            "landscape": _lq_landscape_rows(env, control, evaluation_config),
            "metrics": {
                "objective": float(value.item()),
                "optimal_objective": float(opt_value.item()),
                "objective_gap": float((opt_value - value).item()),
                "lambda": lambda_value,
            },
        }

    lambda_value = float(evaluation_config.get("lambda", train_config.get("lambda", 0.0)))
    means, variances = env.exact_moments(control, lambda_=lambda_value)
    optimal = env.optimal_policy()
    opt_means, opt_variances = env.exact_moments(optimal, lambda_=lambda_value)
    objective = env.exact_objective(control, lambda_=lambda_value)
    optimal_objective = env.exact_objective(optimal, lambda_=lambda_value)
    sample = env.sample_trajectories(control, int(evaluation_config.get("particles", 512)), seed=seed, lambda_=lambda_value)
    terminal = sample["states"][:, -1]
    downside = (terminal < 0.0).to(dtype=terminal.dtype).mean()
    return {
        "time_metrics": _moment_rows(env_name, means, variances, opt_means, opt_variances),
        "policy": _gain_rows(env_name, control, optimal),
        "terminal_samples": _terminal_sample_rows(terminal, "terminal_wealth"),
        "efficient_frontier": _portfolio_frontier_rows(env, control, evaluation_config),
        "metrics": {
            "objective": float(objective.item()),
            "optimal_objective": float(optimal_objective.item()),
            "objective_gap": float((optimal_objective - objective).item()),
            "terminal_mean": float(terminal.mean().item()),
            "terminal_variance": float(terminal.var(unbiased=terminal.numel() > 1).item()) if terminal.numel() > 1 else 0.0,
            "downside_probability": float(downside.item()),
        },
    }



def _moment_rows(
    env_name: str,
    means: torch.Tensor,
    variances: torch.Tensor,
    opt_means: torch.Tensor,
    opt_variances: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows = []
    for t in range(means.numel()):
        rows.append(
            {
                "env": env_name,
                "time": t,
                "mean": float(means[t].item()),
                "variance": float(variances[t].item()),
                "optimal_mean": float(opt_means[t].item()),
                "optimal_variance": float(opt_variances[t].item()),
                "mean_error": float((means[t] - opt_means[t]).abs().item()),
                "variance_error": float((variances[t] - opt_variances[t]).abs().item()),
            }
        )
    return rows



def _gain_rows(env_name: str, control: torch.Tensor, optimal: torch.Tensor) -> List[Dict[str, Any]]:
    rows = []
    theta = control.detach()
    opt = optimal.detach()
    for t in range(theta.shape[0]):
        for coordinate in range(theta.shape[1]):
            rows.append(
                {
                    "env": env_name,
                    "time": t,
                    "coordinate": coordinate,
                    "value": float(theta[t, coordinate].item()),
                    "optimal": float(opt[t, coordinate].item()),
                    "abs_error": float((theta[t, coordinate] - opt[t, coordinate]).abs().item()),
                }
            )
    return rows



def _lq_landscape_rows(env: Any, control: torch.Tensor, evaluation_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    grid_size = int(evaluation_config.get("landscape_grid_size", 21))
    radius = float(evaluation_config.get("landscape_radius", 1.0))
    lambda_value = float(evaluation_config.get("lambda", 0.0))
    t = int(evaluation_config.get("landscape_time", 0))
    t = max(0, min(t, env.config.T - 1))
    theta = control.detach()
    k_grid = torch.linspace(theta[t, 0] - radius, theta[t, 0] + radius, grid_size, dtype=env.config.dtype, device=env.config.device)
    ell_grid = torch.linspace(theta[t, 1] - radius, theta[t, 1] + radius, grid_size, dtype=env.config.dtype, device=env.config.device)
    rows: List[Dict[str, Any]] = []
    for k in k_grid:
        for ell in ell_grid:
            candidate = theta.clone()
            candidate[t, 0] = k
            candidate[t, 1] = ell
            cost, grad = env.exact_gradient(candidate, lambda_=lambda_value)
            rows.append(
                {
                    "time": t,
                    "lambda": lambda_value,
                    "theta0": float(k.item()),
                    "theta1": float(ell.item()),
                    "cost": float(cost.item()),
                    "grad0": float(grad[t, 0].item()),
                    "grad1": float(grad[t, 1].item()),
                }
            )
    return rows



def _portfolio_frontier_rows(env: Any, control: torch.Tensor, evaluation_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    scales = torch.as_tensor(
        evaluation_config.get("frontier_scales", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        dtype=env.config.dtype,
        device=env.config.device,
    )
    optimal = env.optimal_policy()
    rows: List[Dict[str, Any]] = []
    for scale in scales:
        theta = control.detach() + scale * (optimal - control.detach())
        means, variances = env.exact_moments(theta, lambda_=0.0)
        objective = env.exact_objective(theta, lambda_=0.0)
        terminal_mean = means[-1]
        terminal_variance = variances[-1]
        rows.append(
            {
                "scale_to_oracle": float(scale.item()),
                "terminal_mean": float(terminal_mean.item()),
                "terminal_variance": float(terminal_variance.item()),
                "objective": float(objective.item()),
                "frontier_distance_proxy": float(torch.linalg.norm(theta - optimal).item()),
            }
        )
    return rows


__all__: list[str] = []
