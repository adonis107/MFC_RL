from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import torch

from ..core.controls import control_parameters, control_payload, initialize_control, load_control
from ..core.registry import environment_spec
from ..core.session import set_seed
from .common import (
    _finite_policy_rows_from_flow,
    _finite_population_rows_from_flow,
    _finite_time_rows_from_flow,
    _randomize_trainable_control,
    _reference_policy_rows_from_matrix,
)


def _twostate_reference_outputs(env: Any, mu0: torch.Tensor, horizon: int) -> Dict[str, Any]:
    optimal_policy = env.optimal_policy().detach().clamp(1e-12, 1.0 - 1e-12)
    theta = torch.logit(optimal_policy[:, 1])
    flow = env.exact_population_flow(theta, mu0, horizon).detach()
    value = env.exact_value(theta, mu0, horizon).detach()
    return {
        "outputs": {
            "reference_population_flow": _finite_population_rows_from_flow(env.n_states, flow, method="analytic_optimal_policy"),
            "reference_time_metrics": _finite_time_rows_from_flow(
                "twostate",
                env,
                theta,
                flow,
                method="analytic_optimal_policy",
            ),
            "reference_policy": _reference_policy_rows_from_matrix(
                "twostate",
                optimal_policy,
                horizon,
                method="analytic_optimal_policy",
            ),
        },
        "metrics": {
            "reference_kind": "analytic_optimal_policy",
            "reference_value": float(value.item()),
        },
    }



def _advertising_reference_outputs(
    env: Any,
    control: Any,
    mu0: torch.Tensor,
    horizon: int,
    evaluation_config: Mapping[str, Any],
) -> Dict[str, Any]:
    grid_size = int(evaluation_config.get("oracle_grid_size", evaluation_config.get("advertising_oracle_grid_size", 201)))
    action_grid_size = int(
        evaluation_config.get("oracle_action_grid_size", evaluation_config.get("advertising_oracle_action_grid_size", grid_size))
    )
    oracle = env.finite_horizon_dp_oracle(grid_size=max(2, grid_size), action_grid_size=max(2, action_grid_size))
    p_grid = oracle["p_grid"].detach()
    policy = oracle["policy"].detach()
    values = oracle["values"].detach()
    p = mu0[env.config.CUSTOMER].detach().clone()
    learned_value = env.exact_value(control, mu0, horizon).detach()
    oracle_value = env._interp_on_grid(p, p_grid, values[0]).detach()  # noqa: SLF001 - benchmark DP reference

    population_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    for t in range(horizon + 1):
        mu = torch.stack([1.0 - p, p]).detach()
        for state in range(env.n_states):
            population_rows.append(
                {
                    "law_index": 0,
                    "time": t,
                    "state": state,
                    "mass": float(mu[state].item()),
                    "method": "finite_horizon_dp_oracle",
                }
            )
        row: Dict[str, Any] = {
            "env": "advertising",
            "law_index": 0,
            "time": t,
            "customer_fraction": float(p.item()),
            "method": "finite_horizon_dp_oracle",
        }
        if t < horizon:
            policy_t = policy[min(t, policy.shape[0] - 1)]
            q = env._interp_on_grid(p, p_grid, policy_t).clamp(0.0, 1.0)  # noqa: SLF001
            row["advertising_rate"] = float(q.item())
            row["advertising_cost"] = float((env.config.c_ad * q).item())
            row["population_gain"] = float(p.item())
            p = p + q * torch.minimum(torch.as_tensor(env.config.kappa_ad, dtype=p.dtype, device=p.device), 1.0 - p)
        time_rows.append(row)

    max_policy_rows = int(evaluation_config.get("oracle_policy_max_rows", 5000))
    stride = max(1, int(math.ceil((policy.numel()) / max(1, max_policy_rows))))
    infinite_reference = env.infinite_horizon_reference_policy(p_grid).detach()
    policy_rows: List[Dict[str, Any]] = []
    written = 0
    for t in range(policy.shape[0]):
        for grid_idx in range(0, p_grid.numel(), stride):
            policy_rows.append(
                {
                    "env": "advertising",
                    "time": t,
                    "grid_index": grid_idx,
                    "customer_fraction": float(p_grid[grid_idx].item()),
                    "oracle_ad_probability": float(policy[t, grid_idx].item()),
                    "infinite_horizon_ad_probability": float(infinite_reference[grid_idx].item()),
                    "method": "finite_horizon_dp_oracle",
                }
            )
            written += 1
            if written >= max_policy_rows:
                break
        if written >= max_policy_rows:
            break

    return {
        "outputs": {
            "reference_population_flow": population_rows,
            "reference_time_metrics": time_rows,
            "reference_policy": policy_rows,
        },
        "metrics": {
            "reference_kind": "finite_horizon_dp_oracle",
            "reference_value": float(oracle_value.item()),
            "reference_value_gap": float((oracle_value - learned_value).item()),
            "oracle_grid_size": max(2, grid_size),
            "oracle_action_grid_size": max(2, action_grid_size),
        },
    }



def _finite_model_based_reference_outputs(
    env_name: str,
    env: Any,
    learned_control: Any,
    mu0: torch.Tensor,
    horizon: int,
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    steps = int(evaluation_config.get("oracle_steps", evaluation_config.get("reference_steps", 0)))
    if steps <= 0:
        return {"outputs": {}, "metrics": {"reference_kind": "not_configured"}}
    spec = environment_spec(env_name)
    restarts = max(1, int(evaluation_config.get("oracle_restarts", 1)))
    lr = float(evaluation_config.get("oracle_lr", 5e-2))
    init_std = float(evaluation_config.get("oracle_init_std", 0.05))
    best_payload: Optional[Mapping[str, Any]] = None
    best_value = torch.tensor(float("-inf"), dtype=env.config.dtype, device=env.config.device)
    for restart in range(restarts):
        set_seed(seed + 700_000 + restart, env.config.device)
        candidate = initialize_control(spec, env)
        _randomize_trainable_control(candidate, init_std)
        params = list(control_parameters(candidate))
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            value = env.exact_value(candidate, mu0, horizon)
            (-value).backward()
            optimizer.step()
        with torch.no_grad():
            value = env.exact_value(candidate, mu0, horizon).detach()
        if value > best_value:
            best_value = value
            best_payload = control_payload(candidate)
    if best_payload is None:
        return {"outputs": {}, "metrics": {"reference_kind": "unavailable"}}

    reference_control = load_control(spec, env, best_payload, trainable=False)
    flow = env.exact_population_flow(reference_control, mu0, horizon).detach()
    learned_value = env.exact_value(learned_control, mu0, horizon).detach()
    return {
        "outputs": {
            "reference_population_flow": _finite_population_rows_from_flow(
                env.n_states,
                flow,
                method="model_based_exact_flow_reference",
            ),
            "reference_time_metrics": _finite_time_rows_from_flow(
                env_name,
                env,
                reference_control,
                flow,
                method="model_based_exact_flow_reference",
            ),
            "reference_policy": _finite_policy_rows_from_flow(
                env_name,
                env,
                reference_control,
                flow,
                method="model_based_exact_flow_reference",
            ),
        },
        "metrics": {
            "reference_kind": "model_based_exact_flow_reference",
            "reference_value": float(best_value.item()),
            "reference_value_gap": float((best_value - learned_value).item()),
            "reference_steps": steps,
            "reference_restarts": restarts,
        },
    }


__all__: list[str] = []
