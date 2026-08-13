from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch

from ..core.controls import initialize_control, load_control
from ..core.registry import build_environment
from ..core.session import load_checkpoint


def _load_or_initialize(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Optional[Mapping[str, Any]]]:
    checkpoint = config.get("checkpoint")
    if checkpoint:
        restored = load_checkpoint(checkpoint)
        payload = restored["payload"]
        checkpoint_config = {
            "env": payload["env"],
            "algorithm": payload["algorithm"],
            "env_config": payload["env_config"],
            "algorithm_config": payload.get("algorithm_config", {}),
            "train": payload.get("train_config", {}),
            "evaluation": payload.get("evaluation_config", {}),
        }
        merged = {**checkpoint_config, **dict(config)}
        merged["env_config"] = {**checkpoint_config.get("env_config", {}), **dict(config.get("env_config", {}))}
        spec, env = build_environment(merged)
        control = load_control(spec, env, payload["control"], trainable=False)
        return spec, env, control, payload
    spec, env = build_environment(config)
    control = initialize_control(spec, env)
    return spec, env, control, None



def _finite_time_rows(env_name: str, env: Any, control: Any, law_idx: int, t: int, mu: torch.Tensor) -> List[Dict[str, Any]]:
    row: Dict[str, Any] = {"env": env_name, "law_index": law_idx, "time": t}
    if env_name == "twostate":
        row["mass_state_1"] = float(mu[1].item())
        row["target_state_1"] = float(env.target_B[1].item())
        row["target_abs_error"] = abs(row["mass_state_1"] - row["target_state_1"])
    elif env_name == "cybersecurity":
        c = env.config
        row["infected_fraction"] = float((mu[c.DI] + mu[c.UI]).item())
        row["defended_fraction"] = float((mu[c.DI] + mu[c.DS]).item())
        row["running_reward"] = float((mu * env.reward_by_state).sum().item())
        if t < getattr(c, "T_val", t + 1):
            pi = env.action_probabilities(control, t, mu)
            row["update_rate"] = float((mu * pi[..., c.UPDATE]).sum().item())
    elif env_name == "distribution-planning":
        diff = mu - env.target
        row["target_l1"] = float(diff.abs().sum().item())
        row["target_l2"] = float(torch.linalg.norm(diff).item())
        cdf_diff = torch.cumsum(diff, dim=0)
        row["target_w1_ring_proxy"] = float(cdf_diff.abs().sum().item())
        if t < env.config.T:
            pi = env.action_probabilities(control, t, mu)
            row["movement_cost"] = float((mu.unsqueeze(-1) * pi * env.action_costs.to(dtype=mu.dtype)).sum().item())
    elif env_name == "advertising":
        c = env.config
        row["customer_fraction"] = float(mu[c.CUSTOMER].item())
        if t < env.config.T:
            ad_rate = env.advertising_rate(control, t, mu)
            row["advertising_rate"] = float(ad_rate.item())
            row["advertising_cost"] = float((env.config.c_ad * ad_rate).item())
            row["population_gain"] = float(mu[c.CUSTOMER].item())
    return [row]



def _finite_policy_rows(env_name: str, env: Any, control: Any, law_idx: int, t: int, mu: torch.Tensor) -> List[Dict[str, Any]]:
    pi = env.action_probabilities(control, t, mu).detach()
    rows: List[Dict[str, Any]] = []
    if pi.ndim == 3:
        pi = pi[0]
    for state in range(env.n_states):
        for action in range(env.n_actions):
            rows.append(
                {
                    "env": env_name,
                    "law_index": law_idx,
                    "time": t,
                    "state": state,
                    "action": action,
                    "probability": float(pi[state, action].item()),
                }
            )
    return rows



def _randomize_trainable_control(control: Any, std: float) -> None:
    if std <= 0.0:
        return
    with torch.no_grad():
        if isinstance(control, torch.nn.Module):
            for parameter in control.parameters():
                parameter.add_(std * torch.randn_like(parameter))
        elif isinstance(control, torch.Tensor):
            control.add_(std * torch.randn_like(control))



def _finite_population_rows_from_flow(n_states: int, flow: torch.Tensor, *, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in range(flow.shape[0]):
        for state in range(n_states):
            rows.append(
                {
                    "law_index": 0,
                    "time": t,
                    "state": state,
                    "mass": float(flow[t, state].item()),
                    "method": method,
                }
            )
    return rows



def _finite_time_rows_from_flow(env_name: str, env: Any, control: Any, flow: torch.Tensor, *, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t, mu in enumerate(flow):
        for row in _finite_time_rows(env_name, env, control, 0, t, mu):
            row["method"] = method
            rows.append(row)
    return rows



def _finite_policy_rows_from_flow(env_name: str, env: Any, control: Any, flow: torch.Tensor, *, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    horizon = flow.shape[0] - 1
    for t in range(horizon):
        for row in _finite_policy_rows(env_name, env, control, 0, t, flow[t]):
            row["method"] = method
            rows.append(row)
    return rows



def _reference_policy_rows_from_matrix(env_name: str, policy: torch.Tensor, horizon: int, *, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in range(horizon):
        for state in range(policy.shape[0]):
            for action in range(policy.shape[1]):
                rows.append(
                    {
                        "env": env_name,
                        "law_index": 0,
                        "time": t,
                        "state": state,
                        "action": action,
                        "probability": float(policy[state, action].item()),
                        "method": method,
                    }
                )
    return rows



def _terminal_sample_rows(samples: torch.Tensor, name: str) -> List[Dict[str, Any]]:
    samples = samples.detach().cpu()
    levels = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=samples.dtype, device=samples.device)
    quantiles = torch.quantile(samples, levels)
    rows = [{"stat": "mean", "name": name, "value": float(samples.mean().item())}]
    rows.append({"stat": "std", "name": name, "value": float(samples.std(unbiased=samples.numel() > 1).item())})
    for q, value in zip([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], quantiles):
        rows.append({"stat": f"q{q:g}", "name": name, "value": float(value.item())})
    return rows



def _snapshot_times(length: int) -> List[int]:
    if length <= 1:
        return [0]
    candidates = [0, max(0, length // 3), max(0, 2 * length // 3), length - 1]
    return sorted(set(candidates))


__all__: list[str] = []
