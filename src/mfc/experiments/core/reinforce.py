from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional

import torch

from .controls import control_vector
from .registry import EnvironmentSpec
from .runtime import _baseline, _main_batch, _training_horizon


Baseline = None | float | Literal["batch_mean"]


def reinforce_gradient_step(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
    *,
    mu0: Optional[torch.Tensor] = None,
    mu_flow: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Classical policy-score REINFORCE without mean-field score terms."""
    if spec.family == "finite":
        if mu0 is None or mu_flow is None:
            raise ValueError("Finite REINFORCE requires mu0 and mu_flow.")
        return finite_reinforce_gradient(env, control, mu0, mu_flow, algorithm_config, train_config)
    if spec.name in {"lq", "portfolio"}:
        return exact_state_reinforce_gradient(spec, env, control, algorithm_config, train_config, iteration)
    if spec.family == "pathwise":
        if not isinstance(control, torch.nn.Module):
            raise ValueError("Pathwise REINFORCE requires a neural policy module.")
        return particle_reinforce_gradient(spec, env, control, algorithm_config, train_config, iteration)
    raise ValueError(f"Unsupported REINFORCE environment {spec.name!r}.")


def finite_reinforce_gradient(
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    mu0: torch.Tensor,
    mu_flow: torch.Tensor,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    horizon = mu_flow.shape[0] - 1
    batch = _main_batch(env, train_config, algorithm_config)
    param_dim = control_vector(control).numel()
    states = torch.empty(batch, horizon + 1, dtype=torch.long, device=env.config.device)
    actions = torch.empty(batch, horizon, dtype=torch.long, device=env.config.device)
    stage_returns = torch.zeros(batch, horizon, dtype=env.config.dtype, device=env.config.device)

    states[:, 0] = torch.multinomial(mu0, num_samples=batch, replacement=True)
    for t in range(horizon):
        law = mu_flow[t]
        states_t = states[:, t]
        actions_t = env.sample_actions_batch(control, t, states_t, law)
        actions[:, t] = actions_t
        stage_returns[:, t] = _discount(env, t) * env.reward_batch(states_t, law, actions_t)
        states[:, t + 1] = env.sample_next_states_batch(states_t, actions_t, law)
    returns_to_go = torch.zeros(batch, horizon + 1, dtype=env.config.dtype, device=env.config.device)
    returns_to_go[:, horizon] = _discount(env, horizon) * env.terminal_reward_batch(states[:, horizon], mu_flow[horizon])
    for t in range(horizon - 1, -1, -1):
        returns_to_go[:, t] = stage_returns[:, t] + returns_to_go[:, t + 1]

    centered_returns = _center_returns_to_go(returns_to_go, _baseline(algorithm_config))
    grad_flat = torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device)
    for t in range(horizon):
        grad_flat = grad_flat + _weighted_policy_score_sums(
            env,
            control,
            t,
            mu_flow[t],
            states[:, t],
            actions[:, t],
            centered_returns[:, t],
        )
    grad_flat = grad_flat / batch
    returns = returns_to_go[:, 0]
    objective = returns.mean()
    return objective, _format_gradient(control, grad_flat), _diag(objective, grad_flat, returns, objective_kind="reward")


def exact_state_reinforce_gradient(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    batch = _main_batch(env, train_config, algorithm_config)
    seed = int(train_config.get("seed", 0)) + int(iteration)
    theta = control.detach()
    param_dim = theta.numel()

    if spec.name == "lq":
        mean_flow = env.exact_moments(theta)[0].detach()
        sample = env.sample_trajectories(theta, batch, seed=seed, frozen_mean_flow=mean_flow)
        signal = sample["stage_costs"].sum(dim=1) + sample["terminal_costs"]
        weights = _center_signal(signal, _baseline(algorithm_config))
        grad_flat = torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device)
        for t in range(env.config.T):
            scores = env.policy_score_batch(
                theta,
                t,
                sample["states"][:, t],
                sample["law_means"][:, t],
                sample["actions"][:, t],
                action_means=sample["action_means"][:, t],
            ).reshape(batch, param_dim)
            grad_flat = grad_flat + weights @ scores
        grad_flat = grad_flat / batch
        objective = signal.mean()
        return objective, grad_flat.reshape_as(control), _diag(objective, grad_flat, -signal, objective_kind="cost")

    lambda_value = float(algorithm_config.get("lambda", algorithm_config.get("lambda_", train_config.get("lambda", 0.0))))
    mean_flow = env.exact_moments(theta, lambda_=lambda_value)[0].detach()
    sample = env.sample_trajectories(theta, batch, seed=seed, lambda_=lambda_value, frozen_mean_flow=mean_flow)
    signal = sample["returns"]
    weights = _center_signal(signal, _baseline(algorithm_config))
    grad_flat = torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device)
    for t in range(env.config.T):
        scores = env.policy_score_batch(
            theta,
            t,
            sample["states"][:, t],
            sample["perturbed_law_means"][:, t],
            sample["actions"][:, t],
            action_means=sample["action_means"][:, t],
        ).reshape(batch, param_dim)
        grad_flat = grad_flat + weights @ scores
    grad_flat = grad_flat / batch
    objective = signal.mean()
    return objective, grad_flat.reshape_as(control), _diag(objective, grad_flat, signal, objective_kind="reward")


def particle_reinforce_gradient(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.nn.Module,
    algorithm_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    batch = _main_batch(env, train_config, algorithm_config)
    horizon = _training_horizon(env, train_config)
    seed = int(train_config.get("seed", 0)) + int(iteration)
    rollout = env.sample_trajectories(
        control,
        batch,
        seed=seed,
        lambda_=0.0,
        horizon=horizon,
        exploration=True,
    )
    signal = rollout["particle_costs"]
    weights = _center_signal(signal, _baseline(algorithm_config))
    param_dim = control_vector(control).numel()
    grad_flat = torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device)

    for t in range(horizon):
        if spec.name == "cucker-smale":
            grad_flat = grad_flat + env.weighted_policy_score_sums(
                control,
                t,
                rollout["state_flow"][t],
                rollout["state_flow"][t],
                rollout["actions"][:, t],
                weights,
            )
        elif spec.name == "kuramoto":
            grad_flat = grad_flat + env.weighted_policy_score_sums(
                control,
                t,
                rollout["lifted_phase_flow"][t],
                rollout["lifted_phase_flow"][t],
                rollout["actions"][:, t],
                weights,
                frequencies=rollout.get("frequencies"),
            )
        else:
            raise ValueError(f"Unsupported pathwise REINFORCE environment {spec.name!r}.")

    grad_flat = grad_flat / batch
    objective = signal.mean()
    return objective, grad_flat, _diag(objective, grad_flat, -signal, objective_kind="cost")


def _weighted_policy_score_sums(
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    t: int,
    law: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if hasattr(env, "weighted_policy_score_sums"):
        return env.weighted_policy_score_sums(control, t, law, states, actions, weights).reshape(-1)
    scores = env.policy_scores_batch(control, t, law, states, actions).reshape(states.numel(), -1)
    return (weights.reshape(1, -1) @ scores).reshape(-1)


def _center_signal(signal: torch.Tensor, baseline: Baseline) -> torch.Tensor:
    if baseline == "batch_mean":
        return signal - signal.mean()
    if baseline is None:
        return signal
    return signal - torch.as_tensor(float(baseline), dtype=signal.dtype, device=signal.device)


def _center_returns_to_go(returns_to_go: torch.Tensor, baseline: Baseline) -> torch.Tensor:
    if baseline == "batch_mean":
        return returns_to_go - returns_to_go[:, 0].mean()
    if baseline is None:
        return returns_to_go
    return returns_to_go - torch.as_tensor(float(baseline), dtype=returns_to_go.dtype, device=returns_to_go.device)


def _discount(env: Any, t: int) -> float:
    return float(getattr(env.config, "gamma", 1.0) ** t)


def _format_gradient(control: torch.Tensor | torch.nn.Module, grad_flat: torch.Tensor) -> torch.Tensor:
    if isinstance(control, torch.nn.Module):
        return grad_flat
    return grad_flat.reshape_as(control)


def _diag(
    objective: torch.Tensor,
    grad_flat: torch.Tensor,
    signal: torch.Tensor,
    *,
    objective_kind: Literal["cost", "reward"],
) -> Dict[str, torch.Tensor]:
    if objective_kind == "reward":
        mean_return = signal.mean()
        std_return = signal.std(unbiased=False)
    else:
        mean_return = (-signal).mean()
        std_return = (-signal).std(unbiased=False)
    return {
        "objective": objective.detach(),
        "mean_return": mean_return.detach(),
        "std_return": std_return.detach(),
        "grad_norm": torch.linalg.norm(grad_flat.detach()),
    }


__all__ = [
    "exact_state_reinforce_gradient",
    "finite_reinforce_gradient",
    "particle_reinforce_gradient",
    "reinforce_gradient_step",
]
