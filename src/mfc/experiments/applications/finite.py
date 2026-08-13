from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from ..core.evaluation import finite_population_flow
from ..core.gradient_steps import make_algorithm
from ..core.registry import environment_spec, require_algorithm_name
from ..core.runtime import _validation_horizon, validation_laws
from .common import _finite_policy_rows, _finite_time_rows
from .references import (
    _advertising_reference_outputs,
    _finite_model_based_reference_outputs,
    _twostate_reference_outputs,
)


def _finite_application_outputs(
    env_name: str,
    env: Any,
    control: Any,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    spec = environment_spec(env_name)
    algorithm_name = require_algorithm_name({"env": env_name, "algorithm": "simplex"})
    algorithm = make_algorithm(algorithm_name, env, {})
    horizon = _validation_horizon(env, train_config, evaluation_config)
    laws = validation_laws(spec, env, evaluation_config)
    if laws is None:
        raise ValueError("Finite application diagnostics require validation laws.")
    law_batch = laws.unsqueeze(0) if laws.ndim == 1 else laws
    flows = []
    for law_idx, mu0 in enumerate(law_batch):
        flow = finite_population_flow(env, algorithm, control, mu0, horizon, "exact", max(1, env.n_states))
        flows.append((law_idx, flow))

    population_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    for law_idx, flow in flows:
        for t in range(flow.shape[0]):
            mu = flow[t]
            for state in range(env.n_states):
                population_rows.append({"law_index": law_idx, "time": t, "state": state, "mass": float(mu[state].item())})
            time_rows.extend(_finite_time_rows(env_name, env, control, law_idx, t, mu))
            if t < horizon:
                policy_rows.extend(_finite_policy_rows(env_name, env, control, law_idx, t, mu))

    values = torch.stack([env.exact_value(control, mu0, horizon).detach() for mu0 in law_batch])
    metrics = {
        "value_mean": float(values.mean().item()),
        "value_std": float(values.std(unbiased=law_batch.shape[0] > 1).item()) if law_batch.shape[0] > 1 else 0.0,
        "law_count": int(law_batch.shape[0]),
        "horizon": horizon,
    }
    outputs: Dict[str, Any] = {
        "population_flow": population_rows,
        "time_metrics": time_rows,
        "policy": policy_rows,
        "metrics": metrics,
    }
    if env_name == "twostate":
        outputs["landscape"] = _twostate_landscape_rows(env, evaluation_config)
        reference = _twostate_reference_outputs(env, law_batch[0], horizon)
        outputs.update(reference["outputs"])
        metrics.update(reference["metrics"])
    elif env_name == "distribution-planning":
        outputs["transport_flux"] = _distribution_flux_rows(env, control, flows)
        reference = _finite_model_based_reference_outputs(env_name, env, control, law_batch[0], horizon, evaluation_config, seed)
        outputs.update(reference["outputs"])
        metrics.update(reference["metrics"])
    elif env_name == "advertising":
        outputs["finite_population"] = _advertising_finite_population_rows(env, control, law_batch[0], horizon, evaluation_config)
        reference = _advertising_reference_outputs(env, control, law_batch[0], horizon, evaluation_config)
        outputs.update(reference["outputs"])
        metrics.update(reference["metrics"])
    elif env_name == "cybersecurity":
        reference = _finite_model_based_reference_outputs(env_name, env, control, law_batch[0], horizon, evaluation_config, seed)
        outputs.update(reference["outputs"])
        metrics.update(reference["metrics"])
    if "reference_value" in metrics and "value_mean" in metrics and "reference_value_gap" not in metrics:
        metrics["reference_value_gap"] = float(metrics["reference_value"] - metrics["value_mean"])
    return outputs



def _twostate_landscape_rows(env: Any, evaluation_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mu0 = torch.as_tensor(evaluation_config.get("mu0", [0.2, 0.8]), dtype=env.config.dtype, device=env.config.device)
    horizon = int(evaluation_config.get("horizon", env.config.T))
    low = float(evaluation_config.get("theta_min", -4.0))
    high = float(evaluation_config.get("theta_max", 4.0))
    grid_size = int(evaluation_config.get("landscape_grid_size", 21))
    theta0_grid = torch.linspace(low, high, grid_size, dtype=env.config.dtype, device=env.config.device)
    theta1_grid = torch.linspace(low, high, grid_size, dtype=env.config.dtype, device=env.config.device)
    rows: List[Dict[str, Any]] = []
    for theta0 in theta0_grid:
        for theta1 in theta1_grid:
            theta = torch.stack([theta0, theta1]).detach().clone().requires_grad_(True)
            value = env.exact_value(theta, mu0, horizon)
            grad = torch.autograd.grad(value, theta)[0]
            rows.append(
                {
                    "theta0": float(theta0.item()),
                    "theta1": float(theta1.item()),
                    "value": float(value.detach().item()),
                    "grad0": float(grad[0].detach().item()),
                    "grad1": float(grad[1].detach().item()),
                }
            )
    return rows



def _distribution_flux_rows(env: Any, control: Any, flows: List[tuple[int, torch.Tensor]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for law_idx, flow in flows:
            for t in range(flow.shape[0] - 1):
                mu = flow[t]
                pi = env.action_probabilities(control, t, mu)
                for state in range(env.n_states):
                    for action_index, move in enumerate(env.actions.tolist()):
                        next_state = int((state + move) % env.n_states)
                        rows.append(
                            {
                                "law_index": law_idx,
                                "time": t,
                                "from_state": state,
                                "to_state": next_state,
                                "action": int(move),
                                "probability": float(pi[state, action_index].item()),
                                "mass_flux": float((mu[state] * pi[state, action_index]).item()),
                            }
                        )
    return rows



def _advertising_finite_population_rows(
    env: Any,
    control: Any,
    mu0: torch.Tensor,
    horizon: int,
    evaluation_config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    particles_values = [int(value) for value in evaluation_config.get("finite_population_particles", [50, 100, 500])]
    replications = int(evaluation_config.get("finite_population_replications", 8))
    rows: List[Dict[str, Any]] = []
    exact = env.exact_population_flow(control, mu0, horizon)
    for n_particles in particles_values:
        for replication in range(replications):
            generator = torch.Generator(device=env.config.device)
            generator.manual_seed(int(evaluation_config.get("seed", 0)) + 10_000 * n_particles + replication)
            states = torch.multinomial(mu0, n_particles, replacement=True, generator=generator)
            for t in range(horizon + 1):
                counts = torch.bincount(states, minlength=env.n_states).to(dtype=env.config.dtype, device=env.config.device)
                empirical = counts / float(n_particles)
                rows.append(
                    {
                        "particles": n_particles,
                        "replication": replication,
                        "time": t,
                        "customer_fraction": float(empirical[env.config.CUSTOMER].item()),
                        "exact_customer_fraction": float(exact[t, env.config.CUSTOMER].item()),
                        "abs_error": float((empirical - exact[t]).abs().sum().item()),
                    }
                )
                if t == horizon:
                    break
                pi = env.action_probabilities(control, t, empirical)
                actions = torch.empty_like(states)
                next_states = torch.empty_like(states)
                for state in range(env.n_states):
                    mask = states == state
                    if not bool(mask.any()):
                        continue
                    actions[mask] = torch.multinomial(pi[state], int(mask.sum().item()), replacement=True, generator=generator)
                    for idx in torch.where(mask)[0]:
                        probs = env.transition_probs(int(states[idx].item()), int(actions[idx].item()), empirical)
                        next_states[idx] = torch.multinomial(probs, 1, replacement=True, generator=generator)
                states = next_states
    return rows


__all__: list[str] = []
