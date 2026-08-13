from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch

from mfc.algorithms import ContinuousTransportMFREINFORCE, SimplexPerturbedMFREINFORCE

from ..core.controls import control_vector, initialize_control
from .common import _as_list, _float_list, _save_diagnostic_result
from ..core.evaluation import finite_population_flow
from ..core.registry import EnvironmentSpec, build_environment, require_algorithm_name, require_env_name, validate_compatibility
from ..core.runtime import _aux_batch, _lambda_value, _training_horizon, sample_initial_laws
from ..core.session import RunResult, normalize_experiment_config, set_seed


def run_sensitivity_diagnostic(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    spec, env = build_environment(config)
    algorithm_name = require_algorithm_name(config)
    validate_compatibility(require_env_name(config), algorithm_name)
    train_config = dict(config.get("train", {}))
    algorithm_config = dict(config.get("algorithm_config", {}))
    diagnostic = dict(config.get("diagnostic", {}))
    seed = int(train_config.get("seed", 0))
    set_seed(seed, env.config.device)
    control = initialize_control(spec, env)
    horizon = _training_horizon(env, train_config)
    n_aux = _aux_batch(env, train_config, algorithm_config)
    etas = _float_list(diagnostic.get("etas", [algorithm_config.get("eta", _lambda_value(algorithm_config, train_config))]))
    replications = int(diagnostic.get("replications", 8))
    methods = [str(method) for method in _as_list(diagnostic.get("methods", ["auxiliary"]))]
    rows = []
    sample_rows: List[Dict[str, Any]] = []

    if spec.family == "finite":
        algorithm = SimplexPerturbedMFREINFORCE(env, algorithm_config)
        mu0 = sample_initial_laws(spec, env, 1, train_config)[0]
        mu_flow = finite_population_flow(env, algorithm, control, mu0, horizon, "exact", n_aux)
        oracle = oracle_sensitivity(spec, env, control, mu0, horizon)
        oracle_metadata = {"oracle_kind": "exact_population_ad" if oracle is not None else "unavailable"}
        for method in methods:
            for eta in etas:
                estimates = sensitivity_method_estimates(
                    method,
                    spec,
                    env,
                    control,
                    algorithm,
                    mu_flow,
                    oracle,
                    eta,
                    n_aux,
                    replications,
                    seed,
                    diagnostic,
                )
                if estimates is None:
                    rows.append(unavailable_sensitivity_row(method, eta))
                    continue
                stacked = torch.stack(estimates)
                summary_rows = sensitivity_summary_rows(eta, stacked, oracle, method=method)
                for row in summary_rows:
                    row.update(oracle_metadata)
                rows.extend(summary_rows)
                sample_rows.extend(sensitivity_sample_rows(method, eta, stacked, oracle))
        return _save_diagnostic_result(
            "diagnose-sensitivity",
            config,
            rows,
            {"rows": len(rows), "sample_rows": len(sample_rows)},
            extra_tables={"sensitivity_samples.csv": sample_rows},
        )

    if algorithm_name != "continuous-mfreinforce":
        raise ValueError("Continuous sensitivity diagnostics require algorithm='continuous-mfreinforce'.")

    algorithm = ContinuousTransportMFREINFORCE(env, algorithm_config)
    population_particles = int(
        train_config.get(
            "population_particles",
            algorithm_config.get("population_particles", train_config.get("particles", getattr(env.config, "N_pop", n_aux))),
        )
    )
    nominal = algorithm.estimate_coordinate_flow(
        control,
        horizon=horizon,
        particles=population_particles,
        seed=seed + 100_000,
        exploration=algorithm_config.get("coordinate_exploration"),
    )
    oracle, oracle_metadata = oracle_continuous_sensitivity(
        spec,
        env,
        control,
        horizon,
        train_config,
        config.get("evaluation", {}),
        diagnostic,
        seed,
    )
    for method in methods:
        for eta in etas:
            estimates = continuous_sensitivity_method_estimates(
                method,
                spec,
                env,
                control,
                algorithm,
                nominal,
                oracle,
                eta,
                n_aux,
                replications,
                seed,
                algorithm_config,
                diagnostic,
            )
            if estimates is None:
                rows.append(unavailable_sensitivity_row(method, eta))
                continue
            stacked = torch.stack(estimates)
            summary_rows = sensitivity_summary_rows(eta, stacked, oracle, method=method)
            for row in summary_rows:
                row.update(oracle_metadata)
            rows.extend(summary_rows)
            sample_rows.extend(sensitivity_sample_rows(method, eta, stacked, oracle))
    return _save_diagnostic_result(
        "diagnose-sensitivity",
        config,
        rows,
        {"rows": len(rows), "sample_rows": len(sample_rows)},
        extra_tables={"sensitivity_samples.csv": sample_rows},
    )


def oracle_sensitivity(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    mu0: torch.Tensor,
    horizon: int,
) -> Optional[torch.Tensor]:
    if spec.family != "finite":
        return None
    param_dim = control_vector(control).numel()
    if isinstance(control, torch.nn.Module):
        params = tuple(control.parameters())
        flow = env.exact_population_flow(control, mu0, horizon)
        rows = []
        for t in range(horizon + 1):
            components = []
            for state in range(env.n_states - 1):
                if t == 0 or not flow[t, state].requires_grad:
                    components.append(torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device))
                    continue
                grads = torch.autograd.grad(flow[t, state], params, retain_graph=True, allow_unused=True)
                flat = [
                    torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
                    for parameter, grad in zip(params, grads)
                ]
                components.append(torch.cat(flat))
            rows.append(torch.stack(components))
        return torch.stack(rows).detach()
    theta = control.detach().clone().requires_grad_(True)
    flow = env.exact_population_flow(theta, mu0, horizon)
    rows = []
    for t in range(horizon + 1):
        components = []
        for state in range(env.n_states - 1):
            if t == 0 or not flow[t, state].requires_grad:
                components.append(torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device))
                continue
            grad = torch.autograd.grad(flow[t, state], theta, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(theta)
            components.append(grad.reshape(-1))
        rows.append(torch.stack(components))
    return torch.stack(rows).detach()


def oracle_continuous_sensitivity(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    horizon: int,
    train_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    seed: int,
) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if spec.family == "pathwise" and isinstance(control, torch.nn.Module):
        default_particles = int(
            train_config.get(
                "population_particles",
                train_config.get("particles", getattr(env.config, "N_val", getattr(env.config, "N_pop", 32))),
            )
        )
        particles = int(
            diagnostic_config.get(
                "oracle_sensitivity_particles",
                evaluation_config.get("oracle_sensitivity_particles", max(1, default_particles)),
            )
        )
        lambda_value = float(
            diagnostic_config.get("oracle_sensitivity_lambda", evaluation_config.get("oracle_sensitivity_lambda", 0.0))
        )
        exploration = bool(diagnostic_config.get("oracle_sensitivity_exploration", False))
        oracle_seed = int(diagnostic_config.get("oracle_sensitivity_seed", seed + 910_000))
        oracle = pathwise_coordinate_sensitivity(
            spec,
            env,
            control,
            horizon=horizon,
            particles=max(1, particles),
            seed=oracle_seed,
            lambda_=lambda_value,
            exploration=exploration,
        )
        if oracle is not None:
            return oracle, {
                "oracle_kind": "pathwise_ad_reference",
                "oracle_sensitivity_particles": max(1, particles),
                "oracle_sensitivity_lambda": lambda_value,
                "oracle_sensitivity_exploration": exploration,
            }
        return None, {"oracle_kind": "unavailable"}

    if spec.name not in {"lq", "portfolio"}:
        return None, {"oracle_kind": "unavailable"}
    if isinstance(control, torch.nn.Module):
        return None, {"oracle_kind": "unavailable"}

    theta = control.detach().clone().requires_grad_(True)
    if spec.name == "lq":
        coordinates = env.exact_moments(theta)[0][: horizon + 1]
        metadata = {"oracle_kind": "analytic_exact"}
    else:
        coordinates = env.exact_moments(theta, lambda_=0.0)[0][: horizon + 1]
        metadata = {"oracle_kind": "analytic_exact", "oracle_lambda": 0.0}
    param_dim = theta.numel()
    rows = []
    for t in range(horizon + 1):
        if t == 0 or not coordinates[t].requires_grad:
            rows.append(torch.zeros(1, param_dim, dtype=env.config.dtype, device=env.config.device))
            continue
        grad = torch.autograd.grad(coordinates[t], theta, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(theta)
        rows.append(grad.reshape(1, -1))
    return torch.stack(rows).detach(), metadata


def pathwise_coordinate_sensitivity(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.nn.Module,
    *,
    horizon: int,
    particles: int,
    seed: int,
    lambda_: float,
    exploration: bool,
) -> Optional[torch.Tensor]:
    if not hasattr(env, "_simulate_particles"):
        return None
    params = tuple(control.parameters())
    if not params:
        return None
    rollout = env._simulate_particles(  # noqa: SLF001 - intentional oracle path through differentiable simulator
        control,
        int(particles),
        seed=int(seed),
        lambda_=float(lambda_),
        horizon=int(horizon),
        exploration=bool(exploration),
    )
    if spec.name == "cucker-smale":
        coordinates = rollout["state_flow"].mean(dim=1)
    elif spec.name == "kuramoto":
        lifted = rollout["lifted_phase_flow"]
        coordinates = torch.stack([torch.cos(lifted).mean(dim=1), torch.sin(lifted).mean(dim=1)], dim=-1)
    else:
        return None

    param_dim = sum(parameter.numel() for parameter in params)
    rows = []
    for t in range(coordinates.shape[0]):
        components = []
        for coordinate in range(coordinates.shape[1]):
            if t == 0 or not coordinates[t, coordinate].requires_grad:
                components.append(torch.zeros(param_dim, dtype=env.config.dtype, device=env.config.device))
                continue
            grads = torch.autograd.grad(coordinates[t, coordinate], params, retain_graph=True, allow_unused=True)
            flat = [
                torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
                for parameter, grad in zip(params, grads)
            ]
            components.append(torch.cat(flat))
        rows.append(torch.stack(components))
    return torch.stack(rows).detach()


def sensitivity_summary_rows(
    eta: float,
    estimates: torch.Tensor,
    oracle: Optional[torch.Tensor],
    *,
    method: str = "auxiliary",
) -> List[Dict[str, Any]]:
    mean_estimate = estimates.mean(dim=0)
    variance = estimates.var(dim=0, unbiased=estimates.shape[0] > 1) if estimates.shape[0] > 1 else torch.zeros_like(mean_estimate)
    rows = []
    for t in range(mean_estimate.shape[0]):
        error = mean_estimate[t] - oracle[t] if oracle is not None else torch.full_like(mean_estimate[t], float("nan"))
        rows.append(
            {
                "method": method,
                "eta": eta,
                "time": t,
                "replications": int(estimates.shape[0]),
                "coordinate_dim": int(mean_estimate.shape[1]),
                "estimate_norm": float(torch.linalg.norm(mean_estimate[t]).item()),
                "variance_trace": float(variance[t].reshape(-1).sum().item()),
                "mse": float(error.square().mean().item()) if oracle is not None else float("nan"),
                "error_norm": float(torch.linalg.norm(error).item()) if oracle is not None else float("nan"),
            }
        )
    return rows


def sensitivity_method_estimates(
    method: str,
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    algorithm: Any,
    mu_flow: torch.Tensor,
    oracle: Optional[torch.Tensor],
    eta: float,
    n_aux: int,
    replications: int,
    seed: int,
    diagnostic: Mapping[str, Any],
) -> Optional[List[torch.Tensor]]:
    if method in {"oracle", "pathwise_ad", "ad"}:
        return [oracle.detach()] if oracle is not None else None
    if method in {"finite_difference", "common_random_finite_difference"}:
        fd = finite_difference_sensitivity(
            spec,
            env,
            control,
            horizon=mu_flow.shape[0] - 1,
            mu0=mu_flow[0],
            step=float(diagnostic.get("finite_difference_step", 1e-4)),
        )
        return [fd] if fd is not None else None
    local_n_aux = max(1, int(n_aux))
    if method in {"reused_main", "reuse_main"}:
        local_n_aux = max(local_n_aux, int(diagnostic.get("reused_main_n", local_n_aux)))
    if method in {"auxiliary", "independent_auxiliary", "reused_main", "reuse_main"}:
        estimates = []
        for rep in range(replications):
            set_seed(seed + rep, env.config.device)
            estimates.append(algorithm.estimate_sensitivity(control, mu_flow, eta, local_n_aux).detach())
        return estimates
    return None


def continuous_sensitivity_method_estimates(
    method: str,
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    algorithm: Any,
    nominal: Mapping[str, Any],
    oracle: Optional[torch.Tensor],
    eta: float,
    n_aux: int,
    replications: int,
    seed: int,
    algorithm_config: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> Optional[List[torch.Tensor]]:
    if method in {"oracle", "pathwise_ad", "ad"}:
        return [oracle.detach()] if oracle is not None else None
    if method in {"finite_difference", "common_random_finite_difference"}:
        fd = finite_difference_sensitivity(
            spec,
            env,
            control,
            horizon=int(nominal.get("horizon", nominal["coordinates"].shape[0] - 1)),
            mu0=None,
            step=float(diagnostic.get("finite_difference_step", 1e-4)),
        )
        return [fd] if fd is not None else None
    local_n_aux = max(1, int(n_aux))
    if method in {"reused_main", "reuse_main"}:
        local_n_aux = max(local_n_aux, int(diagnostic.get("reused_main_n", local_n_aux)))
    if method in {"auxiliary", "independent_auxiliary", "reused_main", "reuse_main"}:
        estimates = []
        for rep in range(replications):
            set_seed(seed + rep, env.config.device)
            estimates.append(
                algorithm.estimate_sensitivity(
                    control,
                    nominal,
                    eta,
                    local_n_aux,
                    seed=seed + rep,
                    baseline=algorithm_config.get("sensitivity_baseline", "nominal"),
                ).detach()
            )
        return estimates
    return None


def unavailable_sensitivity_row(method: str, eta: float) -> Dict[str, Any]:
    return {
        "method": method,
        "eta": eta,
        "time": "",
        "replications": 0,
        "coordinate_dim": "",
        "estimate_norm": float("nan"),
        "variance_trace": float("nan"),
        "mse": float("nan"),
        "error_norm": float("nan"),
        "status": "unavailable",
    }


def sensitivity_sample_rows(
    method: str,
    eta: float,
    estimates: torch.Tensor,
    oracle: Optional[torch.Tensor],
    *,
    max_rows: int = 1_000_000,
) -> List[Dict[str, Any]]:
    estimates = estimates.detach()
    rows: List[Dict[str, Any]] = []
    written = 0
    for replication in range(estimates.shape[0]):
        for time_idx in range(estimates.shape[1]):
            flat = estimates[replication, time_idx].reshape(-1)
            oracle_flat = oracle[time_idx].reshape(-1) if oracle is not None else None
            for coordinate, value in enumerate(flat):
                if written >= max_rows:
                    return rows
                row = {
                    "method": method,
                    "eta": eta,
                    "replication": replication,
                    "time": time_idx,
                    "coordinate": coordinate,
                    "estimate": float(value.item()),
                }
                if oracle_flat is not None:
                    oracle_value = float(oracle_flat[coordinate].item())
                    row["oracle"] = oracle_value
                    row["error"] = float(value.item()) - oracle_value
                rows.append(row)
                written += 1
    return rows


def finite_difference_sensitivity(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    *,
    horizon: int,
    mu0: Optional[torch.Tensor],
    step: float,
) -> Optional[torch.Tensor]:
    base_vector = control_vector(control).detach()
    if base_vector.numel() == 0:
        return None
    base_coordinates = deterministic_coordinate_flow(spec, env, control, horizon=horizon, mu0=mu0)
    if base_coordinates is None:
        return None
    columns = []
    for coordinate in range(base_vector.numel()):
        direction = torch.zeros_like(base_vector)
        direction[coordinate] = float(step)
        plus = control_with_vector(spec, env, control, base_vector + direction)
        minus = control_with_vector(spec, env, control, base_vector - direction)
        plus_flow = deterministic_coordinate_flow(spec, env, plus, horizon=horizon, mu0=mu0)
        minus_flow = deterministic_coordinate_flow(spec, env, minus, horizon=horizon, mu0=mu0)
        if plus_flow is None or minus_flow is None:
            return None
        columns.append(((plus_flow - minus_flow) / (2.0 * float(step))).reshape(horizon + 1, -1))
    return torch.stack(columns, dim=-1).detach()


def deterministic_coordinate_flow(
    spec: EnvironmentSpec,
    env: Any,
    control: torch.Tensor | torch.nn.Module,
    *,
    horizon: int,
    mu0: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    with torch.no_grad():
        if spec.family == "finite":
            if mu0 is None:
                return None
            flow = env.exact_population_flow(control, mu0, horizon)
            return flow[:, : max(1, env.n_states - 1)].detach()
        if spec.name == "lq":
            means = env.exact_moments(control)[0][: horizon + 1]
            return means.unsqueeze(-1).detach()
        if spec.name == "portfolio":
            means = env.exact_moments(control, lambda_=0.0)[0][: horizon + 1]
            return means.unsqueeze(-1).detach()
    return None


def control_with_vector(
    spec: EnvironmentSpec,
    env: Any,
    template: torch.Tensor | torch.nn.Module,
    vector: torch.Tensor,
) -> torch.Tensor | torch.nn.Module:
    if isinstance(template, torch.nn.Module):
        if spec.policy_cls is None:
            raise ValueError(f"{spec.name!r} has no policy class for vector reconstruction.")
        clone = spec.policy_cls(env.config)
        torch.nn.utils.vector_to_parameters(vector.to(device=env.config.device, dtype=env.config.dtype), clone.parameters())
        clone.eval()
        return clone
    return vector.to(device=env.config.device, dtype=env.config.dtype).reshape_as(template)


_continuous_sensitivity_method_estimates = continuous_sensitivity_method_estimates
_control_with_vector = control_with_vector
_deterministic_coordinate_flow = deterministic_coordinate_flow
_finite_difference_sensitivity = finite_difference_sensitivity
_oracle_continuous_sensitivity = oracle_continuous_sensitivity
_oracle_sensitivity = oracle_sensitivity
_pathwise_coordinate_sensitivity = pathwise_coordinate_sensitivity
_sensitivity_method_estimates = sensitivity_method_estimates
_sensitivity_sample_rows = sensitivity_sample_rows
_sensitivity_summary_rows = sensitivity_summary_rows
_unavailable_sensitivity_row = unavailable_sensitivity_row


__all__ = [
    "_continuous_sensitivity_method_estimates",
    "_control_with_vector",
    "_deterministic_coordinate_flow",
    "_finite_difference_sensitivity",
    "_oracle_continuous_sensitivity",
    "_oracle_sensitivity",
    "_pathwise_coordinate_sensitivity",
    "_sensitivity_method_estimates",
    "_sensitivity_sample_rows",
    "_sensitivity_summary_rows",
    "_unavailable_sensitivity_row",
    "continuous_sensitivity_method_estimates",
    "control_with_vector",
    "deterministic_coordinate_flow",
    "finite_difference_sensitivity",
    "oracle_continuous_sensitivity",
    "oracle_sensitivity",
    "pathwise_coordinate_sensitivity",
    "run_sensitivity_diagnostic",
    "sensitivity_method_estimates",
    "sensitivity_sample_rows",
    "sensitivity_summary_rows",
    "unavailable_sensitivity_row",
]
