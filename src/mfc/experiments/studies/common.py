from __future__ import annotations

import copy
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch

from ..core.artifacts import _json_default, _write_csv
from ..core.controls import control_vector
from ..core.registry import DEFAULT_DEVICE
from ..core.runtime import _validation_horizon, validation_laws
from ..core.session import RunResult


StudyFunction = Callable[[Mapping[str, Any]], RunResult]


def _with_default_grid(config: Mapping[str, Any], key: str, values: Any) -> Dict[str, Any]:
    child = copy.deepcopy(dict(config))
    child.setdefault("study", {}).setdefault("parameters", {key: list(values)})
    return child



def _parameter_product(parameters: Mapping[str, Sequence[Any]]) -> List[tuple[Any, ...]]:
    import itertools

    return list(itertools.product(*[list(values) for values in parameters.values()]))



def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))



def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)



def _control_distance(control: Any, reference: Any) -> float:
    current = control_vector(control).reshape(-1)
    base = control_vector(reference).reshape(-1).to(device=current.device, dtype=current.dtype)
    if current.numel() != base.numel():
        return float("nan")
    return float(torch.linalg.norm(current - base).item())



def _policy_output_distance(spec: Any, env: Any, control: Any, reference: Any, config: Mapping[str, Any]) -> float:
    try:
        if spec.family == "finite":
            laws = validation_laws(spec, env, dict(config.get("evaluation", {})))
            if laws is None:
                return float("nan")
            law_batch = laws.unsqueeze(0) if laws.ndim == 1 else laws
            horizon = _validation_horizon(env, dict(config.get("train", {})), dict(config.get("evaluation", {})))
            values = []
            with torch.no_grad():
                for mu0 in law_batch:
                    flow = env.exact_population_flow(control, mu0, horizon)
                    ref_flow = env.exact_population_flow(reference, mu0, horizon)
                    for t in range(horizon):
                        pi = env.action_probabilities(control, t, flow[t])
                        ref_pi = env.action_probabilities(reference, t, ref_flow[t])
                        values.append((pi - ref_pi).reshape(-1))
            return float(torch.linalg.norm(torch.cat(values)).item()) if values else float("nan")
        return _control_distance(control, reference)
    except Exception:
        return float("nan")



def _trajectory_distance(spec: Any, env: Any, control: Any, reference: Any, config: Mapping[str, Any]) -> float:
    try:
        train_config = dict(config.get("train", {}))
        evaluation_config = dict(config.get("evaluation", {}))
        horizon = _validation_horizon(env, train_config, evaluation_config)
        if spec.family == "finite":
            laws = validation_laws(spec, env, evaluation_config)
            if laws is None:
                return float("nan")
            law_batch = laws.unsqueeze(0) if laws.ndim == 1 else laws
            distances = []
            with torch.no_grad():
                for mu0 in law_batch:
                    flow = env.exact_population_flow(control, mu0, horizon)
                    ref_flow = env.exact_population_flow(reference, mu0, horizon)
                    distances.append(torch.linalg.norm((flow - ref_flow).reshape(-1)))
            return float(torch.stack(distances).mean().item()) if distances else float("nan")
        if spec.name == "lq":
            means, vars_ = env.exact_moments(control)
            ref_means, ref_vars = env.exact_moments(reference)
            return float(torch.linalg.norm(torch.stack([means - ref_means, vars_ - ref_vars], dim=-1).reshape(-1)).item())
        if spec.name == "portfolio":
            means, vars_ = env.exact_moments(control, lambda_=0.0)
            ref_means, ref_vars = env.exact_moments(reference, lambda_=0.0)
            return float(torch.linalg.norm(torch.stack([means - ref_means, vars_ - ref_vars], dim=-1).reshape(-1)).item())
    except Exception:
        return float("nan")
    return float("nan")



def _write_adaptive_lambda_trace(run_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for child in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        history_path = child / "history.csv"
        if not history_path.exists():
            continue
        for row in _read_csv(history_path):
            if "lambda" in row:
                rows.append({"variant": child.name, **row})
    if rows:
        _write_csv(run_dir / "lambda_trace.csv", rows)



def _reset_peak_memory_if_available() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass



def _peak_memory_if_available() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.max_memory_allocated())
    except RuntimeError:
        return 0



def _simulator_budget_proxy(config: Mapping[str, Any]) -> int:
    train = dict(config.get("train", {}))
    algorithm = dict(config.get("algorithm_config", {}))
    B = int(train.get("B", algorithm.get("B", 1)))
    n = int(train.get("n", algorithm.get("n", 0)))
    steps = int(train.get("steps", 1))
    horizon = int(train.get("horizon", config.get("env_config", {}).get("T_train", config.get("env_config", {}).get("T", 1))))
    particles = int(train.get("particles", train.get("population_particles", train.get("flow_particles", 1))))
    return max(1, steps) * max(1, horizon) * max(1, B + n + particles)



def _as_sequence(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]



def _default_robustness_variants(env_name: str) -> List[Dict[str, Any]]:
    if env_name == "twostate":
        return [
            {"label": "nominal"},
            {"label": "low_initial_state1", "evaluation.mu0": [0.8, 0.2]},
            {"label": "high_initial_state1", "evaluation.mu0": [0.05, 0.95]},
            {"label": "longer_horizon", "evaluation.horizon": 4},
        ]
    if env_name == "advertising":
        return [
            {"label": "nominal"},
            {"label": "low_customer_initial", "evaluation.mu0": [0.9, 0.1]},
            {"label": "higher_ad_cost", "env_config.c_ad": 0.25},
            {"label": "lower_ad_effect", "env_config.kappa_ad": 0.1},
        ]
    if env_name == "cybersecurity":
        return [
            {"label": "nominal"},
            {"label": "high_hacker_intensity", "env_config.v_H": 0.9},
            {"label": "low_hacker_intensity", "env_config.v_H": 0.3},
            {"label": "higher_beta", "env_config.beta_UU": 0.45, "env_config.beta_UD": 0.55},
            {"label": "infected_initial", "evaluation.mu0": [0.4, 0.1, 0.4, 0.1]},
        ]
    if env_name == "distribution-planning":
        return [
            {"label": "nominal"},
            {"label": "longer_horizon", "evaluation.horizon": 8},
            {"label": "higher_movement_cost", "env_config.lam": 0.05},
        ]
    if env_name == "portfolio":
        return [
            {"label": "nominal"},
            {"label": "student_t_df_5", "env_config.return_distribution": "student_t", "env_config.student_t_df": 5.0},
            {"label": "student_t_df_3", "env_config.return_distribution": "student_t", "env_config.student_t_df": 3.0},
            {"label": "higher_initial_wealth", "env_config.x0_mean": 1.25},
        ]
    if env_name == "cucker-smale":
        return [
            {"label": "nominal"},
            {"label": "weaker_coupling", "env_config.K": 0.5},
            {"label": "stronger_coupling", "env_config.K": 1.5},
            {"label": "noisy", "env_config.sigma": 0.05},
            {"label": "wider_initial_clusters", "env_config.cluster_position": 3.0},
        ]
    if env_name == "kuramoto":
        return [
            {"label": "nominal"},
            {"label": "weaker_coupling", "env_config.K": 0.15},
            {"label": "stronger_coupling", "env_config.K": 0.6},
            {"label": "frequency_noise", "env_config.sigma_omega": 0.25},
        ]
    return [{"label": "nominal"}]



def _minimal_config_kwargs(spec: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for field in getattr(spec.config_cls, "__dataclass_fields__", {}).values():
        if field.name == "device":
            kwargs["device"] = torch.device(DEFAULT_DEVICE)
        elif field.name == "dtype":
            kwargs["dtype"] = torch.float64
    return kwargs



def _state_space_description(name: str, config: Any) -> str:
    if hasattr(config, "n_states"):
        return f"{getattr(config, 'n_states')} states"
    if name == "cucker-smale":
        return "R^2 particles (position, velocity)"
    if name == "kuramoto":
        return "circle phases"
    return "one-dimensional continuous state"



def _action_space_description(name: str, config: Any) -> str:
    if hasattr(config, "n_actions"):
        return f"{getattr(config, 'n_actions')} actions"
    return "one-dimensional continuous action"



def _signature_description(name: str) -> str:
    return {
        "twostate": "state masses",
        "advertising": "customer share",
        "cybersecurity": "full law; infection/defense reduced signatures",
        "distribution-planning": "full distribution and target distances",
        "lq": "mean and variance",
        "portfolio": "wealth mean and variance",
        "cucker-smale": "position mean, velocity mean, velocity dispersion, diameter",
        "kuramoto": "cos/sin moments, order parameter, circular variance",
    }[name]



def _perturbation_description(family: str, name: str) -> str:
    if family == "finite":
        return "simplex affine or CLR/logistic-normal"
    if name in {"lq", "portfolio"}:
        return "Gaussian perturbation of mean/variance signature"
    return "particle perturbation of empirical law"



def _reference_solution_description(name: str, family: str) -> str:
    return {
        "twostate": "analytic optimal stationary policy",
        "advertising": "finite-horizon dynamic-programming grid oracle plus infinite-horizon threshold reference",
        "cybersecurity": "model-based exact-flow optimized reference policy",
        "distribution-planning": "model-based exact-flow optimized reference policy",
        "lq": "Riccati/closed-form optimal gains",
        "portfolio": "closed-form mean-variance optimal gains",
        "cucker-smale": "pathwise-AD particle reference and uncontrolled/heuristic baselines",
        "kuramoto": "pathwise-AD particle reference and uncontrolled/heuristic baselines",
    }.get(name, "pathwise reference" if family == "pathwise" else "")



def _flatten_config_rows(prefix: str, value: Any) -> List[Dict[str, Any]]:
    value = _json_default(value)
    if isinstance(value, Mapping):
        rows: List[Dict[str, Any]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_config_rows(child_prefix, item))
        return rows
    return [{"key": prefix, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}]


__all__ = ["StudyFunction"]
