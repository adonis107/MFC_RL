from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..application import run_application_diagnostics
from ..core.artifacts import _make_run_dir, _metadata, _set_dotted, _write_csv, _write_json
from ..core.memory import release_memory
from ..core.registry import ENVIRONMENTS
from ..core.session import RunResult, normalize_experiment_config
from ..diagnostics.functional_law import run_functional_law_diagnostic
from ..diagnostics.gradient import run_gradient_diagnostic
from ..diagnostics.perturbation import run_perturbation_diagnostic
from ..diagnostics.sensitivity import run_sensitivity_diagnostic
from ..runner import run_train
from .common import (
    StudyFunction,
    _as_sequence,
    _parameter_product,
    _peak_memory_if_available,
    _read_csv,
    _reset_peak_memory_if_available,
    _simulator_budget_proxy,
)
from .score import run_score_validation


def run_parameter_grid_study(config: Mapping[str, Any], command_name: str = "grid-study") -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    parameters = dict(study_config.get("parameters", config.get("sweep", {}).get("parameters", {})))
    if not parameters:
        raise ValueError(f"{command_name} requires study.parameters.")
    command = str(study_config.get("command", "diagnose-gradient"))
    runner = _command_runner(command)

    run_dir = _make_run_dir(command_name, config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata(command_name, config))

    keys = list(parameters)
    metrics_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    for index, values in enumerate(_parameter_product(parameters)):
        child = copy.deepcopy(dict(config))
        child.setdefault("train", {})["output_dir"] = str(run_dir)
        child["train"]["run_name"] = f"{command}_{index:04d}"
        for key, value in zip(keys, values):
            _set_dotted(child, key, value)
        _reset_peak_memory_if_available()
        start = time.perf_counter()
        result = runner(child)
        elapsed = time.perf_counter() - start
        parameter_payload = {key: value for key, value in zip(keys, values)}
        metrics_row = {"index": index, "run_dir": str(result.run_dir), "elapsed_seconds": elapsed, **parameter_payload}
        metrics_row.update({key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))})
        metrics_row["simulator_budget_proxy"] = _simulator_budget_proxy(child)
        metrics_row["peak_cuda_memory_bytes"] = _peak_memory_if_available()
        metrics_rows.append(metrics_row)
        if result.diagnostics_path and Path(result.diagnostics_path).exists():
            for row in _read_csv(result.diagnostics_path):
                diagnostic_rows.append({"index": index, **parameter_payload, **row})
        release_memory()

    _write_csv(run_dir / "grid_metrics.csv", metrics_rows)
    _write_csv(run_dir / "diagnostics.csv", diagnostic_rows if diagnostic_rows else metrics_rows)
    metrics = {"runs": len(metrics_rows), "diagnostic_rows": len(diagnostic_rows), "command": command}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def run_budget_allocation(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    budgets = study_config.get("budgets")
    if budgets is None:
        B_values = study_config.get("B_values", [16, 32, 64])
        n_values = study_config.get("n_values", [2, 4, 8])
        budgets = [{"train.B": int(B), "train.n": int(n)} for B in B_values for n in n_values]
    return run_variant_grid(config, "budget-allocation", budgets, default_command="diagnose-gradient")



def run_horizon_scaling_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    horizons = [int(value) for value in study_config.get("horizons", [2, 4, 8])]
    components = [str(value) for value in _as_sequence(study_config.get("components", ["diagnose-gradient", "diagnose-sensitivity", "score-validation"]))]
    run_dir = _make_run_dir("horizon-scaling", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("horizon-scaling", config))
    env_name = str(config.get("env", ""))
    horizon_key = "env_config.T_train" if env_name == "cybersecurity" else "env_config.T"

    metrics_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    for horizon in horizons:
        for component in components:
            child = copy.deepcopy(dict(config))
            _set_dotted(child, horizon_key, horizon)
            _set_dotted(child, "train.horizon", horizon)
            _set_dotted(child, "evaluation.horizon", horizon)
            child.setdefault("train", {})["output_dir"] = str(run_dir)
            child["train"]["run_name"] = f"{component.replace('-', '_')}_T{horizon}"
            runner = _command_runner(component)
            _reset_peak_memory_if_available()
            start = time.perf_counter()
            result = runner(child)
            elapsed = time.perf_counter() - start
            metrics_row = {
                "component": component,
                "horizon": horizon,
                "run_dir": str(result.run_dir),
                "elapsed_seconds": elapsed,
                "simulator_budget_proxy": _simulator_budget_proxy(child),
                "peak_cuda_memory_bytes": _peak_memory_if_available(),
            }
            metrics_row.update({key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))})
            metrics_rows.append(metrics_row)
            if result.diagnostics_path and Path(result.diagnostics_path).exists():
                for row in _read_csv(result.diagnostics_path):
                    diagnostic_rows.append({"component": component, "horizon": horizon, **row})
            release_memory()

    _write_csv(run_dir / "grid_metrics.csv", metrics_rows)
    _write_csv(run_dir / "diagnostics.csv", diagnostic_rows)
    metrics = {"runs": len(metrics_rows), "diagnostic_rows": len(diagnostic_rows), "components": len(components)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def run_scaling_study(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    parameters = study_config.get("parameters")
    if parameters is None:
        parameters = {
            "env_config.T": study_config.get("horizons", [2, 4, 8]),
            "train.B": study_config.get("B_values", [16, 32]),
            "train.n": study_config.get("n_values", [2, 4]),
        }
        env_name = str(config.get("env", ""))
        if env_name in {"cucker-smale", "kuramoto"}:
            parameters["env_config.N_pop"] = study_config.get("particles", [32, 64])
            parameters["train.population_particles"] = study_config.get("particles", [32, 64])
    child = copy.deepcopy(dict(config))
    child.setdefault("study", {})["parameters"] = parameters
    child["study"]["command"] = study_config.get("command", "diagnose-gradient")
    return run_parameter_grid_study(child, "scaling")



def run_particle_approximation(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    particles = study_config.get("particles", [32, 64, 128])
    env_name = str(config.get("env", ""))
    if env_name in {"cucker-smale", "kuramoto"}:
        variants = [{"train.particles": int(value), "evaluation.particles": int(value)} for value in particles]
    else:
        variants = [
            {"train.flow_mode": "particle", "train.flow_particles": int(value), "algorithm_config.flow_particles": int(value)}
            for value in particles
        ]
    return run_variant_grid(config, "particle-approximation", variants, default_command="diagnose-gradient")



def run_ablation_study(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    variants = study_config.get("variants", [])
    if not variants:
        env_name = str(config.get("env", ""))
        algorithm_name = str(config.get("algorithm", ""))
        variants = [
            {"label": "baseline_batch_mean", "algorithm_config.baseline": "batch_mean"},
            {"label": "no_baseline", "algorithm_config.baseline": None},
            {"label": "score_diagnostics", "algorithm_config.keep_score_diagnostics": True},
        ]
        if env_name in ENVIRONMENTS and ENVIRONMENTS[env_name].family == "finite":
            variants.extend(
                [
                    {"label": "exact_flow", "train.flow_mode": "exact"},
                    {"label": "particle_flow", "train.flow_mode": "particle"},
                    {"label": "antithetic_perturbations", "algorithm_config.antithetic": True},
                ]
            )
            if algorithm_name != "logits":
                variants.append({"label": "logit_geometry", "algorithm": "logits", "algorithm_config.epsilon": 0.1})
        else:
            variants.extend(
                [
                    {"label": "nominal_sensitivity_baseline", "algorithm_config.sensitivity_baseline": "nominal"},
                    {"label": "zero_sensitivity_baseline", "algorithm_config.sensitivity_baseline": None},
                    {"label": "time_batch_mean_baseline", "algorithm_config.baseline": "time_batch_mean"},
                ]
            )
    return run_variant_grid(config, "ablation", variants, default_command=str(study_config.get("command", "diagnose-gradient")))



def run_variant_grid(
    config: Mapping[str, Any],
    command_name: str,
    variants: Sequence[Mapping[str, Any]],
    default_command: str,
) -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    command = str(study_config.get("command", default_command))
    runner = _command_runner(command)
    run_dir = _make_run_dir(command_name, config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata(command_name, config))

    metrics_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant = dict(variant)
        label = str(variant.pop("label", f"variant_{index:04d}"))
        child = copy.deepcopy(dict(config))
        child.setdefault("train", {})["output_dir"] = str(run_dir)
        child["train"]["run_name"] = label
        for key, value in variant.items():
            if isinstance(value, Mapping):
                existing = child.setdefault(key, {})
                if not isinstance(existing, dict):
                    raise ValueError(f"Variant key {key!r} conflicts with a non-mapping config value.")
                existing.update(value)
            else:
                _set_dotted(child, key, value)
        _reset_peak_memory_if_available()
        start = time.perf_counter()
        result = runner(child)
        elapsed = time.perf_counter() - start
        row = {"variant": label, "run_dir": str(result.run_dir), "elapsed_seconds": elapsed}
        row.update({key: value for key, value in variant.items() if isinstance(value, (int, float, str, bool)) or value is None})
        row.update({key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))})
        row["simulator_budget_proxy"] = _simulator_budget_proxy(child)
        row["peak_cuda_memory_bytes"] = _peak_memory_if_available()
        metrics_rows.append(row)
        for history_row in result.history:
            history_rows.append({"variant": label, **variant, **history_row})
        if result.diagnostics_path and Path(result.diagnostics_path).exists():
            for diagnostic_row in _read_csv(result.diagnostics_path):
                diagnostic_rows.append({"variant": label, **variant, **diagnostic_row})
        release_memory()

    _write_csv(run_dir / "grid_metrics.csv", metrics_rows)
    _write_csv(run_dir / "diagnostics.csv", diagnostic_rows if diagnostic_rows else metrics_rows)
    if history_rows:
        _write_csv(run_dir / "optimization_history.csv", history_rows)
    metrics = {"variants": len(metrics_rows), "diagnostic_rows": len(diagnostic_rows), "history_rows": len(history_rows), "command": command}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def _command_runner(command: str) -> StudyFunction:
    if command == "train":
        return run_train
    if command == "diagnose-gradient":
        return run_gradient_diagnostic
    if command == "diagnose-sensitivity":
        return run_sensitivity_diagnostic
    if command == "diagnose-perturbation":
        return run_perturbation_diagnostic
    if command == "diagnose-functional-law":
        return run_functional_law_diagnostic
    if command == "score-validation":
        return run_score_validation
    if command == "application-diagnostics":
        return run_application_diagnostics
    raise ValueError(f"Unsupported study command {command!r}.")


__all__ = [
    "run_ablation_study",
    "run_budget_allocation",
    "run_horizon_scaling_study",
    "run_parameter_grid_study",
    "run_particle_approximation",
    "run_scaling_study",
    "run_variant_grid",
]
