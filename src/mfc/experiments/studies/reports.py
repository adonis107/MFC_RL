from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from ..core.artifacts import _make_run_dir, _metadata, _set_dotted, _write_csv, _write_json
from ..core.controls import load_control
from ..core.evaluation import evaluate_control
from ..core.memory import release_memory
from ..core.registry import ENVIRONMENTS, build_environment
from ..core.session import RunResult, normalize_experiment_config
from .common import (
    _action_space_description,
    _default_robustness_variants,
    _flatten_config_rows,
    _minimal_config_kwargs,
    _perturbation_description,
    _read_csv,
    _read_json,
    _reference_solution_description,
    _signature_description,
    _state_space_description,
)


def run_robustness_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    checkpoint = config.get("checkpoint")
    if not checkpoint:
        raise ValueError("Robustness studies require config.checkpoint.")
    payload = torch.load(checkpoint, map_location="cpu")
    base_config = {
        "env": payload["env"],
        "algorithm": payload["algorithm"],
        "env_config": payload["env_config"],
        "algorithm_config": payload.get("algorithm_config", {}),
        "train": payload.get("train_config", {}),
        "evaluation": payload.get("evaluation_config", {}),
    }
    variants = config.get("study", {}).get("variants")
    if not variants:
        variants = _default_robustness_variants(str(payload["env"]))
    run_dir = _make_run_dir("robustness", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("robustness", config))
    rows = []
    for index, variant in enumerate(variants):
        variant = dict(variant)
        label = str(variant.pop("label", f"variant_{index:04d}"))
        child = copy.deepcopy(base_config)
        child["env_config"] = {**dict(child.get("env_config", {})), **dict(config.get("env_config", {}))}
        child["evaluation"] = {**dict(child.get("evaluation", {})), **dict(config.get("evaluation", {}))}
        for key, value in variant.items():
            if isinstance(value, Mapping):
                child.setdefault(key, {}).update(value)
            else:
                _set_dotted(child, key, value)
        spec, env = build_environment(child)
        control = load_control(spec, env, payload["control"], trainable=False)
        metrics = evaluate_control(spec, env, control, child.get("train", {}), child.get("evaluation", {}), int(child.get("train", {}).get("seed", 0)))
        row = {"variant": label, **{key: value for key, value in variant.items() if isinstance(value, (int, float, str, bool))}}
        row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
        rows.append(row)
        release_memory()
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"variants": len(rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def run_optimization_summary(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    input_dirs = [Path(path) for path in study_config.get("input_dirs", config.get("input_dirs", []))]
    if not input_dirs:
        raise ValueError("optimization-summary requires study.input_dirs.")
    run_dir = _make_run_dir("optimization-summary", normalize_experiment_config(config))
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("optimization-summary", config))

    history_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for path in input_dirs:
        metrics = _read_json(path / "metrics.json") if (path / "metrics.json").exists() else {}
        run_config = _read_json(path / "config.json") if (path / "config.json").exists() else {}
        run_label = path.name
        for row in _read_csv(path / "history.csv") if (path / "history.csv").exists() else []:
            history_rows.append({"run": run_label, "run_dir": str(path), **row})
        summary = {
            "run": run_label,
            "run_dir": str(path),
            "env": run_config.get("env"),
            "algorithm": run_config.get("algorithm"),
        }
        summary.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
        summary_rows.append(summary)
    _write_csv(run_dir / "optimization_history.csv", history_rows)
    _write_csv(run_dir / "diagnostics.csv", summary_rows)
    metrics = {"runs": len(summary_rows), "history_rows": len(history_rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, history_rows, diagnostics_path=run_dir / "diagnostics.csv")



def run_benchmark_properties(config: Mapping[str, Any]) -> RunResult:
    normalized = normalize_experiment_config(config)
    run_dir = _make_run_dir("benchmark-properties", normalized)
    _write_json(run_dir / "config.json", normalized)
    _write_json(run_dir / "metadata.json", _metadata("benchmark-properties", normalized))
    rows = []
    for name, spec in ENVIRONMENTS.items():
        cfg = spec.config_cls(**_minimal_config_kwargs(spec))
        rows.append(
            {
                "benchmark": name,
                "family": spec.family,
                "objective": spec.objective,
                "state_space": _state_space_description(name, cfg),
                "action_space": _action_space_description(name, cfg),
                "horizon": getattr(cfg, "T", getattr(cfg, "T_train", "")),
                "selected_signature": _signature_description(name),
                "perturbation_geometry": _perturbation_description(spec.family, name),
                "exact_objective": spec.family in {"finite", "exact"},
                "exact_gradient": spec.family == "exact" or spec.family == "finite",
                "exact_optimizer": name in {"twostate", "lq", "portfolio"},
                "reference_solution": _reference_solution_description(name, spec.family),
                "pathwise_oracle": spec.family == "pathwise",
                "particle_approximation_required": spec.family == "pathwise",
            }
        )
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"benchmarks": len(rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def run_hyperparameter_table(config: Mapping[str, Any]) -> RunResult:
    normalized = normalize_experiment_config(config)
    run_dir = _make_run_dir("hyperparameters", normalized)
    _write_json(run_dir / "config.json", normalized)
    _write_json(run_dir / "metadata.json", _metadata("hyperparameters", normalized))
    rows = _flatten_config_rows("", normalized)
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"rows": len(rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")


__all__ = [
    "run_benchmark_properties",
    "run_hyperparameter_table",
    "run_optimization_summary",
    "run_robustness_study",
]
