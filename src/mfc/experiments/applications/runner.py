from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.evaluation import evaluate_control
from ..core.session import RunResult, normalize_experiment_config, set_seed
from .common import _load_or_initialize
from .exact import _exact_application_outputs
from .finite import _finite_application_outputs
from .pathwise import _cucker_smale_outputs, _kuramoto_outputs


def run_application_diagnostics(config: Mapping[str, Any]) -> RunResult:
    """Save benchmark-specific, notebook-ready diagnostics.

    The output is deliberately tabular: notebooks can use the same files to
    create state-flow plots, policy heatmaps, phase snapshots, wealth tables,
    and benchmark-specific summary tables.
    """
    config = normalize_experiment_config(config)
    run_dir = _make_run_dir("application-diagnostics", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("application-diagnostics", config))

    spec, env, control, checkpoint_payload = _load_or_initialize(config)
    train_config = dict(config.get("train", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    seed = int(train_config.get("seed", evaluation_config.get("seed", 0)))
    set_seed(seed, env.config.device)

    if spec.family == "finite":
        outputs = _finite_application_outputs(spec.name, env, control, train_config, evaluation_config, seed)
    elif spec.name in {"lq", "portfolio"}:
        outputs = _exact_application_outputs(spec.name, env, control, train_config, evaluation_config, seed)
    elif spec.name == "cucker-smale":
        outputs = _cucker_smale_outputs(env, control, evaluation_config, seed)
    elif spec.name == "kuramoto":
        outputs = _kuramoto_outputs(env, control, evaluation_config, seed)
    else:
        raise ValueError(f"Unsupported application diagnostics for {spec.name!r}.")

    metrics = outputs.pop("metrics")
    metrics.update(evaluate_control(spec, env, control, train_config, evaluation_config, seed + 50_000))
    if checkpoint_payload is not None:
        metrics["checkpoint_env"] = checkpoint_payload.get("env")
        metrics["checkpoint_algorithm"] = checkpoint_payload.get("algorithm")

    diagnostics_path: Optional[Path] = None
    for name, rows in outputs.items():
        path = run_dir / f"{name}.csv"
        _write_csv(path, rows)
        diagnostics_path = diagnostics_path or path
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=diagnostics_path)


__all__ = ["run_application_diagnostics"]
