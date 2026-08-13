from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Mapping

from ..application import run_application_diagnostics
from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.memory import release_memory
from ..core.registry import ENVIRONMENTS, FINITE_ALGORITHMS, require_algorithm_name, require_env_name
from ..core.session import RunResult, normalize_experiment_config
from ..diagnostics.functional_law import run_functional_law_diagnostic
from ..diagnostics.gradient import run_gradient_diagnostic
from ..diagnostics.perturbation import run_perturbation_diagnostic
from ..diagnostics.sensitivity import run_sensitivity_diagnostic
from .common import StudyFunction
from .score import run_score_validation


def run_core_suite(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    run_dir = _make_run_dir("core-suite", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("core-suite", config))

    commands: List[tuple[str, StudyFunction]] = [
        ("perturbation", run_perturbation_diagnostic),
        ("functional_law", run_functional_law_diagnostic),
        ("gradient", run_gradient_diagnostic),
        ("application", run_application_diagnostics),
    ]
    env_name = require_env_name(config)
    algorithm_name = require_algorithm_name(config)
    if algorithm_name in FINITE_ALGORITHMS and ENVIRONMENTS[env_name].family == "finite":
        commands.insert(3, ("sensitivity", run_sensitivity_diagnostic))
        commands.insert(3, ("score", run_score_validation))
    if algorithm_name == "continuous-mfreinforce" and ENVIRONMENTS[env_name].family != "finite":
        commands.insert(3, ("sensitivity", run_sensitivity_diagnostic))
        commands.insert(3, ("score", run_score_validation))

    rows = []
    for label, fn in commands:
        child = copy.deepcopy(dict(config))
        child.setdefault("train", {})["output_dir"] = str(run_dir)
        child["train"]["run_name"] = label
        start = time.perf_counter()
        try:
            result = fn(child)
            rows.append(
                {
                    "component": label,
                    "status": "ok",
                    "run_dir": str(result.run_dir),
                    "diagnostics_path": str(result.diagnostics_path) if result.diagnostics_path else "",
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
        except Exception as exc:  # Keep long suites inspectable.
            rows.append(
                {
                    "component": label,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            if bool(config.get("study", {}).get("fail_fast", False)):
                raise
        finally:
            release_memory()
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"components": len(rows), "failures": sum(row["status"] != "ok" for row in rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")


__all__ = ["run_core_suite"]
