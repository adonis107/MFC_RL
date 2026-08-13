from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from .configs import CONTINUOUS_BENCHMARKS, DISCRETE_BENCHMARKS


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r") as handle:
        return json.load(handle)



def read_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()



def load_training_histories(bundle: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    return {algorithm: read_csv(Path(path) / "history.csv") for algorithm, path in bundle["train"].items()}



def load_application_data(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for algorithm, path in bundle["application"].items():
        path = Path(path)
        data[algorithm] = {
            "population_flow": read_csv(path / "population_flow.csv"),
            "time_metrics": read_csv(path / "time_metrics.csv"),
            "policy": read_csv(path / "policy.csv"),
            "terminal_samples": read_csv(path / "terminal_samples.csv"),
            "snapshots": read_csv(path / "snapshots.csv"),
            "metrics": pd.DataFrame([read_json(path / "metrics.json")]) if (path / "metrics.json").exists() else pd.DataFrame(),
        }
        for csv_path in sorted(path.glob("*.csv")):
            data[algorithm].setdefault(csv_path.stem, read_csv(csv_path))
    return data



def reference_solution_table(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for algorithm, data in application_data.items():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        record = metrics.iloc[0].to_dict()
        row: Dict[str, Any] = {"algorithm": algorithm}
        if env_name in DISCRETE_BENCHMARKS:
            row.update(
                {
                    "reference_kind": record.get("reference_kind", "exact_population_evaluation"),
                    "learned_value": _first_present(record, "value_mean", "value"),
                    "reference_value": record.get("reference_value"),
                    "reference_value_gap": record.get("reference_value_gap"),
                }
            )
            if env_name == "twostate" and "policy_error" in record:
                row["policy_l1_mean_error"] = record.get("policy_error")
        elif env_name == "lq":
            row.update(
                {
                    "reference_kind": "riccati_exact_optimizer",
                    "learned_cost": _first_present(record, "cost", "objective"),
                    "optimal_cost": record.get("optimal_cost"),
                    "objective_gap": record.get("objective_gap"),
                }
            )
        elif env_name == "portfolio":
            row.update(
                {
                    "reference_kind": "closed_form_mean_variance_optimizer",
                    "learned_objective": _first_present(record, "objective", "value"),
                    "optimal_objective": record.get("optimal_objective"),
                    "objective_gap": record.get("objective_gap"),
                }
            )
        elif env_name in {"cucker-smale", "kuramoto"}:
            row.update(
                {
                    "reference_kind": "pathwise_particle_baselines",
                    "controlled_objective": record.get("objective"),
                    "free_objective": record.get("free_objective"),
                    "heuristic_objective": record.get("heuristic_objective"),
                }
            )
            if "heuristic_best_kappa" in record:
                row["heuristic_best_kappa"] = record.get("heuristic_best_kappa")
            if "heuristic_best_nu" in record:
                row["heuristic_best_nu"] = record.get("heuristic_best_nu")
        rows.append(row)
    return pd.DataFrame(rows)



def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and not pd.isna(value):
            return value
    return None



def load_diagnostic_data(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for algorithm, diagnostics in bundle["diagnostics"].items():
        data[algorithm] = {}
        for name, path in diagnostics.items():
            path = Path(path)
            data[algorithm][name] = read_csv(path / "diagnostics.csv")
            for csv_path in sorted(path.glob("*.csv")):
                if csv_path.name == "diagnostics.csv":
                    continue
                data[algorithm][f"{name}_{csv_path.stem}"] = read_csv(csv_path)
    return data



def load_study_data(bundle: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for name, path in bundle["studies"].items():
        path = Path(path)
        data[name] = read_csv(path / "diagnostics.csv")
        for csv_path in sorted(path.glob("*.csv")):
            if csv_path.name in {"diagnostics.csv", "grid_metrics.csv"}:
                continue
            data[f"{name}_{csv_path.stem}"] = read_csv(csv_path)
    return data



def load_study_grid_metrics(bundle: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    return {name: read_csv(Path(path) / "grid_metrics.csv") for name, path in bundle["studies"].items()}



def load_optimization_history(bundle: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(bundle["studies"]["optimization"]) / "optimization_history.csv"
    return read_csv(path)


__all__ = [
    "load_application_data",
    "load_diagnostic_data",
    "load_optimization_history",
    "load_study_data",
    "load_study_grid_metrics",
    "load_training_histories",
    "read_csv",
    "read_json",
    "reference_solution_table",
]
