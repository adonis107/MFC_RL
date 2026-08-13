from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .core.artifacts import _write_json


RESULT_CATALOG: Dict[str, Dict[str, Any]] = {
    "perturbation_geometry": {
        "command": "diagnose-perturbation",
        "files": ["diagnostics.csv", "perturbation_samples.csv"],
        "plots": [
            "mean perturbation distance vs lambda",
            "distance quantile bands vs lambda",
            "empirical log-log slope vs lambda",
        ],
    },
    "functional_law": {
        "command": "diagnose-functional-law",
        "files": ["diagnostics.csv", "signature_samples.csv", "functional_covariance.csv", "functional_distances.csv"],
        "plots": [
            "Gamma(M^lambda) coordinate histograms",
            "Gamma(M^lambda) pair plots",
            "Gamma(M^lambda) covariance heatmaps",
            "Q-Q plots for standardized Gamma(M^lambda)",
            "standardized functional perturbation summaries",
            "functional covariance trace vs lambda",
        ],
    },
    "score_validation": {
        "study": "score-validation",
        "files": ["diagnostics.csv", "score_coordinates.csv", "score_samples.csv", "score_covariance.csv"],
        "plots": [
            "score mean norm vs sample size",
            "score variance vs lambda",
            "lambda^2 score variance vs lambda",
            "coordinate score variance bars",
        ],
    },
    "gradient_validation": {
        "command": "diagnose-gradient",
        "files": ["diagnostics.csv", "gradient_samples.csv", "gradient_coordinates.csv", "gradient_covariance.csv"],
        "plots": [
            "relative bias vs lambda",
            "variance trace vs lambda",
            "MSE vs lambda",
            "bias/std/RMSE decomposition",
            "cosine similarity vs lambda",
            "norm ratio vs lambda",
            "coordinatewise oracle vs estimated gradient scatter",
            "CI coverage against oracle gradient",
            "gradient covariance heatmap",
        ],
    },
    "sensitivity_validation": {
        "command": "diagnose-sensitivity",
        "files": ["diagnostics.csv", "sensitivity_samples.csv"],
        "plots": [
            "sensitivity MSE vs time",
            "sensitivity variance vs time",
            "sensitivity MSE heatmap over time and eta",
        ],
    },
    "horizon_scaling": {
        "study": "horizon-scaling",
        "files": ["grid_metrics.csv", "diagnostics.csv"],
        "plots": ["gradient MSE vs T", "runtime vs T", "best lambda by T"],
    },
    "budget_allocation": {
        "study": "budget-allocation",
        "files": ["grid_metrics.csv", "diagnostics.csv"],
        "plots": ["MSE heatmap over B and n", "MSE-runtime Pareto curve"],
    },
    "particle_approximation": {
        "study": "particle-approximation",
        "files": ["grid_metrics.csv", "diagnostics.csv"],
        "plots": ["law/signature/gradient error vs N_pop", "runtime vs N_pop"],
    },
    "particle_transfer": {
        "study": "particle-transfer",
        "files": ["diagnostics.csv", "optimization_history.csv"],
        "plots": ["trained-at-N evaluated-at-N' matrix", "policy transfer by N_pop"],
    },
    "ablation": {
        "study": "ablation",
        "files": ["grid_metrics.csv", "diagnostics.csv"],
        "plots": ["component ablation table", "MSE/runtime by ablation"],
    },
    "robustness": {
        "study": "robustness",
        "files": ["diagnostics.csv"],
        "plots": ["objective under shifted environment parameters", "robustness summary table"],
    },
    "signature_ablation": {
        "study": "signature-ablation",
        "files": ["grid_metrics.csv", "diagnostics.csv"],
        "plots": ["full vs reduced vs underspecified signatures", "MSE/runtime vs d_Gamma"],
    },
    "adaptive_lambda": {
        "study": "adaptive-lambda",
        "files": ["grid_metrics.csv", "diagnostics.csv", "lambda_trace.csv"],
        "plots": ["lambda trajectory", "adaptive vs fixed lambda", "final objective vs hindsight fixed lambda"],
    },
    "optimizer_bias": {
        "study": "optimizer-bias",
        "files": ["diagnostics.csv", "optimization_history.csv"],
        "plots": ["theta_lambda_star distance", "unperturbed performance of theta_lambda_star", "trajectory error vs lambda"],
    },
    "optimization": {
        "study": "optimization-summary",
        "files": ["optimization_history.csv", "diagnostics.csv"],
        "plots": [
            "objective gap vs simulator calls",
            "objective gap vs wall-clock",
            "gradient norm vs budget",
            "final objective distribution across seeds",
        ],
    },
    "application": {
        "command": "application-diagnostics",
        "files": [
            "population_flow.csv",
            "time_metrics.csv",
            "policy.csv",
            "landscape.csv",
            "transport_flux.csv",
            "finite_population.csv",
            "efficient_frontier.csv",
            "post_control.csv",
            "snapshots.csv",
            "terminal_samples.csv",
            "metrics.json",
        ],
        "plots": [
            "benchmark-specific state/population trajectory",
            "policy heatmap or gains",
            "application-specific terminal/sample diagnostics",
        ],
    },
    "master_tables": {
        "studies": ["benchmark-properties", "hyperparameters", "optimization-summary"],
        "files": ["diagnostics.csv"],
        "tables": [
            "benchmark properties",
            "gradient estimator accuracy",
            "optimization performance",
            "sensitivity estimation",
            "particle approximation",
            "ablations",
            "robustness",
            "hyperparameters",
        ],
    },
}


def catalog_rows() -> List[Dict[str, Any]]:
    rows = []
    for key, spec in RESULT_CATALOG.items():
        rows.append(
            {
                "result": key,
                "command": spec.get("command", ""),
                "study": spec.get("study", ",".join(spec.get("studies", []))),
                "files": ",".join(spec.get("files", [])),
                "plots": "; ".join(spec.get("plots", spec.get("tables", []))),
            }
        )
    return rows


def write_result_catalog(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, RESULT_CATALOG)
    return path


__all__ = ["RESULT_CATALOG", "catalog_rows", "write_result_catalog"]
