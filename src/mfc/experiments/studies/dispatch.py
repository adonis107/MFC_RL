from __future__ import annotations

from typing import Any, Mapping

from ..application import run_application_diagnostics
from ..core.session import RunResult, normalize_experiment_config
from .advanced import run_adaptive_lambda_study, run_particle_transfer_study, run_signature_ablation_study
from .bias import run_optimizer_bias_study, run_perturbation_bias_study
from .core_suite import run_core_suite
from .grids import (
    run_ablation_study,
    run_budget_allocation,
    run_horizon_scaling_study,
    run_particle_approximation,
    run_scaling_study,
)
from .reports import run_benchmark_properties, run_hyperparameter_table, run_optimization_summary, run_robustness_study
from .score import run_score_validation


def run_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    name = str(study_config.get("name", "core-suite"))
    if name == "core-suite":
        return run_core_suite(config)
    if name == "score-validation":
        return run_score_validation(config)
    if name == "horizon-scaling":
        return run_horizon_scaling_study(config)
    if name == "budget-allocation":
        return run_budget_allocation(config)
    if name == "particle-approximation":
        return run_particle_approximation(config)
    if name == "scaling":
        return run_scaling_study(config)
    if name == "ablation":
        return run_ablation_study(config)
    if name == "robustness":
        return run_robustness_study(config)
    if name == "optimization-summary":
        return run_optimization_summary(config)
    if name == "benchmark-properties":
        return run_benchmark_properties(config)
    if name == "hyperparameters":
        return run_hyperparameter_table(config)
    if name == "application-diagnostics":
        return run_application_diagnostics(config)
    if name == "perturbation-bias":
        return run_perturbation_bias_study(config)
    if name == "optimizer-bias":
        return run_optimizer_bias_study(config)
    if name == "signature-ablation":
        return run_signature_ablation_study(config)
    if name == "adaptive-lambda":
        return run_adaptive_lambda_study(config)
    if name == "particle-transfer":
        return run_particle_transfer_study(config)
    raise ValueError(f"Unknown study.name={name!r}.")


__all__ = ["run_study"]
