from __future__ import annotations

from typing import Any, Mapping

from ..data import (
    load_application_data,
    load_diagnostic_data,
    load_optimization_history,
    load_study_data,
    load_study_grid_metrics,
    load_training_histories,
)
from .continuous import (
    plot_continuous_application_details,
    plot_continuous_policy_and_samples,
    plot_continuous_snapshots,
    plot_continuous_time_metrics,
)
from .diagnostics import (
    plot_functional_law,
    plot_functional_sample_diagnostics,
    plot_functional_signature_means,
    plot_gradient_coordinate_diagnostics,
    plot_gradient_error_decomposition,
    plot_gradient_validation,
    plot_perturbation_geometry,
    plot_perturbation_slopes,
    plot_perturbation_tv_comparison,
    plot_score_coordinate_diagnostics,
    plot_score_validation,
    plot_sensitivity_heatmap,
    plot_sensitivity_validation,
)
from .discrete import plot_discrete_application_details, plot_policy_heatmaps, plot_population_flow, plot_time_metrics
from .studies import plot_budget_and_horizon, plot_budget_pareto, plot_extended_study_summaries, plot_optimization_history, plot_optimization_summary
from .training import plot_training_comparison


def display_continuous_figures(env_name: str, bundle: Mapping[str, Any]) -> None:
    histories = load_training_histories(bundle)
    app = load_application_data(bundle)
    diagnostics = load_diagnostic_data(bundle)
    studies = load_study_data(bundle)
    grid_metrics = load_study_grid_metrics(bundle)
    optimization_history = load_optimization_history(bundle)
    plot_training_comparison(histories)
    plot_continuous_time_metrics(app, env_name)
    plot_continuous_policy_and_samples(app, env_name)
    plot_continuous_snapshots(app, env_name)
    plot_continuous_application_details(app, env_name)
    plot_perturbation_geometry(diagnostics)
    plot_perturbation_slopes(diagnostics)
    plot_functional_law(diagnostics)
    plot_functional_signature_means(diagnostics)
    plot_functional_sample_diagnostics(diagnostics)
    plot_score_validation(diagnostics)
    plot_score_coordinate_diagnostics(diagnostics)
    plot_gradient_validation(diagnostics)
    plot_gradient_error_decomposition(diagnostics)
    plot_gradient_coordinate_diagnostics(diagnostics)
    plot_sensitivity_validation(diagnostics)
    plot_sensitivity_heatmap(diagnostics)
    plot_budget_and_horizon(studies)
    plot_budget_pareto(studies, grid_metrics)
    plot_extended_study_summaries(studies, grid_metrics)
    plot_optimization_history(optimization_history)
    plot_optimization_summary(studies)



def display_all_figures(env_name: str, bundle: Mapping[str, Any]) -> None:
    histories = load_training_histories(bundle)
    app = load_application_data(bundle)
    diagnostics = load_diagnostic_data(bundle)
    studies = load_study_data(bundle)
    grid_metrics = load_study_grid_metrics(bundle)
    optimization_history = load_optimization_history(bundle)
    plot_training_comparison(histories)
    plot_population_flow(app, env_name)
    plot_time_metrics(app, env_name)
    plot_policy_heatmaps(app, env_name)
    plot_discrete_application_details(app, env_name)
    plot_perturbation_geometry(diagnostics)
    plot_perturbation_slopes(diagnostics)
    plot_perturbation_tv_comparison(diagnostics)
    plot_functional_law(diagnostics)
    plot_functional_signature_means(diagnostics)
    plot_functional_sample_diagnostics(diagnostics)
    plot_score_validation(diagnostics)
    plot_score_coordinate_diagnostics(diagnostics)
    plot_gradient_validation(diagnostics)
    plot_gradient_error_decomposition(diagnostics)
    plot_gradient_coordinate_diagnostics(diagnostics)
    plot_sensitivity_validation(diagnostics)
    plot_sensitivity_heatmap(diagnostics)
    plot_budget_and_horizon(studies)
    plot_budget_pareto(studies, grid_metrics)
    plot_extended_study_summaries(studies, grid_metrics)
    plot_optimization_history(optimization_history)
    plot_optimization_summary(studies)


__all__ = ["display_all_figures", "display_continuous_figures"]
