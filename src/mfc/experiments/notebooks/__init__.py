from __future__ import annotations

from . import configs as _configs
from .bundles import *
from .configs import *
from .coverage import *
from .data import *
from .plots import *


def set_default_device(device: str) -> None:
    _configs.set_default_device(device)
    globals()["DEFAULT_DEVICE"] = _configs.DEFAULT_DEVICE


__all__ = ['ALGORITHMS', 'CONTINUOUS_ALGORITHM', 'CONTINUOUS_ALGORITHMS', 'CONTINUOUS_BENCHMARKS', 'DEFAULT_DEVICE', 'DISCRETE_BENCHMARKS', 'benchmark_config', 'bundle_paths', 'continuous_benchmark_config', 'continuous_bundle_paths', 'display_continuous_figures', 'display_all_figures', 'ensure_continuous_benchmark_bundle', 'ensure_discrete_benchmark_bundle', 'figure_coverage_matrix', 'figure_checklist', 'global_figure_gap_table', 'load_application_data', 'load_diagnostic_data', 'load_optimization_history', 'load_study_data', 'load_study_grid_metrics', 'load_training_histories', 'perturbation_tv_comparison_table', 'reference_solution_table', 'set_default_device', 'plot_budget_and_horizon', 'plot_budget_pareto', 'plot_continuous_application_details', 'plot_continuous_policy_and_samples', 'plot_continuous_snapshots', 'plot_continuous_time_metrics', 'plot_discrete_application_details', 'plot_functional_law', 'plot_functional_sample_diagnostics', 'plot_functional_signature_means', 'plot_extended_study_summaries', 'plot_gradient_coordinate_diagnostics', 'plot_gradient_error_decomposition', 'plot_gradient_validation', 'plot_optimization_history', 'plot_optimization_summary', 'plot_perturbation_geometry', 'plot_perturbation_slopes', 'plot_perturbation_tv_comparison', 'plot_policy_heatmaps', 'plot_population_flow', 'plot_score_coordinate_diagnostics', 'plot_score_validation', 'plot_sensitivity_heatmap', 'plot_sensitivity_validation', 'plot_time_metrics', 'plot_training_comparison']
