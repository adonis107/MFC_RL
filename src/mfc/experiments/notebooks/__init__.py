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


__all__ = ['ALGORITHMS', 'CONTINUOUS_ALGORITHM', 'CONTINUOUS_ALGORITHMS', 'CONTINUOUS_ORACLE_SENSITIVITY_ALGORITHM', 'CONTINUOUS_BENCHMARKS', 'DEFAULT_DEVICE', 'DISCRETE_BENCHMARKS', 'benchmark_config', 'bundle_paths', 'continuous_algorithms_for_env', 'continuous_benchmark_config', 'continuous_bundle_paths', 'display_continuous_figures', 'display_all_figures', 'discrete_main_summary_table', 'ensure_continuous_benchmark_bundle', 'ensure_discrete_benchmark_bundle', 'figure_coverage_matrix', 'figure_checklist', 'global_figure_gap_table', 'load_application_data', 'load_diagnostic_data', 'load_optimization_history', 'load_study_data', 'load_study_grid_metrics', 'load_training_histories', 'lq_main_summary_table', 'pathwise_main_summary_table', 'perturbation_tv_comparison_table', 'reference_solution_table', 'set_default_device', 'plot_budget_and_horizon', 'plot_budget_pareto', 'plot_continuous_application_details', 'plot_continuous_diagnostic_appendix', 'plot_continuous_policy_and_samples', 'plot_continuous_snapshots', 'plot_continuous_time_metrics', 'plot_discrete_application_details', 'plot_discrete_diagnostic_appendix', 'plot_discrete_main_results', 'plot_discrete_training_value', 'plot_functional_law', 'plot_functional_sample_diagnostics', 'plot_functional_signature_means', 'plot_extended_study_summaries', 'plot_gradient_coordinate_diagnostics', 'plot_gradient_error_decomposition', 'plot_gradient_validation', 'plot_lambda_training_comparison', 'plot_lq_diagnostic_appendix', 'plot_lq_jlambda_comparison', 'plot_lq_learned_policy_comparison', 'plot_lq_main_results', 'plot_lq_theta_comparison', 'plot_lq_validation_reward', 'plot_optimization_history', 'plot_optimization_summary', 'plot_pathwise_control_energy', 'plot_pathwise_dynamics', 'plot_pathwise_main_results', 'plot_pathwise_snapshots', 'plot_pathwise_training_cost', 'plot_perturbation_geometry', 'plot_perturbation_slopes', 'plot_perturbation_tv_comparison', 'plot_policy_heatmaps', 'plot_population_flow', 'plot_portfolio_jlambda_comparison', 'plot_portfolio_main_results', 'plot_portfolio_policy_comparison', 'plot_portfolio_validation_reward', 'plot_portfolio_wealth_comparison', 'plot_score_coordinate_diagnostics', 'plot_score_validation', 'plot_sensitivity_heatmap', 'plot_sensitivity_validation', 'plot_time_metrics', 'plot_training_comparison', 'portfolio_main_summary_table']
