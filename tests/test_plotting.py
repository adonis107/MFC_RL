from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from mfc.plotting import diagnostics as viz
from mfc.plotting.style import CATEGORICAL, color_for


def _fake_run(lam, seed, n_train=10, alg="simplex"):
    return {
        "lam": lam,
        "alg": alg,
        "seed": seed,
        "validation_iterations": torch.arange(0, n_train + 1, 2),
        "validation_J": torch.linspace(-5.0, -4.0, (n_train // 2) + 1) + 0.01 * seed,
    }


def test_color_for_is_fixed_order_and_wraps():
    assert [color_for(i) for i in range(len(CATEGORICAL))] == CATEGORICAL
    assert color_for(len(CATEGORICAL)) == CATEGORICAL[0]


def test_plot_functions_reuse_a_given_axes():
    fig, ax = plt.subplots()
    out_fig, out_ax = viz.plot_trajectories(torch.tensor([0, 1, 0]), ax=ax)
    assert out_ax is ax
    assert out_fig is ax.figure


def test_plot_validation_curve_groups_by_lambda_and_draws_optimal_line():
    runs = [_fake_run(0.1, seed) for seed in range(3)] + [_fake_run(0.2, seed) for seed in range(3)]
    fig, ax = viz.plot_validation_curve(runs, optimal_J=-3.5)
    assert len(ax.lines) == 2 + 1  # one mean line per lambda + the optimal reference line
    labels = [line.get_label() for line in ax.lines]
    assert "λ=0.1" in labels and "λ=0.2" in labels and "optimal" in labels
    assert ax.get_legend() is not None


def test_plot_validation_curve_labels_by_algorithm_when_no_lambda():
    """Algorithms with no perturbation scale (reinforce, mfreinforce) store
    lam=None for every run; the legend must show the algorithm name, not
    the literal string "λ=None"."""
    runs = [_fake_run(None, seed, alg="reinforce") for seed in range(3)]
    fig, ax = viz.plot_validation_curve(runs)
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["reinforce"]


def test_plot_validation_curve_keeps_multiple_no_lambda_algorithms_separate():
    """Two lambda-less algorithms (reinforce, mfreinforce) both store
    lam=None; grouping on lam alone would silently average their runs
    together into one line (the bug this regression-tests)."""
    runs = [_fake_run(None, seed, alg="reinforce") for seed in range(3)] + [_fake_run(None, seed, alg="mfreinforce") for seed in range(3)]
    fig, ax = viz.plot_validation_curve(runs)
    labels = sorted(line.get_label() for line in ax.lines)
    assert labels == ["mfreinforce", "reinforce"]


def test_plot_state_distribution_has_one_line_per_state_and_integer_ticks():
    mu_flow = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]])
    fig, ax = viz.plot_state_distribution(mu_flow, target_law=torch.tensor([0.6, 0.4]))
    assert sum(1 for line in ax.lines if line.get_label().startswith("state")) == 2
    ax.figure.canvas.draw()
    xticks = ax.get_xticks()
    assert all(float(t).is_integer() for t in xticks)


def test_plot_state_distribution_uses_custom_state_labels():
    mu_flow = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.4, 0.2, 0.3, 0.1]])
    fig, ax = viz.plot_state_distribution(mu_flow, state_labels=["DI", "DS", "UI", "US"])
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["DI", "DS", "UI", "US"]


def test_plot_population_fractions_one_line_per_series():
    series = {"infected": torch.tensor([0.5, 0.4, 0.3]), "defended": torch.tensor([0.2, 0.3, 0.4])}
    fig, ax = viz.plot_population_fractions(series)
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["infected", "defended"]


def test_plot_distribution_comparison_one_bar_group_per_name():
    distributions = {"initial": torch.tensor([0.5, 0.5, 0.0]), "target": torch.tensor([0.2, 0.3, 0.5])}
    fig, ax = viz.plot_distribution_comparison(distributions, state_labels=["a", "b", "c"])
    assert len(ax.patches) == 2 * 3  # 2 distributions x 3 states
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_plot_objective_gap_draws_J_per_lambda_not_a_flat_reference():
    """J is evaluated at each lambda's own learned theta, so it must vary
    across the sweep like J^lambda does — a flat line would silently use
    one lambda's theta for every point (the bug this regression-tests)."""
    gaps = {
        0.1: {"J": torch.tensor(-2.0), "J_lambda_mean": torch.tensor(-2.1), "J_lambda_se": torch.tensor(0.05)},
        0.2: {"J": torch.tensor(-3.5), "J_lambda_mean": torch.tensor(-2.3), "J_lambda_se": torch.tensor(0.05)},
    }
    fig, ax = viz.plot_objective_gap(gaps)
    J_lines = [line for line in ax.lines if line.get_label().startswith("J (exact")]
    assert len(J_lines) == 1
    assert list(J_lines[0].get_ydata()) == [-2.0, -3.5]


def test_plot_gradient_and_theta_diagnostics_one_series_per_component():
    by_lambda = {
        0.1: {"bias": torch.tensor([0.1, -0.2]), "std": torch.tensor([0.3, 0.4])},
        0.2: {"bias": torch.tensor([0.15, -0.1]), "std": torch.tensor([0.25, 0.35])},
    }
    fig, ax = viz.plot_gradient_diagnostics(by_lambda)
    assert sum(1 for line in ax.lines if line.get_label().startswith("θ_")) == 2

    theta_by_lambda = {
        0.1: {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([0.1, 0.2])},
        0.2: {"mean": torch.tensor([1.1, 1.9]), "std": torch.tensor([0.1, 0.2])},
    }
    fig, ax = viz.plot_theta_diagnostics(theta_by_lambda, optimal_theta=torch.tensor([1.0, 2.0]))
    assert ax.get_legend() is not None


def test_plot_policy_error_has_one_bar_group_per_state():
    errors = {0.1: torch.tensor([0.05, 0.1]), 0.2: torch.tensor([0.08, 0.12])}
    fig, ax = viz.plot_policy_error(errors)
    assert len(ax.patches) == 2 * 2  # 2 lambdas x 2 states


def test_plot_perturbation_coverage_draws_bound_line():
    results = [
        {"mu": torch.tensor([0.5, 0.5]), "mean_dTV": torch.tensor(0.05), "max_dTV": torch.tensor(0.09)},
        {"mu": torch.tensor([0.2, 0.8]), "mean_dTV": torch.tensor(0.03), "max_dTV": torch.tensor(0.07)},
    ]
    fig, ax = viz.plot_perturbation_coverage(results, lam=0.1)
    bound_lines = [line for line in ax.lines if "bound" in line.get_label()]
    assert len(bound_lines) == 1
    assert bound_lines[0].get_ydata()[0] == 0.1


def test_plot_generalization_one_bar_per_scenario():
    results = [{"name": "baseline", "J": torch.tensor(-1.0)}, {"name": "shifted", "J": torch.tensor(-1.5)}]
    fig, ax = viz.plot_generalization(results)
    assert len(ax.patches) == 2


def test_plot_horizon_scaling_overlays_series_on_shared_axes():
    fig, ax = viz.plot_horizon_scaling({1: -1.0, 2: -1.5, 5: -2.0}, label="a", color_index=0, ax=None)
    viz.plot_horizon_scaling({1: -0.5, 2: -0.8, 5: -1.2}, label="b", color_index=1, ax=ax)
    labels = [line.get_label() for line in ax.lines]
    assert "a" in labels and "b" in labels


def test_plot_validation_curve_names_the_algorithm_when_several_carry_a_lambda():
    """Both continuous-state algorithms are trained on the perturbed
    process, so both store a lambda; labeling by lambda alone would give two
    different lines the same legend entry."""
    runs = [_fake_run(0.2, 0, alg="simplex"), _fake_run(0.2, 0, alg="reinforce")]
    fig, ax = viz.plot_validation_curve(runs)
    labels = sorted(line.get_label() for line in ax.lines)
    assert labels == ["reinforce λ=0.2", "simplex λ=0.2"]
