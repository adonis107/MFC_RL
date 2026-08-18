"""
Plotting functions for the diagnostics produced by scripts/train.py and
scripts/test.py. Each function takes the data structures those scripts
already return (a list of `train_run` dicts, a `scripts.test.*` result, or a
`{key: result}` mapping swept over lambda/T) and draws one figure — so a
notebook cell is `data = load(...); plot_xxx(data)`, per context.md.

Environment-agnostic except where the data itself is (state distributions,
policy errors: any number of states/components). Pass an existing `ax` to
draw into a subplot grid; otherwise a new figure is created and returned
alongside it.
"""

from __future__ import annotations

import torch
from matplotlib.ticker import MaxNLocator

from .style import INK, apply_style, color_for, new_figure, style_legend


def _fig_ax(ax, *, figsize=(6.0, 4.0)):
    return (ax.figure, ax) if ax is not None else new_figure(figsize=figsize)


def _integer_xaxis(ax):
    """Time/horizon axes are inherently integer-valued; avoid fractional ticks on short ranges."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_validation_curve(runs: list[dict], *, optimal_J: float | None = None, ax=None):
    """
    Evolution of the validation objective with training steps (context.md:
    "evolution of validation rewards ... with training steps"). `runs` is a
    list of `scripts.train.train_run` dicts; grouped by `lam` into one line
    each (algorithms with no perturbation scale, e.g. reinforce, store
    `lam=None` for every run and are labeled by algorithm name instead),
    with seeds sharing a lambda averaged and shown as a +-1 std band.
    `optimal_J`, if given (e.g. `simplex.exact_objective` at
    `env.optimal_theta()`), is drawn as a reference line.
    """
    fig, ax = _fig_ax(ax)
    groups: dict[float | None, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["lam"], []).append(r)

    for i, lam in enumerate(sorted(groups, key=lambda x: (x is None, x))):
        group = groups[lam]
        iters = group[0]["validation_iterations"]
        J = torch.stack([g["validation_J"] for g in group])  # (n_seeds, K)
        mean, std = J.mean(dim=0), J.std(dim=0)
        color = color_for(i)
        label = f"λ={lam}" if lam is not None else group[0].get("alg", "run")
        ax.plot(iters, mean, color=color, linewidth=2, label=label)
        if len(group) > 1:
            ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    if optimal_J is not None:
        ax.axhline(optimal_J, color=INK["muted"], linestyle="--", linewidth=1.5, label="optimal")

    apply_style(ax, xlabel="training iteration", ylabel="validation objective J")
    style_legend(ax)
    return fig, ax


def plot_state_distribution(mu_flow, *, target_law=None, optimal_flow=None, ax=None):
    """
    State distribution over time under a policy (context.md: "state
    distribution over time using learned policy (with optimal benchmark
    values whenever available)"). `mu_flow`: shape (T+1, n_states), e.g.
    from `scripts.test.state_distribution`. One line per state;
    `optimal_flow` (same shape, e.g. under `env.optimal_policy()` via
    `scripts.test.constant_policy_fn`), if given, drawn dashed for
    comparison; `target_law`, if given, as horizontal references.
    """
    fig, ax = _fig_ax(ax)
    T1, n_states = mu_flow.shape
    t = range(T1)
    for x in range(n_states):
        color = color_for(x)
        ax.plot(t, mu_flow[:, x], color=color, linewidth=2, marker="o", markersize=5, label=f"state {x}")
        if optimal_flow is not None:
            ax.plot(t, optimal_flow[:, x], color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        if target_law is not None:
            ax.axhline(target_law[x].item(), color=color, linestyle=":", linewidth=1, alpha=0.5)

    apply_style(ax, xlabel="t", ylabel="P(X_t = x)")
    ax.set_ylim(-0.02, 1.02)
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_objective_gap(gaps_by_lambda: dict[float, dict], *, ax=None):
    """
    J vs J^lambda across the perturbation grid (context.md: "J^lambda vs J,
    at learned theta, optimal theta"). `gaps_by_lambda`: {lam:
    `scripts.test.objective_gap(...)` dict}, one entry per lambda's own
    learned theta_lambda. J is *not* a function of lambda — it only varies
    across the x-axis here because it's evaluated at a different theta per
    point (each lambda's own trained result), which is why it's drawn as
    its own line rather than a single reference: a flat line would silently
    take one lambda's theta and mislabel it as valid for every other point.
    J^lambda carries its Monte Carlo standard error.
    """
    fig, ax = _fig_ax(ax)
    lambdas = sorted(gaps_by_lambda)
    Js = [gaps_by_lambda[lam]["J"].item() for lam in lambdas]
    means = [gaps_by_lambda[lam]["J_lambda_mean"].item() for lam in lambdas]
    ses = [gaps_by_lambda[lam]["J_lambda_se"].item() for lam in lambdas]

    ax.plot(lambdas, Js, color=INK["muted"], linestyle="--", linewidth=1.5, marker="o", markersize=5, label="J (exact, at that λ's θ)")
    ax.errorbar(lambdas, means, yerr=ses, color=color_for(0), linewidth=2, marker="o", markersize=6, capsize=3, label="J^λ (Monte Carlo)")

    apply_style(ax, xlabel="λ", ylabel="objective")
    style_legend(ax)
    return fig, ax


def plot_gradient_diagnostics(gd_by_lambda: dict[float, dict], *, component_labels=None, ax=None):
    """
    Gradient estimator bias vs. lambda, ±1 std band (context.md: "gradient
    bias, variance"). `gd_by_lambda`: {lam:
    `scripts.test.gradient_diagnostics(...)` dict}. One line per theta
    component.
    """
    fig, ax = _fig_ax(ax)
    lambdas = sorted(gd_by_lambda)
    D = gd_by_lambda[lambdas[0]]["bias"].shape[0]
    labels = component_labels or [f"θ_{i}" for i in range(D)]

    for i in range(D):
        bias = [gd_by_lambda[lam]["bias"][i].item() for lam in lambdas]
        std = [gd_by_lambda[lam]["std"][i].item() for lam in lambdas]
        color = color_for(i)
        ax.plot(lambdas, bias, color=color, linewidth=2, marker="o", markersize=6, label=labels[i])
        lo = [b - s for b, s in zip(bias, std)]
        hi = [b + s for b, s in zip(bias, std)]
        ax.fill_between(lambdas, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.axhline(0.0, color=INK["axis"], linewidth=1)
    apply_style(ax, xlabel="λ", ylabel="gradient bias (±1 std)")
    style_legend(ax)
    return fig, ax


def plot_theta_diagnostics(theta_by_lambda: dict[float, dict], *, optimal_theta=None, component_labels=None, ax=None):
    """
    Learned theta (mean ± std across seeds) vs. lambda (context.md: "theta
    bias, variance"). `theta_by_lambda`: {lam:
    `scripts.test.theta_diagnostics(...)` dict}. `optimal_theta`, if given,
    drawn as a per-component reference line.
    """
    fig, ax = _fig_ax(ax)
    lambdas = sorted(theta_by_lambda)
    D = theta_by_lambda[lambdas[0]]["mean"].shape[0]
    labels = component_labels or [f"θ_{i}" for i in range(D)]

    for i in range(D):
        mean = [theta_by_lambda[lam]["mean"][i].item() for lam in lambdas]
        std = [theta_by_lambda[lam]["std"][i].item() for lam in lambdas]
        color = color_for(i)
        ax.errorbar(lambdas, mean, yerr=std, color=color, linewidth=2, marker="o", markersize=6, capsize=3, label=labels[i])
        if optimal_theta is not None:
            ax.axhline(optimal_theta[i].item(), color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    apply_style(ax, xlabel="λ", ylabel="learned θ (mean ± std across seeds)")
    style_legend(ax)
    return fig, ax


def plot_policy_error(errors_by_lambda: dict[float, object], *, state_labels=None, ax=None):
    """
    Policy error vs. lambda, grouped by state (context.md: "average absolute
    error of learned policy"). `errors_by_lambda`: {lam:
    `scripts.test.policy_error(...)` tensor of shape (n_states,)}.
    """
    fig, ax = _fig_ax(ax)
    lambdas = sorted(errors_by_lambda)
    n_states = errors_by_lambda[lambdas[0]].shape[0]
    labels = state_labels or [f"state {x}" for x in range(n_states)]
    width = 0.8 / n_states
    positions = range(len(lambdas))

    for x in range(n_states):
        vals = [errors_by_lambda[lam][x].item() for lam in lambdas]
        offsets = [p + (x - (n_states - 1) / 2) * width for p in positions]
        ax.bar(offsets, vals, width=width * 0.9, color=color_for(x), label=labels[x])

    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(lam) for lam in lambdas])
    apply_style(ax, xlabel="λ", ylabel="policy error |π^θ - π*|")
    style_legend(ax)
    return fig, ax


def plot_perturbation_coverage(results: list[dict], lam: float, *, mu_labels=None, ax=None):
    """
    d_TV(M^lambda, mu) per representative mu, against the theorem's bound
    lambda (context.md: check d_TV(M^lambda,mu)<lambda always for simplex).
    `results`: `scripts.test.perturbation_coverage(...)` output.
    """
    fig, ax = _fig_ax(ax)
    labels = mu_labels or [f"μ_{i}" for i in range(len(results))]
    positions = range(len(results))
    means = [r["mean_dTV"].item() for r in results]
    maxes = [r["max_dTV"].item() for r in results]
    width = 0.35

    ax.bar([p - width / 2 for p in positions], means, width=width, color=color_for(0), label="mean d_TV")
    ax.bar([p + width / 2 for p in positions], maxes, width=width, color=color_for(1), label="max d_TV")
    ax.axhline(lam, color=INK["muted"], linestyle="--", linewidth=1.5, label=f"λ={lam} bound")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    apply_style(ax, xlabel="population law", ylabel="d_TV(M^λ, μ)")
    style_legend(ax)
    return fig, ax


def plot_generalization(results: list[dict], *, ax=None):
    """
    J across generalization scenarios, without retraining (context.md:
    "generalization" — different mu0, interaction strength, horizon,
    model misspecification). `results`:
    `scripts.test.generalization_eval(...)` output.
    """
    fig, ax = _fig_ax(ax)
    names = [r["name"] for r in results]
    values = [r["J"].item() for r in results]
    ax.bar(range(len(names)), values, color=[color_for(i) for i in range(len(names))])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    apply_style(ax, ylabel="J (no retraining)")
    return fig, ax


def plot_trajectories(learned, optimal=None, *, ax=None):
    """
    Learned vs. optimal sample state trajectory (context.md: "learned vs
    optimal trajectories whenever applicable"). `learned`/`optimal`: shape
    (T+1,), e.g. from `scripts.test.rollout`.
    """
    fig, ax = _fig_ax(ax)
    t = range(len(learned))
    ax.step(t, learned, where="mid", color=color_for(0), linewidth=2, marker="o", markersize=6, label="learned")
    if optimal is not None:
        ax.step(t, optimal, where="mid", color=color_for(1), linewidth=2, marker="s", markersize=6, alpha=0.8, label="optimal")

    apply_style(ax, xlabel="t", ylabel="state")
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_horizon_scaling(metric_by_T: dict[int, float], *, ylabel: str = "metric", label: str | None = None, color_index: int = 0, ax=None):
    """
    A scalar metric vs. horizon T (context.md: "horizon scaling tests
    (gradient, theta, value vs time)"). Call once per metric/lambda on the
    same `ax` to overlay several series (pass a distinct `color_index` each
    time).
    """
    fig, ax = _fig_ax(ax)
    Ts = sorted(metric_by_T)
    values = [metric_by_T[T] for T in Ts]
    ax.plot(Ts, values, color=color_for(color_index), linewidth=2, marker="o", markersize=6, label=label)

    apply_style(ax, xlabel="T", ylabel=ylabel)
    _integer_xaxis(ax)
    if label:
        style_legend(ax)
    return fig, ax
