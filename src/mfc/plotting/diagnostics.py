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
from matplotlib.ticker import LogLocator, MaxNLocator, PercentFormatter, ScalarFormatter

from .style import apply_style, color_for, new_figure, style_legend


def _fig_ax(ax, *, figsize=(6.0, 4.0)):
    return (ax.figure, ax) if ax is not None else new_figure(figsize=figsize)


def _cpu(t):
    """matplotlib needs numpy-convertible arrays; move CUDA tensors to CPU
    first (`.item()` on a scalar already does this implicitly, so this is
    only needed where a whole tensor is handed to a plot/fill call)."""
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def _integer_xaxis(ax):
    """Time/horizon axes are inherently integer-valued; avoid fractional ticks on short ranges."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _set_probability_axis(ax):
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _maybe_log_lambda_axis(ax, values):
    positive = [float(v) for v in values if float(v) > 0.0]
    if len(positive) == len(values) and len(positive) >= 3 and max(positive) / min(positive) >= 4.0:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)


def plot_validation_curve(runs: list[dict], *, optimal_J: float | None = None, ax=None):
    """
    Evolution of the validation objective with training steps (context.md:
    "evolution of validation rewards ... with training steps"). `runs` is a
    list of `scripts.train.train_run` dicts; grouped by `(alg, lam)` into
    one line each (algorithms with no perturbation scale, e.g. reinforce and
    mfreinforce, store `lam=None` for every run and are labeled by algorithm
    name instead — grouping on `alg` too, not just `lam`, keeps several
    such algorithms compared side by side from colliding into one averaged
    line), with seeds sharing a group averaged and shown as a +-1 std band.
    When several algorithms *do* carry a lambda (the continuous benchmarks,
    where both compared algorithms are trained on the perturbed process),
    the label names both, so the legend still identifies each line.
    `optimal_J`, if given (e.g. `simplex.exact_objective` at
    `env.optimal_theta()`), is drawn as a reference line.
    """
    fig, ax = _fig_ax(ax)
    groups: dict[tuple[str, float | None], list[dict]] = {}
    for r in runs:
        groups.setdefault((r.get("alg", "run"), r["lam"]), []).append(r)

    several_algs = len({alg for alg, _ in groups}) > 1
    for key in sorted(groups, key=lambda k: (k[1] is None, k[1], k[0])):
        alg, lam = key
        group = groups[key]
        iters = _cpu(group[0]["validation_iterations"])
        J = torch.stack([g["validation_J"] for g in group])  # (n_seeds, K)
        mean, std = _cpu(J.mean(dim=0)), _cpu(J.std(dim=0, unbiased=False))
        label = alg if lam is None else (f"{alg} λ={lam}" if several_algs else f"λ={lam}")
        (line,) = ax.plot(iters, mean, label=label)
        if len(group) > 1:
            ax.fill_between(iters, mean - std, mean + std, color=line.get_color(), alpha=0.2)

    if optimal_J is not None:
        ax.axhline(optimal_J, color="0.15", linestyle="--", linewidth=1.4, label="optimal")

    apply_style(ax, xlabel="training iteration", ylabel="validation objective")
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_state_distribution(mu_flow, *, target_law=None, optimal_flow=None, state_labels=None, ax=None):
    """
    State distribution over time under a policy (context.md: "state
    distribution over time using learned policy (with optimal benchmark
    values whenever available)"). `mu_flow`: shape (T+1, n_states), e.g.
    from `scripts.test.state_distribution`. One line per state (labeled
    `state_labels[x]` if given, else "state x"); `optimal_flow` (same
    shape, e.g. under `env.optimal_policy()` via
    `scripts.test.constant_policy_fn`), if given, drawn dashed for
    comparison; `target_law`, if given, as horizontal references.
    """
    fig, ax = _fig_ax(ax)
    mu_flow = _cpu(mu_flow)
    optimal_flow = _cpu(optimal_flow)
    T1, n_states = mu_flow.shape
    labels = state_labels or [f"state {x}" for x in range(n_states)]
    t = range(T1)
    for x in range(n_states):
        (line,) = ax.plot(t, mu_flow[:, x], marker="o", label=labels[x])
        color = line.get_color()
        if optimal_flow is not None:
            ax.plot(t, optimal_flow[:, x], color=color, linestyle="--", alpha=0.8)
        if target_law is not None:
            ax.axhline(target_law[x].item(), color=color, linestyle=":", alpha=0.8)

    apply_style(ax, xlabel="time step", ylabel="population share")
    _set_probability_axis(ax)
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_population_fractions(series: dict, *, ax=None):
    """
    One or more scalar population-level fractions over time — e.g.
    cybersecurity's aggregate infected I_t=mu_t(DI)+mu_t(UI), defended
    D_t=mu_t(DI)+mu_t(DS) (`CyberSecurity.aggregate_fractions`), or the
    population-averaged intervention probability
    A_t=sum_x mu_t(x)pi_t(1|x,mu_t) (`scripts.test.intervention_probability`)
    — beyond the per-state flow `plot_state_distribution` already covers.
    `series`: {name: tensor of shape (T+1,)}, one line each.
    """
    fig, ax = _fig_ax(ax)
    for name, values in series.items():
        values = _cpu(values)
        t = range(len(values))
        ax.plot(t, values, marker="o", label=name)

    apply_style(ax, xlabel="time step", ylabel="population share")
    _set_probability_axis(ax)
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_distribution_comparison(distributions: dict, *, state_labels=None, ax=None):
    """
    Several named laws over the same state space, grouped by state (e.g.
    distribution planning's reference "Evaluation criteria": initial,
    target, and terminal population distributions side by side).
    `distributions`: {name: tensor of shape (n_states,)}, one bar-color per name.
    """
    fig, ax = _fig_ax(ax)
    names = list(distributions)
    n_states = distributions[names[0]].shape[0]
    labels = state_labels or [str(x) for x in range(n_states)]
    width = 0.8 / len(names)
    positions = range(n_states)

    for i, name in enumerate(names):
        values = _cpu(distributions[name])
        offsets = [p + (i - (len(names) - 1) / 2) * width for p in positions]
        ax.bar(offsets, values, width=width * 0.9, label=name)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    apply_style(ax, xlabel="state", ylabel="population share")
    _set_probability_axis(ax)
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

    ax.plot(lambdas, Js, linestyle="--", marker="o", label="J (exact, at that λ's θ)")
    ax.errorbar(lambdas, means, yerr=ses, marker="o", capsize=3, label="J^λ (Monte Carlo)")

    _maybe_log_lambda_axis(ax, lambdas)
    apply_style(ax, xlabel="perturbation scale λ", ylabel="objective")
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
        (line,) = ax.plot(lambdas, bias, marker="o", label=labels[i])
        lo = [b - s for b, s in zip(bias, std)]
        hi = [b + s for b, s in zip(bias, std)]
        ax.fill_between(lambdas, lo, hi, color=line.get_color(), alpha=0.2)

    ax.axhline(0.0, color="0.25", linewidth=1.0)
    _maybe_log_lambda_axis(ax, lambdas)
    apply_style(ax, xlabel="perturbation scale λ", ylabel="gradient bias (band: ±1 std)")
    style_legend(ax)
    return fig, ax


def plot_theta_diagnostics(theta_by_lambda: dict[float, dict], *, optimal_theta=None, component_labels=None, ax=None):
    """
    Learned theta (mean +/- std across seeds) vs. lambda. `theta_by_lambda`:
    {lam: `scripts.test.theta_diagnostics(...)` dict}. `optimal_theta`, if
    given, drawn as a per-component dashed reference line, labeled
    "{component} (optimal)" in the legend.
    """
    fig, ax = _fig_ax(ax)
    lambdas = sorted(theta_by_lambda)
    D = theta_by_lambda[lambdas[0]]["mean"].shape[0]
    labels = component_labels or [f"θ_{i}" for i in range(D)]

    for i in range(D):
        mean = [theta_by_lambda[lam]["mean"][i].item() for lam in lambdas]
        std = [theta_by_lambda[lam]["std"][i].item() for lam in lambdas]
        container = ax.errorbar(lambdas, mean, yerr=std, marker="o", capsize=3, label=labels[i])
        if optimal_theta is not None:
            color = container.lines[0].get_color()
            ax.axhline(optimal_theta[i].item(), color=color, linestyle="--", alpha=0.8, label=f"{labels[i]} (optimal)")

    _maybe_log_lambda_axis(ax, lambdas)
    apply_style(ax, xlabel="perturbation scale λ", ylabel="learned θ (mean ± std across seeds)")
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
        ax.bar(offsets, vals, width=width * 0.9, label=labels[x])

    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(lam) for lam in lambdas])
    apply_style(ax, xlabel="perturbation scale λ", ylabel="mean absolute policy error")
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

    ax.bar([p - width / 2 for p in positions], means, width=width, label="mean d_TV")
    ax.bar([p + width / 2 for p in positions], maxes, width=width, label="max d_TV")
    ax.axhline(lam, color="0.15", linestyle="--", linewidth=1.4, label=f"λ={lam} bound")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    apply_style(ax, xlabel="population law", ylabel="total variation distance")
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
    ax.bar(range(len(names)), values, color=[color_for(0)] + [color_for(1)] * max(0, len(names) - 1))
    if values:
        ax.axhline(values[0], color="0.25", linestyle="--", linewidth=1.2, label=f"baseline: {values[0]:.3g}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    apply_style(ax, ylabel="objective (no retraining)")
    if values:
        style_legend(ax)
    return fig, ax


def plot_trajectories(learned, optimal=None, *, ax=None):
    """
    Learned vs. optimal sample state trajectory (context.md: "learned vs
    optimal trajectories whenever applicable"). `learned`/`optimal`: shape
    (T+1,), e.g. from `scripts.test.rollout`.
    """
    fig, ax = _fig_ax(ax)
    learned = _cpu(learned)
    optimal = _cpu(optimal)
    t = range(len(learned))
    ax.step(t, learned, where="post", marker="o", label="learned")
    if optimal is not None:
        ax.step(t, optimal, where="post", marker="s", alpha=0.8, label="optimal")

    apply_style(ax, xlabel="time step", ylabel="state")
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_gaussian_flow(mu, Sigma, *, optimal_mu=None, optimal_Sigma=None, label="learned", optimal_label="optimal", ax=None):
    """
    Mean +-1 std trajectory of a time-indexed Gaussian law over states
    (context.md: "state distribution over time using learned policy"), e.g.
    `mfc.environments.lq.LQ.forward_moments`'s (mu_t^theta, Sigma_t^theta).
    `mu`/`Sigma`: shape (T+1,). `optimal_mu`/`optimal_Sigma`, if given (e.g.
    under `LQ.riccati_optimal`), drawn dashed for comparison.
    """
    fig, ax = _fig_ax(ax)
    mu, Sigma = _cpu(mu), _cpu(Sigma)
    std = Sigma.clamp_min(0).sqrt()
    t = range(len(mu))
    (line,) = ax.plot(t, mu, marker="o", label=label)
    ax.fill_between(t, mu - std, mu + std, color=line.get_color(), alpha=0.2)
    if optimal_mu is not None:
        optimal_mu, optimal_Sigma = _cpu(optimal_mu), _cpu(optimal_Sigma)
        optimal_std = optimal_Sigma.clamp_min(0).sqrt()
        (optimal_line,) = ax.plot(t, optimal_mu, linestyle="--", marker="s", label=optimal_label)
        ax.fill_between(t, optimal_mu - optimal_std, optimal_mu + optimal_std, color=optimal_line.get_color(), alpha=0.15)

    apply_style(ax, xlabel="time step", ylabel="state mean with ±1 std band")
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_lq_theta(theta, *, optimal_theta=None, ax=None):
    """
    theta_t^1 (self gain), theta_t^2 (population gain) vs t, for
    `mfc.environments.lq`'s genuinely time-indexed (T,2) parametrization
    (context.md: "theta bias, variance", specialized here to a single seed's
    per-t trajectory rather than a bias/variance-across-seeds summary, since
    each t is its own free parameter pair). `theta`/`optimal_theta`
    (e.g. `LQ.riccati_optimal()`, dashed): shape (T,2).
    """
    fig, ax = _fig_ax(ax)
    theta = _cpu(theta)
    t = range(theta.shape[0])
    labels = ["θ^1 (self gain)", "θ^2 (population gain)"]
    for i in range(2):
        (line,) = ax.plot(t, theta[:, i], marker="o", label=labels[i])
        if optimal_theta is not None:
            ax.plot(t, _cpu(optimal_theta)[:, i], color=line.get_color(), linestyle="--", alpha=0.8)

    apply_style(ax, xlabel="time step", ylabel="policy parameter value")
    _integer_xaxis(ax)
    style_legend(ax)
    return fig, ax


def plot_horizon_scaling(
    metric_by_T: dict, *, xlabel: str = "T", ylabel: str = "metric", label: str | None = None, color_index: int = 0, integer_xaxis: bool = True, ax=None
):
    """
    A scalar metric vs. a swept parameter — horizon T by default
    (context.md: "horizon scaling tests (gradient, theta, value vs time)"),
    or any other scalar sweep axis via `xlabel` (e.g. lambda: pass
    `integer_xaxis=False`, since perturbation scales aren't integer-valued).
    Call once per metric/lambda on the same `ax` to overlay several series
    (pass a distinct `color_index` each time).
    """
    fig, ax = _fig_ax(ax)
    xs = sorted(metric_by_T)
    values = [metric_by_T[x] for x in xs]
    ax.plot(xs, values, color=color_for(color_index), marker="o", label=label)

    apply_style(ax, xlabel=xlabel, ylabel=ylabel)
    if integer_xaxis:
        _integer_xaxis(ax)
    else:
        _maybe_log_lambda_axis(ax, xs)
    if label:
        style_legend(ax)
    return fig, ax
