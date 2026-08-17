from __future__ import annotations

from typing import Any, Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from .common import _spaced_indices, _state_names, _time_metric_columns


def _algorithm_order(mapping: Mapping[str, Any]) -> list[str]:
    preferred = ["simplex", "logits", "reinforce"]
    return [algorithm for algorithm in preferred if algorithm in mapping] + [algorithm for algorithm in mapping if algorithm not in preferred]


def _algorithm_label(algorithm: str) -> str:
    labels = {
        "simplex": "MF-REINFORCE simplex",
        "logits": "MF-REINFORCE logits",
        "reinforce": "classical REINFORCE",
    }
    return labels.get(str(algorithm), str(algorithm).replace("_", " "))


def _to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)


def _first_metric(row: Mapping[str, Any], *columns: str) -> float | None:
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _reference_value(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> float | None:
    for data in application_data.values():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        value = _first_metric(metrics.iloc[0].to_dict(), "reference_value")
        if value is not None:
            return value
    return None


def discrete_main_summary_table(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in _algorithm_order(application_data):
        data = application_data[algorithm]
        metrics = data.get("metrics", pd.DataFrame())
        record = metrics.iloc[0].to_dict() if not metrics.empty else {}
        row: dict[str, Any] = {
            "algorithm": _algorithm_label(algorithm),
            "learned_value_J": _first_metric(record, "value_mean", "value"),
            "reference_value_J_star": _first_metric(record, "reference_value"),
            "value_gap": _first_metric(record, "reference_value_gap"),
        }
        time_metrics = data.get("time_metrics", pd.DataFrame()).copy()
        if not time_metrics.empty:
            time_metrics = time_metrics.sort_values("time") if "time" in time_metrics else time_metrics
            final = time_metrics.iloc[-1].to_dict()
            if env_name == "twostate":
                row["final_mu_T_1"] = _first_metric(final, "mass_state_1")
                row["target_error"] = _first_metric(final, "target_abs_error")
                row["policy_l1_mean_error"] = _first_metric(record, "policy_error")
            elif env_name == "advertising":
                row["final_informed_fraction_p_T"] = _first_metric(final, "customer_fraction")
                row["cumulative_population_gain"] = float(_to_numeric(time_metrics, "population_gain").sum()) if "population_gain" in time_metrics else None
                row["cumulative_advertising_cost"] = float(_to_numeric(time_metrics, "advertising_cost").sum()) if "advertising_cost" in time_metrics else None
            elif env_name == "cybersecurity":
                row["final_infected_fraction_I_T"] = _first_metric(final, "infected_fraction")
                row["final_defended_fraction_D_T"] = _first_metric(final, "defended_fraction")
                row["cumulative_running_reward"] = float(_to_numeric(time_metrics, "running_reward").sum()) if "running_reward" in time_metrics else None
            elif env_name == "distribution-planning":
                row["terminal_target_L1"] = _first_metric(final, "target_l1")
                row["terminal_target_W1_proxy"] = _first_metric(final, "target_w1_ring_proxy")
                row["cumulative_movement_cost"] = float(_to_numeric(time_metrics, "movement_cost").sum()) if "movement_cost" in time_metrics else None
        rows.append(row)
    return pd.DataFrame(rows)


def plot_discrete_main_results(
    env_name: str,
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    plot_discrete_training_value(env_name, histories, application_data)
    if env_name == "twostate":
        _plot_twostate_main(application_data)
    elif env_name == "advertising":
        _plot_advertising_main(application_data)
    elif env_name == "cybersecurity":
        _plot_cybersecurity_main(application_data)
    elif env_name == "distribution-planning":
        _plot_distribution_planning_main(application_data)


def plot_discrete_training_value(
    env_name: str,
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    plotted = False
    for algorithm in _algorithm_order(histories):
        history = histories[algorithm].copy()
        if history.empty:
            continue
        x = _to_numeric(history, "episode") if "episode" in history else pd.Series(range(len(history)))
        y_column = "value" if "value" in history and _to_numeric(history, "value").notna().any() else "objective"
        if y_column not in history:
            continue
        y = _to_numeric(history, y_column)
        work = pd.DataFrame({"episode": x, "value": y}).dropna()
        if work.empty:
            continue
        ax.plot(work["episode"], work["value"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        plotted = True
    reference = _reference_value(application_data or {})
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=1.8, label="reference $J(\\theta^\\star)$")
    if not plotted and reference is None:
        plt.close(fig)
        return
    ax.set_title(f"{env_name}: validation value during training $J(\\theta_k)$")
    ax.set_xlabel("training episode $k$")
    ax.set_ylabel("validation value $J(\\theta_k)$; higher is better")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()


def plot_discrete_diagnostic_appendix(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    _plot_diagnostic_line_panel(
        axes[0, 0],
        diagnostics,
        table="perturbation",
        y="distance_mean",
        title="Perturbation geometry $E[d(M^\\lambda,\\mu)]$",
        ylabel="mean perturbation distance",
    )
    _plot_diagnostic_line_panel(
        axes[0, 1],
        diagnostics,
        table="gradient",
        y="mse",
        title="Gradient MSE $E[\\|\\widehat g-g\\|^2]$",
        ylabel="gradient MSE",
        log_y=True,
    )
    _plot_diagnostic_line_panel(
        axes[1, 0],
        diagnostics,
        table="score",
        y="variance_trace",
        title="Score variance $\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$",
        ylabel="score variance trace",
        log_y=True,
    )
    _plot_sensitivity_appendix_panel(axes[1, 1], diagnostics)
    fig.suptitle("Estimator diagnostics appendix", y=1.02)
    fig.tight_layout()


def _plot_diagnostic_line_panel(
    ax: Any,
    diagnostics: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    table: str,
    y: str,
    title: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    plotted = False
    for algorithm in _algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get(table, pd.DataFrame()).copy()
        if frame.empty or not {"lambda", y}.issubset(frame.columns):
            continue
        frame["lambda"] = _to_numeric(frame, "lambda")
        frame[y] = _to_numeric(frame, y)
        frame = frame.dropna(subset=["lambda", y]).sort_values("lambda")
        if frame.empty:
            continue
        ax.plot(frame["lambda"], frame[y], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("perturbation scale $\\lambda$ or $\\epsilon$")
    ax.set_ylabel(ylabel)
    if log_y and plotted:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize="small")


def _plot_sensitivity_appendix_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    plotted = False
    for algorithm in _algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get("sensitivity", pd.DataFrame()).copy()
        if frame.empty or not {"time", "eta", "mse"}.issubset(frame.columns):
            continue
        frame["time"] = _to_numeric(frame, "time")
        frame["eta"] = _to_numeric(frame, "eta")
        frame["mse"] = _to_numeric(frame, "mse")
        frame = frame.dropna(subset=["time", "eta", "mse"])
        if frame.empty:
            continue
        eta = sorted(frame["eta"].unique())[0]
        subset = frame[frame["eta"] == eta].sort_values("time")
        ax.plot(subset["time"], subset["mse"], marker="o", linewidth=2.0, label=f"{_algorithm_label(algorithm)}, $\\eta={eta:g}$")
        plotted = True
    ax.set_title("Sensitivity MSE $E[\\|\\widehat D_t-D_t\\|^2]$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("sensitivity MSE")
    if plotted:
        ax.set_yscale("log")
        ax.legend(fontsize="small")
    ax.grid(alpha=0.25)


def _plot_twostate_main(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    reference = _first_reference_frame(application_data, "reference_time_metrics", {"time", "mass_state_1"})
    target_drawn = False
    for algorithm in _algorithm_order(application_data):
        metrics = application_data[algorithm].get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty or not {"time", "mass_state_1"}.issubset(metrics.columns):
            continue
        metrics = metrics.sort_values("time")
        axes[0].plot(metrics["time"], metrics["mass_state_1"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        if "target_state_1" in metrics and not target_drawn:
            axes[0].plot(metrics["time"], metrics["target_state_1"], color="black", linestyle=":", linewidth=1.8, label="target $p$")
            target_drawn = True
    if not reference.empty:
        reference = reference.sort_values("time")
        axes[0].plot(reference["time"], reference["mass_state_1"], color="black", linestyle="--", linewidth=2.0, label="exact optimum")

    _plot_action_one_probabilities(axes[1], application_data, title="Learned policy $\\pi(a=1\\mid x)$")
    axes[0].set_title("Population objective: state-1 mass $\\mu_t(1)$")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("$\\mu_t(1)$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()

    landscape = _first_nonempty_application(application_data, "landscape", {"theta0", "theta1", "value"})
    if not landscape.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        pivot = landscape.pivot_table(index="theta1", columns="theta0", values="value", aggfunc="mean").sort_index()
        image = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_title("Exact two-parameter objective landscape $J(\\theta_0,\\theta_1)$")
        ax.set_xlabel("$\\theta_0$")
        ax.set_ylabel("$\\theta_1$")
        fig.colorbar(image, ax=ax, label="value $J$")
        fig.tight_layout()


def _plot_advertising_main(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    reference = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
    for algorithm in _algorithm_order(application_data):
        metrics = application_data[algorithm].get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty:
            continue
        metrics = metrics.sort_values("time")
        if "customer_fraction" in metrics:
            axes[0].plot(metrics["time"], metrics["customer_fraction"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        if "advertising_rate" in metrics:
            axes[1].plot(metrics["time"], metrics["advertising_rate"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        gain = _to_numeric(metrics, "population_gain").cumsum() if "population_gain" in metrics else None
        cost = _to_numeric(metrics, "advertising_cost").cumsum() if "advertising_cost" in metrics else None
        if gain is not None:
            axes[2].plot(metrics["time"], gain, marker="o", linewidth=2.0, label=f"{_algorithm_label(algorithm)} gain")
        if cost is not None:
            axes[2].plot(metrics["time"], cost, linestyle="--", linewidth=1.6, label=f"{_algorithm_label(algorithm)} ad cost")
    if not reference.empty:
        reference = reference.sort_values("time")
        if "customer_fraction" in reference:
            axes[0].plot(reference["time"], reference["customer_fraction"], color="black", linestyle="--", linewidth=2.0, label="DP reference")
        if "advertising_rate" in reference:
            axes[1].plot(reference["time"], reference["advertising_rate"], color="black", linestyle="--", linewidth=2.0, label="DP reference")
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
    axes[0].axhline(0.8, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
    axes[0].set_title("Informed fraction $p_t$")
    axes[0].set_ylabel("$p_t=P(X_t=1)$")
    axes[1].set_title("Advertising probability/intensity")
    axes[1].set_ylabel("advertising rate")
    axes[2].set_title("Objective components $\\sum_t(g_t-c_t)$")
    axes[2].set_ylabel("cumulative component")
    for ax in axes:
        ax.set_xlabel("time $t$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def _plot_cybersecurity_main(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    reference = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
    for algorithm in _algorithm_order(application_data):
        metrics = application_data[algorithm].get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty:
            continue
        metrics = metrics.sort_values("time")
        if "infected_fraction" in metrics:
            axes[0].plot(metrics["time"], metrics["infected_fraction"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        if "defended_fraction" in metrics:
            axes[1].plot(metrics["time"], metrics["defended_fraction"], marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
        if "running_reward" in metrics:
            axes[2].plot(metrics["time"], _to_numeric(metrics, "running_reward").cumsum(), marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
    if not reference.empty:
        reference = reference.sort_values("time")
        if "infected_fraction" in reference:
            axes[0].plot(reference["time"], reference["infected_fraction"], color="black", linestyle="--", linewidth=2.0, label="reference")
        if "defended_fraction" in reference:
            axes[1].plot(reference["time"], reference["defended_fraction"], color="black", linestyle="--", linewidth=2.0, label="reference")
        if "running_reward" in reference:
            axes[2].plot(reference["time"], _to_numeric(reference, "running_reward").cumsum(), color="black", linestyle="--", linewidth=2.0, label="reference")
    axes[0].set_title("Infection level $I_t=\\mu_t(DI)+\\mu_t(UI)$")
    axes[0].set_ylabel("infected fraction")
    axes[1].set_title("Protection level $D_t=\\mu_t(DI)+\\mu_t(DS)$")
    axes[1].set_ylabel("defended fraction")
    axes[2].set_title("Cumulative reward $\\sum_{s\\leq t} r_s$")
    axes[2].set_ylabel("cumulative reward")
    for ax in axes:
        ax.set_xlabel("time $t$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def _plot_distribution_planning_main(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    algorithms = _algorithm_order(application_data)
    reference = _first_reference_frame(application_data, "reference_population_flow", {"time", "state", "mass"})
    panels = algorithms + ([] if reference.empty else ["reference"])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    for ax, algorithm in zip(axes[0, : len(algorithms)], algorithms):
        flow = application_data[algorithm].get("population_flow", pd.DataFrame()).copy()
        _plot_population_heatmap(ax, flow, title=f"{_algorithm_label(algorithm)}: $\\mu_t(x)$")
    if not reference.empty:
        _plot_population_heatmap(axes[0, -1], reference, title="reference $\\mu_t^\\star(x)$")
    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    reference_metrics = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
    for algorithm in algorithms:
        metrics = application_data[algorithm].get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty:
            continue
        metrics = metrics.sort_values("time")
        for column, linestyle in [("target_l1", "-"), ("target_w1_ring_proxy", "--")]:
            if column in metrics:
                axes[0].plot(metrics["time"], metrics[column], linestyle=linestyle, marker="o", linewidth=2.0, label=f"{_algorithm_label(algorithm)} {column}")
        if "movement_cost" in metrics:
            axes[1].plot(metrics["time"], _to_numeric(metrics, "movement_cost").cumsum(), marker="o", linewidth=2.0, label=_algorithm_label(algorithm))
    if not reference_metrics.empty:
        reference_metrics = reference_metrics.sort_values("time")
        if "target_l1" in reference_metrics:
            axes[0].plot(reference_metrics["time"], reference_metrics["target_l1"], color="black", linestyle="-", linewidth=2.0, label="reference $L^1$")
        if "target_w1_ring_proxy" in reference_metrics:
            axes[0].plot(reference_metrics["time"], reference_metrics["target_w1_ring_proxy"], color="black", linestyle="--", linewidth=2.0, label="reference $W_1$")
        if "movement_cost" in reference_metrics:
            axes[1].plot(reference_metrics["time"], _to_numeric(reference_metrics, "movement_cost").cumsum(), color="black", linestyle="--", linewidth=2.0, label="reference")
    axes[0].set_title("Distance to target law $\\mu_{target}$")
    axes[0].set_ylabel("$\\|\\mu_t-\\mu_{target}\\|$ / ring $W_1$")
    axes[1].set_title("Cumulative movement cost $\\sum_{s\\leq t} c_s$")
    axes[1].set_ylabel("movement cost")
    for ax in axes:
        ax.set_xlabel("time $t$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def _plot_action_one_probabilities(ax: Any, application_data: Mapping[str, Mapping[str, pd.DataFrame]], *, title: str) -> None:
    rows: list[dict[str, Any]] = []
    for algorithm in _algorithm_order(application_data):
        policy = application_data[algorithm].get("policy", pd.DataFrame()).copy()
        if policy.empty or not {"state", "action", "probability"}.issubset(policy.columns):
            continue
        action_one = policy[pd.to_numeric(policy["action"], errors="coerce") == 1]
        for state, subset in action_one.groupby("state"):
            rows.append({"algorithm": _algorithm_label(algorithm), "state": int(state), "probability": float(_to_numeric(subset, "probability").mean())})
    reference = _first_reference_frame(application_data, "reference_policy", {"state", "action", "probability"})
    if not reference.empty:
        action_one = reference[pd.to_numeric(reference["action"], errors="coerce") == 1]
        for state, subset in action_one.groupby("state"):
            rows.append({"algorithm": "exact optimum", "state": int(state), "probability": float(_to_numeric(subset, "probability").mean())})
    table = pd.DataFrame(rows)
    if table.empty:
        ax.set_title(title)
        return
    states = sorted(table["state"].unique())
    algorithms = list(table["algorithm"].drop_duplicates())
    width = 0.8 / max(1, len(algorithms))
    offsets = [idx - (len(algorithms) - 1) / 2 for idx in range(len(algorithms))]
    for offset, algorithm in zip(offsets, algorithms):
        subset = table[table["algorithm"] == algorithm].set_index("state")
        values = [subset.loc[state, "probability"] if state in subset.index else 0.0 for state in states]
        ax.bar([state + offset * width for state in states], values, width=width, label=algorithm, alpha=0.78)
    ax.set_title(title)
    ax.set_xlabel("state $x$")
    ax.set_ylabel("$\\pi(a=1\\mid x)$")
    ax.set_xticks(states, [_state_names("twostate").get(int(state), str(state)) for state in states])
    ax.set_ylim(0.0, 1.0)


def _first_nonempty_application(
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
    key: str,
    required: set[str],
) -> pd.DataFrame:
    for data in application_data.values():
        frame = data.get(key, pd.DataFrame())
        if not frame.empty and required.issubset(frame.columns):
            return frame
    return pd.DataFrame()


def _plot_population_heatmap(ax: Any, flow: pd.DataFrame, *, title: str) -> None:
    if flow.empty or not {"time", "state", "mass"}.issubset(flow.columns):
        ax.axis("off")
        ax.set_title(title)
        return
    pivot = flow.pivot_table(index="state", columns="time", values="mass", aggfunc="mean").sort_index()
    image = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=max(1e-12, float(pivot.values.max())))
    ax.set_title(title)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("state $x$")
    ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mass")


def data_reference_flow(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    reference = data.get("reference_population_flow", pd.DataFrame())
    if reference.empty or not {"time", "state", "mass"}.issubset(reference.columns):
        return pd.DataFrame()
    return reference



def _first_reference_frame(application_data: Mapping[str, Mapping[str, pd.DataFrame]], key: str, required: set[str]) -> pd.DataFrame:
    for data in application_data.values():
        frame = data.get(key, pd.DataFrame())
        if not frame.empty and required.issubset(frame.columns):
            return frame
    return pd.DataFrame()



def _has_state_action_reference(data: Mapping[str, pd.DataFrame]) -> bool:
    reference = data.get("reference_policy", pd.DataFrame())
    return not reference.empty and {"state", "action", "probability"}.issubset(reference.columns)



def plot_population_flow(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    state_names = _state_names(env_name)
    algorithms = list(application_data)
    reference = _first_reference_frame(application_data, "reference_population_flow", {"time", "state", "mass"})
    if env_name == "distribution-planning":
        panels = [(algorithm, application_data[algorithm]["population_flow"]) for algorithm in algorithms]
        if not reference.empty:
            panels.append(("reference", reference))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), squeeze=False)
        for ax, (label, flow) in zip(axes[0], panels):
            if flow.empty:
                ax.axis("off")
                continue
            pivot = flow.pivot_table(index="state", columns="time", values="mass", aggfunc="mean").sort_index()
            image = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=max(1e-12, float(pivot.values.max())))
            ax.set_title(f"{label}: population law $\\mu_t(x)$")
            ax.set_xlabel("time $t$")
            ax.set_ylabel("state $x$")
            ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [state_names.get(int(idx), str(idx)) for idx in pivot.index])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mass")
        fig.tight_layout()
        return

    n_rows = len(algorithms) + (0 if reference.empty else 1)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), squeeze=False)
    for ax, algorithm in zip(axes[:, 0], algorithms):
        flow = application_data[algorithm]["population_flow"]
        if flow.empty:
            continue
        grouped = flow.groupby(["time", "state"], as_index=False)["mass"].mean()
        for state, subset in grouped.groupby("state"):
            label = state_names.get(int(state), f"state {state}")
            ax.plot(subset["time"], subset["mass"], marker="o", label=label)
        ax.set_title(f"{algorithm}: learned population law $\\mu_t$")
        ax.set_xlabel("time $t$")
        ax.set_ylabel("state mass $\\mu_t(x)$")
        ax.legend()
    if not reference.empty:
        ax = axes[-1, 0]
        reference_grouped = reference.groupby(["time", "state"], as_index=False)["mass"].mean()
        for state, subset in reference_grouped.groupby("state"):
            label = state_names.get(int(state), f"state {state}")
            ax.plot(subset["time"], subset["mass"], linestyle="--", linewidth=1.5, label=label)
        ax.set_title("reference population law $\\mu_t^\\star$")
        ax.set_xlabel("time $t$")
        ax.set_ylabel("state mass $\\mu_t^\\star(x)$")
        ax.legend()
    fig.tight_layout()



def plot_time_metrics(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    columns = _time_metric_columns(env_name)
    if not columns:
        return
    reference = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 3.2 * len(columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            if metrics.empty or column not in metrics:
                continue
            grouped = metrics.groupby("time", as_index=False)[column].mean()
            ax.plot(grouped["time"], grouped[column], marker="o", label=algorithm)
        if not reference.empty and column in reference:
            reference_grouped = reference.groupby("time", as_index=False)[column].mean()
            ax.plot(reference_grouped["time"], reference_grouped[column], linestyle="--", linewidth=1.8, color="black", label="reference")
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(column.replace("_", " "))
        ax.legend()
    fig.tight_layout()



def plot_policy_heatmaps(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    algorithms = list(application_data)
    reference = _first_reference_frame(application_data, "reference_policy", {"state", "action", "probability"})
    panels = algorithms + ([] if reference.empty else ["reference"])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, algorithm in zip(axes[0, : len(algorithms)], algorithms):
        policy = application_data[algorithm]["policy"]
        if policy.empty:
            continue
        table = policy.groupby(["state", "action"], as_index=False)["probability"].mean().pivot(index="state", columns="action", values="probability")
        image = ax.imshow(table.values, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{algorithm}: mean policy")
        ax.set_xlabel("action")
        ax.set_ylabel("state")
        ax.set_xticks(range(table.shape[1]), [str(col) for col in table.columns])
        ax.set_yticks(range(table.shape[0]), [_state_names(env_name).get(int(idx), str(idx)) for idx in table.index])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if not reference.empty:
        ax = axes[0, -1]
        table = reference.groupby(["state", "action"], as_index=False)["probability"].mean().pivot(index="state", columns="action", values="probability")
        image = ax.imshow(table.values, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title("reference policy $\\pi^\\star(a|x)$")
        ax.set_xlabel("action $a$")
        ax.set_ylabel("state $x$")
        ax.set_xticks(range(table.shape[1]), [str(col) for col in table.columns])
        ax.set_yticks(range(table.shape[0]), [_state_names(env_name).get(int(idx), str(idx)) for idx in table.index])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()



def plot_discrete_application_details(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name == "distribution-planning":
        algorithms = list(application_data)
        reference = _first_reference_frame(application_data, "reference_population_flow", {"time", "state", "mass"})
        panels = algorithms + ([] if reference.empty else ["reference"])
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
        for ax, algorithm in zip(axes[0, : len(algorithms)], algorithms):
            flow = application_data[algorithm]["population_flow"]
            if flow.empty:
                continue
            pivot = flow.pivot_table(index="state", columns="time", values="mass", aggfunc="mean")
            image = ax.imshow(pivot.values, aspect="auto", vmin=0.0)
            ax.set_title(f"{algorithm}: $\\mu_t(x)$")
            ax.set_xlabel("time $t$")
            ax.set_ylabel("state $x$")
            ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
            fig.colorbar(image, ax=ax, label="mass")
        if not reference.empty:
            ax = axes[0, -1]
            pivot = reference.pivot_table(index="state", columns="time", values="mass", aggfunc="mean")
            image = ax.imshow(pivot.values, aspect="auto", vmin=0.0)
            ax.set_title("reference $\\mu_t^\\star(x)$")
            ax.set_xlabel("time $t$")
            ax.set_ylabel("state $x$")
            ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
            fig.colorbar(image, ax=ax, label="mass")
        fig.tight_layout()
        fig, ax = plt.subplots(figsize=(9, 4))
        any_flux = False
        for algorithm, data in application_data.items():
            flux = data.get("transport_flux", pd.DataFrame())
            if flux.empty:
                continue
            grouped = flux.groupby("time", as_index=False)["mass_flux"].sum()
            ax.plot(grouped["time"], grouped["mass_flux"], marker="o", label=algorithm)
            any_flux = True
        if any_flux:
            ax.set_title("Transport flux $\\sum_x |\\Phi_t(x)|$ around the ring")
            ax.set_xlabel("time $t$")
            ax.set_ylabel("total mass flux")
            ax.legend()
            fig.tight_layout()
        else:
            plt.close(fig)
        return

    if env_name == "advertising":
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        reference = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"].copy()
            if metrics.empty:
                metrics = pd.DataFrame()
            else:
                metrics = metrics.sort_values("time")
                if "customer_fraction" in metrics:
                    axes[0].plot(metrics["time"], metrics["customer_fraction"], marker="o", label=f"{algorithm}: customer")
                for column, label in [("advertising_cost", "ad cost"), ("population_gain", "population gain")]:
                    if column in metrics:
                        axes[1].plot(metrics["time"], metrics[column].cumsum(), marker="o", label=f"{algorithm}: {label}")
            finite = data.get("finite_population", pd.DataFrame())
            if not finite.empty and "particles" in finite:
                grouped = finite.groupby("particles", as_index=False)["abs_error"].mean()
                axes[2].plot(grouped["particles"], grouped["abs_error"], marker="o", label=algorithm)
        if not reference.empty:
            reference = reference.sort_values("time")
            if "customer_fraction" in reference:
                axes[0].plot(reference["time"], reference["customer_fraction"], linestyle="--", linewidth=1.8, color="black", label="DP reference")
            for column, label in [("advertising_cost", "DP ad cost"), ("population_gain", "DP population gain")]:
                if column in reference:
                    axes[1].plot(reference["time"], reference[column].cumsum(), linestyle="--", linewidth=1.8, label=label)
        axes[0].axhline(0.5, linestyle="--", color="black", linewidth=1.0, alpha=0.5)
        axes[0].axhline(0.8, linestyle=":", color="black", linewidth=1.0, alpha=0.5)
        axes[0].set_title("Informed fraction $p_t$ and target levels")
        axes[0].set_ylabel("$p_t$")
        axes[1].set_title("Cumulative reward-cost components")
        axes[1].set_ylabel("cumulative value")
        axes[2].set_title("Finite-population gap")
        axes[2].set_ylabel("absolute trajectory error")
        for ax in axes:
            ax.set_xlabel("time $t$")
            ax.legend(fontsize="small")
        axes[2].set_xlabel("particles")
        fig.tight_layout()
        reference_policies = {
            algorithm: data.get("reference_policy", pd.DataFrame())
            for algorithm, data in application_data.items()
            if not data.get("reference_policy", pd.DataFrame()).empty
            and {"time", "customer_fraction", "oracle_ad_probability"}.issubset(data.get("reference_policy", pd.DataFrame()).columns)
        }
        if reference_policies:
            fig, axes = plt.subplots(1, len(reference_policies), figsize=(5.5 * len(reference_policies), 4), squeeze=False)
            for ax, (algorithm, policy) in zip(axes[0], reference_policies.items()):
                pivot = policy.pivot_table(index="time", columns="customer_fraction", values="oracle_ad_probability", aggfunc="mean")
                image = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0, origin="lower")
                ax.set_title(f"{algorithm}: finite-horizon DP ad policy")
                ax.set_xlabel("customer fraction")
                ax.set_ylabel("time")
                x_ticks = _spaced_indices(len(pivot.columns), 5)
                ax.set_xticks(x_ticks, [f"{float(pivot.columns[idx]):.2f}" for idx in x_ticks])
                ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
                fig.colorbar(image, ax=ax)
            fig.tight_layout()
        return

    if env_name == "cybersecurity":
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        reference = _first_reference_frame(application_data, "reference_time_metrics", {"time"})
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"].copy()
            if metrics.empty:
                continue
            metrics = metrics.sort_values("time")
            for column, label in [("infected_fraction", "infected"), ("defended_fraction", "defended")]:
                if column in metrics:
                    axes[0].plot(metrics["time"], metrics[column], marker="o", label=f"{algorithm}: {label}")
            if "running_reward" in metrics:
                axes[1].plot(metrics["time"], metrics["running_reward"].cumsum(), marker="o", label=f"{algorithm}: cumulative reward")
            if "update_rate" in metrics:
                axes[1].plot(metrics["time"], metrics["update_rate"], linestyle="--", marker="o", label=f"{algorithm}: update rate")
        if not reference.empty:
            reference = reference.sort_values("time")
            for column, label in [("infected_fraction", "reference infected"), ("defended_fraction", "reference defended")]:
                if column in reference:
                    axes[0].plot(reference["time"], reference[column], linestyle="--", linewidth=1.8, label=label)
            if "running_reward" in reference:
                axes[1].plot(reference["time"], reference["running_reward"].cumsum(), linestyle="--", linewidth=1.8, label="reference cumulative reward")
        axes[0].set_title("Cybersecurity fractions $I_t$ and $D_t$")
        axes[0].set_ylabel("population fraction")
        axes[1].set_title("Cumulative reward and switching rate")
        axes[1].set_ylabel("reward / rate")
        for ax in axes:
            ax.set_xlabel("time $t$")
            ax.legend(fontsize="small")
        fig.tight_layout()
        return

    if env_name == "twostate":
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        reference = _first_reference_frame(application_data, "reference_time_metrics", {"time", "mass_state_1"})
        target_drawn = False
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            policy = data["policy"]
            if not metrics.empty and "mass_state_1" in metrics:
                axes[0].plot(metrics["time"], metrics["mass_state_1"], marker="o", label=f"{algorithm}: mass state 1")
                if "target_state_1" in metrics and not target_drawn:
                    axes[0].plot(metrics["time"], metrics["target_state_1"], linestyle="--", color="black", label="target $p$")
                    target_drawn = True
            if not policy.empty:
                grouped = policy.groupby(["time", "state"], as_index=False)["probability"].max()
                for state, subset in grouped.groupby("state"):
                    axes[1].plot(subset["time"], subset["probability"], marker="o", label=f"{algorithm}: state {state}")
            landscape = data.get("landscape", pd.DataFrame())
            if not landscape.empty:
                pivot = landscape.pivot_table(index="theta1", columns="theta0", values="value")
                image = axes[2].imshow(pivot.values, origin="lower", aspect="auto")
                axes[2].set_title(f"{algorithm}: objective landscape")
                axes[2].set_xlabel("theta0")
                axes[2].set_ylabel("theta1")
                fig.colorbar(image, ax=axes[2])
        if not reference.empty:
            reference = reference.sort_values("time")
            axes[0].plot(reference["time"], reference["mass_state_1"], linestyle=":", linewidth=2.0, color="black", label="exact optimum")
        axes[0].set_title("State-1 mass $\\mu_t(1)$ vs target $p$")
        axes[0].set_ylabel("$\\mu_t(1)$")
        axes[1].set_title("Dominant action probability $\\max_a \\pi_t(a|x)$")
        axes[1].set_ylabel("probability")
        for ax in axes[:2]:
            ax.set_xlabel("time $t$")
            ax.legend(fontsize="small")
        axes[2].legend(fontsize="small")
        fig.tight_layout()


__all__ = [
    "data_reference_flow",
    "discrete_main_summary_table",
    "plot_discrete_application_details",
    "plot_discrete_diagnostic_appendix",
    "plot_discrete_main_results",
    "plot_discrete_training_value",
    "plot_policy_heatmaps",
    "plot_population_flow",
    "plot_time_metrics",
]
