from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ...core.controls import load_control
from ...core.registry import build_environment


def _first_nonempty(application_data: Mapping[str, Mapping[str, pd.DataFrame]], key: str, required: set[str]) -> pd.DataFrame:
    for data in application_data.values():
        frame = data.get(key, pd.DataFrame())
        if not frame.empty and required.issubset(frame.columns):
            return frame
    return pd.DataFrame()


def _continuous_algorithm_order(mapping: Mapping[str, Any]) -> list[str]:
    preferred = ["continuous-mfreinforce", "continuous-oracle-sensitivity", "reinforce", "exact-gradient"]
    return [algorithm for algorithm in preferred if algorithm in mapping] + [algorithm for algorithm in mapping if algorithm not in preferred]


def _continuous_algorithm_label(algorithm: str) -> str:
    labels = {
        "continuous-mfreinforce": "continuous MF-REINFORCE",
        "continuous-oracle-sensitivity": "MF-REINFORCE, oracle $D_t$",
        "reinforce": "classical REINFORCE",
        "exact-gradient": "exact gradient",
    }
    return labels.get(str(algorithm), str(algorithm).replace("_", " "))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)


def lq_main_summary_table(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm, data in application_data.items():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        cost = _first_numeric(row, ["cost", "objective"])
        optimal_cost = _first_numeric(row, ["optimal_cost"])
        rows.append(
            {
                "algorithm": _lq_algorithm_label(algorithm),
                "learned_cost_J": cost,
                "riccati_cost_J_star": optimal_cost,
                "cost_gap": _first_numeric(row, ["objective_gap"]),
            }
        )
    return pd.DataFrame(rows)


def plot_lq_main_results(
    bundle: Mapping[str, Any],
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
    studies: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    plot_lq_validation_reward(histories, application_data)
    plot_lq_jlambda_comparison(bundle, studies=studies)
    plot_lq_theta_comparison(application_data)
    plot_lq_learned_policy_comparison(application_data)


def plot_lq_validation_reward(
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    for algorithm in _lq_algorithm_order(histories):
        history = histories.get(algorithm, pd.DataFrame()).copy()
        if history.empty or "episode" not in history:
            continue
        reward = _lq_validation_reward_series(history)
        if reward is None:
            continue
        history["episode"] = pd.to_numeric(history["episode"], errors="coerce")
        history["validation_reward"] = reward
        history = history.dropna(subset=["episode", "validation_reward"])
        if history.empty:
            continue
        ax.plot(
            history["episode"],
            history["validation_reward"],
            marker="o",
            linewidth=2.0,
            label=_lq_algorithm_label(algorithm),
        )
        plotted = True
    optimal_reward = _lq_optimal_reward(application_data or {})
    if optimal_reward is not None:
        ax.axhline(optimal_reward, color="black", linestyle="--", linewidth=1.8, label="Riccati optimum $-J(\\theta^\\star)$")
    if not plotted and optimal_reward is None:
        plt.close(fig)
        return
    ax.set_title("Validation reward during training: $-J(\\theta_k)$")
    ax.set_xlabel("training episode $k$")
    ax.set_ylabel("validation reward $-J(\\theta_k)$; higher is better")
    ax.legend(fontsize="small", ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_lq_jlambda_comparison(
    bundle: Mapping[str, Any],
    *,
    studies: Mapping[str, pd.DataFrame] | None = None,
    lambdas: Iterable[float] | None = None,
) -> None:
    controls = _lq_controls_from_bundle(bundle)
    if not controls:
        return
    lambda_values = list(lambdas) if lambdas is not None else _lq_lambda_grid_from_studies(studies or {})
    if 0.0 not in lambda_values:
        lambda_values = [0.0, *lambda_values]
    lambda_values = sorted({float(value) for value in lambda_values})
    if not lambda_values:
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    oracle_drawn = False
    for algorithm, env, control in controls:
        costs = [float(env.exact_cost(control, lambda_=lambda_value).detach().cpu().item()) for lambda_value in lambda_values]
        ax.plot(lambda_values, costs, marker="o", linewidth=2.0, label=_lq_algorithm_label(algorithm))
        if not oracle_drawn and hasattr(env, "riccati_policy"):
            optimal = env.riccati_policy()
            optimal_costs = [float(env.exact_cost(optimal, lambda_=lambda_value).detach().cpu().item()) for lambda_value in lambda_values]
            ax.plot(lambda_values, optimal_costs, color="black", linestyle="--", linewidth=1.8, label="Riccati $\\theta^\\star$")
            oracle_drawn = True
    ax.set_title("Perturbed objective comparison: $J^\\lambda(\\hat\\theta)$")
    ax.set_xlabel("law perturbation scale $\\lambda$")
    ax.set_ylabel("cost $J^\\lambda(\\theta)$; lower is better")
    ax.legend(fontsize="small", ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_lq_theta_comparison(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    optimal_drawn: set[int] = set()
    plotted = False
    names = {
        0: "state-feedback gain $\\theta_{t,0}$",
        1: "mean-field gain $\\theta_{t,1}$",
    }
    for algorithm in _lq_algorithm_order(application_data):
        policy = application_data.get(algorithm, {}).get("policy", pd.DataFrame()).copy()
        if policy.empty or not {"time", "coordinate", "value"}.issubset(policy.columns):
            continue
        policy["time"] = pd.to_numeric(policy["time"], errors="coerce")
        policy["coordinate"] = pd.to_numeric(policy["coordinate"], errors="coerce")
        for coordinate, subset in policy.dropna(subset=["time", "coordinate"]).groupby("coordinate"):
            coordinate_int = int(coordinate)
            if coordinate_int not in {0, 1}:
                continue
            ax = axes[coordinate_int]
            subset = subset.sort_values("time")
            ax.plot(subset["time"], subset["value"], marker="o", linewidth=2.0, label=_lq_algorithm_label(algorithm))
            if coordinate_int not in optimal_drawn and "optimal" in subset:
                ax.plot(
                    subset["time"],
                    subset["optimal"],
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    label="Riccati $\\theta^\\star$",
                )
                optimal_drawn.add(coordinate_int)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    for coordinate, ax in enumerate(axes):
        ax.set_title(names[coordinate])
        ax.set_xlabel("time $t$")
        ax.set_ylabel(names[coordinate])
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def plot_lq_learned_policy_comparison(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    reference_drawn = False
    plotted = False
    for algorithm in _lq_algorithm_order(application_data):
        metrics = application_data.get(algorithm, {}).get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty or not {"time", "mean", "variance"}.issubset(metrics.columns):
            continue
        metrics["time"] = pd.to_numeric(metrics["time"], errors="coerce")
        metrics = metrics.dropna(subset=["time"]).sort_values("time")
        axes[0].plot(metrics["time"], metrics["mean"], marker="o", linewidth=2.0, label=_lq_algorithm_label(algorithm))
        axes[1].plot(metrics["time"], metrics["variance"], marker="o", linewidth=2.0, label=_lq_algorithm_label(algorithm))
        if not reference_drawn and {"optimal_mean", "optimal_variance"}.issubset(metrics.columns):
            axes[0].plot(metrics["time"], metrics["optimal_mean"], color="black", linestyle="--", linewidth=1.8, label="Riccati")
            axes[1].plot(metrics["time"], metrics["optimal_variance"], color="black", linestyle="--", linewidth=1.8, label="Riccati")
            reference_drawn = True
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Mean trajectory $m_t=E[X_t]$")
    axes[0].set_ylabel("population mean $m_t$")
    axes[1].set_title("Variance trajectory $v_t=\\operatorname{Var}(X_t)$")
    axes[1].set_ylabel("population variance $v_t$")
    for ax in axes:
        ax.set_xlabel("time $t$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def plot_lq_diagnostic_appendix(
    diagnostics: Mapping[str, Mapping[str, pd.DataFrame]],
    studies: Mapping[str, pd.DataFrame] | None = None,
    grid_metrics: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    _plot_lq_perturbation_panel(axes[0, 0], diagnostics)
    _plot_lq_gradient_panel(axes[0, 1], diagnostics)
    _plot_lq_score_panel(axes[1, 0], diagnostics)
    _plot_lq_sensitivity_panel(axes[1, 1], diagnostics)
    fig.suptitle("Estimator diagnostics appendix for LQ", y=1.02)
    fig.tight_layout()


def _plot_lq_perturbation_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    for algorithm in _lq_algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get("perturbation", pd.DataFrame()).copy()
        if frame.empty or not {"lambda", "distance_mean"}.issubset(frame.columns):
            continue
        frame["lambda"] = pd.to_numeric(frame["lambda"], errors="coerce")
        frame["distance_mean"] = pd.to_numeric(frame["distance_mean"], errors="coerce")
        frame = frame.dropna(subset=["lambda", "distance_mean"]).sort_values("lambda")
        if frame.empty:
            continue
        ax.plot(frame["lambda"], frame["distance_mean"], marker="o", label=_lq_algorithm_label(algorithm))
    ax.set_title("Perturbation geometry $E[d(M^\\lambda,\\mu)]$")
    ax.set_xlabel("perturbation scale $\\lambda$")
    ax.set_ylabel("mean law-coordinate distance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")


def _plot_lq_gradient_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    for algorithm in _lq_algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get("gradient", pd.DataFrame()).copy()
        if frame.empty or not {"lambda", "relative_bias"}.issubset(frame.columns):
            continue
        frame["lambda"] = pd.to_numeric(frame["lambda"], errors="coerce")
        frame["relative_bias"] = pd.to_numeric(frame["relative_bias"], errors="coerce")
        frame = frame.dropna(subset=["lambda", "relative_bias"]).sort_values("lambda")
        if frame.empty:
            continue
        ax.plot(frame["lambda"], frame["relative_bias"], marker="o", label=_lq_algorithm_label(algorithm))
    ax.set_title("Gradient bias $\\|E[\\hat g]-g\\|/\\|g\\|$")
    ax.set_xlabel("perturbation scale $\\lambda$")
    ax.set_ylabel("relative bias")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")


def _plot_lq_score_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    for algorithm in _lq_algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get("score", pd.DataFrame()).copy()
        if frame.empty or not {"lambda", "variance_trace"}.issubset(frame.columns):
            continue
        frame["lambda"] = pd.to_numeric(frame["lambda"], errors="coerce")
        frame["variance_trace"] = pd.to_numeric(frame["variance_trace"], errors="coerce")
        frame = frame.dropna(subset=["lambda", "variance_trace"]).sort_values("lambda")
        if frame.empty:
            continue
        ax.plot(frame["lambda"], frame["variance_trace"], marker="o", label=_lq_algorithm_label(algorithm))
    ax.set_title("Score variance $\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$")
    ax.set_xlabel("perturbation scale $\\lambda$")
    ax.set_ylabel("variance trace")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")


def _plot_lq_sensitivity_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frame = diagnostics.get("continuous-mfreinforce", {}).get("sensitivity", pd.DataFrame()).copy()
    if frame.empty or not {"time", "eta", "mse"}.issubset(frame.columns):
        ax.set_title("Sensitivity MSE by time")
        return
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["eta"] = pd.to_numeric(frame["eta"], errors="coerce")
    frame["mse"] = pd.to_numeric(frame["mse"], errors="coerce")
    for eta, subset in frame.dropna(subset=["time", "eta", "mse"]).groupby("eta"):
        subset = subset.sort_values("time")
        ax.plot(subset["time"], subset["mse"], marker="o", label=f"$\\eta={eta:g}$")
    ax.set_title("Sensitivity error $E[\\|\\hat D_t-D_t\\|^2]$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")


def _lq_controls_from_bundle(bundle: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    controls: list[tuple[str, Any, Any]] = []
    train_paths = bundle.get("train", {}) if isinstance(bundle, Mapping) else {}
    for algorithm, path in train_paths.items():
        checkpoint = Path(path) / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("env") != "lq":
            continue
        env_config = dict(payload.get("env_config", {}))
        env_config["device"] = "cpu"
        config = {
            "env": "lq",
            "algorithm": payload.get("algorithm", algorithm),
            "env_config": env_config,
            "algorithm_config": payload.get("algorithm_config", {}),
        }
        spec, env = build_environment(config)
        control = load_control(spec, env, payload["control"], trainable=False)
        controls.append((str(algorithm), env, control))
    return controls


def _lq_algorithm_order(mapping: Mapping[str, Any]) -> list[str]:
    preferred = ["continuous-mfreinforce", "continuous-oracle-sensitivity", "reinforce", "exact-gradient"]
    return [algorithm for algorithm in preferred if algorithm in mapping] + [algorithm for algorithm in mapping if algorithm not in preferred]


def _lq_algorithm_label(algorithm: str) -> str:
    labels = {
        "continuous-mfreinforce": "continuous MF-REINFORCE",
        "continuous-oracle-sensitivity": "MF-REINFORCE, oracle $D_t$",
        "reinforce": "classical REINFORCE",
        "exact-gradient": "exact gradient",
    }
    return labels.get(str(algorithm), str(algorithm).replace("_", " "))


def _lq_validation_reward_series(frame: pd.DataFrame) -> pd.Series | None:
    if "value" in frame and pd.to_numeric(frame["value"], errors="coerce").notna().any():
        return pd.to_numeric(frame["value"], errors="coerce")
    if "cost" in frame and pd.to_numeric(frame["cost"], errors="coerce").notna().any():
        return -pd.to_numeric(frame["cost"], errors="coerce")
    if "objective" in frame and pd.to_numeric(frame["objective"], errors="coerce").notna().any():
        return -pd.to_numeric(frame["objective"], errors="coerce")
    return None


def _lq_optimal_reward(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> float | None:
    for data in application_data.values():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        optimal_objective = _first_numeric(row, ["optimal_objective"])
        if optimal_objective is not None:
            return optimal_objective
        optimal_cost = _first_numeric(row, ["optimal_cost"])
        if optimal_cost is not None:
            return -optimal_cost
    return None


def _first_numeric(row: Mapping[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _lq_lambda_grid_from_studies(studies: Mapping[str, pd.DataFrame]) -> list[float]:
    values: set[float] = set()
    for key in ("lambda_training", "optimizer_bias"):
        frame = studies.get(key, pd.DataFrame())
        for column in ("lambda", "algorithm_config.lambda"):
            if frame.empty or column not in frame:
                continue
            parsed = pd.to_numeric(frame[column], errors="coerce").dropna()
            values.update(float(value) for value in parsed.tolist())
    return sorted(values or {0.025, 0.1, 0.2})


def portfolio_main_summary_table(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in _continuous_algorithm_order(application_data):
        metrics = application_data[algorithm].get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        record = metrics.iloc[0].to_dict()
        rows.append(
            {
                "algorithm": _continuous_algorithm_label(algorithm),
                "learned_objective_J": _first_numeric(record, ["objective", "value"]),
                "closed_form_objective_J_star": _first_numeric(record, ["optimal_objective"]),
                "objective_gap": _first_numeric(record, ["objective_gap"]),
                "terminal_mean": _first_numeric(record, ["terminal_mean"]),
                "terminal_variance": _first_numeric(record, ["terminal_variance"]),
                "downside_probability": _first_numeric(record, ["downside_probability"]),
            }
        )
    return pd.DataFrame(rows)


def plot_portfolio_main_results(
    bundle: Mapping[str, Any],
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
    studies: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    plot_portfolio_validation_reward(histories, application_data)
    plot_portfolio_jlambda_comparison(bundle, studies=studies)
    plot_portfolio_policy_comparison(application_data)
    plot_portfolio_wealth_comparison(application_data)


def plot_portfolio_validation_reward(
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    for algorithm in _continuous_algorithm_order(histories):
        history = histories.get(algorithm, pd.DataFrame()).copy()
        if history.empty or "episode" not in history:
            continue
        y_column = "value" if "value" in history and _numeric(history, "value").notna().any() else "objective"
        if y_column not in history:
            continue
        work = pd.DataFrame({"episode": _numeric(history, "episode"), "value": _numeric(history, y_column)}).dropna()
        if work.empty:
            continue
        ax.plot(work["episode"], work["value"], marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
        plotted = True
    optimal = _portfolio_optimal_objective(application_data or {})
    if optimal is not None:
        ax.axhline(optimal, color="black", linestyle="--", linewidth=1.8, label="closed-form optimum $J(\\theta^\\star)$")
    if not plotted and optimal is None:
        plt.close(fig)
        return
    ax.set_title("Validation objective during training: $J(\\theta_k)$")
    ax.set_xlabel("training episode $k$")
    ax.set_ylabel("mean-variance objective $J(\\theta_k)$; higher is better")
    ax.legend(fontsize="small", ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_portfolio_jlambda_comparison(
    bundle: Mapping[str, Any],
    *,
    studies: Mapping[str, pd.DataFrame] | None = None,
    lambdas: Iterable[float] | None = None,
) -> None:
    controls = _exact_controls_from_bundle(bundle, "portfolio")
    if not controls:
        return
    lambda_values = list(lambdas) if lambdas is not None else _lq_lambda_grid_from_studies(studies or {})
    if 0.0 not in lambda_values:
        lambda_values = [0.0, *lambda_values]
    lambda_values = sorted({float(value) for value in lambda_values})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    oracle_drawn = False
    for algorithm, env, control in controls:
        if not hasattr(env, "exact_objective"):
            continue
        values = [float(env.exact_objective(control, lambda_=lambda_value).detach().cpu().item()) for lambda_value in lambda_values]
        ax.plot(lambda_values, values, marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
        if not oracle_drawn and hasattr(env, "optimal_policy"):
            optimal = env.optimal_policy()
            optimal_values = [float(env.exact_objective(optimal, lambda_=lambda_value).detach().cpu().item()) for lambda_value in lambda_values]
            ax.plot(lambda_values, optimal_values, color="black", linestyle="--", linewidth=1.8, label="closed-form $\\theta^\\star$")
            oracle_drawn = True
    ax.set_title("Perturbed objective comparison: $J^\\lambda(\\hat\\theta)$")
    ax.set_xlabel("law perturbation scale $\\lambda$")
    ax.set_ylabel("mean-variance objective $J^\\lambda(\\theta)$; higher is better")
    ax.legend(fontsize="small", ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_portfolio_policy_comparison(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    optimal_drawn: set[int] = set()
    plotted = False
    names = {
        0: "wealth-feedback allocation $k_t$",
        1: "baseline allocation $\\ell_t$",
    }
    for algorithm in _continuous_algorithm_order(application_data):
        policy = application_data.get(algorithm, {}).get("policy", pd.DataFrame()).copy()
        if policy.empty or not {"time", "coordinate", "value"}.issubset(policy.columns):
            continue
        policy["time"] = _numeric(policy, "time")
        policy["coordinate"] = _numeric(policy, "coordinate")
        for coordinate, subset in policy.dropna(subset=["time", "coordinate"]).groupby("coordinate"):
            coordinate_int = int(coordinate)
            if coordinate_int not in {0, 1}:
                continue
            ax = axes[coordinate_int]
            subset = subset.sort_values("time")
            ax.plot(subset["time"], subset["value"], marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
            if coordinate_int not in optimal_drawn and "optimal" in subset:
                ax.plot(subset["time"], subset["optimal"], color="black", linestyle="--", linewidth=1.8, label="closed-form $\\theta^\\star$")
                optimal_drawn.add(coordinate_int)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    for coordinate, ax in enumerate(axes):
        ax.set_title(names[coordinate])
        ax.set_xlabel("time $t$")
        ax.set_ylabel(names[coordinate])
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def plot_portfolio_wealth_comparison(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    reference_drawn = False
    plotted = False
    for algorithm in _continuous_algorithm_order(application_data):
        metrics = application_data.get(algorithm, {}).get("time_metrics", pd.DataFrame()).copy()
        if not metrics.empty and {"time", "mean", "variance"}.issubset(metrics.columns):
            metrics = metrics.sort_values("time")
            axes[0].plot(metrics["time"], metrics["mean"], marker="o", linewidth=2.0, label=f"{_continuous_algorithm_label(algorithm)} mean")
            axes[0].plot(metrics["time"], metrics["variance"], linestyle="--", marker="o", linewidth=1.6, label=f"{_continuous_algorithm_label(algorithm)} variance")
            if not reference_drawn and {"optimal_mean", "optimal_variance"}.issubset(metrics.columns):
                axes[0].plot(metrics["time"], metrics["optimal_mean"], color="black", linestyle="-", linewidth=2.0, label="oracle mean")
                axes[0].plot(metrics["time"], metrics["optimal_variance"], color="black", linestyle="--", linewidth=2.0, label="oracle variance")
                reference_drawn = True
            plotted = True
        frontier = application_data.get(algorithm, {}).get("efficient_frontier", pd.DataFrame()).copy()
        if not frontier.empty and {"terminal_mean", "terminal_variance"}.issubset(frontier.columns):
            axes[1].plot(
                frontier["terminal_variance"],
                frontier["terminal_mean"],
                marker="o",
                linewidth=2.0,
                label=_continuous_algorithm_label(algorithm),
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Wealth moments under learned policy")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("wealth mean / variance")
    axes[1].set_title("Terminal mean-variance frontier")
    axes[1].set_xlabel("$\\operatorname{Var}(W_T)$")
    axes[1].set_ylabel("$E[W_T]$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()


def pathwise_main_summary_table(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in _continuous_algorithm_order(application_data):
        metrics = application_data[algorithm].get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        record = metrics.iloc[0].to_dict()
        row = {
            "algorithm": _continuous_algorithm_label(algorithm),
            "controlled_cost_J": _first_numeric(record, ["objective"]),
            "free_cost": _first_numeric(record, ["free_objective"]),
            "heuristic_cost": _first_numeric(record, ["heuristic_objective"]),
            "cumulative_control_energy": _first_numeric(record, ["cumulative_control_energy"]),
        }
        if env_name == "cucker-smale":
            row["alignment_time"] = _first_numeric(record, ["alignment_time"])
            row["heuristic_best_kappa"] = _first_numeric(record, ["heuristic_best_kappa"])
        elif env_name == "kuramoto":
            row["synchronization_time"] = _first_numeric(record, ["synchronization_time"])
            row["phase_locking_time"] = _first_numeric(record, ["phase_locking_time"])
            row["heuristic_best_kappa"] = _first_numeric(record, ["heuristic_best_kappa"])
            row["heuristic_best_nu"] = _first_numeric(record, ["heuristic_best_nu"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pathwise_main_results(
    env_name: str,
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    plot_pathwise_training_cost(env_name, histories, application_data)
    plot_pathwise_dynamics(env_name, application_data)
    plot_pathwise_snapshots(env_name, application_data)
    plot_pathwise_control_energy(env_name, application_data)


def plot_pathwise_training_cost(
    env_name: str,
    histories: Mapping[str, pd.DataFrame],
    application_data: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    for algorithm in _continuous_algorithm_order(histories):
        history = histories.get(algorithm, pd.DataFrame()).copy()
        if history.empty:
            continue
        if "objective" in history and _numeric(history, "objective").notna().any():
            y = _numeric(history, "objective")
        elif "cost" in history and _numeric(history, "cost").notna().any():
            y = _numeric(history, "cost")
        elif "value" in history and _numeric(history, "value").notna().any():
            y = -_numeric(history, "value")
        else:
            continue
        x = _numeric(history, "episode") if "episode" in history else pd.Series(range(len(history)))
        work = pd.DataFrame({"episode": x, "cost": y}).dropna()
        if work.empty:
            continue
        ax.plot(work["episode"], work["cost"], marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
        plotted = True
    free, heuristic = _pathwise_reference_costs(application_data or {})
    if free is not None:
        ax.axhline(free, color="0.25", linestyle=":", linewidth=1.8, label="free dynamics")
    if heuristic is not None:
        ax.axhline(heuristic, color="black", linestyle="--", linewidth=1.8, label="heuristic control")
    if not plotted and free is None and heuristic is None:
        plt.close(fig)
        return
    ax.set_title(f"{env_name}: validation cost during training $J(\\theta_k)$")
    ax.set_xlabel("training episode $k$")
    ax.set_ylabel("control cost $J(\\theta_k)$; lower is better")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()


def plot_pathwise_dynamics(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    if env_name == "cucker-smale":
        panels = [
            ("velocity_dispersion", "Velocity dispersion $N^{-1}\\sum_i |v_i-\\bar v|^2$"),
            ("spatial_diameter", "Spatial diameter"),
        ]
    elif env_name == "kuramoto":
        panels = [
            ("order_parameter", "Order parameter $R_t=|N^{-1}\\sum_j e^{i\\theta_j}|$"),
            ("target_aligned_order", "Target-aligned order parameter"),
        ]
    else:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, (metric, title) in zip(axes, panels):
        _plot_pathwise_metric(ax, application_data, metric, title=title)
    fig.tight_layout()


def plot_pathwise_snapshots(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    algorithms = _continuous_algorithm_order(application_data)
    snapshot_sets = [(algorithm, application_data[algorithm].get("snapshots", pd.DataFrame()).copy()) for algorithm in algorithms]
    snapshot_sets = [(algorithm, frame) for algorithm, frame in snapshot_sets if not frame.empty]
    if not snapshot_sets:
        return
    times = sorted(snapshot_sets[0][1]["time"].drop_duplicates().tolist())
    times = [times[idx] for idx in _evenly_spaced_indices(len(times), max_count=4)]
    fig, axes = plt.subplots(len(snapshot_sets), len(times), figsize=(4 * len(times), 3.6 * len(snapshot_sets)), squeeze=False)
    for row, (algorithm, snapshots) in enumerate(snapshot_sets):
        for col, time_value in enumerate(times):
            ax = axes[row, col]
            subset = snapshots[snapshots["time"] == time_value]
            if env_name == "cucker-smale":
                ax.scatter(subset["position"], subset["velocity"], s=16, alpha=0.75)
                ax.set_xlabel("position $x$")
                ax.set_ylabel("velocity $v$")
            elif env_name == "kuramoto":
                ax.scatter(subset["cos_phase"], subset["sin_phase"], s=16, alpha=0.75)
                circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linewidth=1.0)
                ax.add_patch(circle)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("$\\cos\\theta$")
                ax.set_ylabel("$\\sin\\theta$")
            ax.set_title(f"{_continuous_algorithm_label(algorithm)}, $t={time_value}$")
            ax.grid(alpha=0.2)
    fig.suptitle("Representative controlled particle snapshots", y=1.02)
    fig.tight_layout()


def plot_pathwise_control_energy(env_name: str, application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _plot_pathwise_metric(
        ax,
        application_data,
        "control_energy",
        title="Cumulative control energy $\\sum_{s\\leq t} E[|u_s|^2]$",
        cumulative=True,
    )
    fig.tight_layout()


def plot_continuous_diagnostic_appendix(
    env_name: str,
    diagnostics: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    _plot_continuous_diag_panel(
        axes[0, 0],
        diagnostics,
        table="perturbation",
        y="distance_mean",
        title="Perturbation geometry $E[d(M^\\lambda,\\mu)]$",
        ylabel="mean law distance",
    )
    _plot_continuous_diag_panel(
        axes[0, 1],
        diagnostics,
        table="gradient",
        y="mse",
        title="Gradient MSE $E[\\|\\widehat g-g\\|^2]$",
        ylabel="gradient MSE",
        log_y=True,
    )
    _plot_continuous_diag_panel(
        axes[1, 0],
        diagnostics,
        table="score",
        y="variance_trace",
        title="Score variance $\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$",
        ylabel="score variance trace",
        log_y=True,
    )
    _plot_continuous_sensitivity_panel(axes[1, 1], diagnostics)
    fig.suptitle(f"Estimator diagnostics appendix for {env_name}", y=1.02)
    fig.tight_layout()


def _exact_controls_from_bundle(bundle: Mapping[str, Any], env_name: str) -> list[tuple[str, Any, Any]]:
    controls: list[tuple[str, Any, Any]] = []
    train_paths = bundle.get("train", {}) if isinstance(bundle, Mapping) else {}
    for algorithm, path in train_paths.items():
        checkpoint = Path(path) / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("env") != env_name:
            continue
        env_config = dict(payload.get("env_config", {}))
        env_config["device"] = "cpu"
        config = {
            "env": env_name,
            "algorithm": payload.get("algorithm", algorithm),
            "env_config": env_config,
            "algorithm_config": payload.get("algorithm_config", {}),
        }
        spec, env = build_environment(config)
        control = load_control(spec, env, payload["control"], trainable=False)
        controls.append((str(algorithm), env, control))
    return controls


def _portfolio_optimal_objective(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> float | None:
    for data in application_data.values():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        value = _first_numeric(metrics.iloc[0].to_dict(), ["optimal_objective"])
        if value is not None:
            return value
    return None


def _pathwise_reference_costs(application_data: Mapping[str, Mapping[str, pd.DataFrame]]) -> tuple[float | None, float | None]:
    free = None
    heuristic = None
    for data in application_data.values():
        metrics = data.get("metrics", pd.DataFrame())
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        if free is None:
            free = _first_numeric(row, ["free_objective"])
        if heuristic is None:
            heuristic = _first_numeric(row, ["heuristic_objective"])
    return free, heuristic


def _plot_pathwise_metric(
    ax: Any,
    application_data: Mapping[str, Mapping[str, pd.DataFrame]],
    metric: str,
    *,
    title: str,
    cumulative: bool = False,
) -> None:
    baseline_drawn: set[str] = set()
    for algorithm in _continuous_algorithm_order(application_data):
        metrics = application_data[algorithm].get("time_metrics", pd.DataFrame()).copy()
        if metrics.empty or metric not in metrics or "time" not in metrics:
            continue
        metrics = metrics.sort_values("time")
        if "method" not in metrics:
            y = _numeric(metrics, metric).cumsum() if cumulative else _numeric(metrics, metric)
            ax.plot(metrics["time"], y, marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
            continue
        for method, subset in metrics.groupby("method", sort=False):
            subset = subset.sort_values("time")
            y = _numeric(subset, metric).cumsum() if cumulative else _numeric(subset, metric)
            if method == "controlled":
                label = _continuous_algorithm_label(algorithm)
                style = "-"
                color = None
            else:
                if str(method) in baseline_drawn:
                    continue
                baseline_drawn.add(str(method))
                label = str(method)
                style = "--" if method == "heuristic" else ":"
                color = "black" if method == "heuristic" else "0.35"
            ax.plot(subset["time"], y, marker="o", linestyle=style, color=color, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("cumulative energy" if cumulative else metric.replace("_", " "))
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")


def _evenly_spaced_indices(length: int, max_count: int) -> list[int]:
    if length <= max_count:
        return list(range(length))
    if max_count <= 1:
        return [0]
    return sorted({round(i * (length - 1) / (max_count - 1)) for i in range(max_count)})


def _plot_continuous_diag_panel(
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
    for algorithm in _continuous_algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get(table, pd.DataFrame()).copy()
        if frame.empty or not {"lambda", y}.issubset(frame.columns):
            continue
        frame["lambda"] = _numeric(frame, "lambda")
        frame[y] = _numeric(frame, y)
        frame = frame.dropna(subset=["lambda", y]).sort_values("lambda")
        if frame.empty:
            continue
        ax.plot(frame["lambda"], frame[y], marker="o", linewidth=2.0, label=_continuous_algorithm_label(algorithm))
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("perturbation scale $\\lambda$")
    ax.set_ylabel(ylabel)
    if log_y and plotted:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize="small")


def _plot_continuous_sensitivity_panel(ax: Any, diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    plotted = False
    for algorithm in _continuous_algorithm_order(diagnostics):
        frame = diagnostics.get(algorithm, {}).get("sensitivity", pd.DataFrame()).copy()
        if frame.empty or not {"time", "eta", "mse"}.issubset(frame.columns):
            continue
        frame["time"] = _numeric(frame, "time")
        frame["eta"] = _numeric(frame, "eta")
        frame["mse"] = _numeric(frame, "mse")
        frame = frame.dropna(subset=["time", "eta", "mse"])
        if frame.empty:
            continue
        eta = sorted(frame["eta"].unique())[0]
        subset = frame[frame["eta"] == eta].sort_values("time")
        ax.plot(subset["time"], subset["mse"], marker="o", linewidth=2.0, label=f"{_continuous_algorithm_label(algorithm)}, $\\eta={eta:g}$")
        plotted = True
    ax.set_title("Sensitivity MSE $E[\\|\\widehat D_t-D_t\\|^2]$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("sensitivity MSE")
    if plotted:
        ax.set_yscale("log")
        ax.legend(fontsize="small")
    ax.grid(alpha=0.25)



def plot_continuous_time_metrics(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name in {"lq", "portfolio"}:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        reference = _first_nonempty(application_data, "time_metrics", {"time", "optimal_mean", "optimal_variance"})
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            if metrics.empty:
                continue
            axes[0].plot(metrics["time"], metrics["mean"], marker="o", label=f"{algorithm} mean")
            axes[1].plot(metrics["time"], metrics["variance"], marker="o", label=f"{algorithm} variance")
        if not reference.empty:
            axes[0].plot(reference["time"], reference["optimal_mean"], linestyle="--", linewidth=1.8, color="black", label="optimal mean")
            axes[1].plot(reference["time"], reference["optimal_variance"], linestyle="--", linewidth=1.8, color="black", label="optimal variance")
        axes[0].set_title("Mean trajectory $m_t$")
        axes[0].set_ylabel("$m_t$")
        axes[1].set_title("Variance trajectory $\\operatorname{Var}(X_t)$")
        axes[1].set_ylabel("$\\operatorname{Var}(X_t)$")
        for ax in axes:
            ax.set_xlabel("time $t$")
            ax.legend()
        fig.tight_layout()
        return

    columns = {
        "cucker-smale": ["velocity_dispersion", "spatial_diameter", "control_energy"],
        "kuramoto": ["order_parameter", "target_aligned_order", "synchronization_cost", "control_energy"],
    }.get(env_name, [])
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 3.2 * len(columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            if metrics.empty or column not in metrics:
                continue
            if "method" in metrics:
                for method, subset in metrics.groupby("method"):
                    ax.plot(subset["time"], subset[column], marker="o", label=f"{algorithm}: {method}")
            else:
                ax.plot(metrics["time"], metrics[column], marker="o", label=algorithm)
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("time $t$")
        ax.set_ylabel(column.replace("_", " "))
        ax.legend()
    fig.tight_layout()



def plot_continuous_policy_and_samples(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name not in {"lq", "portfolio"}:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    optimal_drawn: set[int] = set()
    for algorithm, data in application_data.items():
        policy = data["policy"]
        if not policy.empty:
            for coordinate, subset in policy.groupby("coordinate"):
                label = f"{algorithm}: theta[{int(coordinate)}]"
                axes[int(coordinate)].plot(subset["time"], subset["value"], marker="o", label=label)
                if int(coordinate) not in optimal_drawn and "optimal" in subset:
                    axes[int(coordinate)].plot(
                        subset["time"],
                        subset["optimal"],
                        linestyle="--",
                        linewidth=1.8,
                        color="black",
                        label=f"optimal theta[{int(coordinate)}]",
                    )
                    optimal_drawn.add(int(coordinate))
        samples = data["terminal_samples"]
        if not samples.empty:
            ordered = samples[samples["stat"].astype(str).str.startswith("q")]
            axes[2].bar(ordered["stat"], ordered["value"], alpha=0.7, label=algorithm)
        if env_name == "portfolio":
            frontier = data.get("efficient_frontier", pd.DataFrame())
            if not frontier.empty:
                axes[3].plot(frontier["terminal_variance"], frontier["terminal_mean"], marker="o", label=algorithm)
        else:
            landscape = data.get("landscape", pd.DataFrame())
            if not landscape.empty:
                grouped = landscape.groupby("theta0", as_index=False)["cost"].min()
                axes[3].plot(grouped["theta0"], grouped["cost"], marker="o", label=algorithm)
    axes[0].set_title("State/fluctuation gain $\\theta_{t,0}$")
    axes[1].set_title("Mean/level gain $\\theta_{t,1}$")
    axes[2].set_title("Terminal quantiles")
    axes[3].set_title("Efficient frontier or objective landscape")
    for idx, ax in enumerate(axes):
        ax.set_xlabel("time $t$" if idx < 2 else "")
        ax.legend()
    fig.tight_layout()



def plot_continuous_snapshots(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name not in {"cucker-smale", "kuramoto"}:
        return
    for algorithm, data in application_data.items():
        snapshots = data["snapshots"]
        if snapshots.empty:
            continue
        times = list(snapshots["time"].drop_duplicates())
        fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4), squeeze=False)
        for ax, time_value in zip(axes[0], times):
            subset = snapshots[snapshots["time"] == time_value]
            if env_name == "cucker-smale":
                ax.scatter(subset["position"], subset["velocity"], s=18, alpha=0.75)
                ax.set_xlabel("position")
                ax.set_ylabel("velocity")
            else:
                ax.scatter(subset["cos_phase"], subset["sin_phase"], s=18, alpha=0.75)
                circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linewidth=1.0)
                ax.add_patch(circle)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("cos phase")
                ax.set_ylabel("sin phase")
            ax.set_title(f"{algorithm}: t={time_value}")
        fig.tight_layout()
        post = data.get("post_control", pd.DataFrame())
        if not post.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            metric = "velocity_dispersion" if env_name == "cucker-smale" else "order_parameter"
            if metric in post:
                ax.plot(post["time"], post[metric], marker="o", label=f"{algorithm}: post-control")
                ax.set_title("Post-control persistence")
                ax.set_xlabel("post-control time")
                ax.legend()
                fig.tight_layout()
            else:
                plt.close(fig)



def plot_continuous_application_details(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name in {"lq", "portfolio"}:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"].copy()
            policy = data["policy"].copy()
            samples = data["terminal_samples"]
            if not metrics.empty:
                metrics = metrics.sort_values("time")
                if "mean_error" in metrics:
                    axes[0].plot(metrics["time"], metrics["mean_error"], marker="o", label=f"{algorithm}: mean")
                if "variance_error" in metrics:
                    axes[0].plot(metrics["time"], metrics["variance_error"], marker="o", label=f"{algorithm}: variance")
            if not policy.empty and "abs_error" in policy:
                grouped = policy.groupby("time", as_index=False)["abs_error"].mean()
                axes[1].plot(grouped["time"], grouped["abs_error"], marker="o", label=algorithm)
            if not samples.empty:
                table = samples.set_index("stat")["value"]
                if "mean" in table and "std" in table:
                    axes[2].bar([f"{algorithm} mean", f"{algorithm} std"], [table["mean"], table["std"]], alpha=0.75)
        axes[0].set_title("Moment error vs oracle")
        axes[0].set_xlabel("time $t$")
        axes[0].set_ylabel("absolute error")
        axes[1].set_title("Mean absolute gain error $|\\theta-\\theta^\\star|$")
        axes[1].set_xlabel("time $t$")
        axes[1].set_ylabel("mean absolute error")
        axes[2].set_title("Terminal mean and spread")
        for ax in axes[:2]:
            ax.legend(fontsize="small")
        axes[2].tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return

    if env_name == "cucker-smale":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"].copy()
            snapshots = data["snapshots"]
            if not metrics.empty and "control_energy" in metrics:
                controlled = metrics[metrics.get("method", "controlled") == "controlled"] if "method" in metrics else metrics
                controlled = controlled.sort_values("time")
                axes[0].plot(controlled["time"], controlled["control_energy"].cumsum(), marker="o", label=algorithm)
            if not snapshots.empty and "velocity" in snapshots:
                for time_value, subset in snapshots.groupby("time"):
                    axes[1].hist(subset["velocity"], bins=12, alpha=0.35, label=f"t={time_value}")
                axes[2].scatter(snapshots["position"], snapshots["velocity"], c=snapshots["time"], s=18, alpha=0.7)
        axes[0].set_title("Cumulative control energy $\\sum_t \\|u_t\\|^2$")
        axes[0].set_xlabel("time $t$")
        axes[0].set_ylabel("energy")
        axes[1].set_title("Velocity histograms")
        axes[1].set_xlabel("velocity")
        axes[2].set_title("Phase-space samples colored by time")
        axes[2].set_xlabel("position")
        axes[2].set_ylabel("velocity")
        axes[0].legend(fontsize="small")
        axes[1].legend(fontsize="small")
        fig.tight_layout()
        return

    if env_name == "kuramoto":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"].copy()
            snapshots = data["snapshots"]
            if not metrics.empty and "control_energy" in metrics:
                controlled = metrics[metrics.get("method", "controlled") == "controlled"] if "method" in metrics else metrics
                controlled = controlled.sort_values("time")
                axes[0].plot(controlled["time"], controlled["control_energy"].cumsum(), marker="o", label=algorithm)
            if not snapshots.empty and "phase" in snapshots:
                for time_value, subset in snapshots.groupby("time"):
                    axes[1].hist(subset["phase"], bins=12, alpha=0.35, label=f"t={time_value}")
                axes[2].scatter(snapshots["cos_phase"], snapshots["sin_phase"], c=snapshots["time"], s=18, alpha=0.7)
                axes[2].set_aspect("equal", adjustable="box")
        axes[0].set_title("Cumulative control energy $\\sum_t \\|u_t\\|^2$")
        axes[0].set_xlabel("time $t$")
        axes[0].set_ylabel("energy")
        axes[1].set_title("Phase histograms")
        axes[1].set_xlabel("phase")
        axes[2].set_title("Phase circle samples colored by time")
        axes[2].set_xlabel("cos phase")
        axes[2].set_ylabel("sin phase")
        axes[0].legend(fontsize="small")
        axes[1].legend(fontsize="small")
        fig.tight_layout()


__all__ = [
    "lq_main_summary_table",
    "pathwise_main_summary_table",
    "plot_continuous_diagnostic_appendix",
    "plot_lq_diagnostic_appendix",
    "plot_lq_jlambda_comparison",
    "plot_lq_learned_policy_comparison",
    "plot_lq_main_results",
    "plot_lq_theta_comparison",
    "plot_lq_validation_reward",
    "plot_pathwise_control_energy",
    "plot_pathwise_dynamics",
    "plot_pathwise_main_results",
    "plot_pathwise_snapshots",
    "plot_pathwise_training_cost",
    "plot_portfolio_jlambda_comparison",
    "plot_portfolio_main_results",
    "plot_portfolio_policy_comparison",
    "plot_portfolio_validation_reward",
    "plot_portfolio_wealth_comparison",
    "portfolio_main_summary_table",
    "plot_continuous_application_details",
    "plot_continuous_policy_and_samples",
    "plot_continuous_snapshots",
    "plot_continuous_time_metrics",
]
