from __future__ import annotations

import re
from typing import Any, List, Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from .common import _numeric_sorted


def plot_budget_and_horizon(studies: Mapping[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    budget = studies.get("budget", pd.DataFrame())
    budget_metric = _preferred_study_metric(budget)
    if not budget.empty and budget_metric is not None:
        budget_label = _metric_label(budget_metric)
        budget = budget.copy()
        budget["B"] = pd.to_numeric(_series_or_variant_number(budget, ["train.B", "B"], r"B(\d+)"), errors="coerce")
        budget["n"] = pd.to_numeric(_series_or_variant_number(budget, ["train.n", "n"], r"n(\d+)"), errors="coerce")
        pivot = budget.pivot_table(index="n", columns="B", values=budget_metric, aggfunc="mean")
        if not pivot.empty:
            image = axes[0].imshow(pivot.values, aspect="auto")
            axes[0].set_title(f"{budget_label} over main/auxiliary budget")
            axes[0].set_xlabel("main batch $B$")
            axes[0].set_ylabel("auxiliary batch $n$")
            axes[0].set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            axes[0].set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
            fig.colorbar(image, ax=axes[0], label=budget_label)
    horizon = studies.get("horizon", pd.DataFrame())
    horizon_metric = _preferred_study_metric(horizon)
    if not horizon.empty and horizon_metric is not None:
        horizon_label = _metric_label(horizon_metric)
        horizon = horizon.copy()
        label_col = "env_config.T_train" if "env_config.T_train" in horizon else "env_config.T"
        horizon["horizon"] = pd.to_numeric(_series_or_variant_number(horizon, [label_col, "T"], r"T(\d+)"), errors="coerce")
        grouped = horizon.dropna(subset=["horizon"]).groupby("horizon", as_index=False)[horizon_metric].mean()
        if not grouped.empty:
            axes[1].plot(grouped["horizon"], grouped[horizon_metric], marker="o")
        axes[1].set_title(f"Gradient {horizon_label} vs horizon")
        axes[1].set_xlabel("horizon $T$")
        axes[1].set_ylabel(horizon_label)
    fig.tight_layout()



def plot_budget_pareto(studies: Mapping[str, pd.DataFrame], grid_metrics: Mapping[str, pd.DataFrame]) -> None:
    budget = studies.get("budget", pd.DataFrame())
    grid = grid_metrics.get("budget", pd.DataFrame())
    metric = _preferred_study_metric(budget)
    if budget.empty or grid.empty or metric is None:
        return
    metric_label = _metric_label(metric)
    merged = budget.copy()
    if "variant" in merged and "variant" in grid:
        merged = merged.merge(grid[["variant", "elapsed_seconds"]], on="variant", how="left")
    elif "index" in merged and "index" in grid:
        merged = merged.merge(grid[["index", "elapsed_seconds"]], on="index", how="left")
    else:
        return
    if "elapsed_seconds" not in merged:
        return
    merged[metric] = pd.to_numeric(merged[metric], errors="coerce")
    merged["elapsed_seconds"] = pd.to_numeric(merged["elapsed_seconds"], errors="coerce")
    if "train.B" in merged:
        merged["B"] = pd.to_numeric(merged["train.B"], errors="coerce")
    else:
        merged["B"] = pd.to_numeric(_series_or_variant_number(merged, ["B"], r"B(\d+)"), errors="coerce")
    if "train.n" in merged:
        merged["n"] = pd.to_numeric(merged["train.n"], errors="coerce")
    else:
        merged["n"] = pd.to_numeric(_series_or_variant_number(merged, ["n"], r"n(\d+)"), errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, subset in merged.groupby("variant" if "variant" in merged else "index"):
        axes[0].scatter(subset["elapsed_seconds"], subset[metric], label=str(label), alpha=0.8)
    axes[0].set_title(f"{metric_label} vs runtime")
    axes[0].set_xlabel("elapsed seconds")
    axes[0].set_ylabel(metric_label)
    if merged["B"].notna().any():
        grouped = merged.groupby("B", as_index=False)[metric].mean()
        axes[1].plot(grouped["B"], grouped[metric], marker="o")
    axes[1].set_title(f"{metric_label} vs main batch $B$")
    axes[1].set_xlabel("main batch $B$")
    if merged["n"].notna().any():
        grouped = merged.groupby("n", as_index=False)[metric].mean()
        axes[2].plot(grouped["n"], grouped[metric], marker="o")
    axes[2].set_title(f"{metric_label} vs auxiliary batch $n$")
    axes[2].set_xlabel("auxiliary batch $n$")
    axes[0].legend(fontsize="small")
    fig.tight_layout()



def plot_extended_study_summaries(studies: Mapping[str, pd.DataFrame], grid_metrics: Mapping[str, pd.DataFrame]) -> None:
    plot_lambda_training_comparison(studies)
    plot_optimizer_bias_summary(studies)
    plot_robustness_summary(studies)
    plot_adaptive_lambda_summary(studies, grid_metrics)
    plot_ablation_and_signature_summary(studies, grid_metrics)
    plot_particle_and_scaling_summary(studies, grid_metrics)



def plot_lambda_training_comparison(studies: Mapping[str, pd.DataFrame]) -> None:
    history = studies.get("lambda_training_optimization_history", pd.DataFrame())
    if history.empty:
        return
    history = history.copy()
    if "episode" not in history:
        return
    history["episode"] = pd.to_numeric(history["episode"], errors="coerce")
    reward = _validation_reward_series(history)
    if reward is None:
        return
    history["validation_reward"] = reward
    history = history.dropna(subset=["episode", "validation_reward"])
    if history.empty:
        return
    label_col = "variant" if "variant" in history else "run" if "run" in history else None
    if label_col is None:
        history["variant"] = "run"
        label_col = "variant"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, subset in history.groupby(label_col):
        grouped = subset.groupby("episode", as_index=False)["validation_reward"].mean()
        ax.plot(grouped["episode"], grouped["validation_reward"], marker="o", label=_lambda_variant_label(str(label)))
    ax.set_title("Validation reward vs training step by perturbation level")
    ax.set_xlabel("training episode $k$")
    ax.set_ylabel("mean validation reward $-J(\\theta_k)$ for LQ")
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()



def plot_optimizer_bias_summary(studies: Mapping[str, pd.DataFrame]) -> None:
    frame = _numeric_sorted(studies.get("optimizer_bias", pd.DataFrame()), "lambda")
    if frame.empty:
        return
    metric_columns = [
        column
        for column in (
            "control_distance",
            "policy_output_distance",
            "trajectory_distance",
            "optimal_value_bias_proxy",
            "unperturbed_objective",
        )
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if not metric_columns:
        return
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(5 * len(metric_columns), 4), squeeze=False)
    for ax, column in zip(axes[0], metric_columns):
        ax.plot(frame["lambda"], pd.to_numeric(frame[column], errors="coerce"), marker="o")
        ax.set_title(_metric_label(column))
        ax.set_xlabel("perturbation scale $\\lambda$")
        ax.set_ylabel(_metric_label(column))
    fig.tight_layout()



def plot_robustness_summary(studies: Mapping[str, pd.DataFrame]) -> None:
    frame = studies.get("robustness", pd.DataFrame())
    if frame.empty:
        return
    metric = _preferred_outcome_metric(frame)
    if metric is None:
        return
    grouped = _group_for_category_plot(frame, "variant", metric)
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(grouped)), 4))
    ax.bar(grouped["variant"].astype(str), grouped[metric])
    ax.set_title(f"Robustness: {_metric_label(metric)}")
    ax.set_ylabel(_metric_label(metric))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()



def plot_adaptive_lambda_summary(studies: Mapping[str, pd.DataFrame], grid_metrics: Mapping[str, pd.DataFrame]) -> None:
    summary = _study_frame(studies, grid_metrics, "adaptive")
    trace = studies.get("adaptive_lambda_trace", pd.DataFrame())
    if summary.empty and trace.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if not summary.empty:
        metric = _preferred_outcome_metric(summary)
        if metric is not None:
            grouped = _group_for_category_plot(summary, "variant", metric)
            if not grouped.empty:
                axes[0].bar(grouped["variant"].astype(str), grouped[metric])
                axes[0].set_title(f"Adaptive vs fixed: {_metric_label(metric)}")
                axes[0].set_ylabel(_metric_label(metric))
                axes[0].tick_params(axis="x", rotation=35)
    if not trace.empty and {"variant", "episode", "lambda"}.issubset(trace.columns):
        trace = trace.copy()
        trace["episode"] = pd.to_numeric(trace["episode"], errors="coerce")
        trace["lambda"] = pd.to_numeric(trace["lambda"], errors="coerce")
        for variant, subset in trace.dropna(subset=["episode", "lambda"]).groupby("variant"):
            axes[1].plot(subset["episode"], subset["lambda"], marker="o", label=str(variant))
        axes[1].set_title("Lambda trajectory")
        axes[1].set_xlabel("episode")
        axes[1].legend(fontsize="small")
    fig.tight_layout()



def plot_ablation_and_signature_summary(studies: Mapping[str, pd.DataFrame], grid_metrics: Mapping[str, pd.DataFrame]) -> None:
    ablation = _study_frame(studies, grid_metrics, "ablation")
    signature = _study_frame(studies, grid_metrics, "signature")
    if ablation.empty and signature.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _plot_variant_metric(axes[0], ablation, "Ablation")
    if not signature.empty:
        metric = _preferred_outcome_metric(signature)
        x = _first_existing_column(signature, ["diagnostic.signature_dim", "signature_dim"])
        if metric is not None and x is not None:
            signature = signature.copy()
            signature[x] = pd.to_numeric(signature[x], errors="coerce")
            grouped = signature.dropna(subset=[x]).groupby(x, as_index=False)[metric].mean()
            if not grouped.empty:
                axes[1].plot(grouped[x], grouped[metric], marker="o")
                axes[1].set_xlabel("signature dimension $d_\\Gamma$")
                axes[1].set_ylabel(_metric_label(metric))
        else:
            _plot_variant_metric(axes[1], signature, "Signature")
        axes[1].set_title("Signature ablation")
    fig.tight_layout()



def plot_particle_and_scaling_summary(studies: Mapping[str, pd.DataFrame], grid_metrics: Mapping[str, pd.DataFrame]) -> None:
    particle = _study_frame(studies, grid_metrics, "particle")
    transfer = _study_frame(studies, grid_metrics, "particle_transfer")
    scaling = _study_frame(studies, grid_metrics, "scaling")
    if particle.empty and transfer.empty and scaling.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    _plot_particle_metric(axes[0], particle)
    _plot_transfer_heatmap(axes[1], transfer)
    _plot_scaling_metric(axes[2], scaling)
    fig.tight_layout()



def _preferred_study_metric(frame: pd.DataFrame) -> str | None:
    for column in ("mse", "variance_trace", "estimate_norm"):
        if column in frame and frame[column].notna().any():
            return column
    return None



def _validation_reward_series(frame: pd.DataFrame) -> pd.Series | None:
    if "value" in frame and pd.to_numeric(frame["value"], errors="coerce").notna().any():
        return pd.to_numeric(frame["value"], errors="coerce")
    if "cost" in frame and pd.to_numeric(frame["cost"], errors="coerce").notna().any():
        return -pd.to_numeric(frame["cost"], errors="coerce")
    if "objective" in frame and pd.to_numeric(frame["objective"], errors="coerce").notna().any():
        return -pd.to_numeric(frame["objective"], errors="coerce")
    return None



def _lambda_variant_label(label: str) -> str:
    match = re.search(r"lambda_([0-9.]+)", label)
    if label.startswith("mf_lambda_") and match:
        return f"MF-REINFORCE $\\lambda={match.group(1)}$"
    if label.startswith("oracle_sens_lambda_") and match:
        return f"oracle sensitivity $\\lambda={match.group(1)}$"
    if label == "exact_gradient":
        return "exact gradient oracle"
    return label.replace("_", " ")



def _metric_label(metric: str) -> str:
    labels = {
        "mse": "MSE $E[\\|\\hat g-g\\|^2]$",
        "relative_bias": "relative bias $\\|E[\\hat g]-g\\|/\\|g\\|$",
        "variance_trace": "variance trace $\\operatorname{tr}\\operatorname{Cov}(\\hat g)$",
        "estimate_norm": "estimate norm $\\|\\hat g\\|$",
        "standardized_norm_mean": "$E[\\|(\\Gamma(M^\\lambda)-\\Gamma(\\mu))/\\lambda\\|]$",
        "covariance_trace": "$\\operatorname{tr}\\operatorname{Cov}(\\Gamma(M^\\lambda))$",
        "signature_distance_mean": "mean signature distance",
        "distance_mean": "mean perturbation distance $E[d(M^\\lambda,\\mu)]$",
        "objective_gap": "objective gap",
        "optimal_value_bias_proxy": "optimal-value bias proxy",
        "control_distance": "control distance",
        "trajectory_distance": "trajectory distance",
        "policy_output_distance": "policy-output distance",
        "value": "value $J(\\theta)$",
        "objective": "objective $J(\\theta)$",
        "cost": "cost",
        "metric_value": "value $J(\\theta)$",
        "metric_objective": "objective $J(\\theta)$",
        "metric_cost": "cost",
        "terminal_mean": "terminal mean",
        "terminal_variance": "terminal variance",
        "elapsed_seconds": "elapsed seconds",
    }
    return labels.get(metric, metric.replace("_", " "))



def _preferred_outcome_metric(frame: pd.DataFrame) -> str | None:
    for column in (
        "mse",
        "relative_bias",
        "variance_trace",
        "standardized_norm_mean",
        "covariance_trace",
        "signature_distance_mean",
        "distance_mean",
        "objective_gap",
        "optimal_value_bias_proxy",
        "control_distance",
        "trajectory_distance",
        "policy_output_distance",
        "value",
        "objective",
        "cost",
        "metric_value",
        "metric_objective",
        "metric_cost",
        "terminal_mean",
        "terminal_variance",
        "elapsed_seconds",
    ):
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return _preferred_study_metric(frame)



def _study_frame(
    studies: Mapping[str, pd.DataFrame],
    grid_metrics: Mapping[str, pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    grid = grid_metrics.get(name, pd.DataFrame())
    if not grid.empty:
        return grid
    return studies.get(name, pd.DataFrame())



def _first_existing_column(frame: pd.DataFrame, columns: List[str]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None



def _group_for_category_plot(frame: pd.DataFrame, category: str, metric: str) -> pd.DataFrame:
    if category not in frame or metric not in frame:
        return pd.DataFrame()
    grouped = frame.copy()
    grouped[metric] = pd.to_numeric(grouped[metric], errors="coerce")
    return grouped.dropna(subset=[metric]).groupby(category, as_index=False)[metric].mean()



def _plot_variant_metric(ax: Any, frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        ax.set_title(title)
        return
    metric = _preferred_outcome_metric(frame)
    if metric is None:
        ax.set_title(title)
        return
    category = "variant" if "variant" in frame else "index" if "index" in frame else None
    if category is None:
        ax.set_title(title)
        return
    grouped = _group_for_category_plot(frame, category, metric)
    if grouped.empty:
        ax.set_title(title)
        return
    ax.bar(grouped[category].astype(str), grouped[metric])
    ax.set_title(f"{title}: {_metric_label(metric)}")
    ax.tick_params(axis="x", rotation=35)



def _plot_particle_metric(ax: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        ax.set_title("Particle approximation")
        return
    metric = _preferred_outcome_metric(frame)
    x_col = _first_existing_column(
        frame,
        [
            "train.flow_particles",
            "algorithm_config.flow_particles",
            "train.particles",
            "evaluation.particles",
            "env_config.N_pop",
            "train.population_particles",
        ],
    )
    if metric is None or x_col is None:
        _plot_variant_metric(ax, frame, "Particle approximation")
        return
    plotted = frame.copy()
    plotted[x_col] = pd.to_numeric(plotted[x_col], errors="coerce")
    plotted[metric] = pd.to_numeric(plotted[metric], errors="coerce")
    grouped = plotted.dropna(subset=[x_col, metric]).groupby(x_col, as_index=False)[metric].mean()
    if grouped.empty:
        _plot_variant_metric(ax, frame, "Particle approximation")
        return
    ax.plot(grouped[x_col], grouped[metric], marker="o")
    ax.set_title(f"Particle approximation: {_metric_label(metric)}")
    ax.set_xlabel("particles")
    ax.set_ylabel(_metric_label(metric))



def _plot_transfer_heatmap(ax: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        ax.set_title("Particle transfer")
        return
    metric = _preferred_outcome_metric(frame)
    required = {"train_particles", "eval_particles"}
    if metric is None or not required.issubset(frame.columns):
        _plot_variant_metric(ax, frame, "Particle transfer")
        return
    plotted = frame.copy()
    for column in ("train_particles", "eval_particles", metric):
        plotted[column] = pd.to_numeric(plotted[column], errors="coerce")
    pivot = plotted.pivot_table(index="train_particles", columns="eval_particles", values=metric, aggfunc="mean")
    if pivot.empty:
        _plot_variant_metric(ax, frame, "Particle transfer")
        return
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(f"Particle transfer: {_metric_label(metric)}")
    ax.set_xlabel("eval particles")
    ax.set_ylabel("train particles")
    ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=_metric_label(metric))



def _plot_scaling_metric(ax: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        ax.set_title("Scaling")
        return
    metric = _preferred_outcome_metric(frame)
    x_col = _first_existing_column(frame, ["simulator_budget_proxy", "elapsed_seconds", "env_config.T", "env_config.T_train", "train.B"])
    if metric is None or x_col is None:
        _plot_variant_metric(ax, frame, "Scaling")
        return
    plotted = frame.copy()
    plotted[x_col] = pd.to_numeric(plotted[x_col], errors="coerce")
    plotted[metric] = pd.to_numeric(plotted[metric], errors="coerce")
    grouped = plotted.dropna(subset=[x_col, metric]).groupby(x_col, as_index=False)[metric].mean()
    if grouped.empty:
        _plot_variant_metric(ax, frame, "Scaling")
        return
    ax.plot(grouped[x_col], grouped[metric], marker="o")
    ax.set_title(f"Scaling: {_metric_label(metric)}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(_metric_label(metric))



def _series_or_variant_number(frame: pd.DataFrame, columns: List[str], pattern: str) -> pd.Series:
    for column in columns:
        if column in frame:
            return frame[column]
    if "variant" not in frame:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    regex = re.compile(pattern)
    return frame["variant"].astype(str).map(lambda value: _first_float_match(regex, value))



def _first_float_match(regex: re.Pattern[str], value: str) -> float:
    match = regex.search(value)
    return float(match.group(1)) if match else float("nan")



def plot_optimization_summary(studies: Mapping[str, pd.DataFrame]) -> None:
    frame = studies.get("optimization", pd.DataFrame())
    if frame.empty:
        return
    metric = "value" if "value" in frame else "objective"
    fig, ax = plt.subplots(figsize=(8, 4))
    if metric in frame:
        ax.bar(frame["algorithm"], pd.to_numeric(frame[metric], errors="coerce"))
    ax.set_title("Final training comparison")
    ax.set_ylabel(metric)
    fig.tight_layout()



def plot_optimization_history(history: pd.DataFrame) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x_col = "episode" if "episode" in history else None
    for run, subset in history.groupby("run" if "run" in history else history.index):
        x = subset[x_col] if x_col else range(len(subset))
        metric = "value" if "value" in subset and subset["value"].notna().any() else "objective"
        if metric in subset:
            axes[0].plot(x, subset[metric], marker="o", label=str(run))
        if "elapsed_seconds" in subset and metric in subset:
            axes[1].plot(subset["elapsed_seconds"], subset[metric], marker="o", label=str(run))
        if "grad_norm" in subset:
            axes[2].plot(x, subset["grad_norm"], marker="o", label=str(run))
    axes[0].set_title("Objective/value vs iteration")
    axes[0].set_xlabel("episode")
    axes[1].set_title("Objective/value vs wall-clock")
    axes[1].set_xlabel("elapsed seconds")
    axes[2].set_title("Gradient norm vs iteration")
    axes[2].set_xlabel("episode")
    for ax in axes:
        ax.legend(fontsize="small")
    fig.tight_layout()


__all__ = [
    "plot_budget_and_horizon",
    "plot_budget_pareto",
    "plot_extended_study_summaries",
    "plot_lambda_training_comparison",
    "plot_optimization_history",
    "plot_optimization_summary",
]
