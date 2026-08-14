from __future__ import annotations

from typing import Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def _first_nonempty(application_data: Mapping[str, Mapping[str, pd.DataFrame]], key: str, required: set[str]) -> pd.DataFrame:
    for data in application_data.values():
        frame = data.get(key, pd.DataFrame())
        if not frame.empty and required.issubset(frame.columns):
            return frame
    return pd.DataFrame()



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
    "plot_continuous_application_details",
    "plot_continuous_policy_and_samples",
    "plot_continuous_snapshots",
    "plot_continuous_time_metrics",
]
