from __future__ import annotations

from typing import Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from .common import _spaced_indices, _state_names, _time_metric_columns


def data_reference_flow(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    reference = data.get("reference_population_flow", pd.DataFrame())
    if reference.empty or not {"time", "state", "mass"}.issubset(reference.columns):
        return pd.DataFrame()
    return reference



def _has_state_action_reference(data: Mapping[str, pd.DataFrame]) -> bool:
    reference = data.get("reference_policy", pd.DataFrame())
    return not reference.empty and {"state", "action", "probability"}.issubset(reference.columns)



def plot_population_flow(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    state_names = _state_names(env_name)
    algorithms = list(application_data)
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(10, 3.5 * len(algorithms)), squeeze=False)
    for ax, algorithm in zip(axes[:, 0], algorithms):
        flow = application_data[algorithm]["population_flow"]
        if flow.empty:
            continue
        grouped = flow.groupby(["time", "state"], as_index=False)["mass"].mean()
        for state, subset in grouped.groupby("state"):
            label = state_names.get(int(state), f"state {state}")
            ax.plot(subset["time"], subset["mass"], marker="o", label=label)
        reference = data_reference_flow(application_data[algorithm])
        if not reference.empty:
            reference_grouped = reference.groupby(["time", "state"], as_index=False)["mass"].mean()
            for state, subset in reference_grouped.groupby("state"):
                label = state_names.get(int(state), f"state {state}")
                ax.plot(subset["time"], subset["mass"], linestyle="--", linewidth=1.5, label=f"ref: {label}")
        ax.set_title(f"{algorithm}: population flow")
        ax.set_xlabel("time")
        ax.set_ylabel("mass")
        ax.legend()
    fig.tight_layout()



def plot_time_metrics(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    columns = _time_metric_columns(env_name)
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 3.2 * len(columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            if metrics.empty or column not in metrics:
                continue
            grouped = metrics.groupby("time", as_index=False)[column].mean()
            ax.plot(grouped["time"], grouped[column], marker="o", label=algorithm)
            reference = data.get("reference_time_metrics", pd.DataFrame())
            if not reference.empty and column in reference:
                reference_grouped = reference.groupby("time", as_index=False)[column].mean()
                ax.plot(
                    reference_grouped["time"],
                    reference_grouped[column],
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{algorithm}: reference",
                )
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("time")
        ax.legend()
    fig.tight_layout()



def plot_policy_heatmaps(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    algorithms = list(application_data)
    has_reference = any(_has_state_action_reference(data) for data in application_data.values())
    n_rows = 2 if has_reference else 1
    fig, axes = plt.subplots(n_rows, len(algorithms), figsize=(5 * len(algorithms), 4 * n_rows), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
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
    if has_reference:
        for ax, algorithm in zip(axes[1], algorithms):
            policy = application_data[algorithm].get("reference_policy", pd.DataFrame())
            if policy.empty or not {"state", "action", "probability"}.issubset(policy.columns):
                ax.axis("off")
                continue
            table = policy.groupby(["state", "action"], as_index=False)["probability"].mean().pivot(index="state", columns="action", values="probability")
            image = ax.imshow(table.values, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(f"{algorithm}: reference policy")
            ax.set_xlabel("action")
            ax.set_ylabel("state")
            ax.set_xticks(range(table.shape[1]), [str(col) for col in table.columns])
            ax.set_yticks(range(table.shape[0]), [_state_names(env_name).get(int(idx), str(idx)) for idx in table.index])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()



def plot_discrete_application_details(application_data: Mapping[str, Mapping[str, pd.DataFrame]], env_name: str) -> None:
    if env_name == "distribution-planning":
        algorithms = list(application_data)
        has_reference = any(not data_reference_flow(data).empty for data in application_data.values())
        n_rows = 2 if has_reference else 1
        fig, axes = plt.subplots(n_rows, len(algorithms), figsize=(5.5 * len(algorithms), 4 * n_rows), squeeze=False)
        for ax, algorithm in zip(axes[0], algorithms):
            flow = application_data[algorithm]["population_flow"]
            if flow.empty:
                continue
            pivot = flow.pivot_table(index="state", columns="time", values="mass", aggfunc="mean")
            image = ax.imshow(pivot.values, aspect="auto", vmin=0.0)
            ax.set_title(f"{algorithm}: state-time mass")
            ax.set_xlabel("time")
            ax.set_ylabel("state")
            ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
            fig.colorbar(image, ax=ax)
        if has_reference:
            for ax, algorithm in zip(axes[1], algorithms):
                flow = data_reference_flow(application_data[algorithm])
                if flow.empty:
                    ax.axis("off")
                    continue
                pivot = flow.pivot_table(index="state", columns="time", values="mass", aggfunc="mean")
                image = ax.imshow(pivot.values, aspect="auto", vmin=0.0)
                ax.set_title(f"{algorithm}: reference state-time mass")
                ax.set_xlabel("time")
                ax.set_ylabel("state")
                ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
                ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
                fig.colorbar(image, ax=ax)
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
            ax.set_title("Transport flux around the ring")
            ax.set_xlabel("time")
            ax.legend()
            fig.tight_layout()
        else:
            plt.close(fig)
        return

    if env_name == "advertising":
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
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
            reference = data.get("reference_time_metrics", pd.DataFrame())
            if not reference.empty:
                reference = reference.sort_values("time")
                if "customer_fraction" in reference:
                    axes[0].plot(
                        reference["time"],
                        reference["customer_fraction"],
                        linestyle="--",
                        linewidth=1.5,
                        label=f"{algorithm}: DP customer",
                    )
                for column, label in [("advertising_cost", "DP ad cost"), ("population_gain", "DP population gain")]:
                    if column in reference:
                        axes[1].plot(
                            reference["time"],
                            reference[column].cumsum(),
                            linestyle="--",
                            linewidth=1.5,
                            label=f"{algorithm}: {label}",
                        )
            finite = data.get("finite_population", pd.DataFrame())
            if not finite.empty and "particles" in finite:
                grouped = finite.groupby("particles", as_index=False)["abs_error"].mean()
                axes[2].plot(grouped["particles"], grouped["abs_error"], marker="o", label=algorithm)
        axes[0].axhline(0.5, linestyle="--", color="black", linewidth=1.0, alpha=0.5)
        axes[0].axhline(0.8, linestyle=":", color="black", linewidth=1.0, alpha=0.5)
        axes[0].set_title("Customer fraction and target levels")
        axes[1].set_title("Cumulative objective components")
        axes[2].set_title("Finite-population gap")
        for ax in axes:
            ax.set_xlabel("time")
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
            reference = data.get("reference_time_metrics", pd.DataFrame())
            if not reference.empty:
                reference = reference.sort_values("time")
                for column, label in [("infected_fraction", "ref infected"), ("defended_fraction", "ref defended")]:
                    if column in reference:
                        axes[0].plot(
                            reference["time"],
                            reference[column],
                            linestyle="--",
                            linewidth=1.5,
                            label=f"{algorithm}: {label}",
                        )
                if "running_reward" in reference:
                    axes[1].plot(
                        reference["time"],
                        reference["running_reward"].cumsum(),
                        linestyle="--",
                        linewidth=1.5,
                        label=f"{algorithm}: ref cumulative reward",
                    )
        axes[0].set_title("Infected and defended fractions")
        axes[1].set_title("Reward and switching diagnostics")
        for ax in axes:
            ax.set_xlabel("time")
            ax.legend(fontsize="small")
        fig.tight_layout()
        return

    if env_name == "twostate":
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        for algorithm, data in application_data.items():
            metrics = data["time_metrics"]
            policy = data["policy"]
            if not metrics.empty and "mass_state_1" in metrics:
                axes[0].plot(metrics["time"], metrics["mass_state_1"], marker="o", label=f"{algorithm}: mass state 1")
                if "target_state_1" in metrics:
                    axes[0].plot(metrics["time"], metrics["target_state_1"], linestyle="--", label=f"{algorithm}: target")
            reference = data.get("reference_time_metrics", pd.DataFrame())
            if not reference.empty and "mass_state_1" in reference:
                reference = reference.sort_values("time")
                axes[0].plot(
                    reference["time"],
                    reference["mass_state_1"],
                    linestyle=":",
                    linewidth=2.0,
                    label=f"{algorithm}: exact optimum",
                )
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
        axes[0].set_title("State 1 mass vs target")
        axes[1].set_title("Dominant action probability")
        for ax in axes[:2]:
            ax.set_xlabel("time")
            ax.legend(fontsize="small")
        axes[2].legend(fontsize="small")
        fig.tight_layout()


__all__ = [
    "data_reference_flow",
    "plot_discrete_application_details",
    "plot_policy_heatmaps",
    "plot_population_flow",
    "plot_time_metrics",
]
