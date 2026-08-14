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
    "plot_discrete_application_details",
    "plot_policy_heatmaps",
    "plot_population_flow",
    "plot_time_metrics",
]
