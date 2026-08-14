from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from ..configs import CONTINUOUS_ALGORITHM
from .common import _numeric_sorted


def _row_number(row: pd.Series, *names: str) -> float:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return float("nan")



def _row_text(row: pd.Series, name: str, default: str) -> str:
    if name not in row.index:
        return default
    value = row[name]
    if pd.isna(value):
        return default
    return str(value)



def _finite_tv_samples(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"lambda", "sample", "delta"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["lambda", "sample", "d_TV"])
    work = frame.copy()
    if "family" in work:
        work = work[work["family"].astype(str) == "finite"]
    work["lambda"] = pd.to_numeric(work["lambda"], errors="coerce")
    work["sample"] = pd.to_numeric(work["sample"], errors="coerce")
    work["abs_delta"] = pd.to_numeric(work["delta"], errors="coerce").abs()
    work = work.dropna(subset=["lambda", "sample", "abs_delta"])
    if work.empty:
        return pd.DataFrame(columns=["lambda", "sample", "d_TV"])
    grouped = work.groupby(["lambda", "sample"], as_index=False)["abs_delta"].sum()
    grouped["d_TV"] = 0.5 * grouped["abs_delta"]
    return grouped[["lambda", "sample", "d_TV"]]



def _tv_samples_at_lambda(samples: pd.DataFrame, lambda_value: float) -> pd.DataFrame:
    if samples.empty or math.isnan(lambda_value):
        return samples.iloc[0:0]
    tolerance = max(1e-12, abs(lambda_value) * 1e-9)
    return samples[(samples["lambda"] - lambda_value).abs() <= tolerance]



def perturbation_tv_comparison_table(
    diagnostics: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    violation_tolerance: float = 1e-12,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for algorithm, data in diagnostics.items():
        summary = _numeric_sorted(data.get("perturbation", pd.DataFrame()), "lambda")
        if summary.empty:
            continue
        samples = _finite_tv_samples(data.get("perturbation_perturbation_samples", pd.DataFrame()))
        for _, row in summary.iterrows():
            lambda_value = _row_number(row, "lambda")
            parameter_value = _row_number(row, "lambda_or_epsilon")
            if math.isnan(parameter_value):
                parameter_value = lambda_value
            parameter_name = _row_text(row, "parameter_name", "epsilon" if algorithm == "logits" else "lambda")
            sample_group = _tv_samples_at_lambda(samples, lambda_value)

            mean_tv = _row_number(row, "tv_mean", "distance_mean")
            median_tv = _row_number(row, "tv_median", "distance_q50")
            p90_tv = _row_number(row, "tv_p90", "distance_q90")
            max_tv = _row_number(row, "tv_max", "distance_max")
            sample_count = _row_number(row, "num_samples")
            if not sample_group.empty:
                if math.isnan(mean_tv):
                    mean_tv = float(sample_group["d_TV"].mean())
                if math.isnan(median_tv):
                    median_tv = float(sample_group["d_TV"].median())
                if math.isnan(p90_tv):
                    p90_tv = float(sample_group["d_TV"].quantile(0.90))
                if math.isnan(max_tv):
                    max_tv = float(sample_group["d_TV"].max())
                if math.isnan(sample_count):
                    sample_count = float(sample_group.shape[0])

            row_data: Dict[str, Any] = {
                "algorithm": algorithm,
                "parameter": parameter_name,
                "value": parameter_value,
                "samples": int(sample_count) if not math.isnan(sample_count) else pd.NA,
                "E_d_TV": mean_tv,
                "median_d_TV": median_tv,
                "p90_d_TV": p90_tv,
                "max_d_TV": max_tv,
                "comparison_kind": "",
                "comparison_radius": float("nan"),
                "pathwise_bound": _row_number(row, "pathwise_bound"),
                "violation_count": pd.NA,
                "violation_rate": float("nan"),
                "simplex_expected_unit_radius": _row_number(row, "simplex_expected_unit_radius", "simplex_mean_over_bound"),
                "simplex_min_slack": _row_number(row, "simplex_min_slack"),
                "logit_reference_size": _row_number(row, "logit_reference_size"),
                "logit_mean_over_reference": _row_number(row, "logit_mean_over_reference"),
                "logit_min_reference_slack": _row_number(row, "logit_min_reference_slack"),
            }

            if algorithm == "logits" or parameter_name == "epsilon":
                reference = _row_number(row, "logit_reference_size", "comparison_radius")
                if math.isnan(reference):
                    reference = 0.5 * parameter_value
                violations = _row_number(row, "logit_reference_violation_count")
                violation_rate = _row_number(row, "logit_reference_violation_rate")
                if not sample_group.empty:
                    mask = sample_group["d_TV"] > reference + violation_tolerance
                    if math.isnan(violations):
                        violations = float(mask.sum())
                    if math.isnan(violation_rate):
                        violation_rate = float(mask.mean())
                    if math.isnan(row_data["logit_min_reference_slack"]):
                        row_data["logit_min_reference_slack"] = float((reference - sample_group["d_TV"]).min())
                row_data["comparison_kind"] = "logit reference epsilon/2"
                row_data["comparison_radius"] = reference
                row_data["logit_reference_size"] = reference
                row_data["violation_count"] = int(violations) if not math.isnan(violations) else pd.NA
                row_data["violation_rate"] = violation_rate
                if math.isnan(row_data["logit_mean_over_reference"]) and reference > 0.0 and not math.isnan(mean_tv):
                    row_data["logit_mean_over_reference"] = mean_tv / reference
            else:
                bound = _row_number(row, "pathwise_bound", "comparison_radius")
                if math.isnan(bound):
                    bound = parameter_value
                violations = _row_number(row, "simplex_violation_count")
                violation_rate = _row_number(row, "simplex_violation_rate")
                if not sample_group.empty:
                    mask = sample_group["d_TV"] > bound + violation_tolerance
                    if math.isnan(violations):
                        violations = float(mask.sum())
                    if math.isnan(violation_rate):
                        violation_rate = float(mask.mean())
                    if math.isnan(row_data["simplex_min_slack"]):
                        row_data["simplex_min_slack"] = float((bound - sample_group["d_TV"]).min())
                row_data["comparison_kind"] = "simplex pathwise bound lambda"
                row_data["comparison_radius"] = bound
                row_data["pathwise_bound"] = bound
                row_data["violation_count"] = int(violations) if not math.isnan(violations) else pd.NA
                row_data["violation_rate"] = violation_rate
                if math.isnan(row_data["simplex_expected_unit_radius"]) and bound > 0.0 and not math.isnan(mean_tv):
                    row_data["simplex_expected_unit_radius"] = mean_tv / bound
            rows.append(row_data)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    order = {"simplex": 0, "logits": 1}
    table["_order"] = table["algorithm"].map(order).fillna(99)
    table = table.sort_values(["value", "_order", "algorithm"]).drop(columns=["_order"])
    return table.reset_index(drop=True)



def plot_perturbation_tv_comparison(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    table = perturbation_tv_comparison_table(diagnostics)
    if table.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for algorithm, group in table.groupby("algorithm"):
        group = group.sort_values("value")
        x = pd.to_numeric(group["value"], errors="coerce")
        mean_tv = pd.to_numeric(group["E_d_TV"], errors="coerce")
        radius = pd.to_numeric(group["comparison_radius"], errors="coerce")
        violation_rate = pd.to_numeric(group["violation_rate"], errors="coerce")
        axes[0].plot(x, mean_tv, marker="o", label=f"{algorithm}: E[d_TV]")
        if radius.notna().any():
            axes[0].plot(x, radius, linestyle="--", alpha=0.65, label=f"{algorithm}: comparison radius")
        axes[1].plot(x, violation_rate, marker="o", label=algorithm)
    axes[0].set_title("Total-variation perturbation calibration")
    axes[0].set_xlabel("lambda or epsilon")
    axes[0].set_ylabel("d_TV")
    axes[1].set_title("Comparison-radius violation rate")
    axes[1].set_xlabel("lambda or epsilon")
    axes[1].set_ylabel("violation rate")
    for ax in axes:
        ax.legend(fontsize="small")
    fig.tight_layout()



def plot_perturbation_geometry(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for algorithm, data in diagnostics.items():
        frame = data.get("perturbation", pd.DataFrame())
        if frame.empty:
            continue
        axes[0].loglog(frame["lambda"], frame["distance_mean"], marker="o", label=algorithm)
        lower = frame.get("distance_q10", frame["distance_mean"])
        upper = frame.get("distance_q90", frame["distance_mean"])
        axes[1].plot(frame["lambda"], frame["distance_mean"], marker="o", label=algorithm)
        axes[1].fill_between(frame["lambda"], lower, upper, alpha=0.2)
    axes[0].set_title("Mean perturbation distance $E[d(M^\\lambda,\\mu)]$")
    axes[0].set_xlabel("perturbation scale $\\lambda$ or $\\epsilon$")
    axes[0].set_ylabel("$E[d(M^\\lambda,\\mu)]$")
    axes[0].legend()
    axes[1].set_title("Distance quantile bands for $d(M^\\lambda,\\mu)$")
    axes[1].set_xlabel("perturbation scale $\\lambda$ or $\\epsilon$")
    axes[1].set_ylabel("$d(M^\\lambda,\\mu)$")
    axes[1].legend()
    fig.tight_layout()



def plot_perturbation_slopes(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    any_data = False
    for algorithm, data in diagnostics.items():
        frame = _numeric_sorted(data.get("perturbation", pd.DataFrame()), "lambda")
        if frame.empty or "distance_mean" not in frame or frame.shape[0] < 2:
            continue
        slopes = []
        lambdas = []
        for idx in range(1, frame.shape[0]):
            lam0 = float(frame.iloc[idx - 1]["lambda"])
            lam1 = float(frame.iloc[idx]["lambda"])
            dist0 = float(frame.iloc[idx - 1]["distance_mean"])
            dist1 = float(frame.iloc[idx]["distance_mean"])
            if lam0 > 0.0 and lam1 > 0.0 and dist0 > 0.0 and dist1 > 0.0:
                denominator = math.log(lam1) - math.log(lam0)
                if denominator == 0.0:
                    continue
                slopes.append((math.log(dist1) - math.log(dist0)) / denominator)
                lambdas.append((lam0 * lam1) ** 0.5)
        if slopes:
            ax.plot(lambdas, slopes, marker="o", label=algorithm)
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax.set_title("Empirical perturbation distance slope")
    ax.set_xlabel("perturbation midpoint")
    ax.set_ylabel("local slope of $\\log E[d]$ vs $\\log \\lambda$")
    ax.legend()
    fig.tight_layout()



def plot_functional_law(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for algorithm, data in diagnostics.items():
        frame = data.get("functional_law", pd.DataFrame())
        if frame.empty:
            continue
        axes[0].plot(frame["lambda"], frame["standardized_norm_mean"], marker="o", label=algorithm)
        axes[1].plot(frame["lambda"], frame["covariance_trace"], marker="o", label=algorithm)
    axes[0].set_title("Mean standardized signature perturbation")
    axes[0].set_xlabel("perturbation scale $\\lambda$")
    axes[0].set_ylabel("$E[\\| (\\Gamma(M^\\lambda)-\\Gamma(\\mu))/\\lambda \\|]$")
    axes[0].legend()
    axes[1].set_title("Functional-law covariance trace")
    axes[1].set_xlabel("perturbation scale $\\lambda$")
    axes[1].set_ylabel("$\\operatorname{tr}\\operatorname{Cov}(\\Gamma(M^\\lambda))$")
    axes[1].legend()
    fig.tight_layout()



def plot_functional_signature_means(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    long_frames = []
    dim_frames = []
    for algorithm, data in diagnostics.items():
        frame = _numeric_sorted(data.get("functional_law", pd.DataFrame()), "lambda")
        columns = [column for column in frame.columns if column.startswith("signature_mean_")]
        if frame.empty or not columns:
            continue
        long = frame.melt(id_vars=["lambda"], value_vars=columns, var_name="coordinate", value_name="mean")
        long["coordinate"] = long["coordinate"].astype(str).str.removeprefix("signature_mean_").astype(int)
        long["algorithm"] = algorithm
        long_frames.append(long)
        if "signature_dim" in frame:
            dim = frame[["lambda", "signature_dim"]].copy()
            dim["algorithm"] = algorithm
            dim_frames.append(dim)
    if not long_frames:
        return

    combined = pd.concat(long_frames, ignore_index=True)
    max_coordinate_count = int(combined.groupby("algorithm")["coordinate"].nunique().max())
    if max_coordinate_count > 6:
        algorithms = list(combined["algorithm"].dropna().unique())
        fig, axes = plt.subplots(1, len(algorithms), figsize=(5.5 * len(algorithms), 4), squeeze=False)
        for ax, algorithm in zip(axes[0], algorithms):
            subset = combined[combined["algorithm"] == algorithm]
            pivot = subset.pivot_table(index="coordinate", columns="lambda", values="mean", aggfunc="mean").sort_index()
            if pivot.empty:
                continue
            image = ax.imshow(pivot.values, aspect="auto")
            ax.set_title(f"{algorithm}: $E[\\Gamma_i(M^\\lambda)]$")
            ax.set_xlabel("perturbation scale $\\lambda$")
            ax.set_ylabel("signature coordinate $i$")
            ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
            fig.colorbar(image, ax=ax, label="$E[\\Gamma_i(M^\\lambda)]$")
        fig.tight_layout()
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for (algorithm, coordinate), subset in combined.groupby(["algorithm", "coordinate"]):
        axes[0].plot(subset["lambda"], subset["mean"], marker="o", label=f"{algorithm}: $i={int(coordinate)}$")
    if dim_frames:
        dim_combined = pd.concat(dim_frames, ignore_index=True)
        for algorithm, subset in dim_combined.groupby("algorithm"):
            axes[1].plot(subset["lambda"], subset["signature_dim"], marker="o", label=algorithm)
    axes[0].set_title("Signature coordinate means $E[\\Gamma_i(M^\\lambda)]$")
    axes[0].set_xlabel("perturbation scale $\\lambda$")
    axes[0].set_ylabel("$E[\\Gamma_i(M^\\lambda)]$")
    axes[0].legend(fontsize="small", ncols=2)
    axes[1].set_title("Signature dimension $d_\\Gamma$")
    axes[1].set_xlabel("perturbation scale $\\lambda$")
    axes[1].set_ylabel("$d_\\Gamma$")
    axes[1].legend()
    fig.tight_layout()



def plot_functional_sample_diagnostics(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frames = []
    for algorithm, data in diagnostics.items():
        frame = data.get("functional_law_signature_samples", pd.DataFrame())
        if frame.empty:
            continue
        frame = frame.copy()
        frame["algorithm"] = algorithm
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    for column in ("lambda", "sample", "coordinate", "value", "standardized"):
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    coordinates = sorted(combined["coordinate"].dropna().unique())[:4]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for algorithm, subset in combined.groupby("algorithm"):
        coord0 = subset[subset["coordinate"] == coordinates[0]] if coordinates else subset
        axes[0].hist(coord0["value"].dropna(), bins=24, alpha=0.45, label=algorithm)
        axes[1].hist(coord0["standardized"].dropna(), bins=24, alpha=0.45, label=algorithm)
        if len(coordinates) >= 2:
            pair = subset[subset["coordinate"].isin(coordinates[:2])].pivot_table(
                index=["lambda", "sample"], columns="coordinate", values="standardized"
            )
            if len(pair.columns) >= 2:
                axes[2].scatter(pair.iloc[:, 0], pair.iloc[:, 1], alpha=0.6, label=algorithm)
    axes[0].set_title("Samples of one signature coordinate $\\Gamma_i(M^\\lambda)$")
    axes[0].set_xlabel("$\\Gamma_i(M^\\lambda)$")
    axes[1].set_title("Standardized samples $(\\Gamma_i(M^\\lambda)-\\Gamma_i(\\mu))/\\lambda$")
    axes[1].set_xlabel("standardized value")
    axes[2].set_title("Pair scatter of standardized signature coordinates")
    axes[2].set_xlabel("coordinate 1 standardized value")
    axes[2].set_ylabel("coordinate 2 standardized value")
    for ax in axes:
        ax.legend()
    fig.tight_layout()



def plot_score_validation(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for algorithm, data in diagnostics.items():
        frame = data.get("score", pd.DataFrame())
        if frame.empty:
            continue
        axes[0].plot(frame["lambda"], frame["mean_norm"], marker="o", label=algorithm)
        axes[1].loglog(frame["lambda"], frame["variance_trace"], marker="o", label=algorithm)
        axes[2].plot(frame["lambda"], frame["lambda2_variance_trace"], marker="o", label=algorithm)
    axes[0].set_title("Score mean norm $\\|E[S_\\lambda]\\|$")
    axes[0].set_ylabel("$\\|E[S_\\lambda]\\|$")
    axes[1].set_title("Score variance trace $\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$")
    axes[1].set_ylabel("$\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$")
    axes[2].set_title("Scaled score variance $\\lambda^2\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$")
    axes[2].set_ylabel("$\\lambda^2\\operatorname{tr}\\operatorname{Cov}(S_\\lambda)$")
    for ax in axes:
        ax.set_xlabel("perturbation scale $\\lambda$")
        ax.legend()
    fig.tight_layout()



def plot_score_coordinate_diagnostics(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frames = []
    for algorithm, data in diagnostics.items():
        frame = data.get("score_score_coordinates", data.get("score_coordinates", pd.DataFrame()))
        if frame.empty:
            continue
        frame = frame.copy()
        frame["algorithm"] = algorithm
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    for column in ("lambda", "coordinate", "mean", "variance", "second_moment"):
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.replace([math.inf, -math.inf], pd.NA).dropna(subset=["coordinate", "variance"])
    if combined.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for algorithm, subset in combined.groupby("algorithm"):
        positive = subset[pd.to_numeric(subset["variance"], errors="coerce") > 0.0]
        if positive.empty:
            continue
        grouped = positive.groupby("coordinate", as_index=False)["variance"].mean().sort_values("variance", ascending=False).head(20)
        axes[0].plot(grouped["coordinate"].astype(int).astype(str), grouped["variance"], marker="o", label=algorithm)
        axes[1].scatter(positive["mean"], positive["variance"], label=algorithm, alpha=0.65)
    axes[0].set_title("Largest score-coordinate variances $\\operatorname{Var}(S_i)$")
    axes[0].set_xlabel("score coordinate $i$")
    axes[0].set_ylabel("$\\operatorname{Var}(S_i)$")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("Score-coordinate mean vs variance")
    axes[1].set_xlabel("$E[S_i]$")
    axes[1].set_ylabel("$\\operatorname{Var}(S_i)$")
    axes[1].set_xscale("symlog", linthresh=1e-12)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend()
    fig.tight_layout()



def plot_gradient_validation(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    columns = [
        ("relative_bias", "Relative bias $\\|E[\\hat g]-g\\|/\\|g\\|$"),
        ("variance_trace", "Gradient variance trace $\\operatorname{tr}\\operatorname{Cov}(\\hat g)$"),
        ("mse", "Gradient MSE $E[\\|\\hat g-g\\|^2]$"),
        ("bias_norm", "Gradient bias norm $\\|E[\\hat g]-g\\|$"),
        ("cosine_similarity", "Cosine similarity $\\cos(\\hat g,g)$"),
        ("norm_ratio", "Norm ratio $\\|E[\\hat g]\\|/\\|g\\|$"),
    ]
    for ax, (column, title) in zip(axes.reshape(-1), columns):
        plot_title = title
        for algorithm, data in diagnostics.items():
            frame = data.get("gradient", pd.DataFrame())
            plot_column = column
            if frame.empty:
                continue
            if plot_column not in frame and column in {"relative_bias", "bias_norm", "mse", "cosine_similarity", "norm_ratio"}:
                plot_column = "estimate_norm"
                plot_title = "Estimate norm $\\|\\hat g\\|$"
            if plot_column not in frame:
                continue
            ax.plot(frame["lambda"], frame[plot_column], marker="o", label=algorithm)
        ax.set_title(plot_title)
        ax.set_xlabel("diagnostic scale $\\lambda$ or $\\epsilon$")
        ax.legend()
    fig.tight_layout()



def plot_gradient_error_decomposition(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    any_data = False
    for algorithm, data in diagnostics.items():
        frame = _numeric_sorted(data.get("gradient", pd.DataFrame()), "lambda")
        if frame.empty:
            continue
        if "bias_norm" in frame:
            ax.plot(frame["lambda"], frame["bias_norm"], marker="o", label=f"{algorithm}: bias")
            any_data = True
        if "variance_trace" in frame:
            std = pd.to_numeric(frame["variance_trace"], errors="coerce").clip(lower=0.0).pow(0.5)
            ax.plot(frame["lambda"], std, marker="o", label=f"{algorithm}: std")
            any_data = True
        if "mse" in frame:
            rmse = pd.to_numeric(frame["mse"], errors="coerce").clip(lower=0.0).pow(0.5)
            ax.plot(frame["lambda"], rmse, marker="o", label=f"{algorithm}: RMSE")
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_title("Gradient error decomposition vs perturbation scale")
    ax.set_xlabel("diagnostic scale $\\lambda$ or $\\epsilon$")
    ax.set_ylabel("bias norm, $\\sqrt{\\operatorname{tr}\\operatorname{Cov}}$, or RMSE")
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()



def plot_gradient_coordinate_diagnostics(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frames = []
    for algorithm, data in diagnostics.items():
        coord = data.get("gradient_gradient_coordinates", pd.DataFrame())
        if coord.empty:
            continue
        coord = coord.copy()
        coord["algorithm"] = algorithm
        frames.append(coord)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    for column in ("lambda", "coordinate", "mean", "oracle", "mse", "ci_covers_oracle", "sign_accuracy"):
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if {"mean", "oracle"}.issubset(combined.columns):
        for algorithm, subset in combined.groupby("algorithm"):
            axes[0].scatter(subset["oracle"], subset["mean"], label=algorithm, alpha=0.7)
    if "ci_covers_oracle" in combined:
        coverage = combined.groupby(["algorithm", "lambda"], as_index=False)["ci_covers_oracle"].mean()
        for algorithm, subset in coverage.groupby("algorithm"):
            axes[1].plot(subset["lambda"], subset["ci_covers_oracle"], marker="o", label=algorithm)
    if "sign_accuracy" in combined:
        sign = combined.groupby(["algorithm", "lambda"], as_index=False)["sign_accuracy"].mean()
        for algorithm, subset in sign.groupby("algorithm"):
            axes[2].plot(subset["lambda"], subset["sign_accuracy"], marker="o", label=algorithm)
    axes[0].set_title("Oracle vs estimated gradient coordinates")
    axes[0].set_xlabel("oracle coordinate $g_i$")
    axes[0].set_ylabel("estimated mean coordinate $E[\\hat g_i]$")
    axes[1].set_title("Coordinate CI coverage")
    axes[1].set_xlabel("diagnostic scale $\\lambda$ or $\\epsilon$")
    axes[1].set_ylabel("coverage probability")
    axes[2].set_title("Coordinate sign accuracy")
    axes[2].set_xlabel("diagnostic scale $\\lambda$ or $\\epsilon$")
    axes[2].set_ylabel("$P[\\operatorname{sign}(\\hat g_i)=\\operatorname{sign}(g_i)]$")
    for ax in axes:
        ax.legend()
    fig.tight_layout()



def plot_sensitivity_validation(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frame = diagnostics.get("simplex", {}).get("sensitivity", pd.DataFrame())
    if frame.empty:
        frame = diagnostics.get(CONTINUOUS_ALGORITHM, {}).get("sensitivity", pd.DataFrame())
    if frame.empty:
        for data in diagnostics.values():
            frame = data.get("sensitivity", pd.DataFrame())
            if not frame.empty:
                break
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    error_column = "mse" if "mse" in frame and frame["mse"].notna().any() else "estimate_norm"
    for eta, subset in frame.groupby("eta"):
        axes[0].plot(subset["time"], subset[error_column], marker="o", label=f"eta={eta}")
        axes[1].plot(subset["time"], subset["variance_trace"], marker="o", label=f"eta={eta}")
    axes[0].set_title("Sensitivity MSE $E[\\|\\hat D_t-D_t\\|^2]$ vs time" if error_column == "mse" else "Sensitivity estimate norm $\\|\\hat D_t\\|$ vs time")
    axes[1].set_title("Sensitivity variance trace $\\operatorname{tr}\\operatorname{Cov}(\\hat D_t)$")
    for ax in axes:
        ax.set_xlabel("time $t$")
        ax.legend()
    fig.tight_layout()



def plot_sensitivity_heatmap(diagnostics: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    frames = []
    for algorithm, data in diagnostics.items():
        frame = data.get("sensitivity", pd.DataFrame())
        if frame.empty:
            continue
        frame = frame.copy()
        frame["algorithm"] = algorithm
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    metric = "mse" if "mse" in combined and combined["mse"].notna().any() else "estimate_norm"
    if metric not in combined:
        return
    for column in ("time", "eta", metric):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    algorithms = list(combined["algorithm"].dropna().unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5.5 * len(algorithms), 4), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = combined[combined["algorithm"] == algorithm]
        pivot = subset.pivot_table(index="eta", columns="time", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        image = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"{algorithm}: sensitivity {metric}")
        ax.set_xlabel("time")
        ax.set_ylabel("eta")
        ax.set_xticks(range(len(pivot.columns)), [str(col) for col in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
        fig.colorbar(image, ax=ax)
    fig.tight_layout()


__all__ = [
    "perturbation_tv_comparison_table",
    "plot_functional_law",
    "plot_functional_sample_diagnostics",
    "plot_functional_signature_means",
    "plot_gradient_coordinate_diagnostics",
    "plot_gradient_error_decomposition",
    "plot_gradient_validation",
    "plot_perturbation_geometry",
    "plot_perturbation_slopes",
    "plot_perturbation_tv_comparison",
    "plot_score_coordinate_diagnostics",
    "plot_score_validation",
    "plot_sensitivity_heatmap",
    "plot_sensitivity_validation",
]
