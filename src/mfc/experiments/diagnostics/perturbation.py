from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from .common import (
    _base_law_or_particles,
    _distance_metrics,
    _float_list,
    _perturb_base,
    _perturbation_sample_rows,
    _save_diagnostic_result,
)
from ..core.gradient_steps import make_algorithm
from ..core.registry import FINITE_ALGORITHMS, build_environment, require_algorithm_name
from ..core.session import RunResult, normalize_experiment_config, set_seed


def run_perturbation_diagnostic(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    spec, env = build_environment(config)
    algorithm_name = require_algorithm_name(config)
    algorithm = (
        make_algorithm(algorithm_name if algorithm_name in FINITE_ALGORITHMS else "simplex", env, config.get("algorithm_config", {}))
        if spec.family == "finite"
        else None
    )
    diagnostic = dict(config.get("diagnostic", {}))
    train_config = dict(config.get("train", {}))
    seed = int(train_config.get("seed", 0))
    set_seed(seed, env.config.device)
    samples = int(diagnostic.get("samples", 256))
    lambdas = _float_list(diagnostic.get("lambdas", config.get("lambdas", [0.01, 0.03, 0.1, 0.3])))
    max_raw_samples = int(diagnostic.get("max_raw_samples", samples))
    violation_tolerance = float(diagnostic.get("violation_tolerance", 1e-12))
    rows = []
    sample_rows: List[Dict[str, Any]] = []
    base = _base_law_or_particles(spec, env, samples, seed)
    for lambda_value in lambdas:
        distances: List[float] = []
        metric_rows: List[Dict[str, Any]] = []
        for sample_idx in range(samples):
            perturbed = _perturb_base(spec, env, algorithm, base, lambda_value)
            metrics = _distance_metrics(spec, env, base, perturbed)
            distances.append(float(metrics["distance"]))
            metric_rows.append(metrics)
            if sample_idx < max_raw_samples:
                sample_rows.extend(_perturbation_sample_rows(spec, base, perturbed, lambda_value, sample_idx))
        tensor = torch.as_tensor(distances, dtype=torch.float64)
        row: Dict[str, Any] = {
            "lambda": lambda_value,
            "num_samples": int(tensor.numel()),
            "distance_mean": float(tensor.mean().item()),
            "distance_std": float(tensor.std(unbiased=tensor.numel() > 1).item()) if tensor.numel() > 1 else 0.0,
            "distance_q10": float(torch.quantile(tensor, 0.10).item()),
            "distance_q50": float(torch.quantile(tensor, 0.50).item()),
            "distance_q90": float(torch.quantile(tensor, 0.90).item()),
            "distance_max": float(tensor.max().item()),
        }
        if spec.family == "finite":
            parameter_name = "epsilon" if algorithm_name == "logits" else "lambda"
            row["parameter_name"] = parameter_name
            row["lambda_or_epsilon"] = lambda_value
            row["tv_median"] = row["distance_q50"]
            row["tv_p90"] = row["distance_q90"]
            row["tv_max"] = row["distance_max"]
            if algorithm_name == "logits":
                reference = 0.5 * lambda_value
                row["pathwise_bound"] = float("nan")
                row["comparison_radius"] = reference
                row["logit_reference_size"] = reference
                row["logit_reference_violation_count"] = int((tensor > reference + violation_tolerance).sum().item())
                row["logit_reference_violation_rate"] = float((tensor > reference + violation_tolerance).to(torch.float64).mean().item())
                row["logit_min_reference_slack"] = float((reference - tensor).min().item())
                row["logit_mean_over_reference"] = float(tensor.mean().item() / reference) if reference > 0.0 else float("nan")
            else:
                bound = lambda_value
                row["pathwise_bound"] = bound
                row["comparison_radius"] = bound
                row["simplex_violation_count"] = int((tensor > bound + violation_tolerance).sum().item())
                row["simplex_violation_rate"] = float((tensor > bound + violation_tolerance).to(torch.float64).mean().item())
                row["simplex_min_slack"] = float((bound - tensor).min().item())
                row["simplex_expected_d_tv"] = float(tensor.mean().item())
                row["simplex_expected_unit_radius"] = float(tensor.mean().item() / lambda_value) if lambda_value > 0.0 else float("nan")
                row["simplex_mean_over_bound"] = float(tensor.mean().item() / bound) if bound > 0.0 else float("nan")
        for key in sorted(metric_rows[0]):
            if key == "distance":
                continue
            values = torch.as_tensor([float(metric[key]) for metric in metric_rows], dtype=torch.float64)
            row[f"{key}_mean"] = float(values.mean().item())
        rows.append(row)
    return _save_diagnostic_result(
        "diagnose-perturbation",
        config,
        rows,
        {"rows": len(rows), "sample_rows": len(sample_rows)},
        extra_tables={"perturbation_samples.csv": sample_rows},
    )


__all__ = ["run_perturbation_diagnostic"]
