from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from .common import (
    _base_law_or_particles,
    _float_list,
    _functional_distance_rows,
    _functional_sample_rows,
    _matrix_rows,
    _perturb_base,
    _safe_covariance,
    _save_diagnostic_result,
    _signature,
)
from ..core.gradient_steps import make_algorithm
from ..core.memory import release_memory
from ..core.registry import FINITE_ALGORITHMS, build_environment, require_algorithm_name
from ..core.session import RunResult, normalize_experiment_config, set_seed


def run_functional_law_diagnostic(config: Mapping[str, Any]) -> RunResult:
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
    lambdas = _float_list(diagnostic.get("lambdas", [0.01, 0.03, 0.1, 0.3]))
    base = _base_law_or_particles(spec, env, max(samples, int(diagnostic.get("particles", 128))), seed)
    base_signature = _signature(spec, env, base, diagnostic)
    rows = []
    sample_rows: List[Dict[str, Any]] = []
    covariance_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    for lambda_value in lambdas:
        signatures = []
        for sample_idx in range(samples):
            perturbed = _perturb_base(spec, env, algorithm, base, lambda_value)
            signature = _signature(spec, env, perturbed, diagnostic)
            signatures.append(signature)
            sample_rows.extend(_functional_sample_rows(lambda_value, sample_idx, signature, base_signature))
        stacked = torch.stack(signatures)
        standardized = (stacked - base_signature) / max(lambda_value, 1e-12)
        covariance = _safe_covariance(standardized)
        covariance_rows.extend(_matrix_rows(lambda_value, covariance, "covariance"))
        distance_rows.extend(_functional_distance_rows(lambda_value, standardized))
        row: Dict[str, Any] = {
            "lambda": lambda_value,
            "signature_dim": int(stacked.shape[1]),
            "standardized_norm_mean": float(torch.linalg.norm(standardized, dim=1).mean().item()),
            "standardized_norm_std": float(torch.linalg.norm(standardized, dim=1).std(unbiased=samples > 1).item()) if samples > 1 else 0.0,
            "covariance_trace": float(torch.trace(covariance).item()) if covariance.ndim == 2 else 0.0,
        }
        for idx, value in enumerate(stacked.mean(dim=0).detach().cpu().tolist()):
            row[f"signature_mean_{idx}"] = value
        rows.append(row)
        del signatures, stacked, standardized, covariance
        release_memory()
    return _save_diagnostic_result(
        "diagnose-functional-law",
        config,
        rows,
        {
            "rows": len(rows),
            "sample_rows": len(sample_rows),
            "covariance_rows": len(covariance_rows),
            "distance_rows": len(distance_rows),
        },
        extra_tables={
            "signature_samples.csv": sample_rows,
            "functional_covariance.csv": covariance_rows,
            "functional_distances.csv": distance_rows,
        },
    )


__all__ = ["run_functional_law_diagnostic"]
