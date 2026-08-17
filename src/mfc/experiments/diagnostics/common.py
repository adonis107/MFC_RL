from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from mfc.algorithms import LogitsPerturbedMFREINFORCE

from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.registry import EnvironmentSpec
from ..core.runtime import sample_initial_laws
from ..core.session import RunResult


def save_diagnostic_result(
    command: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    extra_tables: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> RunResult:
    run_dir = _make_run_dir(command, config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata(command, config))
    diagnostics_path = run_dir / "diagnostics.csv"
    _write_csv(diagnostics_path, rows)
    metrics_payload = dict(metrics)
    for filename, table_rows in dict(extra_tables or {}).items():
        _write_csv(run_dir / filename, table_rows)
        metrics_payload[f"{Path(filename).stem}_rows"] = len(table_rows)
    _write_json(run_dir / "metrics.json", metrics_payload)
    return RunResult(run_dir, dict(config), metrics_payload, [], diagnostics_path=diagnostics_path)


def float_list(raw: Any) -> List[float]:
    return [float(value) for value in as_list(raw)]


def as_list(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            return list(json.loads(stripped))
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [raw]


def base_law_or_particles(spec: EnvironmentSpec, env: Any, count: int, seed: int) -> torch.Tensor:
    if spec.family == "finite":
        laws = sample_initial_laws(spec, env, 1, {"seed": seed})
        return laws[0]
    if spec.name == "cucker-smale":
        return env.sample_initial_states(count, seed=seed)
    if spec.name == "kuramoto":
        return env.sample_initial_phases(count, seed=seed)
    if spec.name in {"lq", "portfolio"}:
        theta = env.zero_policy()
        means, variances = env.exact_moments(theta)
        return torch.stack([means, variances], dim=-1)
    raise ValueError(f"Unsupported base law for {spec.name!r}.")


def perturb_base(spec: EnvironmentSpec, env: Any, algorithm: Any, base: torch.Tensor, lambda_value: float) -> torch.Tensor:
    if spec.family == "finite":
        if isinstance(algorithm, LogitsPerturbedMFREINFORCE):
            logits = torch.log(base.clamp_min(env.config.q_clip))
            lam = torch.randn(env.n_states, dtype=env.config.dtype, device=env.config.device)
            return algorithm.perturb_law(logits, float(lambda_value), lam)
        q = algorithm.sample_q()
        return (1.0 - float(lambda_value)) * base + float(lambda_value) * q
    if spec.name == "cucker-smale":
        noise = torch.randn_like(base)
        return base + float(lambda_value) * noise
    if spec.name == "kuramoto":
        noise = torch.randn_like(base)
        return env.wrap_phases(base + float(lambda_value) * noise)
    if spec.name in {"lq", "portfolio"}:
        mean = base[..., 0]
        variance = base[..., 1].clamp_min(0.0)
        rho = float(getattr(env.config, "rho", 1.0))
        zeta = rho * torch.randn_like(mean)
        beta = rho * torch.randn_like(mean)
        perturbed_mean = (1.0 + float(lambda_value) * zeta) * mean + float(lambda_value) * beta
        perturbed_variance = (1.0 + float(lambda_value) * zeta).square() * variance
        return torch.stack([perturbed_mean, perturbed_variance], dim=-1)
    raise ValueError(f"Unsupported perturbation for {spec.name!r}.")


def distance(spec: EnvironmentSpec, env: Any, base: torch.Tensor, perturbed: torch.Tensor) -> float:
    return float(distance_metrics(spec, env, base, perturbed)["distance"])


def distance_metrics(spec: EnvironmentSpec, env: Any, base: torch.Tensor, perturbed: torch.Tensor) -> Dict[str, float]:
    if spec.family == "finite":
        delta = base - perturbed
        tv = 0.5 * delta.abs().sum()
        metrics = {
            "distance": float(tv.item()),
            "tv": float(tv.item()),
            "l1": float(delta.abs().sum().item()),
            "l2": float(torch.linalg.norm(delta).item()),
        }
        clipped_base = base.clamp_min(getattr(env.config, "q_clip", 1e-12))
        clipped_perturbed = perturbed.clamp_min(getattr(env.config, "q_clip", 1e-12))
        clr_base = torch.log(clipped_base) - torch.log(clipped_base).mean()
        clr_perturbed = torch.log(clipped_perturbed) - torch.log(clipped_perturbed).mean()
        metrics["aitchison"] = float(torch.linalg.norm(clr_base - clr_perturbed).item())
        return metrics
    if spec.name == "kuramoto":
        delta = torch.remainder(perturbed - base + math.pi, 2.0 * math.pi) - math.pi
        abs_delta = delta.abs()
        return {
            "distance": float(abs_delta.mean().item()),
            "w1_proxy": float(abs_delta.mean().item()),
            "w2_proxy": float(torch.sqrt(delta.square().mean()).item()),
            "signature_distance": float(torch.linalg.norm(signature(spec, env, perturbed) - signature(spec, env, base)).item()),
        }
    if spec.name in {"lq", "portfolio"}:
        delta = (base - perturbed).reshape(-1)
        return {
            "distance": float(torch.linalg.norm(delta).item()),
            "w1_proxy": float(delta.abs().mean().item()),
            "w2_proxy": float(torch.sqrt(delta.square().mean()).item()),
            "signature_distance": float(torch.linalg.norm(signature(spec, env, perturbed) - signature(spec, env, base)).item()),
        }
    delta = (base - perturbed).reshape(base.shape[0], -1)
    norms = torch.linalg.norm(delta, dim=-1)
    return {
        "distance": float(norms.mean().item()),
        "w1_proxy": float(norms.mean().item()),
        "w2_proxy": float(torch.sqrt(norms.square().mean()).item()),
        "signature_distance": float(torch.linalg.norm(signature(spec, env, perturbed) - signature(spec, env, base)).item()),
    }


def signature(
    spec: EnvironmentSpec,
    env: Any,
    value: torch.Tensor,
    diagnostic: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=env.config.dtype, device=env.config.device)
    diagnostic = dict(diagnostic or {})
    if spec.family == "finite":
        full = value.reshape(-1)
        if diagnostic.get("signature_mode") == "infection" and spec.name == "cybersecurity":
            c = env.config
            full = torch.stack([value[c.DI] + value[c.UI], value[c.DI] + value[c.DS]])
        elif diagnostic.get("signature_mode") == "customer" and spec.name == "advertising":
            full = value[getattr(env.config, "CUSTOMER")]
        return select_signature_coordinates(full, diagnostic)
    if spec.name == "cucker-smale":
        stats = env.empirical_stats(value)
        full = torch.stack(
            [
                stats["x_mean"],
                stats["v_mean"],
                stats["v_variance"],
                env.spatial_diameter(value),
            ]
        ).reshape(-1)
        return select_signature_coordinates(full, diagnostic)
    if spec.name == "kuramoto":
        order = env.order_parameter(value)
        cos_mean = torch.cos(value).mean()
        sin_mean = torch.sin(value).mean()
        circ_var = 1.0 - order
        full = torch.stack([cos_mean, sin_mean, order, circ_var]).reshape(-1)
        return select_signature_coordinates(full, diagnostic)
    if spec.name in {"lq", "portfolio"}:
        return select_signature_coordinates(value.reshape(-1), diagnostic)
    raise ValueError(f"Unsupported signature for {spec.name!r}.")


def select_signature_coordinates(signature_value: torch.Tensor, diagnostic: Mapping[str, Any]) -> torch.Tensor:
    if "signature_coordinates" in diagnostic:
        indices = torch.as_tensor([int(idx) for idx in as_list(diagnostic["signature_coordinates"])], device=signature_value.device)
        return signature_value.index_select(0, indices)
    if "signature_dim" in diagnostic:
        return signature_value[: max(1, min(int(diagnostic["signature_dim"]), signature_value.numel()))]
    if diagnostic.get("signature_mode") == "underspecified":
        return signature_value[:1]
    if diagnostic.get("signature_mode") == "reduced":
        return signature_value[: max(1, min(2, signature_value.numel()))]
    return signature_value


def perturbation_sample_rows(
    spec: EnvironmentSpec,
    base: torch.Tensor,
    perturbed: torch.Tensor,
    lambda_value: float,
    sample_idx: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_flat = base.detach().reshape(-1).cpu()
    perturbed_flat = perturbed.detach().reshape(-1).cpu()
    max_coordinates = min(base_flat.numel(), perturbed_flat.numel(), 2048)
    for coordinate in range(max_coordinates):
        base_value = float(base_flat[coordinate].item())
        value = float(perturbed_flat[coordinate].item())
        rows.append(
            {
                "lambda": lambda_value,
                "sample": sample_idx,
                "coordinate": coordinate,
                "base_value": base_value,
                "perturbed_value": value,
                "delta": value - base_value,
                "family": spec.family,
            }
        )
    return rows


def functional_sample_rows(
    lambda_value: float,
    sample_idx: int,
    signature_value: torch.Tensor,
    base_signature: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    values = signature_value.detach().reshape(-1).cpu()
    base = base_signature.detach().reshape(-1).cpu()
    denom = max(float(lambda_value), 1e-12)
    for coordinate, value in enumerate(values.tolist()):
        base_value = float(base[coordinate].item())
        rows.append(
            {
                "lambda": lambda_value,
                "sample": sample_idx,
                "coordinate": coordinate,
                "base_value": base_value,
                "value": float(value),
                "standardized": (float(value) - base_value) / denom,
            }
        )
    return rows


def safe_covariance(samples: torch.Tensor) -> torch.Tensor:
    samples = samples.reshape(samples.shape[0], -1)
    dim = samples.shape[1]
    if samples.shape[0] <= 1:
        return torch.zeros(dim, dim, dtype=samples.dtype, device=samples.device)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(samples.shape[0] - 1)


def matrix_rows(lambda_value: float, matrix: torch.Tensor, name: str) -> List[Dict[str, Any]]:
    matrix = matrix.detach().cpu()
    rows: List[Dict[str, Any]] = []
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    if matrix.ndim == 1:
        matrix = torch.diag(matrix)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            rows.append(
                {
                    "lambda": lambda_value,
                    "row_coordinate": row_idx,
                    "column_coordinate": col_idx,
                    name: float(matrix[row_idx, col_idx].item()),
                }
            )
    return rows


def functional_distance_rows(lambda_value: float, standardized: torch.Tensor) -> List[Dict[str, Any]]:
    standardized = standardized.reshape(standardized.shape[0], -1)
    norms = torch.linalg.norm(standardized, dim=1)
    if standardized.shape[0] > 1:
        sorted_norms = torch.sort(norms).values
        normal_quantiles = torch.distributions.Normal(0.0, 1.0).icdf(
            (torch.arange(1, norms.numel() + 1, dtype=standardized.dtype, device=standardized.device) - 0.5) / norms.numel()
        )
        qq_correlation = torch.corrcoef(torch.stack([sorted_norms, normal_quantiles.abs()]))[0, 1]
    else:
        qq_correlation = torch.tensor(float("nan"), dtype=standardized.dtype, device=standardized.device)
    mean = standardized.mean(dim=0)
    covariance = safe_covariance(standardized)
    gaussian_energy_proxy = torch.linalg.norm(mean) + torch.linalg.norm(
        covariance - torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    )
    return [
        {
            "lambda": lambda_value,
            "distance_name": "standardized_norm_mean",
            "value": float(norms.mean().item()),
        },
        {
            "lambda": lambda_value,
            "distance_name": "qq_norm_correlation",
            "value": float(qq_correlation.item()),
        },
        {
            "lambda": lambda_value,
            "distance_name": "gaussian_moment_proxy",
            "value": float(gaussian_energy_proxy.item()),
        },
    ]


def normal_ci_z(level: float) -> float:
    if level >= 0.995:
        return 2.80703
    if level >= 0.99:
        return 2.57583
    if level >= 0.975:
        return 2.2414
    if level >= 0.95:
        return 1.95996
    if level >= 0.90:
        return 1.64485
    return 1.0


def sign_match(value: float, oracle_value: float) -> bool:
    if abs(oracle_value) <= 1e-12:
        return abs(value) <= 1e-12
    return (value >= 0.0) == (oracle_value >= 0.0)


_as_list = as_list
_base_law_or_particles = base_law_or_particles
_distance = distance
_distance_metrics = distance_metrics
_float_list = float_list
_functional_distance_rows = functional_distance_rows
_functional_sample_rows = functional_sample_rows
_matrix_rows = matrix_rows
_normal_ci_z = normal_ci_z
_perturb_base = perturb_base
_perturbation_sample_rows = perturbation_sample_rows
_safe_covariance = safe_covariance
_save_diagnostic_result = save_diagnostic_result
_select_signature_coordinates = select_signature_coordinates
_signature = signature
_sign_match = sign_match


__all__ = [
    "_as_list",
    "_base_law_or_particles",
    "_distance",
    "_distance_metrics",
    "_float_list",
    "_functional_distance_rows",
    "_functional_sample_rows",
    "_matrix_rows",
    "_normal_ci_z",
    "_perturb_base",
    "_perturbation_sample_rows",
    "_safe_covariance",
    "_save_diagnostic_result",
    "_select_signature_coordinates",
    "_signature",
    "_sign_match",
    "as_list",
    "base_law_or_particles",
    "distance",
    "distance_metrics",
    "float_list",
    "functional_distance_rows",
    "functional_sample_rows",
    "matrix_rows",
    "normal_ci_z",
    "perturb_base",
    "perturbation_sample_rows",
    "safe_covariance",
    "save_diagnostic_result",
    "select_signature_coordinates",
    "signature",
    "sign_match",
]
