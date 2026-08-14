from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Mapping, Optional

import torch

from mfc.environments import (
    AdvertisingConfig,
    AdvertisingMFC,
    AdvertisingPolicy,
    CuckerSmaleConfig,
    CuckerSmaleMFC,
    CuckerSmalePolicy,
    CybersecurityConfig,
    CybersecurityMFC,
    CybersecurityPolicy,
    DistributionPlanningConfig,
    DistributionPlanningMFC,
    DistributionPlanningPolicy,
    KuramotoConfig,
    KuramotoMFC,
    KuramotoPolicy,
    LQConfig,
    LinearQuadraticMFC,
    MeanVariancePortfolioMFC,
    PortfolioConfig,
    TwoStateConfig,
    TwoStateMFC,
)


FINITE_ALGORITHMS = {
    "simplex",
    "logits",
    "finite-adaptive-simplex",
    "consistent-adaptive-simplex",
}
EXACT_ALGORITHMS = {"exact-gradient"}
PATHWISE_ALGORITHMS = {"pathwise-gradient"}
CONTINUOUS_ALGORITHMS = {"continuous-mfreinforce"}
BASELINE_ALGORITHMS = {"reinforce"}
DEFAULT_DEVICE = "cuda"


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    config_cls: type
    env_cls: type
    policy_cls: Optional[type]
    family: str
    objective: str


ENVIRONMENTS: Dict[str, EnvironmentSpec] = {
    "twostate": EnvironmentSpec("twostate", TwoStateConfig, TwoStateMFC, None, "finite", "maximize"),
    "advertising": EnvironmentSpec("advertising", AdvertisingConfig, AdvertisingMFC, AdvertisingPolicy, "finite", "maximize"),
    "cybersecurity": EnvironmentSpec(
        "cybersecurity",
        CybersecurityConfig,
        CybersecurityMFC,
        CybersecurityPolicy,
        "finite",
        "maximize",
    ),
    "distribution-planning": EnvironmentSpec(
        "distribution-planning",
        DistributionPlanningConfig,
        DistributionPlanningMFC,
        DistributionPlanningPolicy,
        "finite",
        "maximize",
    ),
    "lq": EnvironmentSpec("lq", LQConfig, LinearQuadraticMFC, None, "exact", "minimize"),
    "portfolio": EnvironmentSpec("portfolio", PortfolioConfig, MeanVariancePortfolioMFC, None, "exact", "maximize"),
    "cucker-smale": EnvironmentSpec(
        "cucker-smale",
        CuckerSmaleConfig,
        CuckerSmaleMFC,
        CuckerSmalePolicy,
        "pathwise",
        "minimize",
    ),
    "kuramoto": EnvironmentSpec("kuramoto", KuramotoConfig, KuramotoMFC, KuramotoPolicy, "pathwise", "minimize"),
}


def default_algorithm_for_env(env_name: str) -> str:
    spec = environment_spec(env_name)
    if spec.family == "finite":
        return "simplex"
    if spec.family == "exact":
        return "exact-gradient"
    return "pathwise-gradient"


def environment_spec(env_name: str) -> EnvironmentSpec:
    if env_name not in ENVIRONMENTS:
        choices = ", ".join(sorted(ENVIRONMENTS))
        raise ValueError(f"Unknown environment {env_name!r}. Choices: {choices}.")
    return ENVIRONMENTS[env_name]


def validate_compatibility(env_name: str, algorithm_name: str) -> None:
    spec = environment_spec(env_name)
    if algorithm_name in FINITE_ALGORITHMS and spec.family != "finite":
        raise ValueError(f"{algorithm_name!r} requires a finite-state environment, got {env_name!r}.")
    if algorithm_name in EXACT_ALGORITHMS and spec.family != "exact":
        raise ValueError(f"{algorithm_name!r} requires an exact-gradient environment, got {env_name!r}.")
    if algorithm_name in PATHWISE_ALGORITHMS and spec.family != "pathwise":
        raise ValueError(f"{algorithm_name!r} requires a pathwise-gradient environment, got {env_name!r}.")
    if algorithm_name in CONTINUOUS_ALGORITHMS and spec.family == "finite":
        raise ValueError(f"{algorithm_name!r} requires a continuous-state environment, got {env_name!r}.")
    if algorithm_name not in FINITE_ALGORITHMS | EXACT_ALGORITHMS | PATHWISE_ALGORITHMS | CONTINUOUS_ALGORITHMS | BASELINE_ALGORITHMS:
        choices = ", ".join(sorted(FINITE_ALGORITHMS | EXACT_ALGORITHMS | PATHWISE_ALGORITHMS | CONTINUOUS_ALGORITHMS | BASELINE_ALGORITHMS))
        raise ValueError(f"Unknown algorithm {algorithm_name!r}. Choices: {choices}.")


def build_env_config(spec: EnvironmentSpec, raw_config: Mapping[str, Any]) -> Any:
    raw = dict(raw_config)
    device = torch.device(raw.get("device", DEFAULT_DEVICE))
    dtype = torch_dtype(raw.get("dtype", "float64"))
    kwargs = _dataclass_kwargs(spec.config_cls, raw, device, dtype)
    return spec.config_cls(**kwargs)


def build_environment(config: Mapping[str, Any]) -> tuple[EnvironmentSpec, Any]:
    env_name = require_env_name(config)
    spec = environment_spec(env_name)
    env_config = build_env_config(spec, config.get("env_config", {}))
    return spec, spec.env_cls(env_config)


def require_env_name(config: Mapping[str, Any]) -> str:
    env_name = config.get("env")
    if not env_name:
        raise ValueError("Config must define 'env'.")
    return str(env_name)


def require_algorithm_name(config: Mapping[str, Any]) -> str:
    env_name = require_env_name(config)
    return str(config.get("algorithm") or default_algorithm_for_env(env_name))


def torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    normalized = str(name).replace("torch.", "").lower()
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "single": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return aliases[normalized]


def _dataclass_kwargs(config_cls: type, raw: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    raw = dict(raw)
    for field in fields(config_cls):
        if field.name in raw:
            value = raw[field.name]
        elif field.name == "device":
            value = device
        elif field.name == "dtype":
            value = dtype
        elif field.default is not MISSING:
            continue
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue
        else:
            raise ValueError(f"Missing required config field {field.name!r} for {config_cls.__name__}.")

        if field.name == "device":
            value = torch.device(value)
        elif field.name == "dtype":
            value = torch_dtype(value)
        kwargs[field.name] = value
    return kwargs


__all__ = [
    "CONTINUOUS_ALGORITHMS",
    "BASELINE_ALGORITHMS",
    "DEFAULT_DEVICE",
    "ENVIRONMENTS",
    "EXACT_ALGORITHMS",
    "FINITE_ALGORITHMS",
    "PATHWISE_ALGORITHMS",
    "EnvironmentSpec",
    "build_env_config",
    "build_environment",
    "default_algorithm_for_env",
    "environment_spec",
    "require_algorithm_name",
    "require_env_name",
    "torch_dtype",
    "validate_compatibility",
]
