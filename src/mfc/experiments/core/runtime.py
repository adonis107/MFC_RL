from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from .registry import EnvironmentSpec


def training_horizon(env: Any, train_config: Mapping[str, Any]) -> int:
    if "horizon" in train_config:
        return int(train_config["horizon"])
    if hasattr(env.config, "T_train"):
        return int(env.config.T_train)
    return int(getattr(env.config, "T"))


def validation_horizon(env: Any, train_config: Mapping[str, Any], evaluation_config: Mapping[str, Any]) -> int:
    if "horizon" in evaluation_config:
        return int(evaluation_config["horizon"])
    if "validation_horizon" in train_config:
        return int(train_config["validation_horizon"])
    if hasattr(env.config, "T_val"):
        return int(env.config.T_val)
    return training_horizon(env, train_config)


def main_batch(env: Any, train_config: Mapping[str, Any], algorithm_config: Mapping[str, Any]) -> int:
    return int(train_config.get("B", algorithm_config.get("B", getattr(env.config, "N", getattr(env.config, "N_pop", 32)))))


def aux_batch(env: Any, train_config: Mapping[str, Any], algorithm_config: Mapping[str, Any]) -> int:
    return int(train_config.get("n", algorithm_config.get("n", getattr(env.config, "n", 1))))


def lambda_value(algorithm_config: Mapping[str, Any], train_config: Mapping[str, Any], default: float = 0.1) -> float:
    for key in ("lambda", "lambda_", "epsilon"):
        if key in algorithm_config:
            return float(algorithm_config[key])
        if key in train_config:
            return float(train_config[key])
    return default


def eta_value(algorithm_config: Mapping[str, Any], lambda_value_: float) -> float:
    return float(algorithm_config.get("eta", algorithm_config.get("eta_", lambda_value_)))


def baseline(algorithm_config: Mapping[str, Any]) -> Any:
    return algorithm_config.get("baseline", "batch_mean")


def initial_law_from_config(env: Any, config: Mapping[str, Any]) -> Optional[torch.Tensor]:
    raw = config.get("mu0")
    if raw is None:
        return None
    return torch.as_tensor(raw, dtype=env.config.dtype, device=env.config.device)


def sample_initial_laws(spec: EnvironmentSpec, env: Any, count: int, train_config: Mapping[str, Any]) -> Optional[torch.Tensor]:
    fixed = initial_law_from_config(env, train_config)
    if fixed is not None:
        return fixed.unsqueeze(0).expand(count, -1).detach().clone()
    if spec.family != "finite":
        return None
    if spec.name == "twostate":
        p = env.config.low + (env.config.high - env.config.low) * torch.rand(
            count,
            dtype=env.config.dtype,
            device=env.config.device,
        )
        return torch.stack([1.0 - p, p], dim=-1)
    if hasattr(env, "sample_initial_laws"):
        return env.sample_initial_laws(count)
    return torch.full((count, env.n_states), 1.0 / env.n_states, dtype=env.config.dtype, device=env.config.device)


def validation_laws(spec: EnvironmentSpec, env: Any, evaluation_config: Mapping[str, Any]) -> Optional[torch.Tensor]:
    fixed = initial_law_from_config(env, evaluation_config)
    if fixed is not None:
        return fixed
    if spec.family != "finite":
        return None
    if spec.name == "twostate":
        return torch.tensor([0.2, 0.8], dtype=env.config.dtype, device=env.config.device)
    if hasattr(env, "validation_initial_laws"):
        grid_size = evaluation_config.get("validation_grid_size")
        return env.validation_initial_laws(None if grid_size is None else int(grid_size))
    return torch.full((env.n_states,), 1.0 / env.n_states, dtype=env.config.dtype, device=env.config.device)


_training_horizon = training_horizon
_validation_horizon = validation_horizon
_main_batch = main_batch
_aux_batch = aux_batch
_lambda_value = lambda_value
_eta_value = eta_value
_baseline = baseline
_initial_law_from_config = initial_law_from_config


__all__ = [
    "_aux_batch",
    "_baseline",
    "_eta_value",
    "_initial_law_from_config",
    "_lambda_value",
    "_main_batch",
    "_training_horizon",
    "_validation_horizon",
    "aux_batch",
    "baseline",
    "eta_value",
    "initial_law_from_config",
    "lambda_value",
    "main_batch",
    "sample_initial_laws",
    "training_horizon",
    "validation_horizon",
    "validation_laws",
]
