from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .. import presets as experiment_presets
from ..core.registry import DEFAULT_DEVICE as _REGISTRY_DEFAULT_DEVICE


DISCRETE_BENCHMARKS = ["twostate", "advertising", "cybersecurity", "distribution-planning"]
CONTINUOUS_BENCHMARKS = ["lq", "portfolio", "cucker-smale", "kuramoto"]
ALGORITHMS = ["simplex", "logits"]
CONTINUOUS_ALGORITHM = "continuous-mfreinforce"
DEFAULT_DEVICE = _REGISTRY_DEFAULT_DEVICE


def set_default_device(device: str) -> None:
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = str(device)


def benchmark_config(
    env_name: str,
    algorithm: str,
    output_dir: Path | str,
    run_name: str,
    *,
    seed: int = 0,
    steps: int | None = None,
    quick: bool = True,
    preset: str | None = None,
) -> Dict[str, Any]:
    if env_name not in DISCRETE_BENCHMARKS:
        raise ValueError(f"Expected one of {DISCRETE_BENCHMARKS}, got {env_name!r}.")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Expected one of {ALGORITHMS}, got {algorithm!r}.")

    preset_name = experiment_presets.resolve_preset(preset, quick=quick)
    B, n = experiment_presets.batch_sizes(env_name, preset_name)
    steps = experiment_presets.train_steps(env_name, preset_name) if steps is None else int(steps)
    hidden = experiment_presets.hidden_units(env_name, preset_name)
    horizon = experiment_presets.nominal_horizon(env_name, preset_name)
    diagnostic = experiment_presets.diagnostic_config(preset_name)
    config: Dict[str, Any] = {
        "env": env_name,
        "algorithm": algorithm,
        "env_config": {"device": DEFAULT_DEVICE, "dtype": "float64"},
        "algorithm_config": {"lambda": 0.1, "eta": 0.1} if algorithm == "simplex" else {"epsilon": 0.1, "flow_particles": B},
        "train": {
            "output_dir": str(output_dir),
            "run_name": run_name,
            "overwrite": True,
            "seed": seed,
            "steps": steps,
            "lr": 1e-3,
            "B": B,
            "n": n,
            "validate_every": experiment_presets.validate_every(env_name, preset_name),
            "flow_mode": "exact",
            "flow_particles": B,
        },
        "evaluation": {},
        "diagnostic": diagnostic,
    }

    if env_name == "twostate":
        config["env_config"].update({"T": horizon, "N": B, "n": n, "n_train": steps, "validate_every": 1})
        config["train"]["mu0"] = [0.2, 0.8]
        config["evaluation"].update({"mu0": [0.2, 0.8], "horizon": horizon})
    elif env_name == "advertising":
        config["env_config"].update(
            {"T": horizon, "hidden_units": hidden, "N": B, "n": n, "n_train": steps, "validate_every": 1, "validation_grid_size": 11}
        )
        config["train"]["mu0"] = [0.5, 0.5]
        config["evaluation"].update({"mu0": [0.5, 0.5], "horizon": horizon})
    elif env_name == "cybersecurity":
        config["env_config"].update(
            {
                "T_train": horizon,
                "T_val": horizon,
                "hidden_units": hidden,
                "N": B,
                "n": n,
                "n_train": steps,
                "validate_every": 1,
            }
        )
        config["train"]["mu0"] = [0.25, 0.25, 0.25, 0.25]
        config["evaluation"].update({"mu0": [0.25, 0.25, 0.25, 0.25], "horizon": horizon})
    elif env_name == "distribution-planning":
        config["env_config"].update({"T": horizon, "hidden_units": hidden, "N": B, "n": n, "n_train": steps, "validate_every": 1})
        config["train"]["mu0"] = [0.1] * 10
        config["evaluation"].update({"mu0": [0.1] * 10, "horizon": horizon})
    config["evaluation"].update(_finite_reference_defaults(env_name, preset_name))
    return config



def _finite_reference_defaults(env_name: str, preset: str) -> Dict[str, Any]:
    preset = experiment_presets.resolve_preset(preset)
    if env_name == "advertising":
        if preset == "smoke":
            return {"oracle_grid_size": 41, "oracle_action_grid_size": 41, "oracle_policy_max_rows": 1000}
        if preset == "mid":
            return {"oracle_grid_size": 101, "oracle_action_grid_size": 101, "oracle_policy_max_rows": 2000}
        if preset == "high-confidence":
            return {"oracle_grid_size": 801, "oracle_action_grid_size": 801, "oracle_policy_max_rows": 8000}
        return {"oracle_grid_size": 401, "oracle_action_grid_size": 401, "oracle_policy_max_rows": 5000}
    if env_name in {"cybersecurity", "distribution-planning"}:
        if preset == "smoke":
            return {"oracle_steps": 8, "oracle_restarts": 1, "oracle_lr": 5e-2, "oracle_init_std": 0.05}
        if preset == "mid":
            return {"oracle_steps": 40, "oracle_restarts": 1, "oracle_lr": 4e-2, "oracle_init_std": 0.05}
        if preset == "high-confidence":
            return {"oracle_steps": 500, "oracle_restarts": 5, "oracle_lr": 2e-2, "oracle_init_std": 0.05}
        return {"oracle_steps": 300, "oracle_restarts": 3, "oracle_lr": 3e-2, "oracle_init_std": 0.05}
    return {}



def continuous_benchmark_config(
    env_name: str,
    output_dir: Path | str,
    run_name: str,
    *,
    seed: int = 0,
    steps: int | None = None,
    quick: bool = True,
    preset: str | None = None,
) -> Dict[str, Any]:
    if env_name not in CONTINUOUS_BENCHMARKS:
        raise ValueError(f"Expected one of {CONTINUOUS_BENCHMARKS}, got {env_name!r}.")

    preset_name = experiment_presets.resolve_preset(preset, quick=quick)
    B, n = experiment_presets.batch_sizes(env_name, preset_name)
    steps = experiment_presets.train_steps(env_name, preset_name) if steps is None else int(steps)
    hidden = experiment_presets.hidden_units(env_name, preset_name)
    population_particles, validation_particles = experiment_presets.particle_counts(env_name, preset_name)
    horizon = experiment_presets.nominal_horizon(env_name, preset_name)
    diagnostic = experiment_presets.diagnostic_config(preset_name)
    config: Dict[str, Any] = {
        "env": env_name,
        "algorithm": CONTINUOUS_ALGORITHM,
        "env_config": {"device": DEFAULT_DEVICE, "dtype": "float64", "T": horizon},
        "algorithm_config": {"lambda": 0.1, "eta": 0.1},
        "train": {
            "output_dir": str(output_dir),
            "run_name": run_name,
            "overwrite": True,
            "seed": seed,
            "steps": steps,
            "lr": 1e-3,
            "B": B,
            "n": n,
            "validate_every": experiment_presets.validate_every(env_name, preset_name),
        },
        "evaluation": {"horizon": horizon, "particles": validation_particles if validation_particles else max(16, B)},
        "diagnostic": diagnostic,
    }

    if env_name == "portfolio":
        config["env_config"].update({"return_distribution": "normal"})
    elif env_name == "cucker-smale":
        config["env_config"].update({"hidden_units": hidden, "N_pop": population_particles, "N_val": validation_particles})
        config["train"]["population_particles"] = population_particles
        config["algorithm_config"]["population_particles"] = population_particles
        config["diagnostic"].update(_pathwise_oracle_defaults(population_particles, preset_name))
        config["evaluation"].update(_continuous_reference_defaults(env_name, validation_particles, preset_name))
    elif env_name == "kuramoto":
        config["env_config"].update({"hidden_units": hidden, "N_pop": population_particles, "N_val": validation_particles})
        config["train"]["population_particles"] = population_particles
        config["algorithm_config"]["population_particles"] = population_particles
        config["diagnostic"].update(_pathwise_oracle_defaults(population_particles, preset_name))
        config["evaluation"].update(_continuous_reference_defaults(env_name, validation_particles, preset_name))
    return config



def _pathwise_oracle_defaults(population_particles: int, preset: str) -> Dict[str, Any]:
    preset = experiment_presets.resolve_preset(preset)
    if preset == "smoke":
        return {
            "oracle_particles": max(8, population_particles),
            "oracle_sensitivity_particles": max(8, population_particles),
            "oracle_replications": 2,
        }
    if preset == "mid":
        return {
            "oracle_particles": max(64, population_particles),
            "oracle_sensitivity_particles": max(64, population_particles),
            "oracle_replications": 4,
        }
    if preset == "high-confidence":
        return {
            "oracle_particles": max(2048, population_particles),
            "oracle_sensitivity_particles": max(1024, population_particles),
            "oracle_replications": 16,
        }
    return {
        "oracle_particles": max(1024, population_particles),
        "oracle_sensitivity_particles": max(512, population_particles),
        "oracle_replications": 8,
    }



def _continuous_reference_defaults(env_name: str, validation_particles: int, preset: str) -> Dict[str, Any]:
    preset = experiment_presets.resolve_preset(preset)
    if preset == "smoke":
        heuristic_particles = max(4, min(validation_particles, 16))
    elif preset == "mid":
        heuristic_particles = max(32, min(validation_particles, 128))
    elif preset == "high-confidence":
        heuristic_particles = max(256, min(validation_particles, 1024))
    else:
        heuristic_particles = max(128, min(validation_particles, 512))

    if env_name == "cucker-smale":
        if preset == "smoke":
            grid = [0.0, 0.5, 1.0]
        elif preset == "mid":
            grid = [0.0, 0.25, 0.5, 1.0]
        else:
            grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        return {"heuristic_kappa_grid": grid, "heuristic_particles": heuristic_particles}
    if env_name == "kuramoto":
        if preset == "smoke":
            grid = [0.0, 0.5, 1.0]
        elif preset == "mid":
            grid = [0.0, 0.5, 1.0]
        else:
            grid = [0.0, 0.25, 0.5, 1.0, 1.5]
        return {"heuristic_kappa_grid": grid, "heuristic_nu_grid": grid, "heuristic_particles": heuristic_particles}
    return {}


__all__ = [
    "ALGORITHMS",
    "CONTINUOUS_ALGORITHM",
    "CONTINUOUS_BENCHMARKS",
    "DEFAULT_DEVICE",
    "DISCRETE_BENCHMARKS",
    "benchmark_config",
    "continuous_benchmark_config",
    "set_default_device",
]
