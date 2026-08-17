from __future__ import annotations

from typing import Any, Dict, List, Sequence


PRESET_NAMES = ("smoke", "mid", "main", "high-confidence")

LAMBDA_GRID_MAIN = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
LAMBDA_GRID_MID = [0.025, 0.1, 0.2]
LAMBDA_GRID_SMOKE = [0.05, 0.1, 0.8]
SMALL_LAMBDA_FIT = [0.0125, 0.025, 0.05, 0.1]

MAIN_SEEDS = [0, 1, 2, 3, 4]
ADAPTIVE_SEEDS = [0, 1, 2, 3, 4]
HIGH_CONFIDENCE_SEEDS = list(range(10))
MID_SEEDS = [0]
SMOKE_SEEDS = [0]

MAIN_HORIZONS: Dict[str, List[int]] = {
    "twostate": [2, 8, 32],
    "advertising": [5, 20, 40],
    "cybersecurity": [3, 10, 20],
    "distribution-planning": [5, 20, 40],
    "lq": [5, 20, 40],
    "portfolio": [5, 20, 40],
    "cucker-smale": [25, 50, 100],
    "kuramoto": [50, 100, 200],
}

SMOKE_HORIZONS: Dict[str, List[int]] = {
    "twostate": [2, 4, 8],
    "advertising": [2, 5, 10],
    "cybersecurity": [2, 3, 5],
    "distribution-planning": [2, 5, 10],
    "lq": [2, 5, 10],
    "portfolio": [2, 5, 10],
    "cucker-smale": [5, 10, 20],
    "kuramoto": [5, 10, 20],
}

MID_HORIZONS: Dict[str, List[int]] = {
    "twostate": [2, 8],
    "advertising": [3, 10],
    "cybersecurity": [2, 5],
    "distribution-planning": [3, 10],
    "lq": [3, 10],
    "portfolio": [3, 10],
    "cucker-smale": [10, 20],
    "kuramoto": [10, 20],
}

MAIN_TRAIN_STEPS: Dict[str, int] = {
    "twostate": 10_000,
    "advertising": 25_000,
    "cybersecurity": 25_000,
    "distribution-planning": 30_000,
    "lq": 10_000,
    "portfolio": 15_000,
    "cucker-smale": 30_000,
    "kuramoto": 40_000,
}

SMOKE_TRAIN_STEPS = {env_name: 4 for env_name in MAIN_TRAIN_STEPS}

MID_TRAIN_STEPS = {env_name: 1_000 for env_name in MAIN_TRAIN_STEPS}

MAIN_BATCHES: Dict[str, tuple[int, int]] = {
    "twostate": (256, 16),
    "advertising": (256, 16),
    "cybersecurity": (256, 16),
    "distribution-planning": (256, 16),
    "lq": (512, 32),
    "portfolio": (512, 32),
    "cucker-smale": (256, 32),
    "kuramoto": (256, 32),
}

SMOKE_BATCHES: Dict[str, tuple[int, int]] = {
    "twostate": (8, 2),
    "advertising": (8, 2),
    "cybersecurity": (8, 2),
    "distribution-planning": (8, 2),
    "lq": (6, 2),
    "portfolio": (6, 2),
    "cucker-smale": (6, 2),
    "kuramoto": (6, 2),
}

MID_BATCHES: Dict[str, tuple[int, int]] = {
    "twostate": (32, 4),
    "advertising": (32, 4),
    "cybersecurity": (32, 4),
    "distribution-planning": (32, 4),
    "lq": (64, 64),
    "portfolio": (64, 8),
    "cucker-smale": (24, 4),
    "kuramoto": (24, 4),
}

MAIN_HIDDEN_UNITS: Dict[str, int] = {
    "advertising": 32,
    "cybersecurity": 32,
    "distribution-planning": 128,
    "cucker-smale": 64,
    "kuramoto": 64,
}

SMOKE_HIDDEN_UNITS: Dict[str, int] = {
    "advertising": 8,
    "cybersecurity": 8,
    "distribution-planning": 8,
    "cucker-smale": 4,
    "kuramoto": 4,
}

MID_HIDDEN_UNITS: Dict[str, int] = {
    "advertising": 16,
    "cybersecurity": 16,
    "distribution-planning": 32,
    "cucker-smale": 16,
    "kuramoto": 16,
}


def resolve_preset(preset: str | None = None, *, quick: bool | None = None) -> str:
    if preset is None:
        preset = "smoke" if quick is not False else "main"
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown experiment preset {preset!r}. Expected one of {PRESET_NAMES}.")
    return preset


def seeds(preset: str, *, adaptive: bool = False) -> List[int]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return list(SMOKE_SEEDS)
    if preset == "mid":
        return list(MID_SEEDS)
    if preset == "high-confidence":
        return list(HIGH_CONFIDENCE_SEEDS)
    if adaptive:
        return list(ADAPTIVE_SEEDS)
    return list(MAIN_SEEDS)


def lambda_grid(preset: str) -> List[float]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return list(LAMBDA_GRID_SMOKE)
    if preset == "mid":
        return list(LAMBDA_GRID_MID)
    return list(LAMBDA_GRID_MAIN)


def eta_grid(preset: str) -> List[float]:
    return lambda_grid(preset)


def horizons(env_name: str, preset: str) -> List[int]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        grid = SMOKE_HORIZONS
    elif preset == "mid":
        grid = MID_HORIZONS
    else:
        grid = MAIN_HORIZONS
    return list(grid[env_name])


def nominal_horizon(env_name: str, preset: str) -> int:
    values = horizons(env_name, preset)
    return int(values[len(values) // 2])


def train_steps(env_name: str, preset: str) -> int:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return int(SMOKE_TRAIN_STEPS[env_name])
    if preset == "mid":
        return int(MID_TRAIN_STEPS[env_name])
    return int(MAIN_TRAIN_STEPS[env_name])


def batch_sizes(env_name: str, preset: str) -> tuple[int, int]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return SMOKE_BATCHES[env_name]
    if preset == "mid":
        return MID_BATCHES[env_name]
    return MAIN_BATCHES[env_name]


def hidden_units(env_name: str, preset: str) -> int:
    preset = resolve_preset(preset)
    if preset == "smoke":
        table = SMOKE_HIDDEN_UNITS
    elif preset == "mid":
        table = MID_HIDDEN_UNITS
    else:
        table = MAIN_HIDDEN_UNITS
    return int(table.get(env_name, 0))


def particle_counts(env_name: str, preset: str) -> tuple[int, int]:
    preset = resolve_preset(preset)
    if env_name not in {"cucker-smale", "kuramoto"}:
        return (0, 0)
    if preset == "smoke":
        return (6, 16)
    if preset == "mid":
        return (32, 128)
    if preset == "high-confidence":
        return (1024, 8192)
    return (512, 4096)


def diagnostic_config(preset: str) -> Dict[str, Any]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return {"samples": 32, "replications": 4, "lambdas": lambda_grid(preset), "etas": eta_grid(preset)}
    if preset == "mid":
        return {"samples": 96, "replications": 8, "lambdas": lambda_grid(preset), "etas": eta_grid(preset)}
    if preset == "high-confidence":
        return {"samples": 4096, "replications": 512, "lambdas": lambda_grid(preset), "etas": eta_grid(preset)}
    return {"samples": 2048, "replications": 128, "lambdas": lambda_grid(preset), "etas": eta_grid(preset)}


def budget_values(preset: str) -> tuple[List[int], List[int]]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return [4, 8, 16], [1, 2, 4]
    if preset == "mid":
        return [16, 32], [2, 4]
    return [64, 128, 256, 512, 1024], [4, 8, 16, 32, 64]


def particle_grid(preset: str) -> List[int]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return [4, 8, 16]
    if preset == "mid":
        return [16, 32, 64]
    if preset == "high-confidence":
        return [256, 512, 1024, 2048, 4096]
    return [128, 256, 512, 1024, 2048]


def signature_dims(preset: str) -> List[int]:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return [1, 2]
    if preset == "mid":
        return [1, 2, 4]
    return [1, 2, 4, 8, 16]


def validate_every(env_name: str, preset: str) -> int:
    preset = resolve_preset(preset)
    if preset == "smoke":
        return 1
    if preset == "mid":
        return max(5, train_steps(env_name, preset) // 10)
    return max(10, train_steps(env_name, preset) // 100)


def budget_variants(preset: str) -> List[Dict[str, Any]]:
    B_values, n_values = budget_values(preset)
    return [{"label": f"B{B}_n{n}", "train.B": B, "train.n": n} for B in B_values for n in n_values]


def fixed_lambda_variants(preset: str) -> List[float]:
    return lambda_grid(preset)


def parameter_grid(env_name: str, preset: str) -> Dict[str, Sequence[Any]]:
    B_values, n_values = budget_values(preset)
    grid: Dict[str, Sequence[Any]] = {
        "train.B": B_values,
        "train.n": n_values,
    }
    horizon_key = "env_config.T_train" if env_name == "cybersecurity" else "env_config.T"
    grid[horizon_key] = horizons(env_name, preset)
    return grid
