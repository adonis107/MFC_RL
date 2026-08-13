from __future__ import annotations

import csv
from pathlib import Path


def tiny_twostate_config(tmp_path: Path) -> dict:
    return {
        "env": "twostate",
        "algorithm": "simplex",
        "env_config": {
            "device": "cpu",
            "dtype": "float64",
            "T": 2,
            "N": 4,
            "n": 2,
            "n_train": 2,
            "validate_every": 1,
        },
        "algorithm_config": {"lambda": 0.2, "eta": 0.2},
        "train": {
            "output_dir": str(tmp_path),
            "run_name": "twostate",
            "seed": 7,
            "steps": 2,
            "lr": 1e-2,
            "B": 4,
            "n": 2,
            "validate_every": 1,
            "flow_mode": "exact",
            "flow_particles": 4,
            "mu0": [0.2, 0.8],
        },
        "evaluation": {"mu0": [0.2, 0.8], "horizon": 2},
    }


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
