from __future__ import annotations

from pathlib import Path

import pytest

from helpers import read_csv_rows, tiny_twostate_config
from mfc.experiments.runner import load_checkpoint, run_train


def test_train_twostate_writes_artifacts_and_reconstructs(tmp_path: Path) -> None:
    result = run_train(tiny_twostate_config(tmp_path))

    assert result.run_dir.exists()
    assert (result.run_dir / "config.json").exists()
    assert (result.run_dir / "metadata.json").exists()
    assert (result.run_dir / "history.csv").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert read_csv_rows(result.run_dir / "history.csv")

    restored = load_checkpoint(result.checkpoint_path)
    assert restored["payload"]["env"] == "twostate"
    assert restored["payload"]["algorithm"] == "simplex"
    assert restored["control"].shape == (2,)


def test_train_twostate_reinforce_baseline(tmp_path: Path) -> None:
    config = tiny_twostate_config(tmp_path)
    config["algorithm"] = "reinforce"
    config["algorithm_config"] = {"baseline": "batch_mean"}
    config["train"]["run_name"] = "twostate_reinforce"

    result = run_train(config)

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert result.metrics["steps"] == 2


def test_train_lq_exact_gradient(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "lq",
            "algorithm": "exact-gradient",
            "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "lq",
                "seed": 3,
                "steps": 2,
                "lr": 1e-2,
                "validate_every": 1,
            },
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective_gap" in result.metrics


def test_train_lq_reinforce_baseline(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "lq",
            "algorithm": "reinforce",
            "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
            "algorithm_config": {"baseline": "batch_mean"},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "lq_reinforce",
                "seed": 23,
                "steps": 1,
                "lr": 1e-3,
                "B": 4,
                "validate_every": 1,
            },
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective_gap" in result.metrics


def test_train_cucker_smale_pathwise(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "cucker-smale",
            "algorithm": "pathwise-gradient",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N_pop": 4,
                "N_val": 4,
            },
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "cucker",
                "seed": 5,
                "steps": 1,
                "lr": 1e-3,
                "particles": 4,
                "replications": 1,
                "validate_every": 1,
            },
            "evaluation": {"particles": 4, "horizon": 2},
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective" in result.metrics


def test_train_cucker_smale_reinforce_baseline(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "cucker-smale",
            "algorithm": "reinforce",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N_pop": 4,
                "N_val": 4,
            },
            "algorithm_config": {"baseline": "batch_mean"},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "cucker_reinforce",
                "seed": 29,
                "steps": 1,
                "lr": 1e-3,
                "B": 4,
                "horizon": 2,
                "validate_every": 1,
            },
            "evaluation": {"particles": 4, "horizon": 2},
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective" in result.metrics


def test_train_lq_continuous_mfreinforce(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "lq",
            "algorithm": "continuous-mfreinforce",
            "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
            "algorithm_config": {"lambda": 0.1, "eta": 0.1},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "lq_continuous",
                "seed": 13,
                "steps": 1,
                "lr": 1e-3,
                "B": 4,
                "n": 2,
                "validate_every": 1,
            },
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective_gap" in result.metrics


def test_train_lq_continuous_oracle_sensitivity(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "lq",
            "algorithm": "continuous-oracle-sensitivity",
            "env_config": {"device": "cpu", "dtype": "float64", "T": 2, "c": 0.6, "gamma": 2.0, "gamma_T": 3.0},
            "algorithm_config": {"lambda": 0.1, "eta": 0.1, "sensitivity_mode": "oracle"},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "lq_oracle_sensitivity",
                "seed": 31,
                "steps": 1,
                "lr": 1e-3,
                "B": 4,
                "n": 1,
                "validate_every": 1,
            },
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective_gap" in result.metrics


def test_train_cucker_smale_continuous_mfreinforce(tmp_path: Path) -> None:
    result = run_train(
        {
            "env": "cucker-smale",
            "algorithm": "continuous-mfreinforce",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N_pop": 4,
                "N_val": 4,
            },
            "algorithm_config": {"lambda": 0.1, "eta": 0.1, "population_particles": 4},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "cucker_continuous",
                "seed": 17,
                "steps": 1,
                "lr": 1e-3,
                "B": 2,
                "n": 2,
                "validate_every": 1,
                "population_particles": 4,
            },
            "evaluation": {"particles": 4, "horizon": 2},
        }
    )

    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert "objective" in result.metrics


def test_invalid_env_algorithm_combination_fails(tmp_path: Path) -> None:
    config = {
        "env": "lq",
        "algorithm": "simplex",
        "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
        "train": {"output_dir": str(tmp_path), "steps": 1},
    }

    with pytest.raises(ValueError, match="finite-state"):
        run_train(config)
