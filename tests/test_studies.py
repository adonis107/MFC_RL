from __future__ import annotations

from pathlib import Path

from helpers import read_csv_rows, tiny_twostate_config
from mfc.experiments.runner import main as runner_main, run_sweep, run_train
from mfc.experiments.studies import run_study


def test_kuramoto_robustness_keeps_checkpoint_policy_shape(tmp_path: Path) -> None:
    train = run_train(
        {
            "env": "kuramoto",
            "algorithm": "continuous-mfreinforce",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N_pop": 4,
                "N_val": 4,
                "sigma_omega": 0.0,
                "include_frequency": False,
            },
            "algorithm_config": {"lambda": 0.1, "eta": 0.1, "population_particles": 4},
            "train": {
                "output_dir": str(tmp_path),
                "run_name": "kuramoto_train",
                "seed": 23,
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

    result = run_study(
        {
            "env": "kuramoto",
            "algorithm": "continuous-mfreinforce",
            "checkpoint": str(train.checkpoint_path),
            "train": {"output_dir": str(tmp_path), "run_name": "kuramoto_robustness", "overwrite": True},
            "study": {"name": "robustness"},
        }
    )

    assert result.diagnostics_path is not None
    assert read_csv_rows(result.diagnostics_path)


def test_sweep_runs_train_grid(tmp_path: Path) -> None:
    config = tiny_twostate_config(tmp_path)
    config["train"]["run_name"] = "sweep"
    config["train"]["steps"] = 1
    config["sweep"] = {"parameters": {"algorithm_config.lambda": [0.1, 0.2]}}

    result = run_sweep(config)

    assert result.diagnostics_path is not None
    assert len(read_csv_rows(result.diagnostics_path)) == 2


def test_study_grid_and_catalog_cli(tmp_path: Path) -> None:
    config = tiny_twostate_config(tmp_path)
    config["train"]["run_name"] = "study"
    config["diagnostic"] = {"replications": 2, "lambdas": [0.1]}
    config["study"] = {
        "name": "budget-allocation",
        "command": "diagnose-gradient",
        "budgets": [
            {"label": "small", "train.B": 2, "train.n": 1},
            {"label": "medium", "train.B": 3, "train.n": 1},
        ],
    }

    result = run_study(config)
    assert (result.run_dir / "grid_metrics.csv").exists()
    assert read_csv_rows(result.run_dir / "diagnostics.csv")

    exit_code = runner_main(["catalog", "--output-dir", str(tmp_path), "--run-name", "catalog"])
    assert exit_code == 0
    assert (tmp_path / "catalog" / "catalog.json").exists()
    assert read_csv_rows(tmp_path / "catalog" / "catalog.csv")
