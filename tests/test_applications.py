from __future__ import annotations

import json
from pathlib import Path

from helpers import read_csv_rows, tiny_twostate_config
from mfc.experiments.application import run_application_diagnostics
from mfc.experiments.studies import run_score_validation


def test_application_and_score_outputs_are_notebook_ready(tmp_path: Path) -> None:
    config = tiny_twostate_config(tmp_path)
    config["diagnostic"] = {"replications": 2, "lambdas": [0.1]}

    app = run_application_diagnostics({**config, "train": {**config["train"], "run_name": "app"}})
    score = run_score_validation({**config, "train": {**config["train"], "run_name": "score"}})

    assert (app.run_dir / "population_flow.csv").exists()
    assert (app.run_dir / "time_metrics.csv").exists()
    assert (app.run_dir / "policy.csv").exists()
    assert (score.run_dir / "score_coordinates.csv").exists()
    assert read_csv_rows(score.run_dir / "diagnostics.csv")


def test_application_outputs_advertising_dp_reference(tmp_path: Path) -> None:
    result = run_application_diagnostics(
        {
            "env": "advertising",
            "algorithm": "simplex",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N": 4,
                "n": 2,
            },
            "algorithm_config": {"lambda": 0.1, "eta": 0.1},
            "train": {"output_dir": str(tmp_path), "run_name": "advertising_reference", "seed": 31},
            "evaluation": {"mu0": [0.5, 0.5], "horizon": 2, "oracle_grid_size": 11, "oracle_action_grid_size": 11},
        }
    )

    assert (result.run_dir / "reference_time_metrics.csv").exists()
    assert (result.run_dir / "reference_policy.csv").exists()
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["reference_kind"] == "finite_horizon_dp_oracle"
    assert "reference_value_gap" in metrics


def test_application_outputs_model_based_finite_reference(tmp_path: Path) -> None:
    result = run_application_diagnostics(
        {
            "env": "distribution-planning",
            "algorithm": "simplex",
            "env_config": {
                "device": "cpu",
                "dtype": "float64",
                "T": 2,
                "hidden_units": 4,
                "N": 4,
                "n": 2,
            },
            "algorithm_config": {"lambda": 0.1, "eta": 0.1},
            "train": {"output_dir": str(tmp_path), "run_name": "distribution_reference", "seed": 37},
            "evaluation": {"mu0": [0.1] * 10, "horizon": 2, "oracle_steps": 1, "oracle_restarts": 1},
        }
    )

    assert (result.run_dir / "reference_population_flow.csv").exists()
    assert (result.run_dir / "reference_time_metrics.csv").exists()
    assert (result.run_dir / "reference_policy.csv").exists()
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["reference_kind"] == "model_based_exact_flow_reference"
    assert "reference_value_gap" in metrics
