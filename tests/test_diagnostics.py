from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from helpers import read_csv_rows, tiny_twostate_config
from mfc.environments import LQConfig, LinearQuadraticMFC
from mfc.experiments import notebook_helpers as nh
from mfc.experiments.runner import (
    run_functional_law_diagnostic,
    run_gradient_diagnostic,
    run_perturbation_diagnostic,
    run_sensitivity_diagnostic,
)
from mfc.experiments.studies import run_score_validation


def test_diagnostic_commands_write_csv(tmp_path: Path) -> None:
    base = tiny_twostate_config(tmp_path)
    base["diagnostic"] = {"samples": 4, "replications": 2, "lambdas": [0.1], "etas": [0.1]}

    for command, fn in [
        ("perturb", run_perturbation_diagnostic),
        ("functional", run_functional_law_diagnostic),
        ("gradient", run_gradient_diagnostic),
        ("sensitivity", run_sensitivity_diagnostic),
    ]:
        config = {**base, "train": {**base["train"], "run_name": command}}
        result = fn(config)
        assert result.diagnostics_path is not None
        assert read_csv_rows(result.diagnostics_path)


def test_finite_perturbation_tv_table_reports_bounds_and_reference_violations(tmp_path: Path) -> None:
    base = tiny_twostate_config(tmp_path)
    base["diagnostic"] = {"samples": 6, "max_raw_samples": 6, "lambdas": [0.2]}

    simplex = run_perturbation_diagnostic({**base, "train": {**base["train"], "run_name": "simplex_tv"}})
    logits_config = {
        **base,
        "algorithm": "logits",
        "algorithm_config": {"epsilon": 0.2, "flow_particles": 4},
        "train": {**base["train"], "run_name": "logits_tv"},
    }
    logits = run_perturbation_diagnostic(logits_config)

    assert simplex.diagnostics_path is not None
    assert logits.diagnostics_path is not None
    simplex_row = read_csv_rows(simplex.diagnostics_path)[0]
    logits_row = read_csv_rows(logits.diagnostics_path)[0]
    assert "simplex_expected_d_tv" in simplex_row
    assert int(float(simplex_row["simplex_violation_count"])) == 0
    assert "logit_reference_violation_count" in logits_row
    assert logits_row["logit_reference_size"]

    diagnostics = {
        "simplex": {
            "perturbation": pd.read_csv(simplex.diagnostics_path),
            "perturbation_perturbation_samples": pd.read_csv(simplex.run_dir / "perturbation_samples.csv"),
        },
        "logits": {
            "perturbation": pd.read_csv(logits.diagnostics_path),
            "perturbation_perturbation_samples": pd.read_csv(logits.run_dir / "perturbation_samples.csv"),
        },
    }
    table = nh.perturbation_tv_comparison_table(diagnostics)
    assert set(table["algorithm"]) == {"simplex", "logits"}
    simplex_table = table[table["algorithm"] == "simplex"].iloc[0]
    logits_table = table[table["algorithm"] == "logits"].iloc[0]
    assert simplex_table["comparison_kind"] == "simplex pathwise bound lambda"
    assert logits_table["comparison_kind"] == "logit reference epsilon/2"
    assert int(simplex_table["violation_count"]) == 0
    assert pd.notna(simplex_table["simplex_expected_unit_radius"])
    assert pd.notna(logits_table["logit_reference_size"])


def test_continuous_mfreinforce_diagnostics_write_csv(tmp_path: Path) -> None:
    config = {
        "env": "lq",
        "algorithm": "continuous-mfreinforce",
        "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
        "algorithm_config": {"lambda": 0.1, "eta": 0.1},
        "train": {
            "output_dir": str(tmp_path),
            "run_name": "continuous_diag",
            "seed": 19,
            "B": 4,
            "n": 2,
        },
        "diagnostic": {"replications": 2, "lambdas": [0.1], "etas": [0.1]},
    }

    gradient = run_gradient_diagnostic(config)
    sensitivity = run_sensitivity_diagnostic({**config, "train": {**config["train"], "run_name": "continuous_sens"}})
    score = run_score_validation({**config, "train": {**config["train"], "run_name": "continuous_score"}})

    assert gradient.diagnostics_path is not None and read_csv_rows(gradient.diagnostics_path)
    assert sensitivity.diagnostics_path is not None and read_csv_rows(sensitivity.diagnostics_path)
    assert score.diagnostics_path is not None and read_csv_rows(score.diagnostics_path)
    assert (score.run_dir / "score_coordinates.csv").exists()


def test_lq_perturbed_oracle_is_lambda_aware(tmp_path: Path) -> None:
    env = LinearQuadraticMFC(LQConfig(device=torch.device("cpu"), dtype=torch.float64, T=2))
    theta = env.zero_policy()
    cost0, grad0 = env.exact_gradient(theta, lambda_=0.0)
    cost1, grad1 = env.exact_gradient(theta, lambda_=0.2)

    assert cost1 > cost0
    assert not torch.allclose(grad0, grad1)

    result = run_gradient_diagnostic(
        {
            "env": "lq",
            "algorithm": "exact-gradient",
            "env_config": {"device": "cpu", "dtype": "float64", "T": 2},
            "algorithm_config": {},
            "train": {"output_dir": str(tmp_path), "run_name": "lq_lambda_oracle", "seed": 31},
            "diagnostic": {"replications": 1, "lambdas": [0.2]},
        }
    )

    row = read_csv_rows(result.diagnostics_path)[0]
    assert row["oracle_kind"] == "analytic_exact"
    assert float(row["oracle_lambda"]) == 0.2
    assert float(row["mse"]) == 0.0


def test_pathwise_gradient_diagnostic_reports_reference_oracle(tmp_path: Path) -> None:
    result = run_gradient_diagnostic(
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
                "run_name": "pathwise_oracle_gradient",
                "seed": 29,
                "B": 2,
                "n": 2,
                "population_particles": 4,
            },
            "evaluation": {"particles": 4, "horizon": 2},
            "diagnostic": {"replications": 1, "lambdas": [0.1], "oracle_particles": 4, "oracle_replications": 1},
        }
    )

    assert result.diagnostics_path is not None
    row = read_csv_rows(result.diagnostics_path)[0]
    assert row["oracle_kind"] == "pathwise_ad_reference"
    assert row["oracle_norm"]
    coordinate_rows = read_csv_rows(result.run_dir / "gradient_coordinates.csv")
    assert "oracle" in coordinate_rows[0]
