from __future__ import annotations

from pathlib import Path

from mfc.experiments.runner import main as runner_main


def test_cli_overrides_nested_json_fields(tmp_path: Path) -> None:
    exit_code = runner_main(
        [
            "train",
            "--env",
            "twostate",
            "--algorithm",
            "simplex",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "cli",
            "--seed",
            "11",
            "--set",
            "env_config.device=\"cpu\"",
            "--set",
            "env_config.dtype=\"float64\"",
            "--set",
            "env_config.T=1",
            "--set",
            "train.steps=1",
            "--set",
            "train.B=2",
            "--set",
            "train.n=1",
            "--set",
            "train.mu0=[0.2,0.8]",
            "--set",
            "evaluation.mu0=[0.2,0.8]",
            "--set",
            "algorithm_config.lambda=0.1",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli" / "config.json").exists()
