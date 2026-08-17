from __future__ import annotations

from pathlib import Path

from mfc.experiments import notebook_helpers as nh
from mfc.experiments import presets


def test_paper_presets_capture_final_budget_choices(tmp_path: Path) -> None:
    assert "mid" in presets.PRESET_NAMES
    assert presets.seeds("main") == [0, 1, 2, 3, 4]
    assert presets.seeds("high-confidence") == list(range(10))
    assert 0.8 in presets.lambda_grid("main")
    assert presets.SMALL_LAMBDA_FIT == [0.0125, 0.025, 0.05, 0.1]
    assert len(presets.horizons("kuramoto", "main")) == 3

    lq = nh.continuous_benchmark_config("lq", tmp_path, "lq_main", quick=False)
    assert lq["train"]["steps"] == 10_000
    assert lq["train"]["B"] == 512
    assert lq["train"]["n"] == 32
    assert lq["env_config"]["T"] == 20
    assert lq["env_config"]["c"] == 0.60
    assert lq["env_config"]["gamma"] == 2.0
    assert "continuous-oracle-sensitivity" in nh.continuous_algorithms_for_env("lq")
    assert "exact-gradient" in nh.continuous_algorithms_for_env("lq")
    assert lq["diagnostic"]["lambdas"][-1] == 0.8

    smoke = nh.benchmark_config("advertising", "simplex", tmp_path, "adv_smoke", preset="smoke")
    assert smoke["train"]["steps"] == 4
    assert smoke["train"]["B"] == 8
    assert smoke["diagnostic"]["replications"] == 4
    assert smoke["diagnostic"]["lambdas"] == [0.05, 0.1, 0.8]


def test_mid_preset_is_laptop_scale_but_full_shape(tmp_path: Path) -> None:
    assert presets.seeds("mid") == [0]
    assert presets.lambda_grid("mid") == [0.025, 0.1, 0.2]
    assert presets.horizons("cucker-smale", "mid") == [10, 20]
    assert presets.particle_counts("kuramoto", "mid") == (32, 128)
    assert presets.diagnostic_config("mid")["replications"] == 8
    assert presets.diagnostic_config("mid")["samples"] == 96
    assert len(presets.budget_variants("mid")) == 4
    assert presets.signature_dims("mid") == [1, 2, 4]
    assert all(presets.train_steps(env_name, "mid") == 1_000 for env_name in presets.MAIN_TRAIN_STEPS)

    lq = nh.continuous_benchmark_config("lq", tmp_path, "lq_mid", preset="mid")
    assert lq["train"]["steps"] == 1_000
    assert lq["train"]["B"] == 64
    assert lq["train"]["n"] == 64

    advertising = nh.benchmark_config("advertising", "simplex", tmp_path, "adv_mid", preset="mid")
    assert advertising["train"]["steps"] == 1_000
    assert advertising["train"]["B"] == 32
    assert advertising["train"]["n"] == 4
    assert advertising["env_config"]["T"] == 10
    assert advertising["evaluation"]["oracle_grid_size"] == 101

    cucker = nh.continuous_benchmark_config("cucker-smale", tmp_path, "cs_mid", preset="mid")
    assert cucker["train"]["steps"] == 1_000
    assert cucker["train"]["B"] == 24
    assert cucker["train"]["n"] == 4
    assert cucker["env_config"]["T"] == 20
    assert cucker["env_config"]["N_pop"] == 32
    assert cucker["env_config"]["N_val"] == 128
    assert cucker["diagnostic"]["oracle_replications"] == 4


def test_notebook_helper_default_device_override(tmp_path: Path) -> None:
    original = nh.DEFAULT_DEVICE
    try:
        nh.set_default_device("cpu")
        config = nh.benchmark_config("twostate", "simplex", tmp_path, "twostate_cpu")
        assert config["env_config"]["device"] == "cpu"
        assert nh.DEFAULT_DEVICE == "cpu"
    finally:
        nh.set_default_device(original)
