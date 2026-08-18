from __future__ import annotations

import pytest
import torch

from configs.lq import SMOKE as LQ_SMOKE
from configs.twostate import MAIN as TWOSTATE_MAIN
from configs.twostate import SMOKE as TWOSTATE_SMOKE
from configs.twostate import TwoStateRunConfig
from scripts.train import experiment_tag, list_experiments, list_lq_experiments, materialize_duplicates, run_all


def test_list_experiments_with_no_overrides_reproduces_the_full_config_grid():
    experiments, duplicates, skipped_budget, skipped_flows = list_experiments(TWOSTATE_SMOKE, "simplex")
    assert skipped_budget == [] and skipped_flows == []
    assert duplicates == []  # simplex is genuinely budget_mode-dependent: never deduped
    # smoke: 1 budget_mode x 1 flow x 1 horizon x 5 lambdas x 1 seed
    assert len(experiments) == 5
    assert {lam for _, _, _, lam, _ in experiments} == set(TWOSTATE_SMOKE.lambdas)


def test_list_experiments_pins_only_the_given_axes():
    experiments, duplicates, _, _ = list_experiments(TWOSTATE_SMOKE, "simplex", lam=0.2, seed=7)
    assert duplicates == []
    assert len(experiments) == 1
    budget_mode, flow, T, lam, seed = experiments[0]
    assert lam == 0.2
    assert seed == 7  # not required to be a member of cfg.seeds
    assert T == TWOSTATE_SMOKE.horizons[0]


def test_list_experiments_rejects_lambda_for_an_algorithm_without_a_perturbation_scale():
    with pytest.raises(ValueError, match="no perturbation scale"):
        list_experiments(TWOSTATE_SMOKE, "reinforce", lam=0.2)


def test_list_experiments_reinforce_has_no_lambda_axis():
    experiments, duplicates, _, _ = list_experiments(TWOSTATE_SMOKE, "reinforce")
    assert duplicates == []
    assert len(experiments) == 1
    assert experiments[0][3] is None  # lam


def test_twostate_main_groups_matches_the_5_requested_combinations():
    """The specific (budget_mode, flow, T) restriction actually asked for,
    not the full budget_modes x flows x horizons cartesian product
    (2 x 2 x 3 = 12) MAIN's own budget_modes/flows/horizons fields would
    otherwise define."""
    assert TWOSTATE_MAIN.groups == (
        ("equal_parameters", "exact", 2),
        ("equal_budget", "exact", 2),
        ("equal_budget", "particle", 2),
        ("equal_budget", "particle", 5),
        ("equal_budget", "particle", 10),
    )


def test_list_experiments_uses_groups_instead_of_the_cartesian_product_when_set():
    cfg = TwoStateRunConfig(
        name="test",
        budget_modes=("equal_parameters", "equal_budget"),  # would cartesian-product to 4 (budget_mode, flow) pairs...
        flows=("exact", "particle"),
        horizons=(2, 5),
        groups=(("equal_budget", "particle", 5),),  # ...but only this 1 explicit group should be used
        seeds=(0,),
        lambdas=(0.2,),
    )
    experiments, duplicates, _, _ = list_experiments(cfg, "reinforce")
    assert duplicates == []
    assert experiments == [("equal_budget", "particle", 5, None, 0)]


def test_list_experiments_groups_respects_axis_overrides():
    cfg = TwoStateRunConfig(name="test", groups=(("equal_parameters", "exact", 2), ("equal_budget", "particle", 5)), seeds=(0,), lambdas=(0.2,))
    experiments, _, _, _ = list_experiments(cfg, "reinforce", budget_mode="equal_budget")
    assert experiments == [("equal_budget", "particle", 5, None, 0)]


def test_list_experiments_dedups_budget_mode_invariant_algorithm():
    """mfreinforce ignores budget_mode, so of two groups sharing (flow, T)
    only the first requested budget_mode is a real training job; the second
    is reported as a duplicate of the first."""
    cfg = TwoStateRunConfig(name="test", groups=(("equal_parameters", "exact", 2), ("equal_budget", "exact", 2)), seeds=(0, 1))
    experiments, duplicates, _, _ = list_experiments(cfg, "mfreinforce")
    assert len(experiments) == 2  # one real job per seed
    assert {(bm, fl, T, seed) for bm, fl, T, lam, seed in experiments} == {("equal_parameters", "exact", 2, 0), ("equal_parameters", "exact", 2, 1)}
    assert len(duplicates) == 2  # one duplicate per seed
    assert {(bm, fl, T, seed, primary) for bm, fl, T, lam, seed, primary in duplicates} == {
        ("equal_budget", "exact", 2, 0, "equal_parameters"),
        ("equal_budget", "exact", 2, 1, "equal_parameters"),
    }


def test_list_lq_experiments_override_semantics():
    experiments = list_lq_experiments(LQ_SMOKE, lam=0.1, seed=2)
    assert len(experiments) == 1
    T, lam, seed = experiments[0]
    assert lam == 0.1 and seed == 2 and T == LQ_SMOKE.horizons[0]

    full = list_lq_experiments(LQ_SMOKE)
    assert len(full) == len(LQ_SMOKE.horizons) * len(LQ_SMOKE.lambdas) * len(LQ_SMOKE.seeds)


def test_run_all_dtype_tag_prevents_float32_from_colliding_with_float64(tmp_path):
    """Regression test for a real bug: without a dtype-specific filename tag,
    a float32 job silently no-op'd as 'already exists' against a float64 job
    saved under the identical (alg, budget_mode, flow, T, lam, seed) tag,
    permanently losing the float32 result."""
    kwargs = dict(budget_mode="equal_parameters", flow="exact", T=2, lam=0.2, seed=0, output_dir=str(tmp_path))
    run_all("twostate", "simplex", "smoke", dtype="float64", **kwargs)
    run_all("twostate", "simplex", "smoke", dtype="float32", **kwargs)

    saved = sorted(p.name for p in (tmp_path / "twostate" / "smoke").glob("*.pt"))
    assert len(saved) == 2
    assert any("dtypefloat32" in name for name in saved)
    assert any("dtypefloat32" not in name for name in saved)  # the float64 file keeps its original (undecorated) name

    loaded = {torch.load(tmp_path / "twostate" / "smoke" / name, weights_only=False)["dtype"] for name in saved}
    assert loaded == {"torch.float64", "torch.float32"}


def test_run_all_skips_existing_output_unless_overwrite(tmp_path):
    kwargs = dict(budget_mode="equal_parameters", flow="exact", T=2, lam=0.2, seed=0, output_dir=str(tmp_path))
    [result_first] = run_all("twostate", "simplex", "smoke", **kwargs)
    assert result_first["elapsed_seconds"] > 0.0

    skipped = run_all("twostate", "simplex", "smoke", **kwargs)
    assert skipped == []  # skipped, not retrained

    [result_overwritten] = run_all("twostate", "simplex", "smoke", overwrite=True, **kwargs)
    assert result_overwritten["elapsed_seconds"] > 0.0  # actually retrained, not skipped


def test_run_all_materializes_duplicates_in_process_for_mfreinforce(tmp_path, monkeypatch):
    """End-to-end: run_all on a groups-based config with two budget_modes
    sharing (flow, T) trains mfreinforce once and copies the result into the
    second tag, with its own `budget_mode` field correctly relabeled."""
    from scripts.train import CONFIGS

    cfg = TwoStateRunConfig(name="test", groups=(("equal_parameters", "exact", 2), ("equal_budget", "exact", 2)), seeds=(0,), n_train=5, validate_every=5)
    monkeypatch.setitem(CONFIGS["twostate"], "smoke", cfg)

    results = run_all("twostate", "mfreinforce", "smoke", output_dir=str(tmp_path))
    assert len(results) == 2  # 1 trained + 1 copied

    primary_path = tmp_path / "twostate" / "smoke" / f"{experiment_tag('mfreinforce', 'equal_parameters', 'exact', 2, None, 0)}.pt"
    dup_path = tmp_path / "twostate" / "smoke" / f"{experiment_tag('mfreinforce', 'equal_budget', 'exact', 2, None, 0)}.pt"
    assert primary_path.exists() and dup_path.exists()
    primary = torch.load(primary_path, weights_only=False)
    dup = torch.load(dup_path, weights_only=False)
    assert primary["budget_mode"] == "equal_parameters"
    assert dup["budget_mode"] == "equal_budget"
    assert torch.equal(primary["theta_final"], dup["theta_final"])


def test_materialize_duplicates_skips_when_primary_not_yet_trained(tmp_path, monkeypatch):
    cfg = TwoStateRunConfig(name="test", groups=(("equal_parameters", "exact", 2), ("equal_budget", "exact", 2)), seeds=(0,))
    from scripts.train import CONFIGS

    monkeypatch.setitem(CONFIGS["twostate"], "smoke", cfg)

    copied = materialize_duplicates("twostate", "mfreinforce", "smoke", output_dir=str(tmp_path))
    assert copied == []
    assert list((tmp_path / "twostate" / "smoke").glob("*.pt")) == []
