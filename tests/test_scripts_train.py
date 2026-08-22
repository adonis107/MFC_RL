from __future__ import annotations

import pytest
import torch

from configs.lq import SMOKE as LQ_SMOKE
from configs.twostate import MAIN as TWOSTATE_MAIN
from configs.twostate import SMOKE as TWOSTATE_SMOKE
from configs.twostate import TwoStateRunConfig
from scripts.train import experiment_tag, list_continuous_experiments, list_experiments, make_simplex_step, materialize_duplicates, run_all, run_continuous


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


def test_twostate_main_groups_matches_the_4_requested_combinations():
    """The specific (budget_mode, flow, T) restriction actually asked for,
    not the full budget_modes x flows x horizons cartesian product
    (2 x 2 x 3 = 12) MAIN's own budget_modes/flows/horizons fields would
    otherwise define."""
    assert TWOSTATE_MAIN.groups == (
        ("equal_parameters", "exact", 2),
        ("equal_budget", "exact", 2),
        ("equal_budget", "particle", 5),
        ("equal_budget", "particle", 10),
    )
    # the exact flow is only run at the base horizon, so no (budget_mode, T)
    # pair appears under both flows (see the config's inline comment)
    assert {(bm, T) for bm, fl, T in TWOSTATE_MAIN.groups if fl == "exact"}.isdisjoint(
        {(bm, T) for bm, fl, T in TWOSTATE_MAIN.groups if fl == "particle"}
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


def test_list_continuous_experiments_override_semantics():
    experiments = list_continuous_experiments(LQ_SMOKE, "simplex", lam=0.1, seed=2)
    assert len(experiments) == 1
    T, lam, seed = experiments[0]
    assert lam == 0.1 and seed == 2 and T == LQ_SMOKE.horizons[0]

    full = list_continuous_experiments(LQ_SMOKE, "simplex")
    assert len(full) == len(LQ_SMOKE.horizons) * len(LQ_SMOKE.lambdas) * len(LQ_SMOKE.seeds)


def test_list_continuous_experiments_gives_reinforce_no_lambda_axis():
    """The perturbation exists to expose the mean-field sensitivity through a
    likelihood ratio; reinforce drops that term, so it is trained once per
    (T, seed) on the nominal process rather than once per lambda — the same
    convention the discrete grid already uses."""
    experiments = list_continuous_experiments(LQ_SMOKE, "reinforce")
    assert len(experiments) == len(LQ_SMOKE.horizons) * len(LQ_SMOKE.seeds)
    assert {lam for _, lam, _ in experiments} == {None}

    with pytest.raises(ValueError, match="no perturbation scale"):
        list_continuous_experiments(LQ_SMOKE, "reinforce", lam=0.1)


def test_simplex_step_uses_dedicated_aux_budget_when_present(monkeypatch):
    class Cfg:
        n_aux = 10
        simplex_n_aux = 200
        B = 500
        sigma = 1.0

        @staticmethod
        def simplex_B_equal_budget(T):
            return 15300

    seen = {}

    def fake_gradient_step(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    monkeypatch.setattr("scripts.train.simplex.gradient_step", fake_gradient_step)
    step = make_simplex_step(Cfg(), flow="exact", budget_mode="equal_budget")
    step(None, None, torch.zeros(1), torch.ones(2) / 2, T=5, lam=0.2, generator=torch.Generator())

    assert seen["n_aux"] == 200
    assert seen["B"] == 15300


def test_run_continuous_records_reinforce_without_a_lambda(tmp_path):
    """Its saved run must carry lam=None (so `scripts.test.group_by` and the
    notebooks' plots treat it as one baseline, not one line per lambda) and
    its filename must carry no lambda tag."""
    from configs.lq import LQRunConfig
    import scripts.train as train_mod

    cfg = LQRunConfig(name="tiny", horizons=(2,), n_train=3, seeds=(0,), B=8, n_aux=4, validate_every=1)
    train_mod.CONFIGS.setdefault("lq", {})["tiny_no_lambda"] = cfg

    results = run_continuous("lq", "reinforce", "tiny_no_lambda", output_dir=str(tmp_path))
    assert len(results) == 1
    assert results[0]["lam"] is None
    assert (tmp_path / "lq" / "tiny_no_lambda" / "reinforce_T2_seed0.pt").exists()


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


def test_train_run_theta_history_is_capped_and_off_device(tmp_path, monkeypatch):
    """Regression test for a real OOM: `theta_history` used to append every
    iterate on the training device, so a large policy (distribution planning's
    MLP has D~7.7e4) at the reference's n_train=100_000 accumulated ~28.5 GiB
    of GPU memory over a run and starved the gradient step itself."""
    from scripts.train import CONFIGS, MAX_THETA_SNAPSHOTS

    n_train = 4 * MAX_THETA_SNAPSHOTS + 3  # forces a stride > 1, and a final iterate off the stride
    cfg = TwoStateRunConfig(name="test", seeds=(0,), horizons=(2,), lambdas=(0.2,), n_train=n_train, validate_every=n_train)
    monkeypatch.setitem(CONFIGS["twostate"], "smoke", cfg)

    [result] = run_all("twostate", "simplex", "smoke", output_dir=str(tmp_path))
    history, iterations = result["theta_history"], result["theta_history_iterations"]

    assert history.shape[0] <= MAX_THETA_SNAPSHOTS + 1  # capped, not one row per iterate
    assert history.shape[0] == iterations.numel()
    assert history.device.type == "cpu"  # never occupies accelerator memory
    assert iterations[0].item() == 0 and iterations[-1].item() == n_train  # endpoints always kept
    assert torch.equal(history[0], result["theta0"].cpu())
    assert torch.equal(history[-1], result["theta_final"].cpu())
