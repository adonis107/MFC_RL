from __future__ import annotations

from configs.lq import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.lambdas == (0.05, 0.1, 0.2, 0.4, 0.8)
        # the continuous-state simplex estimator against its own REINFORCE ablation;
        # the closed-form exact_gradient is the oracle, not a compared algorithm
        assert cfg.algorithms == ("simplex", "reinforce")
        assert cfg.B == 400
        assert cfg.n_aux == 200
        assert cfg.baseline == "loo"


def test_reinforce_matches_simplexs_per_step_transition_budget():
    """Equal budget (context.md): simplex spends (n_aux+B)*T transitions per
    step, reinforce the same total in one main batch, at every horizon."""
    for cfg in (MAIN, MID, SMOKE):
        for T in (2, 5, 20):
            assert cfg.reinforce_B_equal_budget() * T == cfg.transitions_per_step(T)


def test_main_covers_the_horizon_scaling_comparison():
    assert MAIN.horizons == (3, 5, 10)
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5
    assert MAIN.n_train == 6_000


def test_mid_and_smoke_are_single_seed_reductions_of_main():
    for cfg in (MID, SMOKE):
        assert cfg.seeds == (0,)
        assert cfg.horizons == (5,)  # base horizon only, not MAIN's full (3,5,10) scaling sweep
        assert cfg.lambdas == MAIN.lambdas
        assert cfg.lr == MAIN.lr
    assert MID.n_train == 2_000
    assert SMOKE.n_train == 20
