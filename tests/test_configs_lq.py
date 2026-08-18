from __future__ import annotations

from configs.lq import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.lambdas == (0.05, 0.1, 0.2, 0.4, 0.8)
        assert set(cfg.algorithms) == {"exact_gradient", "reinforce"}
        assert cfg.B == 200


def test_main_covers_the_horizon_scaling_comparison():
    assert MAIN.horizons == (3, 5, 10)
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5
    assert MAIN.n_train == 8_000


def test_mid_and_smoke_are_single_seed_reductions_of_main():
    for cfg in (MID, SMOKE):
        assert cfg.seeds == (0,)
        assert cfg.horizons == (5,)  # base horizon only, not MAIN's full (3,5,10) scaling sweep
        assert cfg.lambdas == MAIN.lambdas
        assert cfg.lr == MAIN.lr
    assert MID.n_train == 2_000
    assert SMOKE.n_train == 20
