from __future__ import annotations

from configs.advertising import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    """horizons, mu0_val, sigma, lambdas, epsilon are fixed here and must not drift."""
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.horizons == (5,)
        assert cfg.mu0_val == (0.5, 0.5)
        assert cfg.sigma == 1.0
        assert cfg.lambdas == (0.1, 0.2, 0.4, 0.8)
        assert cfg.epsilon == 1.0
        assert cfg.validate_every == 10


def test_main_covers_the_full_comparison_surface():
    assert set(MAIN.algorithms) == {"simplex", "mfreinforce", "reinforce"}
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5  # distinct seeds
    assert MAIN.n_train == 10_000


def test_mid_and_smoke_are_single_seed_reductions_of_main():
    for cfg in (MID, SMOKE):
        assert cfg.seeds == (0,)
        assert cfg.horizons == MAIN.horizons
        assert cfg.algorithms == MAIN.algorithms
        assert cfg.lambdas == MAIN.lambdas
        assert cfg.B == MAIN.B  # theta is tiny here (~1.2k params); no memory-practicality reduction needed
    assert MID.n_train == 5000
    assert SMOKE.n_train == 20


def test_logit_transitions_matches_meuniers_stagewise_algorithm():
    for cfg in (MAIN, MID, SMOKE):
        for T in (1, 5, 8):
            expected = cfg.B * T + cfg.B * cfg.n_aux * T * (T + 1) // 2
            assert cfg.logit_transitions(T) == expected


def test_equal_budget_targets_match_mfreinforces_horizon_dependent_cost():
    for cfg in (MAIN, MID, SMOKE):
        for T in (1, 5, 8):
            target = cfg.logit_transitions(T) // T
            assert cfg.equal_budget_target(T) == target
            assert cfg.reinforce_B_equal_budget(T) == target
            assert cfg.simplex_B_equal_budget(T) == target - cfg.n_aux
            assert cfg.n_aux + cfg.simplex_B_equal_budget(T) == cfg.reinforce_B_equal_budget(T)
