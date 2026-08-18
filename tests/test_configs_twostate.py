from __future__ import annotations

from configs.twostate import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    """B, n_aux, lr, validate_every, mu0 range, mu0_val, sigma, epsilon are
    fixed by the reference's training protocol and must not drift."""
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.B == 200
        assert cfg.n_aux == 10
        assert cfg.lr == 1e-3
        assert cfg.validate_every == 10
        assert (cfg.mu0_low, cfg.mu0_high) == (0.1, 0.9)
        assert cfg.mu0_val == (0.2, 0.8)
        assert cfg.sigma == 1.0
        assert cfg.epsilon == 0.2
        assert cfg.lambdas == (0.05, 0.1, 0.2, 0.4, 0.8)


def test_main_covers_the_full_comparison_surface():
    assert set(MAIN.algorithms) == {"simplex", "mfreinforce", "reinforce"}
    assert set(MAIN.budget_modes) == {"equal_parameters", "equal_budget"}
    assert set(MAIN.flows) == {"exact", "particle"}
    assert MAIN.horizons == (2, 5, 10)
    assert 2 in MAIN.horizons  # the reference's base horizon
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5  # distinct seeds
    assert MAIN.n_train == 10_000


def test_mid_is_a_single_seed_base_horizon_full_length_reduction_of_main():
    assert MID.horizons == (2,)
    assert MID.seeds == (0,)
    assert MID.n_train == 5000  # the reference's "full number of episodes" (main now trains longer than this)
    assert MID.algorithms == MAIN.algorithms
    assert MID.lambdas == MAIN.lambdas


def test_smoke_is_a_fast_single_seed_base_horizon_reduction_of_main():
    assert SMOKE.horizons == (2,)
    assert SMOKE.seeds == (0,)
    assert SMOKE.n_train == 20
    assert SMOKE.algorithms == MAIN.algorithms
    assert SMOKE.lambdas == MAIN.lambdas


def test_logit_transitions_matches_meuniers_stagewise_algorithm():
    """
    mfreinforce here is Meunier, Pham & Reisinger's original Algorithm 1+2:
    Algorithm 2's gradient-of-logits estimator (n_aux*T*(T+1)/2 transitions,
    a fresh stagewise batch per target time) is rerun once per main
    trajectory, giving C_logit(T) = B*T + B*n_aux*T*(T+1)/2 — quadratic in
    T, unlike simplex's linear T*(n_aux+B) reusable-batch cost.
    """
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.n_aux == 10 and cfg.B == 200  # mfreinforce's unchanged anchor
        for T in (1, 2, 5, 10):
            expected = cfg.B * T + cfg.B * cfg.n_aux * T * (T + 1) // 2
            assert cfg.logit_transitions(T) == expected


def test_equal_budget_targets_match_mfreinforces_horizon_dependent_cost():
    """
    At each horizon T, simplex's (n_aux+B) and reinforce's B both match
    mfreinforce's true cost at that T (C_logit(T)/T), not its raw (n_aux, B)
    numbers — since mfreinforce's cost is quadratic in T while simplex's own
    formula is linear, matching budgets requires a horizon-dependent
    allocation, growing well past mfreinforce's own B=200 for T>2.
    """
    for cfg in (MAIN, MID, SMOKE):
        for T in (1, 2, 5, 10):
            target = cfg.logit_transitions(T) // T
            assert cfg.equal_budget_target(T) == target
            assert cfg.reinforce_B_equal_budget(T) == target
            assert cfg.simplex_B_equal_budget(T) == target - cfg.n_aux
            # both algorithms' equal-budget per-step total matches mfreinforce's, exactly
            assert cfg.n_aux + cfg.simplex_B_equal_budget(T) == cfg.reinforce_B_equal_budget(T)

    # concrete values at the reference horizon T=2, B=200, n_aux=10:
    # C_logit(2) = 200*2 + 200*10*2*3/2 = 400 + 6000 = 6400; target = 3200
    assert MAIN.equal_budget_target(2) == 3200
    assert MAIN.reinforce_B_equal_budget(2) == 3200
    assert MAIN.simplex_B_equal_budget(2) == 3190
