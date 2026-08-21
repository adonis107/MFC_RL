from __future__ import annotations

from configs.distribution_planning import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    """n_aux, lr, validate_every, horizons, mu0_val, sigma, epsilon are
    fixed by the reference's training and validation protocol and must not
    drift; B is checked separately since mid/smoke deliberately shrink it
    for memory-practicality reasons (see the module docstring)."""
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.n_aux == 10
        assert cfg.lr == 1e-4
        assert cfg.validate_every == 10
        assert cfg.horizons == (5,)
        assert cfg.mu0_val == tuple([0.1] * 10)
        assert cfg.sigma == 1.0
        assert cfg.lambdas == (0.1, 0.2, 0.4)  # trimmed from the reference's (0.1, 0.2, 0.4, 0.8); see the module docstring
        assert cfg.epsilon == 1.0


def test_main_covers_the_full_comparison_surface():
    assert set(MAIN.algorithms) == {"simplex", "mfreinforce", "reinforce"}
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5  # distinct seeds
    assert MAIN.n_train == 100_000
    assert MAIN.B == 500  # the reference's own value; only mid/smoke shrink it
    # only the budget-matched, particle-flow regime, as for cybersecurity and
    # advertising; two-state's main is the only one comparing budget modes and
    # flows against each other
    assert MAIN.budget_modes == ("equal_budget",)
    assert MAIN.flows == ("particle",)


def test_mid_and_smoke_are_single_seed_reductions_of_main():
    for cfg in (MID, SMOKE):
        assert cfg.seeds == (0,)
        assert cfg.horizons == MAIN.horizons
        assert cfg.algorithms == MAIN.algorithms
        assert cfg.lambdas == MAIN.lambdas
        assert cfg.B < MAIN.B  # shrunk for memory practicality, not reference fidelity
    assert MID.n_train == 5000
    assert MID.B == 100
    assert SMOKE.n_train == 20
    assert SMOKE.B == 20


def test_logit_transitions_matches_meuniers_stagewise_algorithm():
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.n_aux == 10
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

    # concrete value at the reference horizon T=5, B=500, n_aux=10:
    # C_logit(5) = 500*5 + 500*10*5*6/2 = 2500 + 75000 = 77500; target = 15500
    assert MAIN.equal_budget_target(5) == 15500
    assert MAIN.reinforce_B_equal_budget(5) == 15500
    assert MAIN.simplex_B_equal_budget(5) == 15490
