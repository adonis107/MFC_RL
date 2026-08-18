from __future__ import annotations

from configs.cybersecurity import MAIN, MID, SMOKE


def test_training_protocol_matches_reference():
    """B, n_aux, lr, validate_every, T_val, mu0_val, sigma, epsilon are
    fixed by the reference's training and validation protocol and must not drift."""
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.B == 200
        assert cfg.n_aux == 1
        assert cfg.lr == 1e-3
        assert cfg.validate_every == 10
        assert cfg.T_val == 50
        assert cfg.mu0_val == (0.25, 0.25, 0.25, 0.25)
        assert cfg.sigma == 1.0
        assert cfg.lambdas == (0.1, 0.2, 0.4, 0.8)
        assert cfg.epsilon == 1.0


def test_main_covers_the_full_comparison_surface():
    """main sweeps 3 horizons (including the reference's own short T_train=3),
    5 seeds, all 3 algorithms, but only budget_mode="equal_budget" and only
    flow="particle" (unlike two-state's main, which sweeps both budget modes
    and both flows) — a deliberate narrower comparison surface for this
    benchmark, per configs/cybersecurity.py's module docstring."""
    assert set(MAIN.algorithms) == {"simplex", "mfreinforce", "reinforce"}
    assert len(MAIN.seeds) == 5
    assert len(set(MAIN.seeds)) == 5  # distinct seeds
    assert MAIN.n_train == 20_000
    assert MAIN.horizons == (3, 6, 12)
    assert 3 in MAIN.horizons  # the reference's own (short) training horizon
    assert MAIN.budget_modes == ("equal_budget",)
    assert MAIN.flows == ("particle",)


def test_mid_and_smoke_are_single_seed_base_horizon_reductions_of_main():
    for cfg in (MID, SMOKE):
        assert cfg.seeds == (0,)
        assert cfg.horizons == (3,)  # base horizon only, not main's full {3,6,12} sweep
        assert cfg.budget_modes == ("equal_parameters",)  # the simpler default, not main's equal_budget-only
        assert cfg.flows == ("exact",)  # the simpler default, not main's particle-only
        assert cfg.algorithms == MAIN.algorithms
        assert cfg.lambdas == MAIN.lambdas
    assert MID.n_train == 5000
    assert SMOKE.n_train == 20


def test_logit_transitions_matches_meuniers_stagewise_algorithm():
    for cfg in (MAIN, MID, SMOKE):
        assert cfg.n_aux == 1 and cfg.B == 200
        for T in (1, 3, 5):
            expected = cfg.B * T + cfg.B * cfg.n_aux * T * (T + 1) // 2
            assert cfg.logit_transitions(T) == expected


def test_equal_budget_targets_match_mfreinforces_horizon_dependent_cost():
    for cfg in (MAIN, MID, SMOKE):
        for T in (1, 3, 5):
            target = cfg.logit_transitions(T) // T
            assert cfg.equal_budget_target(T) == target
            assert cfg.reinforce_B_equal_budget(T) == target
            assert cfg.simplex_B_equal_budget(T) == target - cfg.n_aux
            assert cfg.n_aux + cfg.simplex_B_equal_budget(T) == cfg.reinforce_B_equal_budget(T)

    # concrete values at main's horizon sweep, B=200, n_aux=1 (module docstring):
    # much gentler growth than two-state's, since n_aux=1 keeps mfreinforce's
    # quadratic term small even as T grows.
    assert MAIN.simplex_B_equal_budget(3) == 599
    assert MAIN.simplex_B_equal_budget(6) == 899
    assert MAIN.simplex_B_equal_budget(12) == 1499
