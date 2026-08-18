from __future__ import annotations

import torch

from mfc.environments.cybersecurity import DI, DS, KEEP, SWITCH, UI, US, CyberSecurity


def _random_mu(n=4, *, generator=None):
    u = torch.rand(n, generator=generator).clamp_min(1e-12)
    e = -torch.log(u)
    return e / e.sum()


def test_generator_matrix_rows_sum_to_zero():
    env = CyberSecurity()
    mu = _random_mu()
    states = torch.tensor([DI, DS, UI, US])
    actions = torch.tensor([KEEP, SWITCH, KEEP, SWITCH])
    Q = env._generator_matrix(mu, actions)
    assert torch.allclose(Q.sum(dim=-1), torch.zeros(4, dtype=env.dtype), atol=1e-12)


def test_transition_probs_are_normalized_and_nonnegative():
    env = CyberSecurity()
    mu = _random_mu()
    states = torch.tensor([DI, DS, UI, US, DI, DS, UI, US])
    actions = torch.tensor([KEEP, KEEP, KEEP, KEEP, SWITCH, SWITCH, SWITCH, SWITCH])
    probs = env.transition_probs(states, actions, mu)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8, dtype=env.dtype))
    assert (probs >= -1e-10).all()


def test_transition_probs_match_matrix_exponential_directly():
    env = CyberSecurity()
    mu = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=env.dtype)
    for a in (KEEP, SWITCH):
        Q = env._generator_matrix(mu, torch.tensor(a))
        P_expected = torch.linalg.matrix_exp(env.config.dt * Q)
        for x in range(4):
            got = env.transition_probs(torch.tensor(x), torch.tensor(a), mu)
            assert torch.allclose(got, P_expected[x], atol=1e-10)


def test_reward_matches_reference_cost_formula():
    env = CyberSecurity()
    mu = _random_mu()
    kD, kI = env.config.k_D, env.config.k_I
    dt = env.config.dt
    expected = {DI: -dt * (kD + kI), DS: -dt * kD, UI: -dt * kI, US: 0.0}
    for x, exp in expected.items():
        r = env.reward(torch.tensor(x), torch.tensor(KEEP), mu)
        assert torch.allclose(r, torch.tensor(exp, dtype=env.dtype))
        g = env.terminal_reward(torch.tensor(x), mu)
        assert torch.equal(g, r)
    # action- and mu-independent
    r_keep = env.reward(torch.tensor(DI), torch.tensor(KEEP), mu)
    r_switch = env.reward(torch.tensor(DI), torch.tensor(SWITCH), mu)
    assert torch.equal(r_keep, r_switch)


def test_policy_probs_is_a_valid_distribution_over_actions():
    env = CyberSecurity()
    generator = torch.Generator(device=env.device).manual_seed(0)
    theta = env.init_theta(generator=generator)
    mu = _random_mu(generator=generator)
    for t in range(3):
        for x in range(env.n_states):
            probs = env.policy_probs(theta, t, torch.tensor(x), mu)
            assert probs.shape == (env.n_actions,)
            assert torch.allclose(probs.sum(), torch.tensor(1.0, dtype=env.dtype))
            assert (probs >= 0).all()


def test_init_theta_has_expected_parameter_count():
    env = CyberSecurity()
    theta = env.init_theta()
    H = env.config.hidden_width
    in_dim, out_dim = 1 + env.n_states, env.n_states * env.n_actions
    expected = (H * in_dim + H) + (H * H + H) + (out_dim * H + out_dim)
    assert theta.numel() == expected


def test_aggregate_fractions_matches_reference_definitions():
    env = CyberSecurity()
    mu_flow = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],  # (DI, DS, UI, US)
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=env.dtype,
    )
    infected, defended = env.aggregate_fractions(mu_flow)
    assert torch.allclose(infected, torch.tensor([0.1 + 0.3, 0.25 + 0.25], dtype=env.dtype))
    assert torch.allclose(defended, torch.tensor([0.1 + 0.2, 0.25 + 0.25], dtype=env.dtype))


def test_sample_mu0_is_a_valid_distribution_with_full_support():
    env = CyberSecurity()
    generator = torch.Generator(device=env.device).manual_seed(0)
    mu0 = env.sample_mu0((1000,), generator=generator)
    assert mu0.shape == (1000, env.n_states)
    assert torch.allclose(mu0.sum(dim=-1), torch.ones(1000, dtype=env.dtype))
    assert (mu0 > 0).all()
