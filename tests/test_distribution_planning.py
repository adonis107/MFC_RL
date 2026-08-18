from __future__ import annotations

import pytest
import torch

from mfc.environments.distribution_planning import LEFT, RIGHT, STAY, DistributionPlanning
from scripts.test import average_mismatch_l2, cumulative_movement, cyclic_wasserstein, terminal_mismatch_l2


def _random_mu(n=10, *, generator=None):
    u = torch.rand(n, generator=generator).clamp_min(1e-12)
    e = -torch.log(u)
    return e / e.sum()


def test_transition_probs_match_deterministic_cyclic_shift():
    env = DistributionPlanning()
    for x in (0, 5, 9):
        assert torch.equal(env.transition_probs(torch.tensor(x), torch.tensor(STAY)), torch.nn.functional.one_hot(torch.tensor(x), 10).to(env.dtype))
        assert torch.equal(env.transition_probs(torch.tensor(x), torch.tensor(LEFT)), torch.nn.functional.one_hot(torch.tensor((x - 1) % 10), 10).to(env.dtype))
        assert torch.equal(env.transition_probs(torch.tensor(x), torch.tensor(RIGHT)), torch.nn.functional.one_hot(torch.tensor((x + 1) % 10), 10).to(env.dtype))


def test_transition_wraps_around_the_torus():
    env = DistributionPlanning()
    assert env.sample_next_states(torch.tensor(9), torch.tensor(RIGHT)).item() == 0
    assert env.sample_next_states(torch.tensor(0), torch.tensor(LEFT)).item() == 9


def test_transition_probs_are_normalized():
    env = DistributionPlanning()
    states = torch.arange(10)
    actions = torch.tensor([LEFT, STAY, RIGHT] * 3 + [STAY])
    probs = env.transition_probs(states, actions)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10, dtype=env.dtype))


def test_reward_matches_reference_formula():
    env = DistributionPlanning()
    mu = _random_mu()
    expected_mismatch = (mu - env.target_law).square().sum()
    r_stay = env.reward(torch.tensor(3), torch.tensor(STAY), mu)
    r_move = env.reward(torch.tensor(3), torch.tensor(LEFT), mu)
    assert torch.allclose(r_stay, -expected_mismatch)
    assert torch.allclose(r_move, -env.config.c_mov - expected_mismatch)
    # action-independent movement cost magnitude: LEFT and RIGHT cost the same
    assert torch.equal(env.reward(torch.tensor(3), torch.tensor(LEFT), mu), env.reward(torch.tensor(3), torch.tensor(RIGHT), mu))


def test_terminal_reward_is_state_independent_and_matches_formula():
    env = DistributionPlanning()
    mu = _random_mu()
    expected = -(mu - env.target_law).square().sum()
    states = torch.arange(10)
    g = env.terminal_reward(states, mu)
    assert g.shape == (10,)
    assert torch.allclose(g, expected.expand(10))


def test_policy_probs_is_a_valid_distribution_over_actions():
    env = DistributionPlanning()
    generator = torch.Generator(device=env.device).manual_seed(0)
    theta = env.init_theta(generator=generator)
    mu = _random_mu(generator=generator)
    for t in range(5):
        for x in range(env.n_states):
            probs = env.policy_probs(theta, t, torch.tensor(x), mu)
            assert probs.shape == (env.n_actions,)
            assert torch.allclose(probs.sum(), torch.tensor(1.0, dtype=env.dtype))
            assert (probs >= 0).all()


def test_init_theta_has_expected_parameter_count():
    env = DistributionPlanning()
    theta = env.init_theta()
    H = env.config.hidden_width
    in_dim, out_dim = 1 + env.n_states, env.n_states * env.n_actions
    expected = (H * in_dim + H) + (H * H + H) + (out_dim * H + out_dim)
    assert theta.numel() == expected


def test_sample_mu0_is_a_valid_distribution_with_full_support():
    env = DistributionPlanning()
    generator = torch.Generator(device=env.device).manual_seed(0)
    mu0 = env.sample_mu0((1000,), generator=generator)
    assert mu0.shape == (1000, env.n_states)
    assert torch.allclose(mu0.sum(dim=-1), torch.ones(1000, dtype=env.dtype))
    assert (mu0 > 0).all()


def test_cyclic_distance_matches_reference_definition():
    env = DistributionPlanning()
    assert env.cyclic_distance(torch.tensor(0), torch.tensor(9)).item() == 1
    assert env.cyclic_distance(torch.tensor(2), torch.tensor(7)).item() == 5
    assert env.cyclic_distance(torch.tensor(0), torch.tensor(5)).item() == 5
    assert env.cyclic_distance(torch.tensor(3), torch.tensor(3)).item() == 0


def test_cyclic_wasserstein_matches_point_mass_ground_truth():
    """For point masses, W_1 on the cycle is exactly the cyclic distance
    between the two points — the ground truth used to derive/verify the
    closed-form (median-of-CDF-difference) implementation."""
    env = DistributionPlanning()
    N = env.n_states
    for k in range(N):
        mu = torch.zeros(N, dtype=env.dtype)
        mu[0] = 1.0
        target = torch.zeros(N, dtype=env.dtype)
        target[k] = 1.0
        w1 = cyclic_wasserstein(mu, target)
        assert torch.allclose(w1, env.cyclic_distance(torch.tensor(0), torch.tensor(k)).to(env.dtype))


def test_cyclic_wasserstein_is_zero_for_identical_laws():
    env = DistributionPlanning()
    mu = _random_mu(env.n_states)
    assert cyclic_wasserstein(mu, mu).item() == pytest.approx(0.0, abs=1e-10)


def test_mismatch_l2_is_zero_when_flow_sits_exactly_on_target():
    """Construct a degenerate policy that always keeps mu_t = mu_target by
    starting exactly there and forcing STAY everywhere; both L2 mismatch
    diagnostics should then be exactly zero."""
    env = DistributionPlanning()
    stay_policy = lambda theta, t, state, mu: torch.tensor([0.0, 1.0, 0.0], dtype=env.dtype)  # always STAY
    mu0 = env.target_law
    theta = env.init_theta()
    assert terminal_mismatch_l2(env, stay_policy, theta, mu0, T=5).item() == pytest.approx(0.0, abs=1e-10)
    assert average_mismatch_l2(env, stay_policy, theta, mu0, T=5).item() == pytest.approx(0.0, abs=1e-10)


def test_cumulative_movement_is_zero_under_always_stay_and_positive_under_always_move():
    env = DistributionPlanning()
    stay_policy = lambda theta, t, state, mu: torch.tensor([0.0, 1.0, 0.0], dtype=env.dtype)
    move_policy = lambda theta, t, state, mu: torch.tensor([1.0, 0.0, 0.0], dtype=env.dtype)  # always LEFT
    mu0 = env.sample_mu0(generator=torch.Generator(device=env.device).manual_seed(0))
    theta = env.init_theta()
    assert cumulative_movement(env, stay_policy, theta, mu0, T=5, stay_action=STAY).item() == pytest.approx(0.0, abs=1e-10)
    assert cumulative_movement(env, move_policy, theta, mu0, T=5, stay_action=STAY).item() == pytest.approx(5.0, abs=1e-10)
