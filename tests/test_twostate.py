from __future__ import annotations

import torch

from mfc.environments.twostate import ST, MV, TwoState


def test_transition_probs_are_normalized():
    env = TwoState()
    states = torch.tensor([0, 0, 1, 1])
    actions = torch.tensor([ST, MV, ST, MV])
    probs = env.transition_probs(states, actions)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, dtype=env.dtype))


def test_transition_probs_match_reference_formula():
    env = TwoState()
    # ST never moves.
    assert torch.equal(env.transition_probs(torch.tensor(0), torch.tensor(ST)), torch.tensor([1.0, 0.0], dtype=env.dtype))
    assert torch.equal(env.transition_probs(torch.tensor(1), torch.tensor(ST)), torch.tensor([0.0, 1.0], dtype=env.dtype))
    # MV switches with probability lambda_x.
    assert torch.allclose(env.transition_probs(torch.tensor(0), torch.tensor(MV)), torch.tensor([1 - 0.5, 0.5], dtype=env.dtype))
    assert torch.allclose(env.transition_probs(torch.tensor(1), torch.tensor(MV)), torch.tensor([0.8, 1 - 0.8], dtype=env.dtype))


def test_reward_matches_reference_formula():
    env = TwoState()
    mu = torch.tensor([0.4, 0.6], dtype=env.dtype)
    r0 = env.reward(torch.tensor(0), None, mu)
    r1 = env.reward(torch.tensor(1), None, mu)
    expected_penalty = mu[1] ** 2 + 10.0 * abs(mu[1] - 0.4)
    assert torch.allclose(r0, -expected_penalty)
    assert torch.allclose(r1, 1.0 - expected_penalty)
    assert torch.equal(env.terminal_reward(torch.tensor(1), mu), r1)


def test_optimal_policy_matches_reference_values():
    env = TwoState()
    pi = env.optimal_policy()
    assert torch.allclose(pi[0, ST], torch.tensor(0.2, dtype=env.dtype))
    assert torch.allclose(pi[1, ST], torch.tensor(0.25, dtype=env.dtype))
    assert torch.allclose(pi.sum(dim=-1), torch.ones(2, dtype=env.dtype))


def test_optimal_theta_realizes_optimal_policy_under_policy_probs():
    env = TwoState()
    theta_star = env.optimal_theta()
    pi = env.optimal_policy()
    mu = env.target_law
    for x in range(2):
        probs = env.policy_probs(theta_star, 0, torch.tensor(x), mu)
        assert torch.allclose(probs, pi[x], atol=1e-12)


def test_optimal_policy_is_a_fixed_point_at_target_law():
    """Under pi*, every initial law maps to bar_mu=(p,1-p) after one step and stays there."""
    env = TwoState()
    pi = env.optimal_policy()

    def step(mu: torch.Tensor) -> torch.Tensor:
        # K[x, x'] = sum_a pi(a|x) P(x'|x,a); mu_next = mu @ K
        kernel = torch.zeros(2, 2, dtype=env.dtype)
        for x in range(2):
            for a in (ST, MV):
                kernel[x] += pi[x, a] * env.transition_probs(torch.tensor(x), torch.tensor(a))
        return mu @ kernel

    for m0 in (0.1, 0.5, 0.9):
        mu0 = torch.tensor([1 - m0, m0], dtype=env.dtype)
        mu1 = step(mu0)
        assert torch.allclose(mu1, env.target_law, atol=1e-10)
        mu2 = step(mu1)
        assert torch.allclose(mu2, env.target_law, atol=1e-10)
