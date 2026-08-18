from __future__ import annotations

import torch

from mfc.algorithms._common import exact_objective
from mfc.environments.twostate import TwoState


def test_exact_objective_gamma_discount_matches_manual_sum():
    """gamma=1.0 (default) must reproduce the undiscounted objective exactly;
    gamma<1 must match a hand-rolled discounted sum of the same per-step
    exact rewards (env-agnostic: this is `_common.exact_objective`'s own
    discounting, not anything environment-specific)."""
    env = TwoState()
    theta = env.optimal_theta()
    mu0 = torch.tensor([0.2, 0.8], dtype=env.dtype)
    T = 3
    gamma = 0.5

    undiscounted = exact_objective(env, env.policy_probs, theta, mu0, T)
    assert torch.allclose(undiscounted, exact_objective(env, env.policy_probs, theta, mu0, T, gamma=1.0))

    discounted = exact_objective(env, env.policy_probs, theta, mu0, T, gamma=gamma)

    # manual per-step reconstruction from the exact population flow
    from mfc.algorithms._common import eval_batched, exact_population_flow

    mu_flow = exact_population_flow(env, env.policy_probs, theta, mu0, T)
    states = torch.arange(env.n_states)
    total = torch.zeros((), dtype=env.dtype)
    for t in range(T):
        mu_t = mu_flow[t]
        pi = eval_batched(env.policy_probs, theta, t, states, mu_t)
        for x in range(env.n_states):
            for a in range(env.n_actions):
                total = total + (gamma**t) * mu_t[x] * pi[x, a] * env.reward(torch.tensor(x), torch.tensor(a), mu_t)
    g = env.terminal_reward(states, mu_flow[T])
    total = total + (gamma**T) * (mu_flow[T] * g).sum()

    assert torch.allclose(discounted, total, atol=1e-10)
    assert not torch.allclose(discounted, undiscounted)  # sanity: discounting actually changes the value here
