from __future__ import annotations

import functools

import pytest
import torch

from mfc.algorithms import _common, reinforce, simplex
from mfc.environments.twostate import TwoState

torch.set_default_dtype(torch.float64)


def _reinforce_partial_objective(env, action_probs_fn, theta, mu0, T, frozen_mu_flow):
    """
    Exact, theta-differentiable objective matching what REINFORCE's
    L_t*G_t estimates in expectation: the state-visitation law evolves
    under theta through the policy (as in `exact_population_flow`), but
    the population *context* fed to the policy/reward/transition at each
    step is frozen at `frozen_mu_flow[t]` rather than re-differentiated —
    exactly what `reinforce.gradient_estimate` does by taking an
    already-computed, detached `mu_flow`. This is NOT the same as freezing
    the state-visitation weights themselves (which would trivially zero
    out any action-independent-reward environment like two-state, since
    probabilities sum to 1); actions still shift which states are visited.
    """
    N, A = env.n_states, env.n_actions
    states_grid = torch.arange(N).unsqueeze(-1).expand(N, A)
    actions_grid = torch.arange(A).unsqueeze(0).expand(N, A)
    states_all = torch.arange(N)

    nu = mu0
    total = torch.zeros(())
    for t in range(T):
        context = frozen_mu_flow[t]
        pi = _common.eval_batched(action_probs_fn, theta, t, states_all, context)
        P = env.transition_probs(states_grid, actions_grid, context)
        K = torch.einsum("xa,xay->xy", pi, P)
        r = env.reward(states_grid, actions_grid, context)
        total = total + (nu * (pi * r).sum(dim=-1)).sum()
        nu = nu @ K
    g = env.terminal_reward(states_all, frozen_mu_flow[T])
    return total + (nu * g).sum()


def test_gradient_estimate_matches_partial_objective_oracle():
    """Averaged over many independent batches, `reinforce.gradient_estimate`
    should recover the exact gradient of the objective it implicitly
    targets (policy differentiated, population context frozen)."""
    torch.manual_seed(0)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 2

    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta0, mu0, T)
    th = theta0.clone().requires_grad_(True)
    oracle = torch.autograd.grad(_reinforce_partial_objective(env, env.policy_probs, th, mu0, T, mu_flow), th)[0]

    reps, B = 400, 400
    samples = torch.stack([reinforce.gradient_estimate(env, env.policy_probs, theta0, mu_flow, mu0, T, B) for _ in range(reps)])
    mean, se = samples.mean(dim=0), samples.std(dim=0) / reps**0.5
    assert torch.all((mean - oracle).abs() < 6 * se + 0.01)


def test_gradient_estimate_misses_the_mean_field_term():
    """
    The whole point of this baseline (context.md: "show missing mean-field
    term"): REINFORCE's gradient direction should differ materially from
    the true gradient (autograd through the *undetached* exact flow, i.e.
    accounting for how theta shifts the population law itself) whenever
    the reward has real population coupling (kappa>0, as in two-state).
    """
    torch.manual_seed(1)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 2

    th = theta0.clone().requires_grad_(True)
    full_gradient = torch.autograd.grad(_common.exact_objective(env, env.policy_probs, th, mu0, T, detach=False), th)[0]

    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta0, mu0, T)
    th2 = theta0.clone().requires_grad_(True)
    partial_gradient = torch.autograd.grad(_reinforce_partial_objective(env, env.policy_probs, th2, mu0, T, mu_flow), th2)[0]

    assert (full_gradient - partial_gradient).norm() > 0.5 * full_gradient.norm()


def test_gradient_estimate_shape():
    env = TwoState()
    theta = torch.tensor([0.1, -0.2])
    mu0 = torch.tensor([0.5, 0.5])
    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta, mu0, T=2)
    g = reinforce.gradient_estimate(env, env.policy_probs, theta, mu_flow, mu0, T=2, B=10)
    assert g.shape == (2,)
    assert torch.isfinite(g).all()


def test_gradient_step_with_particle_flow_runs():
    env = TwoState()
    theta = torch.tensor([0.1, -0.2])
    mu0 = torch.tensor([0.5, 0.5])
    particle_flow_fn = functools.partial(_common.particle_population_flow, n_particles=100)
    g = reinforce.gradient_step(env, env.policy_probs, theta, mu0, T=2, B=50, population_flow_fn=particle_flow_fn)
    assert g.shape == (2,)
    assert torch.isfinite(g).all()


def test_train_runs_stably_and_moves_theta():
    """
    Integration smoke test. Unlike simplex, REINFORCE is *not* expected to
    reliably increase the true objective J from an arbitrary start — its
    gradient can point away from J's ascent direction whenever the missing
    mean-field term matters (see test_gradient_estimate_misses_the_mean_field_term
    and test_reinforce_diverges_more_than_simplex_from_a_poor_start below).
    This only checks the optimizer loop itself is wired correctly: no
    NaN/Inf, and theta actually moves.
    """
    torch.manual_seed(2)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([-2.0, -2.0])

    theta_final, history = reinforce.train(env, env.policy_probs, theta0, mu0, T=2, n_train=300, B=200, lr=5e-2)

    assert torch.isfinite(theta_final).all()
    assert history.shape == (301, 2)
    assert not torch.allclose(theta_final, theta0)


def test_reinforce_diverges_more_than_simplex_from_a_poor_start():
    """
    context.md: "show missing mean-field term by comparison with reinforce".
    From the same poor initialization and budget, REINFORCE's theta should
    end up farther from the known optimum than simplex's — the mean-field
    correction simplex includes (and REINFORCE omits) is precisely what
    steers training toward the true optimum instead of away from it.
    """
    env = TwoState()
    T = 2
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([-2.0, -2.0])
    optimal_theta = env.optimal_theta()

    torch.manual_seed(2)
    theta_reinforce, _ = reinforce.train(env, env.policy_probs, theta0, mu0, T=T, n_train=300, B=200, lr=5e-2)

    torch.manual_seed(2)
    theta_simplex, _ = simplex.train(env, env.policy_probs, theta0, mu0, T=T, n_train=300, n_aux=10, B=200, lam=0.2, lr=5e-2)

    dist_reinforce = (theta_reinforce - optimal_theta).norm()
    dist_simplex = (theta_simplex - optimal_theta).norm()
    assert dist_reinforce > dist_simplex


@pytest.mark.parametrize("T,gamma,use_baseline", [(3, 1.0, False), (3, 0.9, True), (1, 1.0, True), (0, 1.0, False)])
def test_gradient_estimate_matches_the_literal_formula(T, gamma, use_baseline):
    """`reinforce.gradient_estimate` accumulates into one (D,) vector via the
    running-score-sum rearrangement of sum_t L_t G_t, instead of building the
    (T,B,D) score tensor. Driven from the same random stream it must
    reproduce the literal formula, not merely match it in law."""
    env = TwoState()
    theta = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    B, D = 32, theta.numel()
    baseline = torch.linspace(-1.0, 1.0, T + 1) if use_baseline else torch.zeros(T + 1)
    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta, mu0, T)

    # literal transcription of g = (1/B) sum_b sum_t L_t^(b) (G_t^(b) - b_t)
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(1)
    states = torch.multinomial(mu0.expand(B, env.n_states), 1, generator=gen).reshape(B)
    rewards, L = torch.zeros(T, B), torch.zeros(T, B, D)
    terminal_reward = None
    for t in range(T + 1):
        mu_t = mu_flow[t]
        if t < T:
            actions = _common.sample_actions(env.policy_probs, theta, t, states, mu_t, generator=gen)
            L[t] = _common.policy_score(env.policy_probs, theta, t, states, actions, mu_t)
            rewards[t] = env.reward(states, actions, mu_t)
            states = env.sample_next_states(states, actions, mu_t, generator=gen)
        else:
            terminal_reward = env.terminal_reward(states, mu_t)
    G = torch.zeros(T + 1, B)
    G[T] = (gamma**T) * terminal_reward
    for t in range(T - 1, -1, -1):
        G[t] = (gamma**t) * rewards[t] + G[t + 1]
    expected = (L * (G[:T] - baseline[:T].view(-1, 1)).unsqueeze(-1)).sum(dim=(0, 1)) / B

    actual = reinforce.gradient_estimate(
        env, env.policy_probs, theta, mu_flow, mu0, T, B,
        gamma=gamma, baseline=baseline, generator=torch.Generator(device=torch.get_default_device()).manual_seed(1),
    )
    assert torch.allclose(actual, expected, rtol=1e-11, atol=1e-13)
