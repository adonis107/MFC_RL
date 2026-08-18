from __future__ import annotations

import functools

import torch

from mfc.algorithms import _common, mfreinforce
from mfc.environments.twostate import TwoState

torch.set_default_dtype(torch.float64)


def _autograd_logit_sensitivity_oracle(env, action_probs_fn, theta0, mu0, T):
    """D_t = grad_theta log(mu_t^theta), via autograd through the
    differentiable exact population flow. Shape (T+1, n_states, D)."""
    oracle = []
    for t in range(T + 1):
        th = theta0.clone().requires_grad_(True)
        mu_flow = _common.exact_population_flow(env, action_probs_fn, th, mu0, T, detach=False)
        log_mu_t = torch.log(mu_flow[t])
        rows = [torch.autograd.grad(log_mu_t[i], th, retain_graph=True)[0] for i in range(env.n_states)]
        oracle.append(torch.stack(rows))
    return torch.stack(oracle)


def test_estimate_logit_sensitivity_flow_matches_autograd_oracle():
    """
    Algorithm 2 (stagewise-independent, fresh batch per target time),
    averaged over many auxiliary batches, should converge to the exact
    grad_theta log(mu_t^theta).
    """
    torch.manual_seed(0)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 2
    epsilon = 0.2

    oracle = _autograd_logit_sensitivity_oracle(env, env.policy_probs, theta0, mu0, T)
    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta0, mu0, T)

    reps, n_aux = 2000, 20
    acc = torch.zeros_like(oracle)
    for _ in range(reps):
        acc += mfreinforce.estimate_logit_sensitivity_flow(env, env.policy_probs, theta0, mu_flow, mu0, T, n_aux, epsilon)
    mean = acc / reps

    assert torch.allclose(mean, oracle, atol=0.03)


def test_gradient_estimate_satisfies_the_gaussian_stein_identity():
    """
    Isolates the Q_t = Lambda_t @ D_t / epsilon construction (eq. 2.6):
    since z_j = l_t(j) + epsilon*Lambda_j is linear in Lambda,
    E[Q_t_component * z_j] should equal D_t(j, component) exactly (a
    Gaussian-shift Stein identity), independent of any policy/reward.
    """
    torch.manual_seed(1)
    N, D = 2, 2
    epsilon = 0.3
    mu_t = torch.tensor([0.6, 0.4])
    D_val = torch.tensor([[0.37, -0.52], [0.11, 0.9]])
    j = 1

    n = 2_000_000
    Lambda = torch.randn(n, N)
    l = torch.log(mu_t) + epsilon * Lambda
    Q = (Lambda @ D_val) / epsilon
    estimate = (Q * l[:, j].unsqueeze(-1)).mean(dim=0)
    assert torch.allclose(estimate, D_val[j], atol=5e-3)


def _pathwise_J_eps_oracle(env, theta0, mu0, T, epsilon, n_mc, generator=None):
    """
    Exact-over-actions/states, Monte-Carlo-over-Lambda-only oracle for
    J^epsilon(theta) at T=1: since M_t=softmax(log(mu_t^theta)+eps*Lambda)
    is a reparametrized (pathwise-differentiable) function of theta given
    fixed Lambda draws, autograd differentiates it directly — no
    score-function trick needed for the oracle itself. Returns
    grad_theta J^epsilon(theta0).
    """
    assert T == 1
    N = env.n_states
    Lambda0 = torch.randn(n_mc, N, generator=generator)
    Lambda1 = torch.randn(n_mc, N, generator=generator)

    def batched_policy(theta, x, M):
        idx = torch.arange(theta.shape[0])
        onehot = (idx == x).to(theta.dtype)
        p_move = torch.sigmoid((onehot * theta).sum())
        return torch.stack([1 - p_move, p_move]).expand(M.shape[0], 2)

    def J_eps(theta):
        mu_flow = _common.exact_population_flow(env, env.policy_probs, theta, mu0, T, detach=False)
        M0 = torch.softmax(torch.log(mu_flow[0].clamp_min(1e-12)).unsqueeze(0) + epsilon * Lambda0, dim=-1)
        M1 = torch.softmax(torch.log(mu_flow[1].clamp_min(1e-12)).unsqueeze(0) + epsilon * Lambda1, dim=-1)
        total = torch.zeros(n_mc)
        for x0 in range(N):
            pi0 = batched_policy(theta, x0, M0)
            for a0 in range(env.n_actions):
                r0 = env.reward(torch.tensor(x0), torch.tensor(a0), M0)
                trans0 = env.transition_probs(torch.tensor(x0), torch.tensor(a0))
                for x1 in range(N):
                    g1 = env.terminal_reward(torch.tensor(x1), M1)
                    total = total + mu0[x0] * pi0[:, a0] * trans0[x1] * (r0 + g1)
        return total.mean()

    th = theta0.clone().requires_grad_(True)
    return torch.autograd.grad(J_eps(th), th)[0]


def test_gradient_estimate_matches_pathwise_oracle_with_exact_D():
    """
    Averaged over many main-batch draws, gradient_estimate (fed the exact
    autograd D, isolating it from Algorithm 2's own estimation noise)
    should recover grad_theta J^epsilon(theta) — NOT grad_theta J(theta):
    at epsilon=0.3 the perturbation bias is large, exactly like simplex at
    large lambda, so comparing against the unperturbed gradient is not a
    valid check (this reproduces the same pitfall hit while validating
    simplex.py).
    """
    torch.manual_seed(2)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 1
    epsilon = 0.3

    oracle_D = _autograd_logit_sensitivity_oracle(env, env.policy_probs, theta0, mu0, T)
    mu_flow = _common.exact_population_flow(env, env.policy_probs, theta0, mu0, T)

    reps, B = 300, 2000
    samples = torch.stack(
        [mfreinforce.gradient_estimate(env, env.policy_probs, theta0, mu_flow, mu0, oracle_D, T, B, epsilon) for _ in range(reps)]
    )
    mean, se = samples.mean(dim=0), samples.std(dim=0) / reps**0.5

    oracle_grad = _pathwise_J_eps_oracle(env, theta0, mu0, T, epsilon, n_mc=200_000)
    assert torch.all((mean - oracle_grad).abs() < 6 * se + 0.01)


def test_gradient_step_end_to_end_matches_pathwise_oracle():
    """Same check as above, but through the full gradient_step (estimated
    D_hat via Algorithm 2, not the oracle), with extra tolerance for its
    additional Monte Carlo noise."""
    torch.manual_seed(3)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 1
    epsilon = 0.3

    reps = 150
    samples = torch.stack(
        [mfreinforce.gradient_step(env, env.policy_probs, theta0, mu0, T=T, n_aux=30, B=500, epsilon=epsilon) for _ in range(reps)]
    )
    mean, se = samples.mean(dim=0), samples.std(dim=0) / reps**0.5

    oracle_grad = _pathwise_J_eps_oracle(env, theta0, mu0, T, epsilon, n_mc=200_000)
    assert torch.all((mean - oracle_grad).abs() < 8 * se + 0.02)


def test_gradient_step_with_particle_flow_runs():
    env = TwoState()
    theta = torch.tensor([0.1, -0.2])
    mu0 = torch.tensor([0.5, 0.5])
    particle_flow_fn = functools.partial(_common.particle_population_flow, n_particles=100)
    g = mfreinforce.gradient_step(env, env.policy_probs, theta, mu0, T=2, n_aux=10, B=50, epsilon=0.2, population_flow_fn=particle_flow_fn)
    assert g.shape == (2,)
    assert torch.isfinite(g).all()


def test_train_improves_training_objective():
    """Integration test: training should increase the exact population
    objective J(theta;mu0) from a poor initialization, evaluated at the
    same initial law used during training — mirroring simplex's analogous
    test (mfreinforce is expected to converge, unlike reinforce, since it
    does include the mean-field correction, just estimated less
    efficiently than simplex's reusable-batch trick)."""
    torch.manual_seed(4)
    env = TwoState()
    T = 2
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([-2.0, -2.0])

    J_before = _common.exact_objective(env, env.policy_probs, theta0, mu0, T)

    theta_final, _ = mfreinforce.train(env, env.policy_probs, theta0, mu0, T=T, n_train=300, n_aux=10, B=200, epsilon=0.2, lr=5e-2)
    J_after = _common.exact_objective(env, env.policy_probs, theta_final, mu0, T)

    assert J_after > J_before
