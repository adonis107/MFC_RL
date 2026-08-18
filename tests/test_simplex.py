from __future__ import annotations

import functools

import numpy as np
import torch

from mfc.algorithms import simplex as sx
from mfc.environments.twostate import TwoState

torch.set_default_dtype(torch.float64)


def oracle_population_flow(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """Same recursion as `sx.exact_population_flow`, but keeps theta attached
    for autograd, to serve as an independent ground truth for D_t^theta(k)."""
    N, A = env.n_states, env.n_actions
    states_grid = torch.arange(N).unsqueeze(-1).expand(N, A)
    actions_grid = torch.arange(A).unsqueeze(0).expand(N, A)
    states_all = torch.arange(N)
    mu_flow = [mu0]
    for t in range(T):
        mu_t = mu_flow[-1]
        pi = sx.eval_batched(action_probs_fn, theta, t, states_all, mu_t)
        P = env.transition_probs(states_grid, actions_grid, mu_t)
        K = torch.einsum("xa,xay->xy", pi, P)
        mu_flow.append(mu_t @ K)
    return torch.stack(mu_flow)


def test_sample_perturbation_lands_on_simplex_interior():
    U, q = sx.sample_perturbation(4, (1000,), sigma=1.0, dtype=torch.float64, device=torch.get_default_device())
    assert U.shape == (1000, 3)
    assert q.shape == (1000, 4)
    assert torch.all(q > 0)
    assert torch.allclose(q.sum(dim=-1), torch.ones(1000, dtype=torch.float64))


def test_perturbation_score_satisfies_integration_by_parts_identity():
    """
    E_q[Q_t(D)*F(q)] should equal d/dmu . the underlying linear map, i.e. for
    F(q)=M(k)=(1-lam)*mu(k)+lam*q(k), E_q[Q_t(D)*F(q)] = (1-lam)*(-D(0))*e_k
    summed appropriately; here checked directly against the closed form
    (1-lam)*grad_theta M(1) for a 2-state simplex (eq. discrete-plugin score,
    "Simplex score identity").
    """
    torch.manual_seed(0)
    D = torch.tensor([[0.37, -0.52]])
    mu_t = torch.tensor([0.6, 0.4])
    lam, sigma = 0.3, 1.0

    n = 500_000
    U, q = sx.sample_perturbation(2, (n,), sigma, dtype=torch.float64, device=torch.get_default_device())
    H = sx.perturbation_score(U, q, sigma)
    Q = -((1.0 - lam) / lam) * (H @ D)
    M1 = (1.0 - lam) * mu_t[1] + lam * q[:, 1]
    samples = Q * M1.unsqueeze(-1)
    estimate, se = samples.mean(dim=0), samples.std(dim=0) / n**0.5
    expected = (1.0 - lam) * (-D[0])
    # SE-based, not a magic atol: CPU and CUDA use different RNG streams for
    # the same seed, so the exact Monte Carlo draws (and noise) differ.
    assert torch.all((estimate - expected).abs() < 6 * se)


def test_sensitivity_flow_matches_autograd_oracle():
    """
    The single-batch forward estimator of D_t^theta(k) should, averaged over
    many auxiliary batches, converge to the exact grad_theta mu_t^theta(k)
    obtained by ordinary autograd through the (differentiable) exact
    population recursion.
    """
    torch.manual_seed(1)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 2

    oracle_D = []
    for t in range(T + 1):
        th = theta0.clone().requires_grad_(True)
        mu_flow = oracle_population_flow(env, env.policy_probs, th, mu0, T)
        oracle_D.append(torch.autograd.grad(mu_flow[t, 0], th)[0])
    oracle_D = torch.stack(oracle_D)

    mu_flow = sx.exact_population_flow(env, env.policy_probs, theta0, mu0, T)
    reps, n_aux = 300, 20
    acc = torch.zeros(T + 1, 1, 2)
    for _ in range(reps):
        acc += sx.estimate_sensitivity_flow(env, env.policy_probs, theta0, mu_flow, mu0, T, n=n_aux, eta=0.2, sigma=1.0)
    mc_D = (acc / reps).squeeze(1)

    assert torch.allclose(mc_D, oracle_D, atol=0.03)


def _quadrature_gradient_oracle_T1(env, action_probs_fn, theta0, mu0, lam, nq=300, eps=1e-3):
    """
    Exact grad_theta J^lambda(theta) at T=1, via Gauss-Hermite quadrature
    over the two perturbations (q_0, q_1) plus finite differences (the
    reward's |.| term makes autograd through the quadrature itself unstable
    at the kink, even though the quadrature value converges cleanly).
    """
    nodes, weights = np.polynomial.hermite.hermgauss(nq)
    nodes = torch.tensor(nodes)
    weights = torch.tensor(weights) / np.sqrt(np.pi)

    def q_of_u(u):
        p0 = torch.sigmoid(u)
        return torch.stack([p0, 1 - p0])

    @torch.no_grad()
    def J_lambda(theta):
        mu_flow = oracle_population_flow(env, action_probs_fn, theta, mu0, 1)
        us = np.sqrt(2) * nodes
        e_term = torch.zeros(2)
        for x1 in range(2):
            vals = torch.stack([env.terminal_reward(torch.tensor(x1), (1 - lam) * mu_flow[1] + lam * q_of_u(u)) for u in us])
            e_term[x1] = (weights * vals).sum()
        total = torch.tensor(0.0)
        for x0 in range(2):
            for i, u0 in enumerate(us):
                M0 = (1 - lam) * mu_flow[0] + lam * q_of_u(u0)
                pi0 = action_probs_fn(theta, 0, torch.tensor(x0), M0)
                for a0 in range(2):
                    r0 = env.reward(torch.tensor(x0), torch.tensor(a0), M0)
                    trans0 = env.transition_probs(torch.tensor(x0), torch.tensor(a0), M0)
                    for x1 in range(2):
                        total = total + mu0[x0] * pi0[a0] * trans0[x1] * (r0 + e_term[x1]) * weights[i]
        return total

    grad = torch.zeros(2)
    for i in range(2):
        tp, tm = theta0.clone(), theta0.clone()
        tp[i] += eps
        tm[i] -= eps
        grad[i] = (J_lambda(tp) - J_lambda(tm)) / (2 * eps)
    return grad


def test_gradient_estimate_matches_finite_difference_oracle():
    """
    Averaged over many independent (auxiliary batch, main batch) draws, the
    plug-in simplex gradient estimator should recover grad_theta J^lambda(theta)
    within a few Monte Carlo standard errors, at T=1 where an exact oracle
    is tractable by quadrature.
    """
    torch.manual_seed(2)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T, lam = 1, 0.4

    oracle_grad = _quadrature_gradient_oracle_T1(env, env.policy_probs, theta0, mu0, lam)

    mu_flow = sx.exact_population_flow(env, env.policy_probs, theta0, mu0, T)
    reps, B, n_aux = 150, 400, 40
    samples = torch.zeros(reps, 2)
    for i in range(reps):
        D_hat = sx.estimate_sensitivity_flow(env, env.policy_probs, theta0, mu_flow, mu0, T, n=n_aux, eta=lam, sigma=1.0)
        samples[i] = sx.gradient_estimate(env, env.policy_probs, theta0, mu_flow, mu0, D_hat, T, B=B, lam=lam, sigma=1.0)

    mean = samples.mean(dim=0)
    se = samples.std(dim=0) / reps**0.5
    assert torch.all((mean - oracle_grad).abs() < 6 * se + 0.03)


def test_particle_population_flow_rows_are_valid_probability_vectors():
    torch.manual_seed(4)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    flow = sx.particle_population_flow(env, env.policy_probs, theta0, mu0, T=2, n_particles=50)
    assert flow.shape == (3, 2)
    assert torch.all(flow >= 0)
    assert torch.allclose(flow.sum(dim=-1), torch.ones(3))


def test_particle_population_flow_matches_exact_flow_on_average():
    """
    The empirical nominal-flow estimator (Assumption "Access to the nominal
    population flow": mu_hat_t(i)=(1/Ntilde) sum_r 1{X_t^(r)=i}), averaged
    over many independent particle batches, should converge to the exact
    flow it approximates.
    """
    torch.manual_seed(5)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T = 2

    exact = sx.exact_population_flow(env, env.policy_probs, theta0, mu0, T)

    reps, n_particles = 500, 300
    acc = torch.zeros_like(exact)
    for _ in range(reps):
        acc += sx.particle_population_flow(env, env.policy_probs, theta0, mu0, T, n_particles=n_particles)
    mean_particle = acc / reps

    assert torch.allclose(mean_particle, exact, atol=5e-3)


def test_gradient_step_with_particle_flow_matches_exact_flow_gradient_on_average():
    """
    Swapping `population_flow_fn` for the particle-flow estimator only
    changes how the nominal flow is obtained; averaged over enough particle
    batches, the resulting gradient estimate should agree with the
    exact-flow estimator's, within the extra Monte Carlo noise it
    introduces.
    """
    torch.manual_seed(6)
    env = TwoState()
    theta0 = torch.tensor([0.3, -0.4])
    mu0 = torch.tensor([0.8, 0.2])
    T, lam = 1, 0.4

    reps = 150
    exact_samples = torch.zeros(reps, 2)
    particle_samples = torch.zeros(reps, 2)
    particle_flow_fn = functools.partial(sx.particle_population_flow, n_particles=300)
    for i in range(reps):
        exact_samples[i] = sx.gradient_step(env, env.policy_probs, theta0, mu0, T=T, n_aux=40, B=400, lam=lam)
        particle_samples[i] = sx.gradient_step(
            env, env.policy_probs, theta0, mu0, T=T, n_aux=40, B=400, lam=lam, population_flow_fn=particle_flow_fn
        )

    diff = particle_samples.mean(dim=0) - exact_samples.mean(dim=0)
    combined_se = (exact_samples.std(dim=0) ** 2 / reps + particle_samples.std(dim=0) ** 2 / reps).sqrt()
    assert torch.all(diff.abs() < 6 * combined_se + 0.03)


def test_train_with_particle_flow_improves_training_objective():
    """Same integration check as `test_train_improves_training_objective`,
    but sourcing the nominal flow from `particle_population_flow`."""
    torch.manual_seed(7)
    env = TwoState()
    T = 2
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([-2.0, -2.0])

    def exact_J(theta, mu0):
        mu_flow = sx.exact_population_flow(env, env.policy_probs, theta, mu0, T)
        m1 = mu_flow[:, 1]
        return (m1 - m1**2 - env.config.kappa * (m1 - (1.0 - env.config.p)).abs()).sum()

    J_before = exact_J(theta0, mu0)

    theta_final, _ = sx.train(
        env,
        env.policy_probs,
        theta0,
        mu0,
        T=T,
        n_train=300,
        n_aux=10,
        B=100,
        lam=0.2,
        sigma=1.0,
        lr=5e-2,
        population_flow_fn=functools.partial(sx.particle_population_flow, n_particles=200),
    )
    J_after = exact_J(theta_final, mu0)

    assert J_after > J_before


def test_train_improves_training_objective():
    """Integration test: training should increase the exact population
    objective J(theta;mu0) from a poor initialization, evaluated at the same
    initial law used during training (convergence to the mu0_val-optimal
    stationary policy is a matter of training budget, not checked here)."""
    torch.manual_seed(3)
    env = TwoState()
    T = 2
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([-2.0, -2.0])  # far from optimal (rarely moves)

    def exact_J(theta, mu0):
        mu_flow = sx.exact_population_flow(env, env.policy_probs, theta, mu0, T)
        m1 = mu_flow[:, 1]
        return (m1 - m1**2 - env.config.kappa * (m1 - (1.0 - env.config.p)).abs()).sum()

    J_before = exact_J(theta0, mu0)

    theta_final, _ = sx.train(
        env,
        env.policy_probs,
        theta0,
        mu0,
        T=T,
        n_train=300,
        n_aux=10,
        B=100,
        lam=0.2,
        sigma=1.0,
        lr=5e-2,
    )
    J_after = exact_J(theta_final, mu0)

    assert J_after > J_before
