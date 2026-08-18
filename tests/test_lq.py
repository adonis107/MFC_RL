from __future__ import annotations

import torch

from mfc.algorithms.lq import reinforce_step, train
from mfc.environments.lq import LQ, LQConfig


def test_forward_moments_match_hand_computed_first_step():
    env = LQ(LQConfig(a=0.5, b=1.0, c=0.0, sigma=0.2, tau=0.1, mu0=1.0, Sigma0=0.5, T=1))
    theta = torch.tensor([[0.2, 0.0]], dtype=env.dtype)
    mu, Sigma = env.forward_moments(theta, lam=0.0)
    assert torch.allclose(mu[0], torch.tensor(1.0, dtype=env.dtype))
    assert torch.allclose(mu[1], torch.tensor((0.5 + 0.2) * 1.0, dtype=env.dtype))
    assert torch.allclose(Sigma[1], torch.tensor((0.5 + 0.2) ** 2 * 0.5 + 0.1**2 + 0.2**2, dtype=env.dtype))


def test_exact_gradient_matches_autograd_through_the_closed_form_objective():
    env = LQ()
    T = env.config.T
    generator = torch.Generator(device=env.device).manual_seed(0)
    for lam in (0.0, 0.2, 0.8):
        theta = 0.3 * torch.randn(T, 2, dtype=env.dtype, device=env.device, generator=generator)
        theta_attached = theta.clone().requires_grad_(True)
        J = env.exact_objective(theta_attached, lam)
        (g_autograd,) = torch.autograd.grad(J, theta_attached)
        g_exact = env.exact_gradient(theta, lam)
        assert torch.allclose(g_autograd, g_exact, atol=1e-8, rtol=1e-6)


def test_objective_bias_identity_holds_exactly():
    """J^lambda(theta) - J^0(theta) == lambda^2*rho^2*B(theta) exactly
    (LQ_framework.tex, eq. objective-pointwise-rate)."""
    env = LQ()
    T = env.config.T
    generator = torch.Generator(device=env.device).manual_seed(1)
    theta = 0.3 * torch.randn(T, 2, dtype=env.dtype, device=env.device, generator=generator)
    B = env.objective_bias(theta)
    for lam in (0.1, 0.3, 0.7, 1.5):
        J_lam = env.exact_objective(theta, lam)
        J_0 = env.exact_objective(theta, 0.0)
        assert torch.allclose(J_lam - J_0, (lam * env.config.rho) ** 2 * B, atol=1e-8, rtol=1e-6)


def test_riccati_optimal_beats_zero_theta_and_a_no_coupling_baseline():
    """The Riccati-optimal theta* must (a) achieve strictly lower cost than
    doing nothing, and (b) achieve strictly lower cost, under this
    environment's own coupled objective, than the theta* of a *decoupled*
    (c=0, kappa=kappa_T=0) LQ problem — i.e. the mean-field coupling term is
    strong enough that ignoring it in the design of the optimal policy is
    clearly suboptimal. This is the numeric check for LQConfig's docstring
    claim that the default coupling is "strong enough to matter"."""
    env = LQ()
    T = env.config.T
    theta_star = env.riccati_optimal()
    J_star = env.exact_objective(theta_star, 0.0).item()

    J_zero = env.exact_objective(torch.zeros(T, 2, dtype=env.dtype, device=env.device), 0.0).item()
    assert J_star < J_zero

    env_decoupled = LQ(LQConfig(c=0.0, kappa=0.0, kappa_T=0.0))
    theta_star_decoupled = env_decoupled.riccati_optimal()
    J_decoupled_under_coupling = env.exact_objective(theta_star_decoupled, 0.0).item()
    assert J_star < 0.5 * J_decoupled_under_coupling  # more than 2x worse when the coupling is ignored


def test_riccati_optimal_is_a_stationary_point_of_the_exact_gradient():
    env = LQ()
    theta_star = env.riccati_optimal()
    g = env.exact_gradient(theta_star, 0.0)
    assert g.abs().max().item() < 1e-8


def test_rollout_shapes_and_terminal_time_perturbation():
    env = LQ()
    T = env.config.T
    theta = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    out = env.rollout(theta, lam=0.3, B=64, generator=generator)
    assert out["X"].shape == (T + 1, 64)
    assert out["alpha"].shape == (T, 64)
    assert out["mu_hat"].shape == (T + 1,)
    assert out["running_cost"].shape == (T, 64)
    assert out["terminal_cost"].shape == (64,)
    for t in (out["X"], out["alpha"], out["mu_hat"], out["running_cost"], out["terminal_cost"]):
        assert not t.requires_grad  # theta is detached throughout rollout


def test_reinforce_step_gradient_is_nonzero_and_finite():
    """Regression test for a real bug: sampling `alpha` with the same
    (attached) theta later used to score it made alpha-means cancel to a
    theta-independent constant, silently zeroing the whole estimator."""
    env = LQ()
    theta = env.init_theta().clone().requires_grad_(True)
    generator = torch.Generator(device=env.device).manual_seed(0)
    g = reinforce_step(env, theta, lam=0.2, B=128, generator=generator)
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0.0


def test_train_exact_gradient_converges_close_to_the_riccati_optimum():
    env = LQ()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm="exact_gradient", n_train=1500, lr=0.1, lam=0.2, validate_every=100, generator=generator)
    J_star = env.exact_objective(env.riccati_optimal(), 0.0).item()
    assert result["validation_J"][-1].item() < 1.05 * J_star


def test_train_reinforce_runs_and_returns_expected_fields():
    env = LQ()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm="reinforce", n_train=20, lr=0.05, lam=0.2, B=32, validate_every=5, generator=generator)
    T = env.config.T
    assert result["theta_history"].shape == (21, T, 2)
    assert result["validation_iterations"].numel() == result["validation_J"].numel() > 0
    assert result["elapsed_seconds"] > 0.0
