from __future__ import annotations

import pytest
import torch

from mfc.algorithms.portfolio import reinforce_step, train
from mfc.environments.portfolio import Portfolio, PortfolioConfig


def test_forward_moments_match_hand_computed_first_step():
    env = Portfolio(PortfolioConfig(s=1.0, r_bar=0.02, sigma_R=0.08, tau=0.02, mu0=1.0, Sigma0=0.04, T=1))
    theta = torch.tensor([[0.1, 0.05]], dtype=env.dtype)
    mu, Sigma = env.forward_moments(theta, lam=0.0)
    assert torch.allclose(mu[0], torch.tensor(1.0, dtype=env.dtype))
    assert torch.allclose(mu[1], torch.tensor(1.0 * 1.0 + 0.02 * 0.05, dtype=env.dtype))
    h = 0.02**2 + 0.08**2
    A0 = 1.0**2 + 2 * 1.0 * 0.02 * 0.1 + h * 0.1**2
    expected_Sigma1 = A0 * 0.04 + h * 0.02**2 + 0.08**2 * 0.05**2
    assert torch.allclose(Sigma[1], torch.tensor(expected_Sigma1, dtype=env.dtype))


def test_exact_gradient_matches_autograd_through_the_closed_form_objective():
    env = Portfolio()
    T = env.config.T
    generator = torch.Generator(device=env.device).manual_seed(0)
    for lam in (0.0, 0.1, 0.4):
        theta = 0.2 * torch.randn(T, 2, dtype=env.dtype, device=env.device, generator=generator)
        theta_attached = theta.clone().requires_grad_(True)
        J = env.exact_objective(theta_attached, lam)
        (g_autograd,) = torch.autograd.grad(J, theta_attached)
        g_exact = env.exact_gradient(theta, lam)
        assert torch.allclose(g_autograd, g_exact, atol=1e-8, rtol=1e-6)


def test_perturbation_bias_scales_exactly_as_lambda_squared():
    """J^lambda(theta*) - J^0(theta*) should scale as lambda^2 exactly
    (reference: "all perturbation terms in the variance recursion and
    terminal objective are proportional to lambda^2")."""
    env = Portfolio()
    theta_star = env.optimal_theta()
    J0 = env.exact_objective(theta_star, 0.0).item()
    diffs = {lam: env.exact_objective(theta_star, lam).item() - J0 for lam in (0.025, 0.05, 0.1, 0.2, 0.4)}
    for lam in (0.05, 0.1, 0.2, 0.4):
        ratio = diffs[lam] / diffs[lam / 2]
        assert ratio == pytest.approx(4.0, rel=1e-6)


def test_optimal_theta_is_a_stationary_point_and_beats_zero_theta():
    env = Portfolio()
    T = env.config.T
    theta_star = env.optimal_theta()
    g = env.exact_gradient(theta_star, 0.0)
    assert g.abs().max().item() < 1e-8

    J_star = env.exact_objective(theta_star, 0.0).item()
    J_zero = env.exact_objective(torch.zeros(T, 2, dtype=env.dtype, device=env.device), 0.0).item()
    assert J_star > J_zero


def test_optimal_theta_k_is_time_homogeneous_for_the_default_config():
    """k_t^* = -s*r_bar/h is constant across t when s/r_bar/sigma_R don't
    vary with t (the reference's own baseline config)."""
    env = Portfolio()
    theta_star = env.optimal_theta()
    assert torch.allclose(theta_star[:, 0], theta_star[0, 0].expand_as(theta_star[:, 0]))
    # l_t^* strictly decreases moving away from the terminal time (geometric decay)
    assert (theta_star[:-1, 1] > theta_star[1:, 1]).all()


def test_rollout_shapes_and_terminal_time_perturbation():
    env = Portfolio()
    T = env.config.T
    theta = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    out = env.rollout(theta, lam=0.1, B=64, generator=generator)
    assert out["X"].shape == (T + 1, 64)
    assert out["alpha"].shape == (T, 64)
    assert out["mu_hat"].shape == (T + 1,)
    for t in (out["X"], out["alpha"], out["mu_hat"]):
        assert not t.requires_grad  # theta is detached throughout rollout


def test_sample_returns_gaussian_matches_configured_moments():
    env = Portfolio(PortfolioConfig(return_distribution="gaussian"))
    generator = torch.Generator(device=env.device).manual_seed(0)
    R = env.sample_returns(200_000, generator=generator)
    assert R.mean().item() == pytest.approx(env.config.r_bar, abs=5e-4)
    assert R.std().item() == pytest.approx(env.config.sigma_R, rel=5e-2)


def test_sample_returns_student_t_matches_configured_moments_but_is_heavier_tailed():
    env = Portfolio(PortfolioConfig(return_distribution="student_t"))
    generator = torch.Generator(device=env.device).manual_seed(0)
    R = env.sample_returns(200_000, generator=generator)
    assert R.mean().item() == pytest.approx(env.config.r_bar, abs=5e-4)
    assert R.std().item() == pytest.approx(env.config.sigma_R, rel=5e-2)
    # excess kurtosis of a Gaussian is 0; Student-t(5) has excess kurtosis 6 -- heavier tails
    z = (R - R.mean()) / R.std()
    kurtosis = (z**4).mean().item() - 3.0
    assert kurtosis > 2.0


def test_reinforce_step_gradient_is_nonzero_and_finite():
    env = Portfolio()
    theta = env.init_theta().clone().requires_grad_(True)
    generator = torch.Generator(device=env.device).manual_seed(0)
    g = reinforce_step(env, theta, lam=0.1, B=128, generator=generator)
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0.0


def test_train_exact_gradient_converges_close_to_the_optimum():
    env = Portfolio()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm="exact_gradient", n_train=1000, lr=0.05, lam=0.0, validate_every=100, generator=generator)
    J_star = env.exact_objective(env.optimal_theta(), 0.0).item()
    assert result["validation_J"][-1].item() > 0.99 * J_star


def test_train_reinforce_runs_and_returns_expected_fields():
    env = Portfolio()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm="reinforce", n_train=20, lr=0.01, lam=0.1, B=32, validate_every=5, generator=generator)
    T = env.config.T
    assert result["theta_history"].shape == (21, T, 2)
    assert result["validation_iterations"].numel() == result["validation_J"].numel() > 0
    assert result["elapsed_seconds"] > 0.0
