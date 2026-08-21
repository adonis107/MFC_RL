from __future__ import annotations

import pytest
import torch

from mfc.algorithms.continuous import train
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


def test_rollout_shapes_and_per_replica_perturbation():
    env = LQ()
    T = env.config.T
    theta = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    out = env.rollout(theta, lam=0.3, B=64, generator=generator)
    assert out["X"].shape == (T + 1, 64)
    assert out["alpha"].shape == (T, 64)
    assert out["M"].shape == (T + 1, 64)  # one perturbation draw per replica, not one shared across the batch
    assert out["xi"].shape == (T + 1, 64)
    assert out["mu"].shape == (T + 1,)
    assert out["running"].shape == (T, 64)
    assert out["terminal"].shape == (64,)
    for t in out.values():
        assert not t.requires_grad  # theta is detached throughout rollout
    assert (out["M"][:, 0] != out["M"][:, 1]).all()


def test_rollout_standardized_perturbation_reproduces_the_population_argument():
    """xi is exactly the standardization of M around the nominal coordinate:
    M_t = mu_t + lambda*rho*sqrt(mu_t^2+1)*xi_t, which is the form
    `mfc.algorithms.continuous_simplex.perturbation_score` is written in."""
    env = LQ()
    theta = 0.2 * torch.randn(env.config.T, 2, dtype=env.dtype, device=env.device)
    lam, rho = 0.3, env.config.rho
    out = env.rollout(theta, lam=lam, B=256, generator=torch.Generator(device=env.device).manual_seed(0))
    mu = out["mu"].unsqueeze(-1)
    assert torch.allclose(out["M"], mu + lam * rho * torch.sqrt(mu**2 + 1.0) * out["xi"])
    assert out["xi"].mean().abs().item() < 0.2  # standard normal
    assert abs(out["xi"].std().item() - 1.0) < 0.2


def test_train_exact_gradient_converges_close_to_the_riccati_optimum():
    env = LQ()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm="exact_gradient", n_train=1500, lr=0.1, lam=0.2, validate_every=100, generator=generator)
    J_star = env.exact_objective(env.riccati_optimal(), 0.0).item()
    assert result["validation_J"][-1].item() < 1.05 * J_star


@pytest.mark.parametrize("algorithm,kwargs", [("reinforce", {"B": 32}), ("simplex", {"B": 32, "n_aux": 16})])
def test_train_runs_and_returns_expected_fields(algorithm, kwargs):
    env = LQ()
    theta0 = env.init_theta()
    generator = torch.Generator(device=env.device).manual_seed(0)
    result = train(env, theta0, algorithm=algorithm, n_train=20, lr=0.05, lam=0.2, validate_every=5, generator=generator, **kwargs)
    T = env.config.T
    assert result["theta_history"].shape == (21, T, 2)
    assert result["validation_iterations"].numel() == result["validation_J"].numel() > 0
    assert result["elapsed_seconds"] > 0.0
