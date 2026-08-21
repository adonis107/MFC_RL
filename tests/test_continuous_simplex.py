"""
Correctness of the continuous-state simplex-perturbed MF-REINFORCE
estimator, against the closed forms both continuous benchmarks provide.

These are the checks the discrete side can only approximate: here
`env.exact_gradient(theta, lam)` is grad J^lambda in closed form, so the
estimator's unbiasedness can be tested directly rather than inferred.
"""

from __future__ import annotations

import pytest
import torch

from mfc.algorithms import continuous_reinforce, continuous_simplex
from mfc.environments.lq import LQ
from mfc.environments.portfolio import Portfolio

from scripts.test import exact_coordinate_sensitivity


def _theta(env, T, seed=0, scale=0.2):
    generator = torch.Generator(device=env.device).manual_seed(seed)
    return scale * torch.randn(T, 2, dtype=env.dtype, device=env.device, generator=generator)


def test_perturbation_score_is_centered_and_scales_as_one_over_lambda():
    """E[h_t]=0 (what the causality and baseline arguments rest on), and its
    dominant term is the reference's lambda^{-1} transport score."""
    # explicit CPU device: tests/conftest.py sets a CUDA default device where available,
    # which torch.Generator() does not follow (see its own comment)
    xi = torch.randn(200_000, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0))
    mu = torch.tensor(2.0, device="cpu")
    for lam in (0.05, 0.2):
        h = continuous_simplex.perturbation_score(xi, mu, lam, rho=0.3)
        assert abs(h.mean().item()) < 0.05 * h.std().item()
    h_small = continuous_simplex.perturbation_score(xi, mu, 0.05, rho=0.3)
    h_large = continuous_simplex.perturbation_score(xi, mu, 0.2, rho=0.3)
    assert h_small.std().item() == pytest.approx(4.0 * h_large.std().item(), rel=0.05)


@pytest.mark.parametrize("env_cls,T,lam", [(LQ, 2, 0.4), (Portfolio, 2, 0.2)])
def test_oracle_sensitivity_estimator_is_unbiased_for_the_perturbed_gradient(env_cls, T, lam):
    """Fed the exact coordinate sensitivities, the estimator's mean is
    grad_theta J^lambda(theta) exactly (Theorem "Perturbed policy-gradient
    formula"). Checked as a z-score against the Monte Carlo standard error of
    the reps-average, so this tests unbiasedness rather than a tolerance."""
    env = env_cls(device="cpu")
    theta = _theta(env, T)
    D_exact = exact_coordinate_sensitivity(env, theta)
    target = env.exact_gradient(theta, lam)

    generator = torch.Generator(device="cpu").manual_seed(0)
    reps, B = 30, 20_000
    samples = torch.stack(
        [
            continuous_simplex.gradient_estimate(env, theta, env.rollout(theta, lam, B, generator=generator), D_exact, lam, baseline="loo")
            for _ in range(reps)
        ]
    )
    z = (samples.mean(dim=0) - target) / (samples.std(dim=0) / reps**0.5)
    assert z.abs().max().item() < 4.0


def test_sensitivity_flow_converges_to_the_exact_coordinate_sensitivities():
    """D_hat_t -> D_t^theta = grad_theta mu_t^theta as the auxiliary batch
    grows (Algorithm "Single-batch forward estimation of the coordinate
    sensitivities"), with D_0 = 0 exactly since mu_0 does not depend on
    theta."""
    env = LQ(device="cpu")
    T, lam = 3, 0.2
    theta = _theta(env, T)
    exact = exact_coordinate_sensitivity(env, theta)
    assert exact[0].abs().max().item() == 0.0

    generator = torch.Generator(device="cpu").manual_seed(0)
    errors = {}
    for n in (200, 20_000):
        mean = torch.stack([continuous_simplex.estimate_sensitivity_flow(env, theta, n, lam, generator=generator) for _ in range(8)]).mean(dim=0)
        assert mean[0].abs().max().item() == 0.0
        errors[n] = (mean - exact).norm().item() / exact.norm().item()
    assert errors[20_000] < 0.5 * errors[200]
    assert errors[20_000] < 0.1


def test_plug_in_estimator_beats_reinforce_at_the_known_optimum():
    """At theta*, grad J^0 = 0 exactly. The simplex estimate must be
    consistent with zero within Monte Carlo error, while REINFORCE — which
    drops the population-perturbation score — must not be: this is the
    "missing mean-field term" of context.md, measured rather than argued."""
    env = LQ(device="cpu")
    T, lam, reps = 5, 0.2, 40
    theta_star = env.riccati_optimal(T)
    assert env.exact_gradient(theta_star, 0.0).abs().max().item() < 1e-8

    generator = torch.Generator(device="cpu").manual_seed(0)
    simplex_samples = torch.stack(
        [continuous_simplex.gradient_step(env, theta_star, lam, B=200, n_aux=100, baseline="loo", generator=generator) for _ in range(reps)]
    )
    reinforce_samples = torch.stack(
        [continuous_reinforce.gradient_step(env, theta_star, lam, B=300, baseline="loo", generator=generator) for _ in range(reps)]
    )

    simplex_bias = simplex_samples.mean(dim=0).norm().item()
    reinforce_bias = reinforce_samples.mean(dim=0).norm().item()
    simplex_se = (simplex_samples.std(dim=0) / reps**0.5).norm().item()
    reinforce_se = (reinforce_samples.std(dim=0) / reps**0.5).norm().item()

    assert simplex_bias < 2.0 * simplex_se  # unbiased for grad J^0 up to the O(lambda^2) perturbation term
    assert reinforce_bias > 2.0 * reinforce_se  # structurally biased
    assert reinforce_bias > 2.0 * simplex_bias


def test_reinforce_is_the_simplex_estimator_with_the_sensitivity_flow_zeroed():
    """The ablation is one term, not a different algorithm: on the very same
    rollout, simplex with D_hat=0 reproduces the REINFORCE estimate exactly."""
    env = Portfolio(device="cpu")
    T, lam = 3, 0.1
    theta = _theta(env, T)
    out = env.rollout(theta, lam, 64, generator=torch.Generator(device="cpu").manual_seed(0))
    zero_D = torch.zeros(T + 1, T, 2, dtype=env.dtype, device=env.device)
    assert torch.allclose(
        continuous_simplex.gradient_estimate(env, theta, out, zero_D, lam, baseline="loo"),
        continuous_reinforce.gradient_estimate(env, theta, out, baseline="loo"),
    )


def test_leave_one_out_baseline_leaves_the_estimate_unbiased_but_cuts_variance():
    env = LQ(device="cpu")
    T, lam, reps, B = 2, 0.4, 30, 5_000
    theta = _theta(env, T)
    D_exact = exact_coordinate_sensitivity(env, theta)
    target = env.exact_gradient(theta, lam)

    stats = {}
    for baseline in (None, "loo"):
        generator = torch.Generator(device="cpu").manual_seed(0)
        samples = torch.stack(
            [
                continuous_simplex.gradient_estimate(env, theta, env.rollout(theta, lam, B, generator=generator), D_exact, lam, baseline=baseline)
                for _ in range(reps)
            ]
        )
        stats[baseline] = (samples.mean(dim=0), samples.std(dim=0))

    for baseline, (mean, std) in stats.items():
        assert ((mean - target).abs() < 4.0 * std / reps**0.5).all(), baseline
    assert stats["loo"][1].norm().item() < 0.5 * stats[None][1].norm().item()


def test_estimate_objective_matches_the_closed_form_perturbed_objective():
    env = LQ(device="cpu")
    T, lam = 3, 0.4
    theta = _theta(env, T)
    samples = continuous_simplex.estimate_objective(env, theta, lam, 200_000, generator=torch.Generator(device="cpu").manual_seed(0))
    se = (samples.std() / 200_000**0.5).item()
    assert abs(samples.mean().item() - env.exact_objective(theta, lam).item()) < 4.0 * se
