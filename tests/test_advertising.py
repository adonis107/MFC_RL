from __future__ import annotations

import pytest
import torch

from mfc.environments.advertising import AD, CUSTOMER, NO_AD, NOT_CUSTOMER, Advertising, AdvertisingConfig
from scripts.test import reference_policy_fn, rollout, state_distribution


def test_transition_probs_match_reference_formula():
    env = Advertising()
    kappa = env.config.kappa_ad
    for p1 in (0.1, 0.5, 0.9):
        mu = torch.tensor([1 - p1, p1], dtype=env.dtype)
        for x in (NOT_CUSTOMER, CUSTOMER):
            no_ad = env.transition_probs(torch.tensor(x), torch.tensor(NO_AD), mu)
            ad = env.transition_probs(torch.tensor(x), torch.tensor(AD), mu)
            assert torch.allclose(no_ad[CUSTOMER], torch.tensor(p1, dtype=env.dtype))
            assert torch.allclose(ad[CUSTOMER], torch.tensor(min(p1 + kappa, 1.0), dtype=env.dtype))
            # independent of x: both states see the identical transition law
            assert torch.equal(no_ad, env.transition_probs(torch.tensor(1 - x), torch.tensor(NO_AD), mu))


def test_transition_probs_are_capped_at_one():
    env = Advertising()
    mu = torch.tensor([0.05, 0.95], dtype=env.dtype)
    probs = env.transition_probs(torch.tensor(CUSTOMER), torch.tensor(AD), mu)
    assert torch.allclose(probs.sum(), torch.tensor(1.0, dtype=env.dtype))
    assert probs[CUSTOMER] <= 1.0


def test_reward_matches_reference_formula():
    env = Advertising()
    mu = torch.tensor([0.4, 0.6], dtype=env.dtype)
    c_ad = env.config.c_ad
    assert torch.allclose(env.reward(torch.tensor(NOT_CUSTOMER), torch.tensor(NO_AD), mu), torch.tensor(0.0, dtype=env.dtype))
    assert torch.allclose(env.reward(torch.tensor(CUSTOMER), torch.tensor(NO_AD), mu), torch.tensor(1.0, dtype=env.dtype))
    assert torch.allclose(env.reward(torch.tensor(CUSTOMER), torch.tensor(AD), mu), torch.tensor(1.0 - c_ad, dtype=env.dtype))
    # reward has no direct mu-dependence (only through the transition kernel)
    mu2 = torch.tensor([0.1, 0.9], dtype=env.dtype)
    assert torch.equal(env.reward(torch.tensor(CUSTOMER), torch.tensor(AD), mu), env.reward(torch.tensor(CUSTOMER), torch.tensor(AD), mu2))


def test_terminal_reward_is_always_zero():
    env = Advertising()
    mu = torch.tensor([0.3, 0.7], dtype=env.dtype)
    states = torch.tensor([NOT_CUSTOMER, CUSTOMER])
    assert torch.equal(env.terminal_reward(states, mu), torch.zeros(2, dtype=env.dtype))


def test_policy_probs_is_state_independent_and_valid_distribution():
    env = Advertising()
    generator = torch.Generator(device=env.device).manual_seed(0)
    theta = env.init_theta(generator=generator)
    mu = torch.tensor([0.4, 0.6], dtype=env.dtype)
    for t in range(env.config.T):
        p_not_customer = env.policy_probs(theta, t, torch.tensor(NOT_CUSTOMER), mu)
        p_customer = env.policy_probs(theta, t, torch.tensor(CUSTOMER), mu)
        assert torch.equal(p_not_customer, p_customer)  # state-independent by construction
        assert torch.allclose(p_not_customer.sum(), torch.tensor(1.0, dtype=env.dtype))
        assert (p_not_customer >= 0).all()


def test_init_theta_has_expected_parameter_count():
    env = Advertising()
    theta = env.init_theta()
    H = env.config.hidden_width
    expected = (H * 2 + H) + (H * H + H) + (1 * H + 1)
    assert theta.numel() == expected


def test_init_theta_is_dtype_invariant_at_the_same_seed():
    """Regression test for a real bug: drawing the random init directly at
    `self.dtype` gave float32 and float64 runs at the *same* seed unrelated
    random theta0s (torch.rand consumes the generator's stream differently
    per dtype), silently invalidating any float32-vs-float64 comparison."""
    env64 = Advertising(dtype=torch.float64, device="cpu")
    env32 = Advertising(dtype=torch.float32, device="cpu")
    theta64 = env64.init_theta(generator=torch.Generator(device="cpu").manual_seed(0))
    theta32 = env32.init_theta(generator=torch.Generator(device="cpu").manual_seed(0))
    assert torch.allclose(theta64, theta32.double(), atol=1e-6)


def test_sample_mu0_default_batch_shape_gives_a_single_unbatched_law():
    """`train_run` calls `env.sample_mu0(generator=...)` with no batch_shape
    at all (the default `()`), unlike this file's other tests which always
    pass an explicit batch size — regression test for a real bug where the
    unbatched call crashed (torch.rand(*(), ...) has no size argument)."""
    env = Advertising()
    generator = torch.Generator(device=env.device).manual_seed(0)
    mu0 = env.sample_mu0(generator=generator)
    assert mu0.shape == (2,)
    assert torch.allclose(mu0.sum(), torch.tensor(1.0, dtype=env.dtype))
    assert 0.05 <= mu0[CUSTOMER].item() <= 0.95


def test_sample_mu0_stays_within_the_reference_interval():
    env = Advertising()
    generator = torch.Generator(device=env.device).manual_seed(0)
    mu0 = env.sample_mu0((2000,), generator=generator)
    assert torch.allclose(mu0.sum(dim=-1), torch.ones(2000, dtype=env.dtype))
    p0 = mu0[:, CUSTOMER]
    assert (p0 >= 0.05).all() and (p0 <= 0.95).all()


def test_reference_policy_matches_the_benchmarks_numerical_case():
    """At the reference's own benchmark parameters (kappa_ad=0.2, c_ad=0.15,
    gamma=0.5), c_ad/kappa_ad=0.75 falls in [gamma, gamma/(1-gamma))=[0.5,1),
    giving the reference's stated closed form (eq. advertising-reference-
    policy-numerical): q_hat(p) = 1 for p<0.6, ramps down over [0.6,0.7), is
    1 again on [0.7, 0.85), and 0 for p>=0.85."""
    env = Advertising()
    assert env.config.c_ad / env.config.kappa_ad == pytest.approx(0.75)
    ps = torch.tensor([0.0, 0.3, 0.59, 0.6, 0.65, 0.699, 0.7, 0.8, 0.849, 0.85, 0.95, 1.0], dtype=env.dtype)
    q = env.reference_policy(ps)
    expected = torch.tensor([1, 1, 1, (0.8 - 0.6) / 0.2, (0.8 - 0.65) / 0.2, (0.8 - 0.699) / 0.2, 1, 1, 1, 0, 0, 0], dtype=env.dtype)
    assert torch.allclose(q, expected, atol=1e-9)


def test_reference_policy_constant_zero_when_cost_dominates():
    env = Advertising(AdvertisingConfig(kappa_ad=0.1, c_ad=0.5, gamma=0.5))
    assert env.config.c_ad / env.config.kappa_ad >= env.config.gamma / (1.0 - env.config.gamma)
    ps = torch.tensor([0.0, 0.5, 1.0], dtype=env.dtype)
    assert torch.equal(env.reference_policy(ps), torch.zeros(3, dtype=env.dtype))


def test_reference_policy_threshold_form_when_switching_cheap():
    env = Advertising(AdvertisingConfig(kappa_ad=0.5, c_ad=0.05, gamma=0.5))
    assert env.config.c_ad / env.config.kappa_ad < env.config.gamma
    p_bar = 1.0 - env.config.c_ad * (1.0 - env.config.gamma) / env.config.gamma
    below = env.reference_policy(torch.tensor(p_bar - 0.05, dtype=env.dtype))
    above = env.reference_policy(torch.tensor(p_bar + 0.05, dtype=env.dtype))
    assert below.item() == 1.0
    assert above.item() == 0.0


def test_reference_policy_fn_reproduces_the_closed_form_through_the_generic_machinery():
    env = Advertising()
    action_probs_fn = reference_policy_fn(env)
    theta = env.init_theta()  # ignored by the wrapper
    for p1 in (0.1, 0.4, 0.6, 0.65, 0.75, 0.9):
        mu = torch.tensor([1 - p1, p1], dtype=env.dtype)
        probs = action_probs_fn(theta, 0, torch.tensor(NOT_CUSTOMER), mu)
        assert torch.allclose(probs.sum(), torch.tensor(1.0, dtype=env.dtype))
        assert torch.allclose(probs[AD], env.reference_policy(torch.tensor(p1, dtype=env.dtype)))
        # state-independent, as advertising's own policy_probs is
        assert torch.equal(probs, action_probs_fn(theta, 0, torch.tensor(CUSTOMER), mu))


def test_reference_policy_fn_drives_a_valid_rollout_and_flow():
    env = Advertising()
    action_probs_fn = reference_policy_fn(env)
    theta = env.init_theta()
    mu0 = torch.tensor([0.5, 0.5], dtype=env.dtype)
    T = env.config.T
    flow = state_distribution(env, action_probs_fn, theta, mu0, T)
    assert flow.shape == (T + 1, env.n_states)
    assert torch.allclose(flow.sum(dim=-1), torch.ones(T + 1, dtype=env.dtype))
    traj = rollout(env, action_probs_fn, theta, mu0, T, generator=torch.Generator(device=env.device).manual_seed(0))
    assert traj.shape == (T + 1,)
