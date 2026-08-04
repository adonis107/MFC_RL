import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mfc.environments import LQConfig, LinearQuadraticMFC


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def test_lq_riccati_policy_is_finite_and_improves_zero_policy():
    env = LinearQuadraticMFC(LQConfig(device=DEVICE, dtype=DTYPE))
    theta_zero = env.zero_policy()
    theta_riccati = env.riccati_policy()

    assert theta_riccati.shape == (env.config.T, 2)
    assert torch.isfinite(theta_riccati).all()
    assert env.exact_cost(theta_riccati) < env.exact_cost(theta_zero)


def test_lq_exact_gradient_matches_finite_difference():
    env = LinearQuadraticMFC(LQConfig(device=DEVICE, dtype=DTYPE, T=5))
    theta = torch.linspace(-0.20, 0.15, 10, dtype=DTYPE).reshape(5, 2)
    _, grad = env.exact_gradient(theta)

    eps = 1e-6
    finite_difference = torch.empty_like(theta)
    for index in range(theta.numel()):
        step = torch.zeros_like(theta).reshape(-1)
        step[index] = eps
        step = step.reshape_as(theta)
        finite_difference.reshape(-1)[index] = (
            env.exact_cost(theta + step) - env.exact_cost(theta - step)
        ) / (2.0 * eps)

    assert torch.allclose(grad, finite_difference, atol=1e-7, rtol=1e-6)


def test_lq_exact_moments_shapes_are_finite():
    env = LinearQuadraticMFC(LQConfig(device=DEVICE, dtype=DTYPE, T=6))
    theta = env.riccati_policy()
    mean, variance = env.exact_moments(theta)

    assert mean.shape == (env.config.T + 1,)
    assert variance.shape == (env.config.T + 1,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert (variance >= 0.0).all()


def test_lq_sample_trajectories_shapes_are_finite():
    env = LinearQuadraticMFC(LQConfig(device=DEVICE, dtype=DTYPE, T=4))
    theta = env.riccati_policy()
    batch = env.sample_trajectories(theta, n=16, seed=123)

    assert batch["mean_flow"].shape == (env.config.T + 1,)
    assert batch["law_means"].shape == (16, env.config.T + 1)
    assert batch["states"].shape == (16, env.config.T + 1)
    assert batch["actions"].shape == (16, env.config.T)
    assert batch["action_means"].shape == (16, env.config.T)
    assert batch["residuals"].shape == (16, env.config.T)
    assert batch["stage_costs"].shape == (16, env.config.T)
    assert batch["terminal_costs"].shape == (16,)
    for value in batch.values():
        assert torch.isfinite(value).all()


def test_lq_policy_score_batch_matches_gaussian_formula():
    env = LinearQuadraticMFC(LQConfig(device=DEVICE, dtype=DTYPE, T=4, policy_std=0.5))
    theta = torch.tensor(
        [
            [-0.1, -0.2],
            [-0.3, 0.1],
            [0.2, -0.4],
            [0.3, 0.2],
        ],
        dtype=DTYPE,
    )
    states = torch.tensor([0.2, -0.5, 1.0], dtype=DTYPE)
    law_mean = torch.tensor([0.4, 0.4, 0.4], dtype=DTYPE)
    t = 2
    action_means = env.policy_mean(theta, t, states, law_mean)
    actions = action_means + torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)

    scores = env.policy_score_batch(theta, t, states, law_mean, actions, action_means=action_means)
    coefficient = (actions - action_means) / (env.config.policy_std**2)
    expected = torch.zeros((3, env.config.T, 2), dtype=DTYPE)
    expected[:, t, 0] = coefficient * states
    expected[:, t, 1] = coefficient * law_mean

    assert torch.allclose(scores, expected)
