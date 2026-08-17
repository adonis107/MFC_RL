from __future__ import annotations

import math

import torch

from mfc.environments import (
    CybersecurityConfig,
    CybersecurityPolicy,
    KuramotoConfig,
    KuramotoMFC,
    TwoStateConfig,
    TwoStateMFC,
)


def test_twostate_optimal_policy_matrix_can_be_evaluated_directly() -> None:
    cfg = TwoStateConfig(device=torch.device("cpu"))
    env = TwoStateMFC(cfg)
    mu0 = torch.tensor([1.0, 0.0], dtype=cfg.dtype)

    flow = env.exact_population_flow(env.optimal_policy(), mu0, horizon=1)

    assert torch.allclose(flow[-1], env.target_B)


def test_cybersecurity_time_normalization_horizon_is_explicit() -> None:
    default_cfg = CybersecurityConfig(device=torch.device("cpu"), T_train=3, T_val=11)
    assert default_cfg.time_normalization_horizon == 11

    cfg = CybersecurityConfig(
        device=torch.device("cpu"),
        T_train=3,
        T_val=50,
        time_normalization_horizon=10,
        hidden_units=1,
    )
    policy = CybersecurityPolicy(cfg)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.net[0].weight[0, 0] = 1.0
        policy.net[2].weight[0, 0] = 1.0
        policy.net[4].weight[:, 0] = 1.0

    mu = torch.full((cfg.n_states,), 1.0 / cfg.n_states, dtype=cfg.dtype)
    logits = policy.forward(5, mu)
    expected = torch.tanh(torch.tanh(torch.tensor(0.5, dtype=cfg.dtype)))

    assert torch.allclose(logits, torch.full_like(logits, expected))


def test_kuramoto_perturbation_is_circle_valued() -> None:
    cfg = KuramotoConfig(device=torch.device("cpu"))
    env = KuramotoMFC(cfg)
    phases = torch.tensor([-1.0, 2.0 * math.pi + 0.2, 9.0], dtype=cfg.dtype)
    perturbation = torch.tensor([3.0, -2.0, 1.0], dtype=cfg.dtype)
    lambda_value = 0.7

    perturbed = env._perturb_phases(phases, perturbation, lambda_value)
    expected = torch.remainder(
        phases
        + lambda_value
        * (perturbation[0] + perturbation[1] * torch.cos(phases) + perturbation[2] * torch.sin(phases)),
        2.0 * math.pi,
    )

    assert torch.all(perturbed >= 0.0)
    assert torch.all(perturbed < 2.0 * math.pi)
    assert torch.allclose(perturbed, expected)
    assert torch.allclose(env._perturb_phases(phases, perturbation, 0.0), env.wrap_phases(phases))
