from __future__ import annotations

from types import SimpleNamespace

import torch

from mfc.algorithms.simplex_mfreinforce import SimplexPerturbedMFREINFORCE
from mfc.experiments.core.reinforce import finite_reinforce_gradient


class _DeterministicTwoStepEnv:
    n_states = 2
    n_actions = 1

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            T=2,
            gamma=1.0,
            q_sigma=0.2,
            q_clip=1e-8,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

    def sample_actions_batch(
        self,
        control: torch.Tensor,
        t: int,
        states: torch.Tensor,
        law: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(states)

    def sample_next_states_batch(self, states: torch.Tensor, actions: torch.Tensor, law: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(states)

    def reward_batch(self, states: torch.Tensor, law: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        reward_state_0 = torch.full_like(states, 10.0, dtype=self.config.dtype)
        reward_state_1 = torch.full_like(states, 1.0, dtype=self.config.dtype)
        return torch.where(states == 0, reward_state_0, reward_state_1)

    def terminal_reward_batch(self, states: torch.Tensor, law: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape, dtype=self.config.dtype, device=self.config.device)

    def policy_scores_batch(
        self,
        control: torch.Tensor,
        t: int,
        law: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        values = torch.ones(states.shape + (1,), dtype=self.config.dtype, device=self.config.device)
        return values if t == 1 else torch.zeros_like(values)


def test_simplex_mfreinforce_uses_return_to_go_weights() -> None:
    env = _DeterministicTwoStepEnv()
    algorithm = SimplexPerturbedMFREINFORCE(env)
    control = torch.zeros(1, dtype=env.config.dtype, device=env.config.device)
    mu_flow = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=env.config.dtype)
    sensitivity = torch.zeros(3, 1, 1, dtype=env.config.dtype, device=env.config.device)

    grad, diag = algorithm.gradient_estimate(control, mu_flow, sensitivity, eps_law=0.2, B=4, baseline=None)

    assert torch.allclose(grad, torch.ones_like(grad))
    assert torch.allclose(diag["returns"], torch.full((4,), 11.0, dtype=env.config.dtype))


def test_finite_reinforce_uses_return_to_go_weights() -> None:
    env = _DeterministicTwoStepEnv()
    control = torch.zeros(1, dtype=env.config.dtype, device=env.config.device)
    mu_flow = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=env.config.dtype)

    objective, grad, _diag = finite_reinforce_gradient(
        env,
        control,
        mu_flow[0],
        mu_flow,
        {"baseline": None},
        {"B": 4},
    )

    assert torch.allclose(grad, torch.ones_like(grad))
    assert torch.allclose(objective, torch.tensor(11.0, dtype=env.config.dtype))
