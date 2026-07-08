from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DistributionPlanningConfig:
    device: torch.device
    dtype: torch.dtype

    T: int = 5
    lam: float = 0.01
    n_states: int = 10
    n_actions: int = 3
    hidden_units: int = 256

    q_sigma: float = 1.0
    q_clip: float = 1e-8

    N: int = 500 # Trajectories for MF-REINFORCE
    n: int = 10 # Trajectories for the gradient of logits estimation

    n_train: int = 100_000 # Number of epochs
    lr: float = 1e-4 # Learning rate
    training_runs: int = 5 # Number of independent training runs for each epsilon value
    validate_every: int = 10 # Freeze the policy and sample a validation episode, for which we compute the population reward starting from a fixed initial distribution


class DistributionPlanningPolicy(torch.nn.Module):
    def __init__(self, config: DistributionPlanningConfig):
        super().__init__()
        self.config = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1 + config.n_states, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, config.n_states * config.n_actions),
        ).to(device=config.device, dtype=config.dtype)

    def forward(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        time = torch.tensor([t / max(1, self.config.T)], dtype=mu.dtype, device=mu.device)
        z = torch.cat([time, mu])
        return self.net(z).reshape(self.config.n_states, self.config.n_actions)

    def probs(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(t, mu), dim=-1)


class DistributionPlanningMFC:
    def __init__(self, config: DistributionPlanningConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.actions = torch.tensor([-1, 0, 1], dtype=torch.long, device=config.device)
        x = torch.arange(config.n_states, dtype=config.dtype, device=config.device)
        target = torch.exp(-0.5 * ((x - 4.5) / 2.0).square())
        self.target = target / target.sum()
        
    def distribution_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        return -(mu - self.target).square().sum()

    def reward(self, state: int, mu: torch.Tensor, action: Optional[int] = None) -> torch.Tensor:
        move_cost = 0.0 if action is None else self.config.lam * torch.abs(self.actions[action].to(dtype=mu.dtype))
        return self.distribution_penalty(mu) - move_cost

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.distribution_penalty(mu)

    def transition_probs(self, state: int, action: int, mu: torch.Tensor) -> torch.Tensor:
        probs = torch.zeros(self.n_states, dtype=self.config.dtype, device=self.config.device)
        probs[(state + int(self.actions[action].item())) % self.n_states] = 1.0
        return probs

    def averaged_kernel(self, policy: DistributionPlanningPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = policy.probs(t, mu)
        K = torch.zeros(self.n_states, self.n_states, dtype=self.config.dtype, device=self.config.device)
        for x in range(self.n_states):
            for a in range(self.n_actions):
                K[x] = K[x] + pi[x, a] * self.transition_probs(x, a, mu)
        return K

    def exact_population_flow(self, policy: DistributionPlanningPolicy, mu0: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        if horizon is None:
            horizon = self.config.T
        flow = [mu0]
        for t in range(horizon):
            flow.append(flow[-1] @ self.averaged_kernel(policy, t, flow[-1]))
        return torch.stack(flow)

    def exact_value(self, policy: DistributionPlanningPolicy, mu0: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        if horizon is None:
            horizon = self.config.T
        mu = mu0
        value = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        for t in range(horizon):
            pi = policy.probs(t, mu)
            step_reward = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
            for x in range(self.n_states):
                for a in range(self.n_actions):
                    step_reward = step_reward + mu[x] * pi[x, a] * self.reward(x, mu, a)
            value = value + step_reward
            mu = mu @ self.averaged_kernel(policy, t, mu)
        return value + self.terminal_reward(0, mu)

    @torch.no_grad()
    def sample_action(self, policy: DistributionPlanningPolicy, t: int, state: int, mu: torch.Tensor) -> int:
        return int(torch.multinomial(policy.probs(t, mu)[state], num_samples=1).item())

    @torch.no_grad()
    def sample_next_state(self, state: int, action: int, mu: torch.Tensor) -> int:
        return (state + int(self.actions[action].item())) % self.n_states

    def policy_score(self, policy: DistributionPlanningPolicy, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        logp = torch.log(policy.probs(t, mu)[state, action].clamp_min(1e-12))
        grads = torch.autograd.grad(logp, tuple(policy.parameters()), allow_unused=False)
        return torch.nn.utils.parameters_to_vector([g.detach() for g in grads])