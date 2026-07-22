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
        single = mu.ndim == 1
        mu_in = mu.unsqueeze(0) if single else mu
        time = torch.full(
            (*mu_in.shape[:-1], 1),
            t / max(1, self.config.T),
            dtype=mu.dtype,
            device=mu.device,
        )
        z = torch.cat([time, mu_in], dim=-1)
        logits = self.net(z).reshape(*mu_in.shape[:-1], self.config.n_states, self.config.n_actions)
        return logits.squeeze(0) if single else logits

    def probs(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(t, mu), dim=-1)


class DistributionPlanningMFC:
    def __init__(self, config: DistributionPlanningConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.actions = torch.tensor([-1, 0, 1], dtype=torch.long, device=config.device)
        self.action_costs = config.lam * torch.abs(self.actions.to(dtype=config.dtype))
        target = torch.tensor(
            [0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.0],
            dtype=config.dtype,
            device=config.device,
        )
        # Previous exploratory smooth target:
        # x = torch.arange(config.n_states, dtype=config.dtype, device=config.device)
        # target = torch.exp(-0.5 * ((x - 4.5) / 2.0).square())
        self.target = target / target.sum()
        self._transition_tensor = torch.zeros(
            self.n_actions,
            self.n_states,
            self.n_states,
            dtype=config.dtype,
            device=config.device,
        )
        for action_idx, move in enumerate(self.actions.tolist()):
            next_states = (torch.arange(self.n_states, device=config.device) + move) % self.n_states
            self._transition_tensor[action_idx, torch.arange(self.n_states, device=config.device), next_states] = 1.0
        
    def distribution_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        return -(mu - self.target).square().sum(dim=-1)

    def reward(self, state: int, mu: torch.Tensor, action: Optional[int] = None) -> torch.Tensor:
        action_t = None if action is None else torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.reward_batch(torch.as_tensor(state, dtype=torch.long, device=self.config.device), mu, action_t)

    def reward_batch(self, states: torch.Tensor, mu: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        move_cost = 0.0 if actions is None else self.action_costs.to(dtype=mu.dtype)[actions]
        return self.distribution_penalty(mu) - move_cost

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.distribution_penalty(mu)

    def terminal_reward_batch(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.distribution_penalty(mu)

    def transition_probs(self, state: int, action: int, mu: torch.Tensor) -> torch.Tensor:
        return self._transition_tensor[action, state]

    def action_probabilities(self, policy: DistributionPlanningPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        return policy.probs(t, mu)

    def transition_tensor(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.ndim == 1:
            return self._transition_tensor
        return self._transition_tensor.expand(*mu.shape[:-1], self.n_actions, self.n_states, self.n_states)

    def averaged_kernel(self, policy: DistributionPlanningPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(policy, t, mu)
        transitions = self.transition_tensor(mu)
        return torch.einsum("...xa,...axy->...xy", pi, transitions)

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
            pi = self.action_probabilities(policy, t, mu)
            move_penalty = (mu.unsqueeze(-1) * pi * self.action_costs.to(dtype=mu.dtype)).sum()
            value = value + self.distribution_penalty(mu) - move_penalty
            mu = mu @ self.averaged_kernel(policy, t, mu)
        return value + self.terminal_reward(0, mu)

    @torch.no_grad()
    def sample_action(self, policy: DistributionPlanningPolicy, t: int, state: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        return int(self.sample_actions_batch(policy, t, state_t, mu).item())

    @torch.no_grad()
    def sample_actions_batch(self, policy: DistributionPlanningPolicy, t: int, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(policy, t, mu)
        states_flat = states.reshape(-1)
        if pi.ndim == 2:
            probs = pi[states_flat]
        else:
            probs = pi.reshape(-1, self.n_states, self.n_actions)[
                torch.arange(states_flat.numel(), device=states.device),
                states_flat,
            ]
        return torch.multinomial(probs, num_samples=1).reshape_as(states)

    @torch.no_grad()
    def sample_next_state(self, state: int, action: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        action_t = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return int(self.sample_next_states_batch(state_t, action_t, mu).item())

    @torch.no_grad()
    def sample_next_states_batch(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return (states + self.actions[actions]) % self.n_states

    def policy_score(self, policy: DistributionPlanningPolicy, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        states = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.policy_scores_batch(policy, t, mu, states, actions).reshape(-1)

    def policy_scores_batch(
        self,
        policy: DistributionPlanningPolicy,
        t: int,
        mu: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = tuple(policy.parameters())
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        if mu.ndim == 1:
            mu_flat = mu.unsqueeze(0).expand(states_flat.numel(), self.n_states)
        else:
            mu_flat = mu.reshape(states_flat.numel(), self.n_states)
        if chunk_size is None:
            chunk_size = min(64, states_flat.numel())

        chunks = []
        for start in range(0, states_flat.numel(), chunk_size):
            end = min(start + chunk_size, states_flat.numel())
            mu_chunk = mu_flat[start:end]
            states_chunk = states_flat[start:end]
            actions_chunk = actions_flat[start:end]
            probs = policy.probs(t, mu_chunk)
            idx = torch.arange(end - start, device=mu.device)
            logp = torch.log(probs[idx, states_chunk, actions_chunk].clamp_min(1e-12))
            grad_outputs = torch.eye(end - start, dtype=mu.dtype, device=mu.device)
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                allow_unused=False,
            )
            chunks.append(torch.cat([g.reshape(end - start, -1) for g in grads], dim=1).detach())
        return torch.cat(chunks, dim=0).reshape(*states.shape, -1)
