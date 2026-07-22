from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, List

import torch


# Attributes
@dataclass
class CybersecurityConfig:
    device: torch.device
    dtype: torch.dtype

    beta_UU: float = 0.3
    beta_UD: float = 0.4
    beta_DU: float = 0.3
    beta_DD: float = 0.4
    q_rec_D: float = 0.5
    q_rec_U: float = 0.4
    q_inf_D: float = 0.4
    q_inf_U: float = 0.3
    v_H: float = 0.6
    switch_rate: float = 0.8
    k_D: float = 0.3
    k_I: float = 0.5
    dt: float = 0.2
    gamma: float = 0.5
    T_train: int = 3
    T_val: int = 50
    n_states: int = 4
    n_actions: int = 2
    hidden_units: int = 32
    q_sigma: float = 1.0
    q_clip: float = 1e-8

    cyber_state_names: ClassVar[List[str]] = ["DI", "DS", "UI", "US"]
    DI, DS, UI, US = range(4)
    KEEP, UPDATE = range(2)

    N: int = 200 # Trajectories for MF-REINFORCE
    n: int = 1 # Trajectory for the gradient of logits estimation

    lr: float = 1e-3 # Learning rate
    n_train: int = 20_000 # Number of epochs
    training_runs: int = 5 # Number of independent training runs for each epsilon value
    validate_every: int = 10 # Freeze the policy and sample a validation episode, for which we compute the population reward starting from a fixed initial distribution


# Environment
class CybersecurityPolicy(torch.nn.Module):
    def __init__(self, config: CybersecurityConfig):
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
            t / max(1, self.config.T_val),
            dtype=mu.dtype,
            device=mu.device,
        )
        z = torch.cat([time, mu_in], dim=-1)
        logits = self.net(z).reshape(*mu_in.shape[:-1], self.config.n_states, self.config.n_actions)
        return logits.squeeze(0) if single else logits

    def probs(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(t, mu), dim=-1)


class CybersecurityMFC:
    def __init__(self, config: CybersecurityConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.reward_by_state = torch.tensor(
            [-(config.k_D + config.k_I), -config.k_D, -config.k_I, 0.0],
            dtype=config.dtype,
            device=config.device,
        ) * config.dt

    def generator(self, mu: torch.Tensor, action: int) -> torch.Tensor:
        c = self.config
        zero = torch.zeros((), dtype=mu.dtype, device=mu.device)
        sw = torch.as_tensor(c.switch_rate * float(action), dtype=mu.dtype, device=mu.device)
        q_rec_D = torch.as_tensor(c.q_rec_D, dtype=mu.dtype, device=mu.device)
        q_rec_U = torch.as_tensor(c.q_rec_U, dtype=mu.dtype, device=mu.device)
        inf_D = c.v_H * c.q_inf_D + c.beta_DD * mu[c.DI] + c.beta_UD * mu[c.UI]
        inf_U = c.v_H * c.q_inf_U + c.beta_UU * mu[c.UI] + c.beta_DU * mu[c.DI]

        return torch.stack([
            torch.stack([-(q_rec_D + sw), q_rec_D, sw, zero]),
            torch.stack([inf_D, -(inf_D + sw), zero, sw]),
            torch.stack([sw, zero, -(sw + q_rec_U), q_rec_U]),
            torch.stack([zero, sw, inf_U, -(sw + inf_U)]),
        ])

    def action_probabilities(self, policy: CybersecurityPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        return policy.probs(t, mu)

    def transition_tensor(self, mu: torch.Tensor) -> torch.Tensor:
        c = self.config
        batch_shape = mu.shape[:-1]
        action_shape = (*batch_shape, self.n_actions)
        action_view = (1,) * len(batch_shape) + (self.n_actions,)

        action_values = torch.arange(self.n_actions, dtype=mu.dtype, device=mu.device).reshape(action_view)
        sw = c.switch_rate * action_values.expand(action_shape)
        zero = torch.zeros(action_shape, dtype=mu.dtype, device=mu.device)
        q_rec_D = torch.full(action_shape, c.q_rec_D, dtype=mu.dtype, device=mu.device)
        q_rec_U = torch.full(action_shape, c.q_rec_U, dtype=mu.dtype, device=mu.device)
        inf_D = (
            c.v_H * c.q_inf_D
            + c.beta_DD * mu[..., c.DI]
            + c.beta_UD * mu[..., c.UI]
        ).unsqueeze(-1).expand(action_shape)
        inf_U = (
            c.v_H * c.q_inf_U
            + c.beta_UU * mu[..., c.UI]
            + c.beta_DU * mu[..., c.DI]
        ).unsqueeze(-1).expand(action_shape)

        generator = torch.zeros(
            *batch_shape,
            self.n_actions,
            self.n_states,
            self.n_states,
            dtype=mu.dtype,
            device=mu.device,
        )
        generator[..., 0, 0] = -(q_rec_D + sw)
        generator[..., 0, 1] = q_rec_D
        generator[..., 0, 2] = sw
        generator[..., 0, 3] = zero
        generator[..., 1, 0] = inf_D
        generator[..., 1, 1] = -(inf_D + sw)
        generator[..., 1, 2] = zero
        generator[..., 1, 3] = sw
        generator[..., 2, 0] = sw
        generator[..., 2, 1] = zero
        generator[..., 2, 2] = -(sw + q_rec_U)
        generator[..., 2, 3] = q_rec_U
        generator[..., 3, 0] = zero
        generator[..., 3, 1] = sw
        generator[..., 3, 2] = inf_U
        generator[..., 3, 3] = -(sw + inf_U)
        return torch.matrix_exp(c.dt * generator)

    def transition_matrix(self, mu: torch.Tensor, action: int) -> torch.Tensor:
        return self.transition_tensor(mu)[..., action, :, :]

    def averaged_kernel(self, policy: CybersecurityPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(policy, t, mu)
        transitions = self.transition_tensor(mu)
        return torch.einsum("...xa,...axy->...xy", pi, transitions)

    def exact_population_flow(self, policy: CybersecurityPolicy, mu0: torch.Tensor, horizon: int) -> torch.Tensor:
        flow = [mu0]
        for t in range(horizon):
            flow.append(flow[-1] @ self.averaged_kernel(policy, t, flow[-1]))
        return torch.stack(flow)

    def exact_value(self, policy: CybersecurityPolicy, mu0: torch.Tensor, horizon: int) -> torch.Tensor:
        flow = self.exact_population_flow(policy, mu0, horizon)
        value = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        for t in range(horizon):
            value = value + (self.config.gamma ** t) * (flow[t] * self.reward_by_state).sum()
        value = value + (self.config.gamma ** horizon) * (flow[horizon] * self.reward_by_state).sum()
        return value

    @torch.no_grad()
    def sample_action(self, policy: CybersecurityPolicy, t: int, state: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        return int(self.sample_actions_batch(policy, t, state_t, mu).item())

    @torch.no_grad()
    def sample_actions_batch(self, policy: CybersecurityPolicy, t: int, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
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
        transitions = self.transition_tensor(mu)
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        if transitions.ndim == 3:
            probs = transitions[actions_flat, states_flat]
        else:
            probs = transitions.reshape(-1, self.n_actions, self.n_states, self.n_states)[
                torch.arange(states_flat.numel(), device=states.device),
                actions_flat,
                states_flat,
            ]
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).reshape_as(states)

    def reward(self, state: int, mu: torch.Tensor, action: int | None = None) -> torch.Tensor:
        return self.reward_by_state[state]

    def reward_batch(self, states: torch.Tensor, mu: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        return self.reward_by_state[states]

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.reward(state, mu)

    def terminal_reward_batch(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.reward_batch(states, mu)

    def policy_score(self, policy: CybersecurityPolicy, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        states = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.policy_scores_batch(policy, t, mu, states, actions).reshape(-1)

    def policy_scores_batch(
        self,
        policy: CybersecurityPolicy,
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
            chunk_size = states_flat.numel()

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
