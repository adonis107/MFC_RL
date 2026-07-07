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
        time = torch.tensor([t / max(1, self.config.T_val)], dtype=mu.dtype, device=mu.device)
        z = torch.cat([time, mu])
        return self.net(z).reshape(self.config.n_states, self.config.n_actions)

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

    def transition_matrix(self, mu: torch.Tensor, action: int) -> torch.Tensor:
        return torch.matrix_exp(self.config.dt * self.generator(mu, action))

    def averaged_kernel(self, policy: CybersecurityPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        c = self.config
        pi = policy.probs(t, mu)
        P_keep = self.transition_matrix(mu, c.KEEP)
        P_update = self.transition_matrix(mu, c.UPDATE)
        return pi[:, c.KEEP].unsqueeze(1) * P_keep + pi[:, c.UPDATE].unsqueeze(1) * P_update

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
        return int(torch.multinomial(policy.probs(t, mu)[state], num_samples=1).item())

    @torch.no_grad()
    def sample_next_state(self, state: int, action: int, mu: torch.Tensor) -> int:
        probs = torch.clamp(self.transition_matrix(mu, action)[state], min=0.0)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, num_samples=1).item())

    def reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.reward_by_state[state]

    def policy_score(self, policy: CybersecurityPolicy, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        logp = torch.log(policy.probs(t, mu)[state, action].clamp_min(1e-12))
        grads = torch.autograd.grad(logp, tuple(policy.parameters()), allow_unused=False)
        return torch.nn.utils.parameters_to_vector([g.detach() for g in grads])
