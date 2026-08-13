from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass
class AdvertisingConfig:
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float64

    T: int = 5
    gamma: float = 0.5
    kappa_ad: float = 0.2
    c_ad: float = 0.15
    n_states: int = 2
    n_actions: int = 2
    hidden_units: int = 32

    p0_low: float = 0.05
    p0_high: float = 0.95
    validation_grid_size: int = 101

    q_sigma: float = 1.0
    q_clip: float = 1e-8

    N: int = 500 # Trajectories for MF-REINFORCE
    n: int = 10 # Trajectories for the gradient of logits estimation

    n_train: int = 100_000 # Number of epochs
    lr: float = 1e-4 # Learning rate
    training_runs: int = 5 # Number of independent training runs for each epsilon value
    validate_every: int = 10 # Freeze the policy and sample a validation episode, for which we compute the population reward starting from a fixed initial distribution
    keep_score_diagnostics: bool = False # Store full per-sample score tensors for debugging; expensive for neural policies.

    advertising_state_names: ClassVar[list[str]] = ["N", "C"]
    NONCUSTOMER, CUSTOMER = range(2)
    NO_AD, AD = range(2)

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1).")
        if self.kappa_ad <= 0.0:
            raise ValueError("kappa_ad must be positive.")
        if self.c_ad <= 0.0:
            raise ValueError("c_ad must be positive.")
        if self.n_states != 2:
            raise ValueError("AdvertisingConfig only supports two states.")
        if self.n_actions != 2:
            raise ValueError("AdvertisingConfig only supports two actions.")
        if not 0.0 <= self.p0_low < self.p0_high <= 1.0:
            raise ValueError("p0_low and p0_high must define an interval in [0, 1].")
        if self.validation_grid_size <= 0:
            raise ValueError("validation_grid_size must be positive.")


class AdvertisingPolicy(torch.nn.Module):
    def __init__(self, config: AdvertisingConfig):
        super().__init__()
        self.config = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, 1),
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
        p_customers = mu_in[..., self.config.CUSTOMER : self.config.CUSTOMER + 1]
        z = torch.cat([time, p_customers], dim=-1)
        logits = self.net(z).squeeze(-1)
        return logits.squeeze(0) if single else logits

    def advertising_probability(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(t, mu))

    def probs(self, t: int, mu: torch.Tensor) -> torch.Tensor:
        single = mu.ndim == 1
        q = self.advertising_probability(t, mu)
        q_in = q.unsqueeze(0) if single else q
        action_probs = torch.stack([1.0 - q_in, q_in], dim=-1)
        probs = action_probs.unsqueeze(-2).expand(
            *action_probs.shape[:-1],
            self.config.n_states,
            self.config.n_actions,
        )
        return probs.squeeze(0) if single else probs


class AdvertisingMFC:
    """
    Two-state targeted-advertising mean-field control benchmark.

    State 1 is a customer, action 1 displays an advertisement, and the
    transition kernel depends on the current customer proportion.
    """

    def __init__(self, config: AdvertisingConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions

    def action_probabilities(self, policy: AdvertisingPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        return policy.probs(t, mu)

    def advertising_rate(self, policy: AdvertisingPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(policy, t, mu)
        return (mu * pi[..., self.config.AD]).sum(dim=-1)

    def transition_tensor(self, mu: torch.Tensor) -> torch.Tensor:
        c = self.config
        action_values = torch.arange(self.n_actions, dtype=mu.dtype, device=mu.device)
        p_customers = mu[..., c.CUSTOMER].unsqueeze(-1)
        p_next_customer = torch.clamp(p_customers + c.kappa_ad * action_values, max=1.0)
        rows = torch.stack([1.0 - p_next_customer, p_next_customer], dim=-1)
        return rows.unsqueeze(-2).expand(
            *mu.shape[:-1],
            self.n_actions,
            self.n_states,
            self.n_states,
        )

    def transition_probs(self, state: int, action: int, mu: torch.Tensor) -> torch.Tensor:
        return self.transition_tensor(mu)[..., action, state, :]

    def averaged_kernel(self, policy: AdvertisingPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(policy, t, mu)
        transitions = self.transition_tensor(mu)
        return torch.einsum("...xa,...axy->...xy", pi, transitions)

    def next_law(self, policy: AdvertisingPolicy, t: int, mu: torch.Tensor) -> torch.Tensor:
        q_ad = self.advertising_rate(policy, t, mu)
        p_customers = mu[..., self.config.CUSTOMER]
        available_mass = torch.clamp(1.0 - p_customers, min=0.0)
        max_increment = torch.minimum(
            torch.full_like(p_customers, self.config.kappa_ad),
            available_mass,
        )
        p_next = torch.clamp(p_customers + q_ad * max_increment, min=0.0, max=1.0)
        return torch.stack([1.0 - p_next, p_next], dim=-1)

    def exact_population_flow(
        self,
        policy: AdvertisingPolicy,
        mu0: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        flow = [mu0]
        for t in range(steps):
            flow.append(self.next_law(policy, t, flow[-1]))
        return torch.stack(flow)

    def exact_value(
        self,
        policy: AdvertisingPolicy,
        mu0: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        value = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        mu = mu0
        discount = 1.0
        for t in range(steps):
            p_customers = mu[..., self.config.CUSTOMER]
            q_ad = self.advertising_rate(policy, t, mu)
            value = value + discount * (p_customers - self.config.c_ad * q_ad)
            mu = self.next_law(policy, t, mu)
            discount *= self.config.gamma
        return value

    def reward(self, state: int, mu: torch.Tensor, action: int | None = None) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        action_t = None if action is None else torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.reward_batch(state_t, mu, action_t)

    def reward_batch(
        self,
        states: torch.Tensor,
        mu: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rewards = states.to(dtype=self.config.dtype, device=self.config.device)
        if actions is None:
            return rewards
        return rewards - self.config.c_ad * actions.to(dtype=self.config.dtype, device=self.config.device)

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=self.config.dtype, device=self.config.device)

    def terminal_reward_batch(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape, dtype=self.config.dtype, device=states.device)

    @torch.no_grad()
    def sample_initial_laws(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive.")
        p = self.config.p0_low + (self.config.p0_high - self.config.p0_low) * torch.rand(
            count,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        return torch.stack([1.0 - p, p], dim=-1)

    def validation_initial_laws(self, grid_size: int | None = None) -> torch.Tensor:
        size = self.config.validation_grid_size if grid_size is None else grid_size
        if size <= 0:
            raise ValueError("grid_size must be positive.")
        p = torch.linspace(
            self.config.p0_low,
            self.config.p0_high,
            size,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        return torch.stack([1.0 - p, p], dim=-1)

    @torch.no_grad()
    def sample_action(self, policy: AdvertisingPolicy, t: int, state: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        return int(self.sample_actions_batch(policy, t, state_t, mu).item())

    @torch.no_grad()
    def sample_actions_batch(
        self,
        policy: AdvertisingPolicy,
        t: int,
        states: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
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
        actions_flat = actions.reshape(-1)
        if mu.ndim == 1:
            p_customers = mu[self.config.CUSTOMER].expand(actions_flat.numel())
        else:
            p_customers = mu.reshape(actions_flat.numel(), self.n_states)[:, self.config.CUSTOMER]
        p_next_customer = torch.clamp(
            p_customers + self.config.kappa_ad * actions_flat.to(dtype=mu.dtype),
            max=1.0,
        )
        draws = torch.rand(actions_flat.shape, dtype=mu.dtype, device=actions.device)
        return (draws < p_next_customer).to(dtype=torch.long).reshape_as(actions)

    def policy_score(
        self,
        policy: AdvertisingPolicy,
        t: int,
        mu: torch.Tensor,
        state: int,
        action: int,
    ) -> torch.Tensor:
        states = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.policy_scores_batch(policy, t, mu, states, actions).reshape(-1)

    def policy_scores_batch(
        self,
        policy: AdvertisingPolicy,
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
            actions_chunk = actions_flat[start:end].to(dtype=mu.dtype)
            logits = policy.forward(t, mu_chunk)
            logp = actions_chunk * torch.nn.functional.logsigmoid(logits)
            logp = logp + (1.0 - actions_chunk) * torch.nn.functional.logsigmoid(-logits)
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

    def policy_log_probs_batch(
        self,
        policy: AdvertisingPolicy,
        t: int,
        mu: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1).to(dtype=mu.dtype)
        if mu.ndim == 1:
            mu_flat = mu.unsqueeze(0).expand(states_flat.numel(), self.n_states)
        else:
            mu_flat = mu.reshape(states_flat.numel(), self.n_states)
        logits = policy.forward(t, mu_flat)
        logp = actions_flat * torch.nn.functional.logsigmoid(logits)
        logp = logp + (1.0 - actions_flat) * torch.nn.functional.logsigmoid(-logits)
        return logp.reshape_as(states)

    def weighted_policy_score_sums(
        self,
        policy: AdvertisingPolicy,
        t: int,
        mu: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        if mu.ndim == 1:
            mu_flat = mu.unsqueeze(0).expand(states_flat.numel(), self.n_states)
        else:
            mu_flat = mu.reshape(states_flat.numel(), self.n_states)

        weights_2d = weights.to(dtype=mu.dtype, device=mu.device).reshape(-1, states_flat.numel())
        group_shape = weights.shape[:-1] if weights.shape != states.shape else ()
        if chunk_size is None:
            chunk_size = states_flat.numel()

        linear1 = policy.net[0]
        linear2 = policy.net[2]
        linear3 = policy.net[4]
        params = tuple(policy.parameters())
        param_dim = sum(parameter.numel() for parameter in params)
        result = torch.zeros(weights_2d.shape[0], param_dim, dtype=mu.dtype, device=mu.device)

        for start in range(0, states_flat.numel(), chunk_size):
            end = min(start + chunk_size, states_flat.numel())
            chunk_weights = weights_2d[:, start:end]
            active = chunk_weights.abs().sum(dim=1) > 0
            if not active.any():
                continue

            active_indices = active.nonzero(as_tuple=False).squeeze(1)
            chunk_weights = chunk_weights[active]
            mu_chunk = mu_flat[start:end]
            actions_chunk = actions_flat[start:end].to(dtype=mu.dtype)
            batch = end - start
            groups = chunk_weights.shape[0]

            time = torch.full(
                (batch, 1),
                t / max(1, self.config.T),
                dtype=mu.dtype,
                device=mu.device,
            )
            p_customers = mu_chunk[:, self.config.CUSTOMER : self.config.CUSTOMER + 1]
            z = torch.cat([time, p_customers], dim=-1)
            pre1 = torch.nn.functional.linear(z, linear1.weight, linear1.bias)
            hidden1 = torch.tanh(pre1)
            pre2 = torch.nn.functional.linear(hidden1, linear2.weight, linear2.bias)
            hidden2 = torch.tanh(pre2)
            logits = torch.nn.functional.linear(hidden2, linear3.weight, linear3.bias).squeeze(-1)
            q = torch.sigmoid(logits)

            delta_out = chunk_weights * (actions_chunk - q).unsqueeze(0)
            grad_w3 = torch.einsum("gb,bh->gh", delta_out, hidden2).unsqueeze(1)
            grad_b3 = delta_out.sum(dim=1, keepdim=True)
            delta_hidden2 = delta_out.unsqueeze(-1) * linear3.weight.unsqueeze(0)
            delta_pre2 = delta_hidden2 * (1.0 - hidden2.square()).unsqueeze(0)

            grad_w2 = torch.einsum("gbh,bi->ghi", delta_pre2, hidden1)
            grad_b2 = delta_pre2.sum(dim=1)
            delta_hidden1 = torch.einsum("gbh,hi->gbi", delta_pre2, linear2.weight)
            delta_pre1 = delta_hidden1 * (1.0 - hidden1.square()).unsqueeze(0)

            grad_w1 = torch.einsum("gbh,bi->ghi", delta_pre1, z)
            grad_b1 = delta_pre1.sum(dim=1)
            flat = torch.cat(
                [
                    grad_w1.reshape(groups, -1),
                    grad_b1.reshape(groups, -1),
                    grad_w2.reshape(groups, -1),
                    grad_b2.reshape(groups, -1),
                    grad_w3.reshape(groups, -1),
                    grad_b3.reshape(groups, -1),
                ],
                dim=1,
            )
            result[active_indices] += flat.detach()

        if group_shape:
            return result.reshape(*group_shape, param_dim)
        return result.squeeze(0)

    def infinite_horizon_reference_policy(self, p: torch.Tensor | float) -> torch.Tensor:
        p_tensor = torch.as_tensor(p, dtype=self.config.dtype, device=self.config.device)
        ratio = self.config.c_ad / self.config.kappa_ad
        p_bar = 1.0 - self.config.c_ad * (1.0 - self.config.gamma) / self.config.gamma

        if ratio < self.config.gamma:
            return torch.where(
                p_tensor < p_bar,
                torch.ones_like(p_tensor),
                torch.zeros_like(p_tensor),
            )

        if ratio < self.config.gamma / (1.0 - self.config.gamma):
            first = 1.0 - 2.0 * self.config.kappa_ad
            second = 1.0 - (2.0 - self.config.gamma) * self.config.kappa_ad
            middle = (1.0 - self.config.kappa_ad - p_tensor) / self.config.kappa_ad
            q = torch.zeros_like(p_tensor)
            q = torch.where(p_tensor < first, torch.ones_like(q), q)
            q = torch.where((first <= p_tensor) & (p_tensor < second), middle, q)
            q = torch.where((second <= p_tensor) & (p_tensor < p_bar), torch.ones_like(q), q)
            return torch.clamp(q, min=0.0, max=1.0)

        return torch.zeros_like(p_tensor)

    def finite_horizon_dp_oracle(
        self,
        grid_size: int = 1001,
        action_grid_size: int = 1001,
    ) -> dict[str, torch.Tensor]:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if action_grid_size < 2:
            raise ValueError("action_grid_size must be at least 2.")

        p_grid = torch.linspace(
            0.0,
            1.0,
            grid_size,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        q_grid = torch.linspace(
            0.0,
            1.0,
            action_grid_size,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        values = torch.zeros(self.config.T + 1, grid_size, dtype=self.config.dtype, device=self.config.device)
        policy = torch.zeros(self.config.T, grid_size, dtype=self.config.dtype, device=self.config.device)

        p = p_grid.unsqueeze(1)
        q = q_grid.unsqueeze(0)
        p_next = p + q * torch.minimum(
            torch.full_like(p, self.config.kappa_ad),
            1.0 - p,
        )

        for t in range(self.config.T - 1, -1, -1):
            continuation = self._interp_on_grid(p_next, p_grid, values[t + 1])
            objective = p - self.config.c_ad * q + self.config.gamma * continuation
            best_values, best_indices = objective.max(dim=1)
            values[t] = best_values
            policy[t] = q_grid[best_indices]

        return {"p_grid": p_grid, "q_grid": q_grid, "values": values, "policy": policy}

    def _interp_on_grid(self, x: torch.Tensor, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        x = x.clamp(float(grid[0].item()), float(grid[-1].item()))
        right = torch.searchsorted(grid, x).clamp(1, grid.numel() - 1)
        left = right - 1
        left_grid = grid[left]
        right_grid = grid[right]
        weight = (x - left_grid) / (right_grid - left_grid).clamp_min(torch.finfo(grid.dtype).eps)
        return values[left] * (1.0 - weight) + values[right] * weight
