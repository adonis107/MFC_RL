from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class LQConfig:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    T: int = 8
    a: float = 0.75
    b: float = 0.45
    c: float = 0.10
    noise_std: float = 0.20
    policy_std: float = 0.35

    q: float = 1.0
    r: float = 0.30
    gamma: float = 0.40
    q_T: float = 1.5
    gamma_T: float = 0.60

    x0_mean: float = 0.80
    x0_std: float = 0.50

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.policy_std <= 0.0:
            raise ValueError("policy_std must be positive.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if self.x0_std < 0.0:
            raise ValueError("x0_std must be non-negative.")
        if self.r <= 0.0:
            raise ValueError("r must be positive.")


class LinearQuadraticMFC:
    """
    Continuous one-dimensional linear-quadratic mean-field control benchmark.

    The Gaussian feedback policy has parameters theta[t, 0] and theta[t, 1]:
    U_t ~ N(theta[t, 0] X_t + theta[t, 1] m_t, policy_std^2).
    """

    def __init__(self, config: LQConfig):
        self.config = config

    def _as_theta(self, theta: torch.Tensor) -> torch.Tensor:
        theta = torch.as_tensor(theta, dtype=self.config.dtype, device=self.config.device)
        expected_shape = (self.config.T, 2)
        if tuple(theta.shape) != expected_shape:
            raise ValueError(f"theta must have shape {expected_shape}, got {tuple(theta.shape)}.")
        return theta

    def _as_law_flow(self, mean_flow: torch.Tensor) -> torch.Tensor:
        mean_flow = torch.as_tensor(mean_flow, dtype=self.config.dtype, device=self.config.device)
        expected_shape = (self.config.T + 1,)
        if tuple(mean_flow.shape) != expected_shape:
            raise ValueError(f"mean_flow must have shape {expected_shape}, got {tuple(mean_flow.shape)}.")
        return mean_flow

    def zero_policy(self) -> torch.Tensor:
        return torch.zeros((self.config.T, 2), dtype=self.config.dtype, device=self.config.device)

    def policy_mean(self, theta: torch.Tensor, t: int, states: torch.Tensor, law_mean: torch.Tensor) -> torch.Tensor:
        theta = self._as_theta(theta)
        states = torch.as_tensor(states, dtype=theta.dtype, device=theta.device)
        law_mean = torch.as_tensor(law_mean, dtype=theta.dtype, device=theta.device)
        return theta[t, 0] * states + theta[t, 1] * law_mean

    def exact_moments(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self._as_theta(theta)
        cfg = self.config
        mean = torch.as_tensor(cfg.x0_mean, dtype=theta.dtype, device=theta.device)
        variance = torch.as_tensor(cfg.x0_std**2, dtype=theta.dtype, device=theta.device)
        means = [mean]
        variances = [variance]

        for t in range(cfg.T):
            mean_factor = cfg.a + cfg.b * theta[t, 0] + cfg.b * theta[t, 1] + cfg.c
            fluctuation_factor = cfg.a + cfg.b * theta[t, 0]
            mean = mean_factor * mean
            variance = (
                fluctuation_factor.square() * variance
                + cfg.b**2 * cfg.policy_std**2
                + cfg.noise_std**2
            )
            means.append(mean)
            variances.append(variance)

        return torch.stack(means), torch.stack(variances)

    def exact_cost(self, theta: torch.Tensor) -> torch.Tensor:
        theta = self._as_theta(theta)
        cfg = self.config
        mean, variance = self.exact_moments(theta)
        theta_state = theta[:, 0]
        theta_law = theta[:, 1]

        state_second_moment = variance[:-1] + mean[:-1].square()
        action_second_moment = (
            theta_state.square() * variance[:-1]
            + (theta_state + theta_law).square() * mean[:-1].square()
            + cfg.policy_std**2
        )
        running = (
            cfg.q * state_second_moment
            + cfg.r * action_second_moment
            + cfg.gamma * mean[:-1].square()
        )
        terminal = cfg.q_T * (variance[-1] + mean[-1].square()) + cfg.gamma_T * mean[-1].square()
        return running.sum() + terminal

    def exact_gradient(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_var = self._as_theta(theta).detach().clone().requires_grad_(True)
        cost = self.exact_cost(theta_var)
        grad = torch.autograd.grad(cost, theta_var)[0]
        return cost.detach(), grad.detach()

    def riccati_policy(self) -> torch.Tensor:
        cfg = self.config
        P = torch.empty(cfg.T + 1, dtype=cfg.dtype, device=cfg.device)
        S = torch.empty_like(P)
        theta = torch.empty((cfg.T, 2), dtype=cfg.dtype, device=cfg.device)
        P[cfg.T] = cfg.q_T
        S[cfg.T] = cfg.q_T + cfg.gamma_T

        for t in range(cfg.T - 1, -1, -1):
            theta[t, 0] = -(cfg.a * cfg.b * P[t + 1]) / (cfg.r + cfg.b**2 * P[t + 1])
            mean_gain = -(cfg.b * (cfg.a + cfg.c) * S[t + 1]) / (cfg.r + cfg.b**2 * S[t + 1])
            theta[t, 1] = mean_gain - theta[t, 0]
            P[t] = cfg.q + cfg.a**2 * cfg.r * P[t + 1] / (cfg.r + cfg.b**2 * P[t + 1])
            S[t] = cfg.q + cfg.gamma + (cfg.a + cfg.c) ** 2 * cfg.r * S[t + 1] / (
                cfg.r + cfg.b**2 * S[t + 1]
            )

        return theta

    def running_cost_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_mean: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=self.config.dtype, device=self.config.device)
        law_mean = torch.as_tensor(law_mean, dtype=self.config.dtype, device=self.config.device)
        return self.config.q * states.square() + self.config.r * actions.square() + self.config.gamma * law_mean.square()

    def terminal_cost_batch(self, states: torch.Tensor, law_mean: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_mean = torch.as_tensor(law_mean, dtype=self.config.dtype, device=self.config.device)
        return self.config.q_T * states.square() + self.config.gamma_T * law_mean.square()

    def policy_score_batch(
        self,
        theta: torch.Tensor,
        t: int,
        states: torch.Tensor,
        law_mean: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        theta = self._as_theta(theta)
        states = torch.as_tensor(states, dtype=theta.dtype, device=theta.device)
        law_mean = torch.as_tensor(law_mean, dtype=theta.dtype, device=theta.device)
        actions = torch.as_tensor(actions, dtype=theta.dtype, device=theta.device)
        if action_means is None:
            action_means = self.policy_mean(theta, t, states, law_mean)
        else:
            action_means = torch.as_tensor(action_means, dtype=theta.dtype, device=theta.device)

        law_mean = torch.broadcast_to(law_mean, states.shape)
        coefficient = (actions - action_means) / (self.config.policy_std**2)
        scores = torch.zeros((*states.shape, self.config.T, 2), dtype=theta.dtype, device=theta.device)
        scores[..., t, 0] = coefficient * states
        scores[..., t, 1] = coefficient * law_mean
        return scores

    @torch.no_grad()
    def sample_trajectories(
        self,
        theta: torch.Tensor,
        n: int,
        seed: Optional[int] = None,
        frozen_mean_flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive.")
        theta = self._as_theta(theta).detach()
        cfg = self.config
        generator = None
        if seed is not None:
            generator = torch.Generator(device=cfg.device)
            generator.manual_seed(int(seed))

        if frozen_mean_flow is None:
            mean_flow = self.exact_moments(theta)[0].detach()
        else:
            mean_flow = self._as_law_flow(frozen_mean_flow).detach()

        states = torch.empty((n, cfg.T + 1), dtype=cfg.dtype, device=cfg.device)
        actions = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        action_means = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        residuals = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        stage_costs = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)

        x = cfg.x0_mean + cfg.x0_std * torch.randn((n,), dtype=cfg.dtype, device=cfg.device, generator=generator)
        states[:, 0] = x

        for t in range(cfg.T):
            law_mean = mean_flow[t]
            action_mean = self.policy_mean(theta, t, x, law_mean)
            residual = cfg.policy_std * torch.randn((n,), dtype=cfg.dtype, device=cfg.device, generator=generator)
            action = action_mean + residual
            noise = cfg.noise_std * torch.randn((n,), dtype=cfg.dtype, device=cfg.device, generator=generator)

            actions[:, t] = action
            action_means[:, t] = action_mean
            residuals[:, t] = residual
            stage_costs[:, t] = self.running_cost_batch(x, action, law_mean)

            x = cfg.a * x + cfg.b * action + cfg.c * law_mean + noise
            states[:, t + 1] = x

        terminal_costs = self.terminal_cost_batch(states[:, cfg.T], mean_flow[cfg.T])
        law_means = mean_flow.unsqueeze(0).expand(n, cfg.T + 1)
        return {
            "mean_flow": mean_flow,
            "law_means": law_means,
            "states": states,
            "actions": actions,
            "action_means": action_means,
            "residuals": residuals,
            "stage_costs": stage_costs,
            "terminal_costs": terminal_costs,
        }
