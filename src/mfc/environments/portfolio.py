from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

import torch


TimeSeriesLike = float | Sequence[float] | torch.Tensor


@dataclass
class PortfolioConfig:
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float64

    T: int = 10
    x0_mean: float = 1.0
    x0_std: float = 0.20

    s: TimeSeriesLike = 1.0
    r_bar: TimeSeriesLike = 0.02
    sigma_R: TimeSeriesLike = 0.08
    tau: TimeSeriesLike = 0.02

    chi: float = 10.0
    rho: float = 1.0
    return_distribution: Literal["normal", "student_t"] = "normal"
    student_t_df: float = 5.0

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.x0_std < 0.0:
            raise ValueError("x0_std must be non-negative.")
        if self.chi <= 0.0:
            raise ValueError("chi must be positive.")
        if self.rho < 0.0:
            raise ValueError("rho must be non-negative.")
        if self.return_distribution not in {"normal", "student_t"}:
            raise ValueError("return_distribution must be either 'normal' or 'student_t'.")
        if self.student_t_df <= 2.0:
            raise ValueError("student_t_df must be greater than 2 so that the variance is finite.")


class MeanVariancePortfolioMFC:
    """
    Discrete-time mean-variance portfolio selection benchmark.

    The Gaussian policy has parameters theta[t, 0] and theta[t, 1]:
    alpha_t ~ N(theta[t, 0] (X_t - m_t) + theta[t, 1], tau_t^2),
    where m_t is the population mean wealth, possibly after the randomized
    affine law perturbation used by the benchmark oracle.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.s = self._as_time_vector(config.s, "s")
        self.r_bar = self._as_time_vector(config.r_bar, "r_bar")
        self.sigma_R = self._as_time_vector(config.sigma_R, "sigma_R")
        self.tau = self._as_time_vector(config.tau, "tau")
        if (self.s <= 0.0).any():
            raise ValueError("s must be positive at every time step.")
        if (self.sigma_R <= 0.0).any():
            raise ValueError("sigma_R must be positive at every time step.")
        if (self.tau <= 0.0).any():
            raise ValueError("tau must be positive at every time step.")
        self.h = self.r_bar.square() + self.sigma_R.square()

    def _as_time_vector(self, value: TimeSeriesLike, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=self.config.dtype, device=self.config.device)
        if tensor.ndim == 0:
            return tensor.expand(self.config.T)
        expected_shape = (self.config.T,)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must be scalar or have shape {expected_shape}, got {tuple(tensor.shape)}.")
        return tensor

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

    def _lambda_sq_rho_sq(self, lambda_: float) -> float:
        lambda_value = float(lambda_)
        if lambda_value < 0.0:
            raise ValueError("lambda_ must be non-negative.")
        return (lambda_value * self.config.rho) ** 2

    def zero_policy(self) -> torch.Tensor:
        return torch.zeros((self.config.T, 2), dtype=self.config.dtype, device=self.config.device)

    def policy_mean(self, theta: torch.Tensor, t: int, states: torch.Tensor, law_mean: torch.Tensor) -> torch.Tensor:
        theta = self._as_theta(theta)
        states = torch.as_tensor(states, dtype=theta.dtype, device=theta.device)
        law_mean = torch.as_tensor(law_mean, dtype=theta.dtype, device=theta.device)
        return theta[t, 0] * (states - law_mean) + theta[t, 1]

    def A(self, theta: torch.Tensor) -> torch.Tensor:
        theta = self._as_theta(theta)
        k = theta[:, 0]
        return self.s.square() + 2.0 * self.s * self.r_bar * k + self.h * k.square()

    def exact_moments(self, theta: torch.Tensor, lambda_: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self._as_theta(theta)
        cfg = self.config
        lambda_sq_rho_sq = self._lambda_sq_rho_sq(lambda_)
        A = self.A(theta)

        mean = torch.as_tensor(cfg.x0_mean, dtype=theta.dtype, device=theta.device)
        variance = torch.as_tensor(cfg.x0_std**2, dtype=theta.dtype, device=theta.device)
        means = [mean]
        variances = [variance]

        for t in range(cfg.T):
            k_t = theta[t, 0]
            ell_t = theta[t, 1]
            mean = self.s[t] * mean + self.r_bar[t] * ell_t
            variance = (
                A[t] * variance
                + self.h[t] * self.tau[t].square()
                + self.sigma_R[t].square() * ell_t.square()
                + self.h[t] * k_t.square() * lambda_sq_rho_sq * (means[-1].square() + 1.0)
            )
            means.append(mean)
            variances.append(variance)

        return torch.stack(means), torch.stack(variances)

    def exact_objective(self, theta: torch.Tensor, lambda_: float = 0.0) -> torch.Tensor:
        theta = self._as_theta(theta)
        cfg = self.config
        means, variances = self.exact_moments(theta, lambda_=lambda_)
        terminal_mean = means[-1]
        terminal_variance = variances[-1]
        terminal_perturbation = self._lambda_sq_rho_sq(lambda_) * (terminal_mean.square() + 1.0)
        return terminal_mean - cfg.chi * (terminal_variance + terminal_perturbation)

    def exact_value(self, theta: torch.Tensor, lambda_: float = 0.0) -> torch.Tensor:
        return self.exact_objective(theta, lambda_=lambda_)

    def exact_cost(self, theta: torch.Tensor, lambda_: float = 0.0) -> torch.Tensor:
        return -self.exact_objective(theta, lambda_=lambda_)

    def exact_gradient(self, theta: torch.Tensor, lambda_: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self._as_theta(theta).detach()
        cfg = self.config
        lambda_sq_rho_sq = self._lambda_sq_rho_sq(lambda_)
        means, variances = self.exact_moments(theta, lambda_=lambda_)

        p = torch.empty(cfg.T + 1, dtype=theta.dtype, device=theta.device)
        q = torch.empty_like(p)
        grad = torch.empty_like(theta)

        p[cfg.T] = 1.0 - 2.0 * cfg.chi * lambda_sq_rho_sq * means[cfg.T]
        q[cfg.T] = -cfg.chi
        A = self.A(theta)

        for t in range(cfg.T - 1, -1, -1):
            k_t = theta[t, 0]
            ell_t = theta[t, 1]
            p[t] = self.s[t] * p[t + 1] + 2.0 * self.h[t] * k_t.square() * lambda_sq_rho_sq * means[t] * q[t + 1]
            q[t] = A[t] * q[t + 1]
            grad[t, 0] = 2.0 * q[t + 1] * (
                (self.s[t] * self.r_bar[t] + self.h[t] * k_t) * variances[t]
                + self.h[t] * k_t * lambda_sq_rho_sq * (means[t].square() + 1.0)
            )
            grad[t, 1] = self.r_bar[t] * p[t + 1] + 2.0 * self.sigma_R[t].square() * ell_t * q[t + 1]

        return self.exact_objective(theta, lambda_=lambda_).detach(), grad.detach()

    def exact_cost_gradient(self, theta: torch.Tensor, lambda_: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        objective, grad = self.exact_gradient(theta, lambda_=lambda_)
        return -objective, -grad

    def optimal_policy(self) -> torch.Tensor:
        cfg = self.config
        theta = torch.empty((cfg.T, 2), dtype=cfg.dtype, device=cfg.device)
        B = self.r_bar.square() / self.h
        theta[:, 0] = -self.s * self.r_bar / self.h

        future_product = torch.ones((), dtype=cfg.dtype, device=cfg.device)
        for t in range(cfg.T - 1, -1, -1):
            theta[t, 1] = self.r_bar[t] / (2.0 * cfg.chi * self.sigma_R[t].square()) * future_product
            future_product = future_product / (self.s[t] * (1.0 - B[t]))

        return theta

    def running_reward_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_mean: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        return torch.zeros_like(states)

    def terminal_reward_batch(self, states: torch.Tensor, law_mean: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_mean = torch.as_tensor(law_mean, dtype=states.dtype, device=states.device)
        return states - self.config.chi * (states - law_mean).square()

    def running_cost_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_mean: torch.Tensor,
    ) -> torch.Tensor:
        return -self.running_reward_batch(states, actions, law_mean)

    def terminal_cost_batch(self, states: torch.Tensor, law_mean: torch.Tensor) -> torch.Tensor:
        return -self.terminal_reward_batch(states, law_mean)

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
        coefficient = (actions - action_means) / self.tau[t].square()
        scores = torch.zeros((*states.shape, self.config.T, 2), dtype=theta.dtype, device=theta.device)
        scores[..., t, 0] = coefficient * (states - law_mean)
        scores[..., t, 1] = coefficient
        return scores

    @torch.no_grad()
    def sample_trajectories(
        self,
        theta: torch.Tensor,
        n: int,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
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
            mean_flow = self.exact_moments(theta, lambda_=lambda_)[0].detach()
        else:
            mean_flow = self._as_law_flow(frozen_mean_flow).detach()

        states = torch.empty((n, cfg.T + 1), dtype=cfg.dtype, device=cfg.device)
        actions = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        action_means = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        residuals = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        excess_returns = torch.empty((n, cfg.T), dtype=cfg.dtype, device=cfg.device)
        perturbed_law_means = torch.empty((n, cfg.T + 1), dtype=cfg.dtype, device=cfg.device)
        stage_rewards = torch.zeros((n, cfg.T), dtype=cfg.dtype, device=cfg.device)

        x = cfg.x0_mean + cfg.x0_std * torch.randn((n,), dtype=cfg.dtype, device=cfg.device, generator=generator)
        states[:, 0] = x

        for t in range(cfg.T):
            law_mean = self._sample_perturbed_law_mean(mean_flow[t], n, lambda_, generator)
            action_mean = self.policy_mean(theta, t, x, law_mean)
            residual = self.tau[t] * torch.randn((n,), dtype=cfg.dtype, device=cfg.device, generator=generator)
            action = action_mean + residual
            excess_return = self._sample_excess_returns(t, n, generator)

            perturbed_law_means[:, t] = law_mean
            actions[:, t] = action
            action_means[:, t] = action_mean
            residuals[:, t] = residual
            excess_returns[:, t] = excess_return

            x = self.s[t] * x + excess_return * action
            states[:, t + 1] = x

        terminal_law_mean = self._sample_perturbed_law_mean(mean_flow[cfg.T], n, lambda_, generator)
        perturbed_law_means[:, cfg.T] = terminal_law_mean
        terminal_rewards = self.terminal_reward_batch(states[:, cfg.T], terminal_law_mean)
        law_means = mean_flow.unsqueeze(0).expand(n, cfg.T + 1)
        return {
            "mean_flow": mean_flow,
            "law_means": law_means,
            "perturbed_law_means": perturbed_law_means,
            "states": states,
            "actions": actions,
            "action_means": action_means,
            "residuals": residuals,
            "excess_returns": excess_returns,
            "stage_rewards": stage_rewards,
            "terminal_rewards": terminal_rewards,
            "returns": terminal_rewards,
        }

    def _sample_perturbed_law_mean(
        self,
        mean: torch.Tensor,
        n: int,
        lambda_: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        mean = torch.as_tensor(mean, dtype=self.config.dtype, device=self.config.device)
        lambda_value = float(lambda_)
        if lambda_value < 0.0:
            raise ValueError("lambda_ must be non-negative.")
        if lambda_value == 0.0 or self.config.rho == 0.0:
            return mean.expand(n)
        zeta = self.config.rho * torch.randn((n,), dtype=self.config.dtype, device=self.config.device, generator=generator)
        beta = self.config.rho * torch.randn((n,), dtype=self.config.dtype, device=self.config.device, generator=generator)
        return (1.0 + lambda_value * zeta) * mean + lambda_value * beta

    def _sample_excess_returns(
        self,
        t: int,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.config.return_distribution == "normal":
            standardized = torch.randn((n,), dtype=self.config.dtype, device=self.config.device, generator=generator)
        else:
            standardized = self._sample_standardized_student_t(n, generator)
        return self.r_bar[t] + self.sigma_R[t] * standardized

    def _sample_standardized_student_t(
        self,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        df = float(self.config.student_t_df)
        df_int = int(round(df))
        if abs(df - df_int) > 1e-12:
            distribution = torch.distributions.StudentT(
                torch.as_tensor(df, dtype=self.config.dtype, device=self.config.device)
            )
            sample = distribution.sample((n,))
        else:
            numerator = torch.randn((n,), dtype=self.config.dtype, device=self.config.device, generator=generator)
            denominator_normals = torch.randn(
                (n, df_int),
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            chi_square = denominator_normals.square().sum(dim=-1)
            sample = numerator / torch.sqrt(chi_square / df)
        return sample * torch.sqrt(torch.as_tensor((df - 2.0) / df, dtype=self.config.dtype, device=self.config.device))
