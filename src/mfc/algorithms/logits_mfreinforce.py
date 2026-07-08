from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch


class LogitsPerturbedMFREINFORCE:
    """
    Mean-Field REINFORCE for Finite-Horizon MFMDP.
    From: Meunier, M., Pham, H. and Reisinger, C., 2026. Model-free policy gradient for discrete-time mean-field control. arXiv preprint arXiv:2601.11217.
    """

    def __init__(self, env):
        self.env = env
        self.config = env.config
        self.n_states = env.n_states

    def parameter_vector(self, control) -> torch.Tensor:
        if isinstance(control, torch.nn.Module):
            return torch.nn.utils.parameters_to_vector(control.parameters()).detach()
        return control.detach().reshape(-1)

    def format_gradient(self, control, grad_flat: torch.Tensor) -> torch.Tensor:
        if isinstance(control, torch.nn.Module):
            return grad_flat
        return grad_flat.reshape_as(control)

    def logit(self, mu: torch.Tensor) -> torch.Tensor:
        return torch.log(mu)

    def positive_law(self, mu: torch.Tensor) -> torch.Tensor:
        mu = mu.clamp_min(self.config.q_clip)
        return mu / mu.sum()

    def perturb_law(self, logits: torch.Tensor, epsilon: float, lam: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits + epsilon * lam, dim=-1)

    def sample_lambda(self, horizon: int) -> torch.Tensor:
        return torch.randn(horizon + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)

    @torch.no_grad()
    def estimate_population_flow(
        self,
        control,
        mu0: torch.Tensor,
        n_particles: int,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")

        states = torch.multinomial(mu0, num_samples=n_particles, replacement=True)
        flow = torch.zeros(steps + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)
        flow[0] = self.positive_law(torch.bincount(states, minlength=self.n_states).to(self.config.dtype) / n_particles)

        for t in range(steps):
            mu_t = flow[t]
            next_states = torch.empty_like(states)
            for r in range(n_particles):
                x = int(states[r].item())
                a = self.env.sample_action(control, t, x, mu_t)
                next_states[r] = self.env.sample_next_state(x, a, mu_t)
            states = next_states
            flow[t + 1] = self.positive_law(
                torch.bincount(states, minlength=self.n_states).to(self.config.dtype) / n_particles
            )

        return flow

    def simulate_perturbed_path(
        self,
        control,
        mu0: torch.Tensor,
        logit_flow: torch.Tensor,
        epsilon: float,
        lambdas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = logit_flow.shape[0] - 1
        x = torch.empty(horizon + 1, dtype=torch.long, device=self.config.device)
        y = torch.empty(horizon + 1, dtype=torch.long, device=self.config.device)
        actions_x = torch.empty(horizon, dtype=torch.long, device=self.config.device)
        actions_y = torch.empty(horizon, dtype=torch.long, device=self.config.device)
        rewards_y = torch.zeros(horizon, dtype=self.config.dtype, device=self.config.device)

        x[0] = int(torch.multinomial(mu0, num_samples=1).item())
        y[0] = int(torch.multinomial(mu0, num_samples=1).item())

        for t in range(horizon):
            mu_t = torch.softmax(logit_flow[t], dim=-1)
            m_t = self.perturb_law(logit_flow[t], epsilon, lambdas[t])

            actions_x[t] = self.env.sample_action(control, t, int(x[t].item()), mu_t)
            actions_y[t] = self.env.sample_action(control, t, int(y[t].item()), m_t)
            rewards_y[t] = self.env.reward(int(y[t].item()), m_t, int(actions_y[t].item()))

            x[t + 1] = self.env.sample_next_state(int(x[t].item()), int(actions_x[t].item()), mu_t)
            y[t + 1] = self.env.sample_next_state(int(y[t].item()), int(actions_y[t].item()), m_t)

        return x, y, actions_x, actions_y, rewards_y

    def estimate_logit_gradients(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        if n <= 0:
            raise ValueError("n must be positive.")

        mu_flow = self.estimate_population_flow(control, mu0, flow_particles, steps)
        logit_flow = torch.stack([self.logit(mu_t) for mu_t in mu_flow])
        param_dim = self.parameter_vector(control).numel()
        logit_grads = torch.zeros(steps + 1, self.n_states, param_dim, dtype=self.config.dtype, device=self.config.device)

        for t in range(1, steps + 1):
            grad_mu = torch.zeros(self.n_states, param_dim, dtype=self.config.dtype, device=self.config.device)

            for _ in range(n):
                lambdas = self.sample_lambda(t - 1)
                _, y, _, actions_y, _ = self.simulate_perturbed_path(control, mu0, logit_flow[: t + 1], epsilon, lambdas)

                score_sum = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)
                for s in range(t):
                    m_s = self.perturb_law(logit_flow[s], epsilon, lambdas[s])
                    score_sum = score_sum + (lambdas[s] @ logit_grads[s]) / epsilon
                    score_sum = score_sum + self.env.policy_score(
                        control,
                        s,
                        m_s.detach(),
                        int(y[s].item()),
                        int(actions_y[s].item()),
                    ).detach().reshape(-1)

                grad_mu[int(y[t].item())] = grad_mu[int(y[t].item())] + score_sum

            logit_grads[t] = grad_mu / n / mu_flow[t].unsqueeze(1)

        return logit_grads

    def gradient_sample(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        steps = self.config.T if horizon is None else horizon
        mu_flow = self.estimate_population_flow(control, mu0, flow_particles, steps)
        logit_flow = torch.stack([self.logit(mu_t) for mu_t in mu_flow])
        logit_grads = self.estimate_logit_gradients(control, mu0, epsilon, n, flow_particles, steps)
        lambdas = self.sample_lambda(steps)
        _, y, _, actions_y, rewards_y = self.simulate_perturbed_path(control, mu0, logit_flow, epsilon, lambdas)

        returns = torch.zeros(steps + 1, dtype=self.config.dtype, device=self.config.device)
        m_T = self.perturb_law(logit_flow[steps], epsilon, lambdas[steps])
        returns[steps] = self.env.terminal_reward(int(y[steps].item()), m_T)

        for t in range(steps - 1, -1, -1):
            returns[t] = rewards_y[t] + returns[t + 1]

        param_dim = self.parameter_vector(control).numel()
        grad_flat = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)

        for t in range(steps + 1):
            score = (lambdas[t] @ logit_grads[t]) / epsilon
            if t < steps:
                m_t = self.perturb_law(logit_flow[t], epsilon, lambdas[t])
                score = score + self.env.policy_score(
                    control,
                    t,
                    m_t.detach(),
                    int(y[t].item()),
                    int(actions_y[t].item()),
                ).detach().reshape(-1)
            grad_flat = grad_flat + score * returns[t]

        return grad_flat, {
            "returns": returns,
            "logit_gradients": logit_grads,
            "lambdas": lambdas,
            "y": y,
            "actions_y": actions_y,
        }

    def gradient_estimate(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        N: int,
        n: int,
        flow_particles: int,
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if N <= 0:
            raise ValueError("N must be positive.")

        param_dim = self.parameter_vector(control).numel()
        samples = torch.zeros(N, param_dim, dtype=self.config.dtype, device=self.config.device)
        returns = torch.zeros(N, dtype=self.config.dtype, device=self.config.device)

        for k in range(N):
            grad_k, diag_k = self.gradient_sample(control, mu0, epsilon, n, flow_particles, horizon)
            samples[k] = grad_k
            returns[k] = diag_k["returns"][0]

        grad_flat = samples.mean(dim=0)
        return self.format_gradient(control, grad_flat), {
            "samples": samples,
            "returns": returns,
            "mean_return": returns.mean(),
            "std_return": returns.std(unbiased=False),
            "grad_norm": torch.linalg.norm(grad_flat),
        }
