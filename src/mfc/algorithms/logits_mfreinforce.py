from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch


class LogitsPerturbedMFREINFORCE:
    """
    Mean-Field REINFORCE for Finite-Horizon MFMDP.
    From: Meunier, M., Pham, H. and Reisinger, C., 2026. Model-free policy gradient for discrete-time mean-field control. arXiv preprint arXiv:2601.11217.
    By default this algorithm is model-free and estimates the population flow; callers may also supply a precomputed flow for exact-flow experiments.
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

    def discount(self, t: int) -> float:
        return float(getattr(self.config, "gamma", 1.0) ** t)

    def logit(self, mu: torch.Tensor) -> torch.Tensor:
        return torch.log(mu)

    def positive_law(self, mu: torch.Tensor) -> torch.Tensor:
        mu = mu.clamp_min(self.config.q_clip)
        return mu / mu.sum(dim=-1, keepdim=True)

    def perturb_law(self, logits: torch.Tensor, epsilon: float, lam: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits + epsilon * lam, dim=-1)

    def sample_lambda(self, horizon: int) -> torch.Tensor:
        return torch.randn(horizon + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)

    def _sample_lambda_batch(self, batch_size: int, horizon: int) -> torch.Tensor:
        return torch.randn(batch_size, horizon + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)

    def _score_chunk_size(self, param_dim: int, batch_size: int) -> int:
        configured = getattr(self.config, "score_chunk_size", None)
        if configured is not None:
            return int(configured)
        if param_dim > 20_000:
            return min(batch_size, 32)
        return min(batch_size, 128)

    def _sample_chunk_size(self, control, param_dim: int, N: int) -> int:
        configured = getattr(self.config, "logits_sample_chunk_size", None)
        if configured is not None:
            return max(1, min(int(configured), N))
        if not isinstance(control, torch.nn.Module):
            return N
        if param_dim > 20_000:
            return min(N, 8)
        return min(N, 32)

    @torch.no_grad()
    def _estimate_population_flow_batch(
        self,
        control,
        mu0: torch.Tensor,
        n_particles: int,
        horizon: int,
        batch_size: int,
    ) -> torch.Tensor:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")

        states = torch.multinomial(mu0, num_samples=batch_size * n_particles, replacement=True).reshape(batch_size, n_particles)
        flow = torch.zeros(batch_size, horizon + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)
        flow[:, 0] = self.positive_law(torch.nn.functional.one_hot(states, num_classes=self.n_states).to(self.config.dtype).mean(dim=1))

        for t in range(horizon):
            mu_t = flow[:, t]
            states_flat = states.reshape(-1)
            batch_indices = torch.arange(batch_size, device=self.config.device).repeat_interleave(n_particles)

            action_probs = self.env.action_probabilities(control, t, mu_t)
            if action_probs.ndim == 2:
                action_probs = action_probs.unsqueeze(0).expand(batch_size, -1, -1)
            action_probs = action_probs[batch_indices, states_flat]
            actions = torch.multinomial(action_probs, num_samples=1).reshape(-1)

            transitions = self.env.transition_tensor(mu_t)
            if transitions.ndim == 3:
                transitions = transitions.unsqueeze(0).expand(batch_size, -1, -1, -1)
            next_probs = transitions[batch_indices, actions, states_flat]
            next_probs = torch.clamp(next_probs, min=0.0)
            next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)
            states = torch.multinomial(next_probs, num_samples=1).reshape(batch_size, n_particles)
            flow[:, t + 1] = self.positive_law(
                torch.nn.functional.one_hot(states, num_classes=self.n_states).to(self.config.dtype).mean(dim=1)
            )

        return flow

    @torch.no_grad()
    def _population_flow_batch(
        self,
        control,
        mu0: torch.Tensor,
        n_particles: int,
        horizon: int,
        batch_size: int,
        mu_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mu_flow is None:
            return self._estimate_population_flow_batch(control, mu0, n_particles, horizon, batch_size)

        flow = self.positive_law(mu_flow.to(device=self.config.device, dtype=self.config.dtype))
        if flow.shape[-2:] != (horizon + 1, self.n_states):
            raise ValueError(
                f"mu_flow must have trailing shape {(horizon + 1, self.n_states)}, got {tuple(flow.shape)}."
            )

        if flow.ndim == 2:
            return flow.unsqueeze(0).expand(batch_size, -1, -1)
        if flow.ndim == 3:
            if flow.shape[0] != batch_size:
                raise ValueError(f"Batched mu_flow has batch size {flow.shape[0]}, expected {batch_size}.")
            return flow
        raise ValueError("mu_flow must have shape (T+1, n_states) or (batch_size, T+1, n_states).")

    @torch.no_grad()
    def estimate_population_flow(
        self,
        control,
        mu0: torch.Tensor,
        n_particles: int,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        flow = self._estimate_population_flow_batch(control, mu0, n_particles, steps, batch_size=1)
        return flow[0]

    def _simulate_perturbed_paths_batch(
        self,
        control,
        mu0: torch.Tensor,
        logit_flow: torch.Tensor,
        epsilon: float,
        lambdas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = logit_flow.shape[-2] - 1
        batch_size = lambdas[..., 0, 0].numel()
        flat_lambdas = lambdas.reshape(batch_size, lambdas.shape[-2], self.n_states)
        if flat_lambdas.shape[1] < horizon:
            raise ValueError("lambdas must contain at least one perturbation per transition.")
        if logit_flow.ndim == 2:
            flat_logit_flow = logit_flow.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            flat_logit_flow = logit_flow.reshape(batch_size, horizon + 1, self.n_states)

        x = torch.empty(batch_size, horizon + 1, dtype=torch.long, device=self.config.device)
        y = torch.empty(batch_size, horizon + 1, dtype=torch.long, device=self.config.device)
        actions_x = torch.empty(batch_size, horizon, dtype=torch.long, device=self.config.device)
        actions_y = torch.empty(batch_size, horizon, dtype=torch.long, device=self.config.device)
        rewards_y = torch.zeros(batch_size, horizon, dtype=self.config.dtype, device=self.config.device)

        x[:, 0] = torch.multinomial(mu0, num_samples=batch_size, replacement=True)
        y[:, 0] = torch.multinomial(mu0, num_samples=batch_size, replacement=True)

        for t in range(horizon):
            mu_t = torch.softmax(flat_logit_flow[:, t], dim=-1)
            m_t = self.perturb_law(flat_logit_flow[:, t], epsilon, flat_lambdas[:, t])

            actions_x[:, t] = self.env.sample_actions_batch(control, t, x[:, t], mu_t)
            actions_y[:, t] = self.env.sample_actions_batch(control, t, y[:, t], m_t)
            rewards_y[:, t] = self.env.reward_batch(y[:, t], m_t, actions_y[:, t])

            x[:, t + 1] = self.env.sample_next_states_batch(x[:, t], actions_x[:, t], mu_t)
            y[:, t + 1] = self.env.sample_next_states_batch(y[:, t], actions_y[:, t], m_t)

        out_shape = (*lambdas.shape[:-2], horizon + 1)
        action_shape = (*lambdas.shape[:-2], horizon)
        return (
            x.reshape(out_shape),
            y.reshape(out_shape),
            actions_x.reshape(action_shape),
            actions_y.reshape(action_shape),
            rewards_y.reshape(action_shape),
        )

    def simulate_perturbed_path(
        self,
        control,
        mu0: torch.Tensor,
        logit_flow: torch.Tensor,
        epsilon: float,
        lambdas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self._simulate_perturbed_paths_batch(control, mu0, logit_flow, epsilon, lambdas.unsqueeze(0))
        return tuple(item[0] for item in out)

    def _estimate_logit_gradients_batch(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: int,
        batch_size: int,
        mu_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive.")

        mu_flow = self._population_flow_batch(control, mu0, flow_particles, horizon, batch_size, mu_flow)
        logit_flow = self.logit(mu_flow)
        param_dim = self.parameter_vector(control).numel()
        score_chunk_size = self._score_chunk_size(param_dim, batch_size * n)
        logit_grads = torch.zeros(
            batch_size,
            horizon + 1,
            self.n_states,
            param_dim,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        outer_indices = torch.arange(batch_size, device=self.config.device).repeat_interleave(n)

        for t in range(1, horizon + 1):
            repeated_logit_flow = logit_flow[:, : t + 1].repeat_interleave(n, dim=0)
            lambdas = torch.randn(batch_size * n, t, self.n_states, dtype=self.config.dtype, device=self.config.device)
            _, y, _, actions_y, _ = self._simulate_perturbed_paths_batch(
                control,
                mu0,
                repeated_logit_flow,
                epsilon,
                lambdas,
            )

            score_sum = torch.zeros(batch_size * n, param_dim, dtype=self.config.dtype, device=self.config.device)
            for s in range(t):
                m_s = self.perturb_law(repeated_logit_flow[:, s], epsilon, lambdas[:, s])
                repeated_logit_grads = logit_grads[:, s].repeat_interleave(n, dim=0)
                score_sum = score_sum + torch.einsum("bs,bsp->bp", lambdas[:, s], repeated_logit_grads) / epsilon
                score_sum = score_sum + self.env.policy_scores_batch(
                    control,
                    s,
                    m_s.detach(),
                    y[:, s],
                    actions_y[:, s],
                    chunk_size=score_chunk_size,
                ).reshape(batch_size * n, param_dim)

            grad_mu = torch.zeros(batch_size * self.n_states, param_dim, dtype=self.config.dtype, device=self.config.device)
            target = outer_indices * self.n_states + y[:, t]
            grad_mu.index_add_(0, target, score_sum)
            grad_mu = grad_mu.reshape(batch_size, self.n_states, param_dim)
            logit_grads[:, t] = grad_mu / n / mu_flow[:, t].unsqueeze(-1)

        return logit_grads

    def estimate_logit_gradients(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: Optional[int] = None,
        mu_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        if n <= 0:
            raise ValueError("n must be positive.")

        return self._estimate_logit_gradients_batch(
            control,
            mu0,
            epsilon,
            n,
            flow_particles,
            steps,
            batch_size=1,
            mu_flow=mu_flow,
        )[0]

    def _gradient_samples_batch(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: int,
        batch_size: int,
        mu_flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        main_mu_flow = self._population_flow_batch(control, mu0, flow_particles, horizon, batch_size, mu_flow)
        logit_flow = self.logit(main_mu_flow)
        logit_grads = self._estimate_logit_gradients_batch(
            control,
            mu0,
            epsilon,
            n,
            flow_particles,
            horizon,
            batch_size,
            mu_flow=main_mu_flow,
        )
        lambdas = self._sample_lambda_batch(batch_size, horizon)
        _, y, _, actions_y, rewards_y = self._simulate_perturbed_paths_batch(control, mu0, logit_flow, epsilon, lambdas)

        returns = torch.zeros(batch_size, horizon + 1, dtype=self.config.dtype, device=self.config.device)
        m_T = self.perturb_law(logit_flow[:, horizon], epsilon, lambdas[:, horizon])
        returns[:, horizon] = self.discount(horizon) * self.env.terminal_reward_batch(y[:, horizon], m_T)

        for t in range(horizon - 1, -1, -1):
            returns[:, t] = self.discount(t) * rewards_y[:, t] + returns[:, t + 1]

        param_dim = self.parameter_vector(control).numel()
        score_chunk_size = self._score_chunk_size(param_dim, batch_size)
        samples = torch.zeros(batch_size, param_dim, dtype=self.config.dtype, device=self.config.device)

        for t in range(horizon + 1):
            score = torch.einsum("bs,bsp->bp", lambdas[:, t], logit_grads[:, t]) / epsilon
            if t < horizon:
                m_t = self.perturb_law(logit_flow[:, t], epsilon, lambdas[:, t])
                score = score + self.env.policy_scores_batch(
                    control,
                    t,
                    m_t.detach(),
                    y[:, t],
                    actions_y[:, t],
                    chunk_size=score_chunk_size,
                ).reshape(batch_size, param_dim)
            samples = samples + score * returns[:, t].unsqueeze(1)

        return samples, {
            "returns": returns,
            "logit_gradients": logit_grads,
            "lambdas": lambdas,
            "y": y,
            "actions_y": actions_y,
        }

    def gradient_sample(
        self,
        control,
        mu0: torch.Tensor,
        epsilon: float,
        n: int,
        flow_particles: int,
        horizon: Optional[int] = None,
        mu_flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        steps = self.config.T if horizon is None else horizon
        samples, diag = self._gradient_samples_batch(
            control,
            mu0,
            epsilon,
            n,
            flow_particles,
            steps,
            batch_size=1,
            mu_flow=mu_flow,
        )
        return samples[0], {
            "returns": diag["returns"][0],
            "logit_gradients": diag["logit_gradients"][0],
            "lambdas": diag["lambdas"][0],
            "y": diag["y"][0],
            "actions_y": diag["actions_y"][0],
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
        mu_flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if N <= 0:
            raise ValueError("N must be positive.")

        param_dim = self.parameter_vector(control).numel()
        sample_chunk_size = self._sample_chunk_size(control, param_dim, N)
        samples = torch.zeros(N, param_dim, dtype=self.config.dtype, device=self.config.device)
        returns = torch.zeros(N, dtype=self.config.dtype, device=self.config.device)

        steps = self.config.T if horizon is None else horizon
        if mu_flow is not None and mu_flow.ndim == 3 and mu_flow.shape[0] != N:
            raise ValueError(f"Batched mu_flow has batch size {mu_flow.shape[0]}, expected {N}.")

        for start in range(0, N, sample_chunk_size):
            end = min(start + sample_chunk_size, N)
            chunk_mu_flow = None
            if mu_flow is not None:
                chunk_mu_flow = mu_flow[start:end] if mu_flow.ndim == 3 else mu_flow
            sample_chunk, diag_chunk = self._gradient_samples_batch(
                control,
                mu0,
                epsilon,
                n,
                flow_particles,
                steps,
                batch_size=end - start,
                mu_flow=chunk_mu_flow,
            )
            samples[start:end] = sample_chunk
            returns[start:end] = diag_chunk["returns"][:, 0]

        grad_flat = samples.mean(dim=0)
        return self.format_gradient(control, grad_flat), {
            "samples": samples,
            "returns": returns,
            "mean_return": returns.mean(),
            "std_return": returns.std(unbiased=False),
            "grad_norm": torch.linalg.norm(grad_flat),
        }
