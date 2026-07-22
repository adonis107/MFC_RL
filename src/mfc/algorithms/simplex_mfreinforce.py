from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple, Union

import torch


class SimplexPerturbedMFREINFORCE:
    def __init__(self, env):
        self.env = env
        self.config = env.config
        self.n_states = env.n_states

    def sample_q_batch(self, n: int) -> torch.Tensor:
        u = self.config.q_sigma * torch.randn(n, self.n_states - 1, dtype=self.config.dtype, device=self.config.device,)
        logits = torch.cat([u, torch.zeros(n, 1, dtype=self.config.dtype, device=self.config.device)], dim=-1,)
        q = torch.softmax(logits, dim=-1).clamp_min(self.config.q_clip)
        return q / q.sum(dim=-1, keepdim=True)

    def sample_q(self) -> torch.Tensor:
        return self.sample_q_batch(1).squeeze(0)

    def H(self, q: torch.Tensor) -> torch.Tensor:
        p = q.clamp_min(self.config.q_clip)
        p = p / p.sum(dim=-1, keepdim=True)
        p_first = p[..., :-1]
        p_last = p[..., -1:]
        z = torch.log(p_first / p_last)
        a = -z / (self.config.q_sigma ** 2)
        return a / p_first + a.sum(dim=-1, keepdim=True) / p_last - 1.0 / p_first + 1.0 / p_last

    def parameter_vector(self, control) -> torch.Tensor:
        if isinstance(control, torch.nn.Module): return torch.nn.utils.parameters_to_vector(control.parameters()).detach()
        return control.detach().reshape(-1)

    def format_gradient(self, control, grad_flat: torch.Tensor) -> torch.Tensor:
        if isinstance(control, torch.nn.Module): return grad_flat
        return grad_flat.reshape_as(control)

    def discount(self, t: int) -> float:
        return float(getattr(self.config, "gamma", 1.0) ** t)

    def _score_chunk_size(self, param_dim: int, batch_size: int) -> int:
        configured = getattr(self.config, "score_chunk_size", None)
        if configured is not None:
            return int(configured)
        if param_dim > 20_000:
            return min(batch_size, 32)
        return min(batch_size, 128)

    @torch.no_grad()
    def estimate_population_flow(self, control, mu0: torch.Tensor, n_particles: int, horizon: Optional[int] = None,) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        if n_particles <= 0: raise ValueError("n_particles must be positive.")

        states = torch.multinomial(mu0, num_samples=n_particles, replacement=True)
        flow = torch.zeros(steps + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)
        flow[0] = torch.nn.functional.one_hot(states, num_classes=self.n_states).to(self.config.dtype).mean(dim=0)

        for t in range(steps):
            mu_t = flow[t]
            actions = self.env.sample_actions_batch(control, t, states, mu_t)
            states = self.env.sample_next_states_batch(states, actions, mu_t)
            flow[t + 1] = torch.nn.functional.one_hot(states, num_classes=self.n_states).to(self.config.dtype).mean(dim=0)

        return flow

    def estimate_sensitivity(self, control, mu_flow: torch.Tensor, eta: float, n_aux: int) -> torch.Tensor:
        horizon = mu_flow.shape[0] - 1
        param_dim = self.parameter_vector(control).numel()
        score_chunk_size = self._score_chunk_size(param_dim, n_aux)
        x_aux = torch.zeros(n_aux, horizon + 1, dtype=torch.long, device=self.config.device)
        q_aux = torch.zeros(n_aux, horizon, self.n_states, dtype=self.config.dtype, device=self.config.device)
        psi = torch.zeros(n_aux, horizon, param_dim, dtype=self.config.dtype, device=self.config.device)

        x_aux[:, 0] = torch.multinomial(mu_flow[0], num_samples=n_aux, replacement=True)
        for t in range(horizon):
            q_t = self.sample_q_batch(n_aux)
            M_t = (1.0 - eta) * mu_flow[t].unsqueeze(0) + eta * q_t
            states_t = x_aux[:, t]
            actions_t = self.env.sample_actions_batch(control, t, states_t, M_t)
            x_aux[:, t + 1] = self.env.sample_next_states_batch(states_t, actions_t, M_t)
            q_aux[:, t] = q_t
            psi[:, t] = self.env.policy_scores_batch(
                control,
                t,
                M_t.detach(),
                states_t,
                actions_t,
                chunk_size=score_chunk_size,
            ).reshape(n_aux, param_dim)

        D_hat = torch.zeros(horizon + 1, self.n_states - 1, param_dim, dtype=self.config.dtype, device=self.config.device)
        for t in range(1, horizon + 1):
            H_path = self.H(q_aux[:, :t])
            correction = torch.einsum("rsl,slp->rp", H_path, D_hat[:t])
            psi_prefix = psi[:, :t].sum(dim=1)
            for k in range(self.n_states - 1):
                selected = x_aux[:, t] == k
                if selected.any():
                    values = psi_prefix[selected] - ((1.0 - eta) / eta) * correction[selected]
                    D_hat[t, k] = values.sum(dim=0) / n_aux

        return D_hat

    def gradient_estimate(self,
        control, mu_flow: torch.Tensor, D_hat: torch.Tensor, eps_law: float, B: int, baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        horizon = mu_flow.shape[0] - 1
        param_dim = self.parameter_vector(control).numel()
        score_chunk_size = self._score_chunk_size(param_dim, B)
        q_path = self.sample_q_batch(B * (horizon + 1)).reshape(B, horizon + 1, self.n_states)
        states = torch.zeros(B, horizon + 1, dtype=torch.long, device=self.config.device)
        actions = torch.zeros(B, horizon, dtype=torch.long, device=self.config.device)
        returns = torch.zeros(B, dtype=self.config.dtype, device=self.config.device)
        score_pol = torch.zeros(B, param_dim, dtype=self.config.dtype, device=self.config.device)

        states[:, 0] = torch.multinomial(mu_flow[0], num_samples=B, replacement=True)
        for t in range(horizon):
            M_t = (1.0 - eps_law) * mu_flow[t].unsqueeze(0) + eps_law * q_path[:, t]
            states_t = states[:, t]
            actions_t = self.env.sample_actions_batch(control, t, states_t, M_t)
            actions[:, t] = actions_t
            returns = returns + self.discount(t) * self.env.reward_batch(states_t, M_t, actions_t)
            score_pol = score_pol + self.env.policy_scores_batch(
                control,
                t,
                M_t.detach(),
                states_t,
                actions_t,
                chunk_size=score_chunk_size,
            ).reshape(B, param_dim)
            states[:, t + 1] = self.env.sample_next_states_batch(states_t, actions_t, M_t)

        M_T = (1.0 - eps_law) * mu_flow[horizon].unsqueeze(0) + eps_law * q_path[:, horizon]
        returns = returns + self.discount(horizon) * self.env.terminal_reward_batch(states[:, horizon], M_T)

        score_pert = torch.einsum("btk,tkp->bp", self.H(q_path), D_hat)
        score_pert = -((1.0 - eps_law) / eps_law) * score_pert
        scores = score_pol + score_pert

        if baseline == "batch_mean":
            b0 = returns.mean()
        elif baseline is None:
            b0 = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        else:
            b0 = torch.tensor(float(baseline), dtype=self.config.dtype, device=self.config.device)

        grad_flat = ((returns - b0).unsqueeze(1) * scores).mean(dim=0)
        grad_hat = self.format_gradient(control, grad_flat)
        return grad_hat, {
            "returns": returns,
            "scores": scores,
            "baseline": b0,
            "mean_return": returns.mean(),
            "std_return": returns.std(unbiased=False),
            "grad_norm": torch.linalg.norm(grad_flat),
        }
