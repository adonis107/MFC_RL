from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple, Union

import torch


class SimplexPerturbedMFREINFORCE:
    def __init__(self, env, algorithm_config: Optional[Dict[str, object]] = None):
        self.env = env
        self.config = env.config
        self.n_states = env.n_states
        self.algorithm_config = dict(algorithm_config or {})

    def sample_q_batch(self, n: int) -> torch.Tensor:
        if bool(self.algorithm_config.get("antithetic", False)) and n > 1:
            half = n // 2
            base = self.config.q_sigma * torch.randn(half, self.n_states - 1, dtype=self.config.dtype, device=self.config.device)
            pieces = [base, -base]
            if n % 2:
                pieces.append(self.config.q_sigma * torch.randn(1, self.n_states - 1, dtype=self.config.dtype, device=self.config.device))
            u = torch.cat(pieces, dim=0)
        else:
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

    def _keep_score_diagnostics(self, keep_score_diagnostics: Optional[bool]) -> bool:
        if keep_score_diagnostics is None:
            return bool(getattr(self.config, "keep_score_diagnostics", False))
        return bool(keep_score_diagnostics)

    def _weighted_policy_score_sums(
        self,
        control,
        t: int,
        mu: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if hasattr(self.env, "weighted_policy_score_sums"):
            return self.env.weighted_policy_score_sums(
                control,
                t,
                mu,
                states,
                actions,
                weights,
                chunk_size=chunk_size,
            )

        states_flat = states.reshape(-1)
        scores = self.env.policy_scores_batch(
            control,
            t,
            mu,
            states,
            actions,
            chunk_size=chunk_size,
        ).reshape(states_flat.numel(), -1)
        weights_2d = weights.to(dtype=self.config.dtype, device=self.config.device).reshape(-1, states_flat.numel())
        sums = weights_2d @ scores
        if weights.shape == states.shape:
            return sums.squeeze(0)
        return sums.reshape(*weights.shape[:-1], scores.shape[-1])

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
        actions_aux = torch.zeros(n_aux, horizon, dtype=torch.long, device=self.config.device)
        q_aux = torch.zeros(n_aux, horizon, self.n_states, dtype=self.config.dtype, device=self.config.device)

        x_aux[:, 0] = torch.multinomial(mu_flow[0], num_samples=n_aux, replacement=True)
        for t in range(horizon):
            q_t = self.sample_q_batch(n_aux)
            M_t = (1.0 - eta) * mu_flow[t].unsqueeze(0) + eta * q_t
            states_t = x_aux[:, t]
            actions_t = self.env.sample_actions_batch(control, t, states_t, M_t)
            actions_aux[:, t] = actions_t
            x_aux[:, t + 1] = self.env.sample_next_states_batch(states_t, actions_t, M_t)
            q_aux[:, t] = q_t

        D_hat = torch.zeros(horizon + 1, self.n_states - 1, param_dim, dtype=self.config.dtype, device=self.config.device)
        state_groups = torch.arange(self.n_states - 1, device=self.config.device).unsqueeze(1)
        correction_scale = (1.0 - eta) / eta
        for t in range(1, horizon + 1):
            group_weights = (x_aux[:, t].unsqueeze(0) == state_groups).to(self.config.dtype)
            score_prefix_sums = torch.zeros(
                self.n_states - 1,
                param_dim,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            for s in range(t):
                M_s = (1.0 - eta) * mu_flow[s].unsqueeze(0) + eta * q_aux[:, s]
                score_prefix_sums = score_prefix_sums + self._weighted_policy_score_sums(
                    control,
                    s,
                    M_s.detach(),
                    x_aux[:, s],
                    actions_aux[:, s],
                    group_weights,
                    chunk_size=score_chunk_size,
                ).reshape(self.n_states - 1, param_dim)

            H_path = self.H(q_aux[:, :t])
            grouped_H = torch.einsum("kr,rsl->ksl", group_weights, H_path)
            correction_sums = torch.einsum("ksl,slp->kp", grouped_H, D_hat[:t])
            D_hat[t] = (score_prefix_sums - correction_scale * correction_sums) / n_aux

        return D_hat

    def gradient_estimate(self,
        control,
        mu_flow: torch.Tensor,
        D_hat: torch.Tensor,
        eps_law: float,
        B: int,
        baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
        keep_score_diagnostics: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        horizon = mu_flow.shape[0] - 1
        param_dim = self.parameter_vector(control).numel()
        score_chunk_size = self._score_chunk_size(param_dim, B)
        keep_scores = self._keep_score_diagnostics(keep_score_diagnostics)
        q_path = self.sample_q_batch(B * (horizon + 1)).reshape(B, horizon + 1, self.n_states)
        states = torch.zeros(B, horizon + 1, dtype=torch.long, device=self.config.device)
        actions = torch.zeros(B, horizon, dtype=torch.long, device=self.config.device)
        returns = torch.zeros(B, dtype=self.config.dtype, device=self.config.device)

        states[:, 0] = torch.multinomial(mu_flow[0], num_samples=B, replacement=True)
        for t in range(horizon):
            M_t = (1.0 - eps_law) * mu_flow[t].unsqueeze(0) + eps_law * q_path[:, t]
            states_t = states[:, t]
            actions_t = self.env.sample_actions_batch(control, t, states_t, M_t)
            actions[:, t] = actions_t
            returns = returns + self.discount(t) * self.env.reward_batch(states_t, M_t, actions_t)
            states[:, t + 1] = self.env.sample_next_states_batch(states_t, actions_t, M_t)

        M_T = (1.0 - eps_law) * mu_flow[horizon].unsqueeze(0) + eps_law * q_path[:, horizon]
        returns = returns + self.discount(horizon) * self.env.terminal_reward_batch(states[:, horizon], M_T)

        if baseline == "batch_mean":
            b0 = returns.mean()
        elif baseline is None:
            b0 = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        else:
            b0 = torch.tensor(float(baseline), dtype=self.config.dtype, device=self.config.device)

        centered_returns = returns - b0
        policy_score_sum = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)
        score_pol = None
        if keep_scores:
            score_pol = torch.zeros(B, param_dim, dtype=self.config.dtype, device=self.config.device)

        for t in range(horizon):
            M_t = (1.0 - eps_law) * mu_flow[t].unsqueeze(0) + eps_law * q_path[:, t]
            policy_score_sum = policy_score_sum + self._weighted_policy_score_sums(
                control,
                t,
                M_t.detach(),
                states[:, t],
                actions[:, t],
                centered_returns,
                chunk_size=score_chunk_size,
            ).reshape(param_dim)
            if keep_scores:
                score_pol = score_pol + self.env.policy_scores_batch(
                    control,
                    t,
                    M_t.detach(),
                    states[:, t],
                    actions[:, t],
                    chunk_size=score_chunk_size,
                ).reshape(B, param_dim)

        H_path = self.H(q_path)
        weighted_H = torch.einsum("b,btk->tk", centered_returns, H_path)
        perturbation_score_sum = -((1.0 - eps_law) / eps_law) * torch.einsum("tk,tkp->p", weighted_H, D_hat)
        grad_flat = (policy_score_sum + perturbation_score_sum) / B
        grad_hat = self.format_gradient(control, grad_flat)
        diag = {
            "returns": returns,
            "baseline": b0,
            "mean_return": returns.mean(),
            "std_return": returns.std(unbiased=False),
            "grad_norm": torch.linalg.norm(grad_flat),
        }
        if keep_scores:
            score_pert = -((1.0 - eps_law) / eps_law) * torch.einsum("btk,tkp->bp", H_path, D_hat)
            diag["scores"] = score_pol + score_pert
        return grad_hat, diag

    def complete_gradient_estimate(
        self,
        control,
        mu_flow: torch.Tensor,
        eps_law: float,
        B: int,
        n_aux: int,
        eta: Optional[float] = None,
        baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
        keep_score_diagnostics: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eta_value = eps_law if eta is None else float(eta)
        D_hat = self.estimate_sensitivity(control, mu_flow, eta_value, n_aux)
        grad_hat, diag = self.gradient_estimate(
            control,
            mu_flow,
            D_hat,
            eps_law,
            B,
            baseline=baseline,
            keep_score_diagnostics=keep_score_diagnostics,
        )
        diag = dict(diag)
        diag["sensitivity"] = D_hat
        diag["eta"] = torch.tensor(eta_value, dtype=self.config.dtype, device=self.config.device)
        diag["lambda"] = torch.tensor(float(eps_law), dtype=self.config.dtype, device=self.config.device)
        diag["main_trajectories"] = torch.tensor(int(B), device=self.config.device)
        diag["auxiliary_trajectories"] = torch.tensor(int(n_aux), device=self.config.device)
        return grad_hat, diag
