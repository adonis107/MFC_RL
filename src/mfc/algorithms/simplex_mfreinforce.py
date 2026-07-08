from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Union

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
        p = p / p.sum()
        p_first = p[:-1]
        p_last = p[-1]
        z = torch.log(p_first / p_last)
        a = -z / (self.config.q_sigma ** 2)
        return a / p_first + a.sum() / p_last - 1.0 / p_first + 1.0 / p_last

    def parameter_vector(self, control) -> torch.Tensor:
        if isinstance(control, torch.nn.Module): return torch.nn.utils.parameters_to_vector(control.parameters()).detach()
        return control.detach().reshape(-1)

    def format_gradient(self, control, grad_flat: torch.Tensor) -> torch.Tensor:
        if isinstance(control, torch.nn.Module): return grad_flat
        return grad_flat.reshape_as(control)

    def discount(self, t: int) -> float:
        return float(getattr(self.config, "gamma", 1.0) ** t)

    @torch.no_grad()
    def estimate_population_flow(self, control, mu0: torch.Tensor, n_particles: int, horizon: Optional[int] = None,) -> torch.Tensor:
        steps = self.config.T if horizon is None else horizon
        if n_particles <= 0: raise ValueError("n_particles must be positive.")

        states = torch.multinomial(mu0, num_samples=n_particles, replacement=True)
        flow = torch.zeros(steps + 1, self.n_states, dtype=self.config.dtype, device=self.config.device)
        flow[0] = torch.bincount(states, minlength=self.n_states).to(self.config.dtype) / n_particles

        for t in range(steps):
            mu_t = flow[t]
            next_states = torch.empty_like(states)
            for r in range(n_particles):
                x = int(states[r].item())
                a = self.env.sample_action(control, t, x, mu_t)
                next_states[r] = self.env.sample_next_state(x, a, mu_t)
            states = next_states
            flow[t + 1] = torch.bincount(states, minlength=self.n_states).to(self.config.dtype) / n_particles

        return flow

    def estimate_sensitivity(self, control, mu_flow: torch.Tensor, eta: float, n_aux: int) -> torch.Tensor:
        horizon = mu_flow.shape[0] - 1
        param_dim = self.parameter_vector(control).numel()
        x_aux = torch.zeros(n_aux, horizon + 1, dtype=torch.long, device=self.config.device)
        q_aux = torch.zeros(n_aux, horizon, self.n_states, dtype=self.config.dtype, device=self.config.device)
        psi = torch.zeros(n_aux, horizon, param_dim, dtype=self.config.dtype, device=self.config.device)

        for r in range(n_aux):
            x_aux[r, 0] = int(torch.multinomial(mu_flow[0], num_samples=1).item())
            for t in range(horizon):
                q_t = self.sample_q()
                M_t = (1.0 - eta) * mu_flow[t] + eta * q_t
                x = int(x_aux[r, t].item())
                a = self.env.sample_action(control, t, x, M_t)
                x_aux[r, t + 1] = self.env.sample_next_state(x, a, M_t)
                q_aux[r, t] = q_t
                psi[r, t] = self.env.policy_score(control, t, M_t.detach(), x, a).detach().reshape(-1)

        D_hat = torch.zeros(horizon + 1, self.n_states - 1, param_dim, dtype=self.config.dtype, device=self.config.device)
        for t in range(1, horizon + 1):
            for k in range(self.n_states - 1):
                acc = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)
                for r in range(n_aux):
                    if int(x_aux[r, t].item()) != k:
                        continue
                    correction = torch.zeros_like(acc)
                    for s in range(t):
                        H_s = self.H(q_aux[r, s])
                        for ell in range(self.n_states - 1):
                            correction += H_s[ell] * D_hat[s, ell]
                    acc += psi[r, :t].sum(dim=0) - ((1.0 - eta) / eta) * correction
                D_hat[t, k] = acc / n_aux

        return D_hat

    def gradient_estimate(self,
        control, mu_flow: torch.Tensor, D_hat: torch.Tensor, eps_law: float, B: int, baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        horizon = mu_flow.shape[0] - 1
        param_dim = self.parameter_vector(control).numel()
        returns = torch.zeros(B, dtype=self.config.dtype, device=self.config.device)
        scores = torch.zeros(B, param_dim, dtype=self.config.dtype, device=self.config.device)

        for b in range(B):
            x = int(torch.multinomial(mu_flow[0], num_samples=1).item())
            q_path: List[torch.Tensor] = []
            score_pol = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)
            total_return = torch.zeros((), dtype=self.config.dtype, device=self.config.device)

            for t in range(horizon):
                q_t = self.sample_q()
                M_t = (1.0 - eps_law) * mu_flow[t] + eps_law * q_t
                a = self.env.sample_action(control, t, x, M_t)
                total_return += self.discount(t) * self.env.reward(x, M_t, a)
                score_pol += self.env.policy_score(control, t, M_t.detach(), x, a).detach().reshape(-1)
                x = self.env.sample_next_state(x, a, M_t)
                q_path.append(q_t)

            q_T = self.sample_q()
            M_T = (1.0 - eps_law) * mu_flow[horizon] + eps_law * q_T
            total_return += self.discount(horizon) * self.env.terminal_reward(x, M_T)
            q_path.append(q_T)

            score_pert = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)
            for t in range(horizon + 1):
                H_t = self.H(q_path[t])
                for k in range(self.n_states - 1):
                    score_pert += H_t[k] * D_hat[t, k]
            score_pert *= -((1.0 - eps_law) / eps_law)

            returns[b] = total_return
            scores[b] = score_pol + score_pert

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
