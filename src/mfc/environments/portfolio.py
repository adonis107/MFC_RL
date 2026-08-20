"""
Mean-variance portfolio selection mean-field control benchmark.

Reference: files/reference/continuous_benchmarks.tex, Sec. "Mean-Variance
Portfolio Selection". Like `mfc.environments.lq`, the perturbed objective
J^lambda(theta), its exact policy gradient, and the unperturbed optimal
policy are all available in closed form — here even though the terminal
wealth law is generally non-Gaussian (only its first two moments are used,
and they close exactly), unlike LQ's genuinely Gaussian state law.

Model. Wealth X_t in R, monetary amount invested in the risky asset
alpha_t in R (the rest sits in the risk-free asset). X_{t+1} = s_t*X_t +
R_{t+1}*alpha_t, where R_{t+1} (the risky asset's excess return) is
independent across t, independent of X_0 and the policy noise, with only its
first two moments (r_bar_t, sigma_R_t^2, h_t := r_bar_t^2+sigma_R_t^2) fixed
by config — the return law itself is a runtime choice (`sample_returns`
supports "gaussian" and a variance-matched "student_t" for the reference's
own robustness experiment; the exact objective/gradient are unaffected by
this choice, since they only depend on the first two moments). There is no
running reward; the terminal reward is g(x,m) = x - chi*(x-mbar)^2, so
J^0(theta) = E[X_T] - chi*Var(X_T) — the precommitment mean-variance
criterion, a REWARD to maximize (unlike LQ's cost-to-minimize convention;
see `mfc.algorithms.portfolio`'s module docstring for the resulting sign
convention).

Policy: pi_t^theta(.|x,m) = N(k_t*(x-mbar) + l_t, tau_t^2), theta =
((k_t,l_t))_{t=0}^{T-1} — the same genuinely time-indexed, finite-horizon
(T,2) parametrization as `mfc.environments.lq.LQ`, with the same
consequence: there is no separate validation horizon (see
`mfc.environments.lq`'s module docstring for why).

Perturbation: identical affine randomization of the mean-field argument as
LQ (T_t^lambda(x) = (1+lambda*zeta_t)x + lambda*beta_t), applied to the
wealth law's mean only (there is no "variance" of the mean-field argument
used anywhere in the policy/dynamics here, unlike LQ's transition kernel).

Baseline parameters (reference "Training and evaluation"): T=10, X_0 ~
N(1,0.04), s_t=1, r_bar_t=0.02, sigma_R_t=0.08, chi=10, tau_t=0.02, rho=1 —
used directly as `PortfolioConfig`'s defaults (unlike LQ, whose reference
gives no baseline numeric case, this benchmark's own reference values are
used as-is, not invented).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PortfolioConfig:
    """Model parameters, matching the reference's own baseline case exactly
    (see module docstring)."""

    s: float = 1.0  # risk-free gross return
    r_bar: float = 0.02  # mean excess return of the risky asset
    sigma_R: float = 0.08  # excess-return std
    chi: float = 10.0  # variance-aversion weight in the terminal reward

    tau: float = 0.02  # policy exploration std
    rho: float = 1.0  # perturbation std

    mu0: float = 1.0  # initial wealth mean
    Sigma0: float = 0.04  # initial wealth variance

    return_distribution: str = "gaussian"  # "gaussian" or "student_t" (reference's robustness experiment)

    T: int = 10  # default horizon


class Portfolio:
    """Everything is a pure function of `theta` (shape (T,2): theta[t] =
    (k_t, l_t)) and the perturbation scale `lambda` — see
    `mfc.environments.lq.LQ`'s identical structure and rationale."""

    def __init__(
        self,
        config: PortfolioConfig = PortfolioConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device

    def _h(self) -> float:
        """h_t = r_bar_t^2 + sigma_R_t^2 = E[R_{t+1}^2]; time-homogeneous here."""
        return self.config.r_bar**2 + self.config.sigma_R**2

    def _A(self, k: torch.Tensor) -> torch.Tensor:
        """A_t(k) = s_t^2 + 2*s_t*r_bar_t*k + h_t*k^2 (eq. portfolio-mean-recursion's helper)."""
        cfg = self.config
        return cfg.s**2 + 2.0 * cfg.s * cfg.r_bar * k + self._h() * k**2

    def forward_moments(self, theta: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate (mu_t^{theta,lambda}, Sigma_t^{theta,lambda}), t=0,...,T
        (eqs. portfolio-mean-recursion/portfolio-variance-recursion).
        Returns two (T+1,) tensors. Differentiable in `theta`."""
        cfg = self.config
        T = theta.shape[0]
        h = self._h()
        lam2rho2 = (lam * cfg.rho) ** 2
        mu = [torch.as_tensor(cfg.mu0, dtype=self.dtype, device=self.device)]
        Sigma = [torch.as_tensor(cfg.Sigma0, dtype=self.dtype, device=self.device)]
        for t in range(T):
            k_t, l_t = theta[t, 0], theta[t, 1]
            mu_t, Sigma_t = mu[-1], Sigma[-1]
            mu.append(cfg.s * mu_t + cfg.r_bar * l_t)
            Sigma.append(self._A(k_t) * Sigma_t + h * cfg.tau**2 + cfg.sigma_R**2 * l_t**2 + h * k_t**2 * lam2rho2 * (mu_t**2 + 1.0))
        return torch.stack(mu), torch.stack(Sigma)

    def exact_objective(self, theta: torch.Tensor, lam: float) -> torch.Tensor:
        """Closed-form J^lambda(theta) = mu_T - chi*[Sigma_T + lambda^2*rho^2*(mu_T^2+1)]
        (eq. portfolio-perturbed-objective). A reward to maximize. Scalar, differentiable in `theta`."""
        cfg = self.config
        T = theta.shape[0]
        lam2rho2 = (lam * cfg.rho) ** 2
        mu, Sigma = self.forward_moments(theta, lam)
        mu_T, Sigma_T = mu[T], Sigma[T]
        return mu_T - cfg.chi * (Sigma_T + lam2rho2 * (mu_T**2 + 1.0))

    def exact_gradient(self, theta: torch.Tensor, lam: float) -> torch.Tensor:
        """Exact grad_theta J^lambda(theta) in O(T) via the forward-backward
        adjoint sweep (eqs. portfolio-gradient-k/portfolio-gradient-l).
        Returns shape (T,2). `theta` need not require grad."""
        cfg = self.config
        T = theta.shape[0]
        h = self._h()
        lam2rho2 = (lam * cfg.rho) ** 2
        mu, Sigma = self.forward_moments(theta.detach(), lam)

        p_next = 1.0 - 2.0 * cfg.chi * lam2rho2 * mu[T]
        q_next = torch.as_tensor(-cfg.chi, dtype=self.dtype, device=self.device)

        grad = torch.zeros_like(theta)
        for t in range(T - 1, -1, -1):
            k_t, l_t = theta[t, 0], theta[t, 1]
            mu_t, Sigma_t = mu[t], Sigma[t]

            grad[t, 0] = 2.0 * q_next * ((cfg.s * cfg.r_bar + h * k_t) * Sigma_t + h * k_t * lam2rho2 * (mu_t**2 + 1.0))
            grad[t, 1] = cfg.r_bar * p_next + 2.0 * cfg.sigma_R**2 * l_t * q_next

            p_t = cfg.s * p_next + 2.0 * h * k_t**2 * lam2rho2 * mu_t * q_next
            q_t = self._A(k_t) * q_next
            p_next, q_next = p_t, q_t

        return grad

    def optimal_theta(self, T: int | None = None) -> torch.Tensor:
        """The unperturbed (lambda=0) optimal theta*, shape (T,2) (Sec.
        "Optimal policy at lambda=0"): k_t^* = -s_t*r_bar_t/h_t (constant
        here, time-homogeneous config), l_t^* = r_bar_t/(2*chi*sigma_R_t^2)
        * prod_{j=t+1}^{T-1} 1/(s_j*(1-B_j)), B_t := r_bar_t^2/h_t."""
        cfg = self.config
        T = cfg.T if T is None else T
        h = self._h()
        B = cfg.r_bar**2 / h
        k_star = -cfg.s * cfg.r_bar / h

        theta_star = torch.zeros(T, 2, dtype=self.dtype, device=self.device)
        prod = 1.0
        for t in range(T - 1, -1, -1):
            theta_star[t, 0] = k_star
            theta_star[t, 1] = (cfg.r_bar / (2.0 * cfg.chi * cfg.sigma_R**2)) * prod
            prod = prod / (cfg.s * (1.0 - B))
        return theta_star

    def sample_returns(self, B: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample R_{t+1} ~ the configured return law, matching (r_bar,
        sigma_R) exactly regardless of distribution (reference's robustness
        experiment: "replacing the Gaussian return distribution with a
        centered and rescaled Student distribution with five degrees of
        freedom, keeping r_bar_t and sigma_{R,t}^2 unchanged" — since
        `torch.distributions.StudentT` has no `generator` support, the t_5
        draw is built from `torch.randn` alone via the standard
        normal-over-sqrt(chi-square/5) construction, with chi-square(5)
        itself a sum of 5 squared standard normals). Returns shape (B,)."""
        cfg = self.config
        dtype, device = self.dtype, self.device
        if cfg.return_distribution == "gaussian":
            z = torch.randn(B, dtype=dtype, device=device, generator=generator)
        elif cfg.return_distribution == "student_t":
            nu = 5
            numerator = torch.randn(B, dtype=dtype, device=device, generator=generator)
            chi2 = (torch.randn(nu, B, dtype=dtype, device=device, generator=generator) ** 2).sum(dim=0)
            t_std = numerator / torch.sqrt(chi2 / nu)  # standard Student-t(nu), Var = nu/(nu-2)
            z = t_std / math.sqrt(nu / (nu - 2))  # rescale to unit variance
        else:
            raise ValueError(f"unknown return_distribution {cfg.return_distribution!r}; available: gaussian, student_t")
        return cfg.r_bar + cfg.sigma_R * z

    def rollout(
        self,
        theta: torch.Tensor,
        lam: float,
        B: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Simulate B i.i.d. wealth trajectories under the lambda-perturbed
        dynamics, using the deterministic exact mean flow mu_t^{theta,0} as
        the population argument at every step (see `LQ.rollout`'s identical
        design and rationale). `theta` is fully detached throughout (see
        `LQ.rollout`'s docstring for why sampling must not reuse the
        attached theta). Returns X (T+1,B), alpha (T,B), mu_hat (T+1,) — the
        last entry, mu_hat_T, is only used to evaluate the terminal reward."""
        cfg = self.config
        dtype, device = self.dtype, self.device
        T = theta.shape[0]
        theta = theta.detach()
        mu_flow, _ = self.forward_moments(theta, lam=0.0)

        X = [cfg.mu0 + (cfg.Sigma0**0.5) * torch.randn(B, dtype=dtype, device=device, generator=generator)]
        alpha, mu_hat = [], []
        for t in range(T):
            zeta = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
            beta = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
            mu_hat_t = (1.0 + lam * zeta) * mu_flow[t] + lam * beta
            mu_hat.append(mu_hat_t)

            k_t, l_t = theta[t, 0], theta[t, 1]
            eta = torch.randn(B, dtype=dtype, device=device, generator=generator)
            alpha_t = k_t * (X[t] - mu_hat_t) + l_t + cfg.tau * eta
            alpha.append(alpha_t)

            R = self.sample_returns(B, generator=generator)
            X.append(cfg.s * X[t] + R * alpha_t)

        zeta_T = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
        beta_T = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
        mu_hat.append((1.0 + lam * zeta_T) * mu_flow[T] + lam * beta_T)  # the terminal reward g(x,m) needs mu_hat_T too (reference's (zeta_t,beta_t), t=0,...,T)

        return {"X": torch.stack(X), "alpha": torch.stack(alpha), "mu_hat": torch.stack(mu_hat)}

    def init_theta(self, T: int | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """theta_0=0 (k_t=l_t=0), the reference's own initialization ("The
        policy parameters are initialized at k_t=l_t=0"). `generator` is
        accepted (unused: deterministic) for the `env.init_theta
        (generator=...)` contract shared by every environment in this repo."""
        T = self.config.T if T is None else T
        return torch.zeros(T, 2, dtype=self.dtype, device=self.device)
