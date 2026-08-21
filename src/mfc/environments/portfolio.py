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
criterion, a REWARD to maximize (unlike LQ's cost-to-minimize convention) —
hence `MAXIMIZE = True`, which is where `mfc.algorithms.continuous.train`
reads the resulting sign convention from.

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
N(1,0.04), s_t=1, r_bar_t=0.02, sigma_R_t=0.08, chi=10, tau_t=0.02, rho=1.
This repo keeps that baseline except for `chi`, which is strengthened in
the default config so the terminal mean-field variance penalty is more
visible in simplex-vs-REINFORCE comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PortfolioConfig:
    """Model parameters. Defaults match the reference baseline except for
    `chi`, which is strengthened to emphasize mean-field coupling."""

    s: float = 1.0  # risk-free gross return
    r_bar: float = 0.02  # mean excess return of the risky asset
    sigma_R: float = 0.08  # excess-return std
    chi: float = 20.0  # variance-aversion weight in the terminal reward; reference baseline is 10.0

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

    MAXIMIZE = True  # J^lambda here is a reward: `mfc.algorithms.continuous.train` ascends

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

    def policy_features(self, t: int, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        phi_t(x,m), the feature vector of the Gaussian policy
        pi_t^theta(.|x,m) = N(theta_t . phi_t(x,m), tau^2): the policy mean
        is k_t*(x-mbar) + l_t, so phi_t(x,mbar) = (x-mbar, 1) — see
        `mfc.environments.lq.LQ.policy_features` for the shared contract
        (`t` unused: time-homogeneous features). Returns shape (*x.shape, 2).
        """
        del t
        return torch.stack([x - M, torch.ones_like(x - M)], dim=-1)

    def rollout(
        self,
        theta: torch.Tensor,
        lam: float,
        B: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Simulate B i.i.d. replicas of the lambda-perturbed wealth
        dynamics, each with its own perturbation draw at every t (see
        `mfc.environments.lq.LQ.rollout`'s identical design, rationale and
        detached-theta contract). Returns X (T+1,B), alpha (T,B), M (T+1,B),
        xi (T+1,B), mu (T+1,), running (T,B), terminal (B,) — same keys as
        `LQ.rollout`, but `running`/`terminal` are *rewards* here (this
        environment's own sign convention, matching `exact_objective`), and
        `running` is identically zero: the mean-variance criterion is purely
        terminal, g(x,m) = x - chi*(x-mbar)^2.
        """
        cfg = self.config
        dtype, device = self.dtype, self.device
        T = theta.shape[0]
        theta = theta.detach()
        mu_flow, _ = self.forward_moments(theta, lam=0.0)

        X = [cfg.mu0 + (cfg.Sigma0**0.5) * torch.randn(B, dtype=dtype, device=device, generator=generator)]
        alpha, M, xi = [], [], []
        for t in range(T + 1):
            mu_t = mu_flow[t]
            zeta = cfg.rho * torch.randn(B, dtype=dtype, device=device, generator=generator)
            beta = cfg.rho * torch.randn(B, dtype=dtype, device=device, generator=generator)
            M.append((1.0 + lam * zeta) * mu_t + lam * beta)
            xi.append((zeta * mu_t + beta) / (cfg.rho * torch.sqrt(mu_t**2 + 1.0)))
            if t == T:
                break

            k_t, l_t = theta[t, 0], theta[t, 1]
            eta = torch.randn(B, dtype=dtype, device=device, generator=generator)
            alpha_t = k_t * (X[t] - M[t]) + l_t + cfg.tau * eta
            alpha.append(alpha_t)

            R = self.sample_returns(B, generator=generator)
            X.append(cfg.s * X[t] + R * alpha_t)

        return {
            "X": torch.stack(X),
            "alpha": torch.stack(alpha),
            "M": torch.stack(M),
            "xi": torch.stack(xi),
            "mu": mu_flow,
            "running": torch.zeros(T, B, dtype=dtype, device=device),
            "terminal": X[T] - cfg.chi * (X[T] - M[T]) ** 2,
        }

    def init_theta(self, T: int | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """theta_0=0 (k_t=l_t=0), the reference's own initialization ("The
        policy parameters are initialized at k_t=l_t=0"). `generator` is
        accepted (unused: deterministic) for the `env.init_theta
        (generator=...)` contract shared by every environment in this repo."""
        T = self.config.T if T is None else T
        return torch.zeros(T, 2, dtype=self.dtype, device=self.device)
