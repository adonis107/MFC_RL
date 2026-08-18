"""
One-dimensional linear-quadratic (LQ) mean-field control benchmark.

Reference: files/reference/LQ_framework.tex (the full derivation, matching
files/reference/continuous_benchmarks.tex, Sec. "Linear-Quadratic
Validation"). Unlike every other benchmark in this repo, the perturbed
objective J^lambda(theta), its exact policy gradient, and the unperturbed
(lambda=0) optimal policy are all available in closed form, so this module
implements the reference's forward moment recursion, closed-form objective,
adjoint gradient, and Riccati optimum directly — no Monte Carlo estimation
is needed to train "exactly". A stochastic rollout is provided separately,
only for the classical-REINFORCE ablation and trajectory diagnostics
(see `mfc.algorithms.lq`).

Model (LQ_framework.tex, Secs. 1-3). State and action spaces are R. The
initial law is X_0 ~ N(mu0, Sigma0); the state law m_t^theta is Gaussian at
every t, with mean/variance (mu_t, Sigma_t) propagated deterministically.
The perturbation randomizes the mean-field argument via the affine map
T_t^lambda(x) = (1+lambda*zeta_t)x + lambda*beta_t, (zeta_t,beta_t) ~
N(0,rho^2) x N(0,rho^2). The policy is Gaussian feedback,
pi_t^theta(.|x,m) = N(theta_t^1*x + theta_t^2*mbar, tau^2); the transition
kernel is P_t(.|x,a,m) = N(a*x + b*alpha + c*mbar, sigma^2) (a,b,c are the
model's own transition coefficients, unrelated to the RL discount factor
used elsewhere in this repo — this benchmark has no discounting). Running
and terminal costs are r_t(x,m,alpha) = q*x^2 + r*alpha^2 + kappa*mbar^2 and
g(x,m) = q_T*x^2 + kappa_T*mbar^2 (`kappa`/`kappa_T` are the reference's own
gamma/gamma_T, renamed to avoid clashing with this repo's discount `gamma`
and mirroring `mfc.environments.twostate.TwoStateConfig.kappa`'s own use of
"kappa" for a coupling weight).

theta has shape (T, 2): theta[t] = (theta_t^1, theta_t^2), a genuinely
time-indexed finite-horizon parametrization (unlike every other benchmark's
stationary/NN policy) — there is no way to run a trained theta at a horizon
other than the one it was trained for, so (unlike cybersecurity/advertising)
this benchmark has no separate validation horizon.

Everything below is a *cost* to minimize (matching the reference's own
notation directly), not a reward to maximize like the rest of this repo —
see `mfc.algorithms.lq`'s module docstring for the resulting sign
convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LQConfig:
    """Model parameters. Defaults are chosen so the mean-field coupling
    actually matters: `c` is comparable to `a` (the population feeds back
    into the transition almost as strongly as the agent's own state), and
    `kappa`/`kappa_T` are comparable to (or larger than) `q`/`q_T` (the
    population's mean is penalized at least as heavily as the agent's own
    deviation), with a nonzero `mu0` so there is a real population mean to
    control from the start. See `tests/test_lq.py` for a direct numeric
    check that the mean-field coupling changes the optimal policy and value."""

    a: float = 0.9  # self transition coefficient
    b: float = 1.0  # control effectiveness
    c: float = 0.8  # mean-field feedback in the transition kernel
    sigma: float = 0.3  # state noise std

    q: float = 1.0  # running state cost
    r: float = 0.1  # running control cost
    q_T: float = 5.0  # terminal state cost
    kappa: float = 2.0  # running mean-field coupling cost (reference's gamma)
    kappa_T: float = 5.0  # terminal mean-field coupling cost (reference's gamma_T)

    tau: float = 0.2  # policy exploration std
    rho: float = 0.3  # perturbation std

    mu0: float = 2.0  # initial mean
    Sigma0: float = 1.0  # initial variance

    T: int = 5  # default horizon


class LQ:
    """Everything is a pure function of `theta` (shape (T,2)) and the
    perturbation scale `lambda`; there is no discrete state space, so this
    class holds only `config`/`dtype`/`device` (no `n_states`/`n_actions`
    like `mfc.algorithms`'s discrete environment contract)."""

    def __init__(
        self,
        config: LQConfig = LQConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device

    def forward_moments(self, theta: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate (mu_t^{theta,lambda}, Sigma_t^{theta,lambda}), t=0,...,T
        (eqs. algo-forward-mean/algo-forward-variance). Returns two (T+1,)
        tensors. Differentiable in `theta` via ordinary autograd (used to
        cross-check `exact_gradient` in tests)."""
        cfg = self.config
        T = theta.shape[0]
        lam2rho2 = (lam * cfg.rho) ** 2
        mu = [torch.as_tensor(cfg.mu0, dtype=self.dtype, device=self.device)]
        Sigma = [torch.as_tensor(cfg.Sigma0, dtype=self.dtype, device=self.device)]
        for t in range(T):
            th1, th2 = theta[t, 0], theta[t, 1]
            mu_t, Sigma_t = mu[-1], Sigma[-1]
            mu.append((cfg.a + cfg.b * th1 + cfg.b * th2 + cfg.c) * mu_t)
            Sigma.append(
                (cfg.a + cfg.b * th1) ** 2 * Sigma_t
                + (cfg.b * th2 + cfg.c) ** 2 * lam2rho2 * (mu_t**2 + 1.0)
                + cfg.b**2 * cfg.tau**2
                + cfg.sigma**2
            )
        return torch.stack(mu), torch.stack(Sigma)

    def exact_objective(self, theta: torch.Tensor, lam: float) -> torch.Tensor:
        """Closed-form J^lambda(theta) (eq. closed-form-objective). Scalar,
        differentiable in `theta`."""
        cfg = self.config
        T = theta.shape[0]
        lam2rho2 = (lam * cfg.rho) ** 2
        mu, Sigma = self.forward_moments(theta, lam)

        total = torch.zeros((), dtype=self.dtype, device=self.device)
        for t in range(T):
            th1, th2 = theta[t, 0], theta[t, 1]
            mu_t, Sigma_t = mu[t], Sigma[t]
            total = (
                total
                + cfg.q * (Sigma_t + mu_t**2)
                + cfg.r * ((th1 + th2) ** 2 * mu_t**2 + th1**2 * Sigma_t + th2**2 * lam2rho2 * (mu_t**2 + 1.0) + cfg.tau**2)
                + cfg.kappa * (mu_t**2 + lam2rho2 * (mu_t**2 + 1.0))
            )
        mu_T, Sigma_T = mu[T], Sigma[T]
        total = total + cfg.q_T * (Sigma_T + mu_T**2) + cfg.kappa_T * (mu_T**2 + lam2rho2 * (mu_T**2 + 1.0))
        return total

    def exact_gradient(self, theta: torch.Tensor, lam: float) -> torch.Tensor:
        """Exact grad_theta J^lambda(theta) in O(T) via the forward-backward
        adjoint sweep (eqs. algo-terminal-p/s, algo-backward-p/s,
        grad-theta1/2). Returns shape (T,2). `theta` need not require grad
        (this never uses autograd)."""
        cfg = self.config
        T = theta.shape[0]
        lam2rho2 = (lam * cfg.rho) ** 2
        mu, Sigma = self.forward_moments(theta.detach(), lam)

        p_next = 2.0 * (cfg.q_T + cfg.kappa_T * (1.0 + lam2rho2)) * mu[T]
        s_next = torch.as_tensor(cfg.q_T, dtype=self.dtype, device=self.device)

        grad = torch.zeros_like(theta)
        for t in range(T - 1, -1, -1):
            th1, th2 = theta[t, 0], theta[t, 1]
            mu_t, Sigma_t = mu[t], Sigma[t]

            grad[t, 0] = 2 * cfg.r * (th1 + th2) * mu_t**2 + 2 * cfg.r * th1 * Sigma_t + cfg.b * mu_t * p_next + 2 * cfg.b * (cfg.a + cfg.b * th1) * Sigma_t * s_next
            grad[t, 1] = (
                2 * cfg.r * (th1 + th2) * mu_t**2
                + 2 * cfg.r * th2 * lam2rho2 * (mu_t**2 + 1.0)
                + cfg.b * mu_t * p_next
                + 2 * cfg.b * (cfg.b * th2 + cfg.c) * lam2rho2 * (mu_t**2 + 1.0) * s_next
            )

            p_t = (
                2 * (cfg.q + cfg.r * (th1 + th2) ** 2 + cfg.r * th2**2 * lam2rho2 + cfg.kappa * (1.0 + lam2rho2)) * mu_t
                + (cfg.a + cfg.b * th1 + cfg.b * th2 + cfg.c) * p_next
                + 2 * (cfg.b * th2 + cfg.c) ** 2 * lam2rho2 * mu_t * s_next
            )
            s_t = cfg.q + cfg.r * th1**2 + (cfg.a + cfg.b * th1) ** 2 * s_next
            p_next, s_next = p_t, s_t

        return grad

    def objective_bias(self, theta: torch.Tensor) -> torch.Tensor:
        """B(theta) (eq. objective-bias-functional), the exact O(lambda^2)
        coefficient of J^lambda(theta) - J^0(theta) (eq.
        objective-pointwise-rate): J^lambda(theta) == J^0(theta) +
        lambda^2*rho^2*B(theta), checked in tests/test_lq.py."""
        cfg = self.config
        T = theta.shape[0]
        mu0_flow, _ = self.forward_moments(theta, lam=0.0)

        V = [torch.zeros((), dtype=self.dtype, device=self.device)]
        for t in range(T):
            th1, th2 = theta[t, 0], theta[t, 1]
            V.append((cfg.a + cfg.b * th1) ** 2 * V[-1] + (cfg.b * th2 + cfg.c) ** 2 * (mu0_flow[t] ** 2 + 1.0))

        total = torch.zeros((), dtype=self.dtype, device=self.device)
        for t in range(T):
            th1, th2 = theta[t, 0], theta[t, 1]
            total = total + (cfg.q + cfg.r * th1**2) * V[t] + (cfg.r * th2**2 + cfg.kappa) * (mu0_flow[t] ** 2 + 1.0)
        total = total + cfg.q_T * V[T] + cfg.kappa_T * (mu0_flow[T] ** 2 + 1.0)
        return total

    def riccati_optimal(self, T: int | None = None) -> torch.Tensor:
        """The unperturbed (lambda=0) optimal theta*, shape (T,2), via the
        two decoupled Riccati recursions (Sec. "Optimal Policy via Decoupled
        Riccati Equations"): theta_t^{1,*}=k_t^*, theta_t^{2,*}=l_t^*-k_t^*."""
        cfg = self.config
        T = cfg.T if T is None else T
        P_next = torch.as_tensor(cfg.q_T, dtype=self.dtype, device=self.device)
        R_next = torch.as_tensor(cfg.q_T + cfg.kappa_T, dtype=self.dtype, device=self.device)

        theta_star = torch.zeros(T, 2, dtype=self.dtype, device=self.device)
        for t in range(T - 1, -1, -1):
            k_star = -cfg.a * cfg.b * P_next / (cfg.r + cfg.b**2 * P_next)
            l_star = -cfg.b * (cfg.a + cfg.c) * R_next / (cfg.r + cfg.b**2 * R_next)
            theta_star[t, 0] = k_star
            theta_star[t, 1] = l_star - k_star

            P_next = cfg.q + cfg.r * cfg.a**2 * P_next / (cfg.r + cfg.b**2 * P_next)
            R_next = cfg.q + cfg.kappa + cfg.r * (cfg.a + cfg.c) ** 2 * R_next / (cfg.r + cfg.b**2 * R_next)

        return theta_star

    def rollout(
        self,
        theta: torch.Tensor,
        lam: float,
        B: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Simulate B i.i.d. particles under the lambda-perturbed dynamics,
        using the deterministic exact mean flow mu_t^{theta,0} (Sec. 2.2's
        "assume m_t^theta is deterministic") as the population argument at
        every step — one shared (zeta_t,beta_t) draw per t, not per
        particle, matching the reference's randomization of the whole
        population law. `theta` is fully detached throughout (both for
        `mu_flow` and for sampling `alpha`): every returned tensor is a
        fixed constant with no gradient path to `theta`, so
        `mfc.algorithms.lq.reinforce_step` can score the *sampled* alpha
        against the (separately attached) theta — mirroring the discrete
        algorithms' `sample_actions` (always detached) vs. `policy_score`
        (attached) split; scoring against the *same* attached theta used to
        draw the sample would make alpha-means cancel identically to a
        theta-independent constant, silently zeroing the estimator. Returns
        X (T+1,B), alpha (T,B), mu_hat (T+1,), running_cost (T,B),
        terminal_cost (B,)."""
        cfg = self.config
        dtype, device = self.dtype, self.device
        T = theta.shape[0]
        theta = theta.detach()
        mu_flow, _ = self.forward_moments(theta, lam=0.0)

        X = [cfg.mu0 + (cfg.Sigma0**0.5) * torch.randn(B, dtype=dtype, device=device, generator=generator)]
        alpha, mu_hat, running_cost = [], [], []
        for t in range(T):
            zeta = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
            beta = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
            mu_hat_t = (1.0 + lam * zeta) * mu_flow[t] + lam * beta
            mu_hat.append(mu_hat_t)

            th1, th2 = theta[t, 0], theta[t, 1]
            eta = torch.randn(B, dtype=dtype, device=device, generator=generator)
            alpha_t = th1 * X[t] + th2 * mu_hat_t + cfg.tau * eta
            alpha.append(alpha_t)
            running_cost.append(cfg.q * X[t] ** 2 + cfg.r * alpha_t**2 + cfg.kappa * mu_hat_t**2)

            eps = torch.randn(B, dtype=dtype, device=device, generator=generator)
            X.append(cfg.a * X[t] + cfg.b * alpha_t + cfg.c * mu_hat_t + cfg.sigma * eps)

        zeta_T = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
        beta_T = cfg.rho * torch.randn((), dtype=dtype, device=device, generator=generator)
        mu_hat_T = (1.0 + lam * zeta_T) * mu_flow[T] + lam * beta_T
        mu_hat.append(mu_hat_T)
        terminal_cost = cfg.q_T * X[T] ** 2 + cfg.kappa_T * mu_hat_T**2

        return {
            "X": torch.stack(X),
            "alpha": torch.stack(alpha),
            "mu_hat": torch.stack(mu_hat),
            "running_cost": torch.stack(running_cost),
            "terminal_cost": terminal_cost,
        }

    def init_theta(self, T: int | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """theta_0=0 (no control gain), the natural LQR-style starting
        point, mirroring `mfc.environments.twostate.TwoState.init_theta`'s
        own zero init. `generator` is accepted (unused: deterministic) for
        the same `env.init_theta(generator=...)` contract shared by every
        environment in this repo."""
        T = self.config.T if T is None else T
        return torch.zeros(T, 2, dtype=self.dtype, device=self.device)
