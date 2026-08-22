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
for the model-free algorithms (`mfc.algorithms.continuous_simplex`,
`mfc.algorithms.continuous_reinforce`) and for trajectory diagnostics.

Model (LQ_framework.tex, Secs. 1-3). State and action spaces are R. The
initial law is X_0 ~ N(mu0, Sigma0); the state law m_t^theta is Gaussian at
every t, with mean/variance (mu_t, Sigma_t) propagated deterministically.
The perturbation randomizes the Gaussian mean-field argument via the affine map
T_t^lambda(x) = (1+lambda*zeta_t)x + lambda*beta_t, (zeta_t,beta_t) ~
N(0,rho^2) x N(0,rho^2), producing
N(M_t^{lambda,theta}, Sigma_t^{lambda,theta}). The policy is Gaussian feedback,
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
hence `MAXIMIZE = False`, which is where `mfc.algorithms.continuous.train`
reads the resulting sign convention from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LQConfig:
    """Model parameters. Defaults are chosen so the mean-field coupling
    dominates the comparison: `c` is at least as large as `a` (the population
    feeds back into the transition as strongly as the agent's own state),
    and `kappa`/`kappa_T` are deliberately larger than `q`/`q_T` (the
    population's mean is penalized more heavily than the agent's own
    deviation), with a nonzero `mu0` so there is a real population mean to
    control from the start. See `tests/test_lq.py` for a direct numeric
    check that the mean-field coupling changes the optimal policy and value."""

    a: float = 0.9  # self transition coefficient
    b: float = 1.0  # control effectiveness
    c: float = 1.0  # mean-field feedback in the transition kernel
    sigma: float = 0.3  # state noise std

    q: float = 1.0  # running state cost
    r: float = 0.1  # running control cost
    q_T: float = 5.0  # terminal state cost
    kappa: float = 4.0  # running mean-field coupling cost (reference's gamma)
    kappa_T: float = 10.0  # terminal mean-field coupling cost (reference's gamma_T)

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

    MAXIMIZE = False  # J^lambda here is a cost: `mfc.algorithms.continuous.train` descends

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

    def policy_features(self, t: int, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        phi_t(x,m), the feature vector of the Gaussian feedback policy:
        pi_t^theta(.|x,m) = N(theta_t . phi_t(x,m), tau^2). Here the policy
        mean is theta_t^1*x + theta_t^2*mbar, so phi_t(x,mbar) = (x, mbar).
        `t` is accepted (unused: these features are time-homogeneous) for
        the contract shared with `mfc.environments.portfolio.Portfolio
        .policy_features`, which `mfc.algorithms._continuous.policy_score`
        uses to build grad_theta log p_t^theta generically. Returns shape
        (*x.shape, 2).
        """
        del t
        return torch.stack(torch.broadcast_tensors(x, M), dim=-1)

    def rollout(
        self,
        theta: torch.Tensor,
        lam: float,
        B: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Simulate B i.i.d. replicas of the lambda-perturbed system, each
        with its own perturbation draw (zeta_t^b,beta_t^b) at every t, so
        that the randomized population argument is
        M_t^{lambda,theta,b} = (1+lambda*zeta_t^b)*mu_t^{theta,0} +
        lambda*beta_t^b, centered on the deterministic nominal mean flow
        mu_t^{theta,0} (Sec. 2.2's "assume m_t^theta is deterministic").
        Independent per replica, not one draw shared across the batch: the
        Research_Project.tex's continuous LQ construction samples an affine
        perturbation independently of the state and policy noises; the
        auxiliary and main batches here are i.i.d. samples of that generated
        law. Each replica's *marginal* law is unchanged either way, so every
        closed form here (`exact_objective`, `forward_moments`,
        `exact_gradient`) describes this rollout exactly as before.

        `theta` is fully detached throughout (both for `mu_flow` and for
        sampling `alpha`): every returned tensor is a fixed constant with no
        gradient path to `theta`, so the algorithms can score the *sampled*
        alpha against theta — mirroring the discrete algorithms'
        `sample_actions` (always detached) vs. `policy_score` (attached)
        split; scoring against the *same* attached theta used to draw the
        sample would make alpha-means cancel identically to a
        theta-independent constant, silently zeroing the estimator.

        Returns X (T+1,B), alpha (T,B), M (T+1,B), Sigma (T+1,B),
        zeta/beta/xi (T+1,B), mu/Sigma_nominal (T+1,), running (T,B),
        terminal (B,). `xi` is the standardized marginal-mean perturbation
        (M_t = mu_t + lambda*rho*sqrt(mu_t^2+1)*xi_t with xi_t ~ N(0,1)),
        kept for diagnostics and backward compatibility; the joint
        perturbation score is expressed directly in `zeta` and `beta`.
        `mu` and `Sigma_nominal` are the nominal moment flow
        (mu_t^{theta,0}, Sigma_t^{theta,0}); `running`/
        `terminal` are this environment's *costs* (its own sign convention,
        matching `exact_objective`).
        """
        cfg = self.config
        dtype, device = self.dtype, self.device
        T = theta.shape[0]
        theta = theta.detach()
        mu_flow, Sigma_flow = self.forward_moments(theta, lam=0.0)

        X = [cfg.mu0 + (cfg.Sigma0**0.5) * torch.randn(B, dtype=dtype, device=device, generator=generator)]
        alpha, M, Sigma, zetas, betas, xi, running = [], [], [], [], [], [], []
        for t in range(T + 1):
            mu_t = mu_flow[t]
            Sigma_t = Sigma_flow[t]
            zeta = cfg.rho * torch.randn(B, dtype=dtype, device=device, generator=generator)
            beta = cfg.rho * torch.randn(B, dtype=dtype, device=device, generator=generator)
            factor = 1.0 + lam * zeta
            M.append(factor * mu_t + lam * beta)
            Sigma.append(factor**2 * Sigma_t)
            zetas.append(zeta)
            betas.append(beta)
            xi.append((zeta * mu_t + beta) / (cfg.rho * torch.sqrt(mu_t**2 + 1.0)))
            if t == T:
                break

            th1, th2 = theta[t, 0], theta[t, 1]
            eta = torch.randn(B, dtype=dtype, device=device, generator=generator)
            alpha_t = th1 * X[t] + th2 * M[t] + cfg.tau * eta
            alpha.append(alpha_t)
            running.append(cfg.q * X[t] ** 2 + cfg.r * alpha_t**2 + cfg.kappa * M[t] ** 2)

            eps = torch.randn(B, dtype=dtype, device=device, generator=generator)
            X.append(cfg.a * X[t] + cfg.b * alpha_t + cfg.c * M[t] + cfg.sigma * eps)

        return {
            "X": torch.stack(X),
            "alpha": torch.stack(alpha),
            "M": torch.stack(M),
            "Sigma": torch.stack(Sigma),
            "zeta": torch.stack(zetas),
            "beta": torch.stack(betas),
            "xi": torch.stack(xi),
            "mu": mu_flow,
            "Sigma_nominal": Sigma_flow,
            "running": torch.stack(running),
            "terminal": cfg.q_T * X[T] ** 2 + cfg.kappa_T * M[T] ** 2,
        }

    def init_theta(self, T: int | None = None, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """theta_0=0 (no control gain), the natural LQR-style starting
        point, mirroring `mfc.environments.twostate.TwoState.init_theta`'s
        own zero init. `generator` is accepted (unused: deterministic) for
        the same `env.init_theta(generator=...)` contract shared by every
        environment in this repo."""
        T = self.config.T if T is None else T
        return torch.zeros(T, 2, dtype=self.dtype, device=self.device)
