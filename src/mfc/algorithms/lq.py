"""
Training for the continuous-state LQ benchmark (`mfc.environments.lq`).

Unlike the discrete algorithms in this package (simplex, mfreinforce,
reinforce — all built around a categorical state space, `torch.func.vmap`
policy scoring, and Monte Carlo population-flow estimation), LQ's own
reference (files/reference/LQ_framework.tex) gives J^lambda(theta) and its
gradient in closed form, so there is a genuine "exact" training mode with no
sampling at all: `exact_step` just calls `env.exact_gradient`. The only
Monte Carlo estimator here is `reinforce_step`, the classical-REINFORCE
ablation from context.md ("Show missing mean-field term by comparison with
reinforce"): it simulates trajectories against the *deterministic* nominal
mean flow mu_t^{theta,0} (`LQ.rollout` detaches theta before computing it)
and backpropagates only the direct policy score, exactly as
`mfc.algorithms.reinforce` does for the discrete case — never differentiating
through the population's own dependence on theta, so it is a structurally
biased estimator of grad_theta J^lambda(theta).

Sign convention: everything in `mfc.environments.lq` is a *cost* (matching
the reference's own notation), not a reward to maximize like the rest of
this repo. So, unlike `mfc.algorithms.{simplex,mfreinforce,reinforce}`
(which negate their reward gradient before an ascent step, `theta.grad =
-g_hat`), training here sets `theta.grad = g_hat` directly and lets Adam
descend on cost.
"""

from __future__ import annotations

import time

import torch

__all__ = ["reinforce_step", "train"]


def reinforce_step(
    env,
    theta: torch.Tensor,
    lam: float,
    B: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of grad_theta J^lambda(theta), missing the
    mean-field sensitivity term: g_hat = (1/B) sum_b sum_t
    grad_theta log(pi_t^theta(alpha_t^(b)|X_t^(b),mu_hat_t)) * G_t^(b), with
    G_t^(b) the cost-to-go from t (running costs t..T-1 plus the terminal
    cost). `mu_hat_t` is treated as exogenous (see `LQ.rollout`), so this
    never captures how theta influences the population mean itself — the
    same ablation `mfc.algorithms.reinforce` implements for the discrete
    case. `theta` must require grad. Returns shape (T,2).
    """
    out = env.rollout(theta, lam, B, generator=generator)
    X, alpha, mu_hat = out["X"], out["alpha"], out["mu_hat"]
    running_cost, terminal_cost = out["running_cost"], out["terminal_cost"]

    cost_to_go = torch.flip(torch.cumsum(torch.flip(running_cost, [0]), dim=0), [0]) + terminal_cost.unsqueeze(0)  # (T,B)

    tau = env.config.tau
    means = theta[:, 0:1] * X[:-1] + theta[:, 1:2] * mu_hat[:-1].unsqueeze(-1)  # (T,B)
    log_probs = -0.5 * ((alpha - means) / tau) ** 2  # additive normalizing constant doesn't depend on theta

    surrogate = (log_probs * cost_to_go.detach()).sum()
    (g_hat,) = torch.autograd.grad(surrogate, theta)
    return g_hat / B


def train(
    env,
    theta0: torch.Tensor,
    *,
    algorithm: str,
    n_train: int,
    lr: float,
    lam: float,
    B: int | None = None,
    validate_every: int = 20,
    generator: torch.Generator | None = None,
) -> dict:
    """
    Train theta by gradient descent on J^lambda(theta) (`algorithm=
    "exact_gradient"`, using the closed-form `env.exact_gradient`, no
    sampling) or on its classical-REINFORCE MC estimate (`algorithm=
    "reinforce"`, needs `B`). Validates every `validate_every` steps by
    computing the exact unperturbed objective J^0(theta) — exact and free
    (no Monte Carlo error: `mfc.environments.lq.LQ.exact_objective` is
    closed-form), unlike every other benchmark's validation. Returns
    theta_final, theta_history, validation_iterations, validation_J,
    elapsed_seconds — the same fields `scripts/train.py`'s discrete
    `train_run` returns, so `scripts/test.py`/notebooks can treat LQ runs
    uniformly.
    """
    if algorithm not in ("exact_gradient", "reinforce"):
        raise ValueError(f"unknown algorithm {algorithm!r}; available: exact_gradient, reinforce")
    if algorithm == "reinforce" and B is None:
        raise ValueError("algorithm='reinforce' requires B")

    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    theta_history = [theta.detach().clone()]
    val_iterations, val_J = [], []

    start = time.perf_counter()
    for m in range(n_train):
        if algorithm == "exact_gradient":
            g_hat = env.exact_gradient(theta.detach(), lam)
        else:
            g_hat = reinforce_step(env, theta, lam, B, generator=generator)

        optimizer.zero_grad()
        theta.grad = g_hat
        optimizer.step()
        theta_history.append(theta.detach().clone())

        if m % validate_every == 0 or m == n_train - 1:
            val_iterations.append(m)
            val_J.append(env.exact_objective(theta.detach(), 0.0).item())
    elapsed = time.perf_counter() - start

    return {
        "theta_final": theta.detach().clone(),
        "theta_history": torch.stack(theta_history),
        "validation_iterations": torch.tensor(val_iterations),
        "validation_J": torch.tensor(val_J, dtype=theta.dtype),
        "elapsed_seconds": elapsed,
    }
