"""
Training for the continuous-state portfolio benchmark
(`mfc.environments.portfolio`).

Mirrors `mfc.algorithms.lq`'s structure exactly (see its module docstring
for why this benchmark can't share the discrete simplex/mfreinforce/
reinforce machinery): `exact_gradient` trains directly from the reference's
closed-form gradient (`Portfolio.exact_gradient`, no sampling); `reinforce`
is the classical-REINFORCE ablation (context.md: "Show missing mean-field
term by comparison with reinforce"), a Monte Carlo score-function estimator
against the deterministic nominal mean flow (`Portfolio.rollout` detaches
theta before computing it), never differentiating through the population's
own dependence on theta.

Sign convention: UNLIKE `mfc.environments.lq` (a cost to minimize, matching
its own reference's notation), `mfc.environments.portfolio`'s J^lambda is a
REWARD to maximize (E[X_T] - chi*Var(X_T), "the policy is selected... to
maximize the terminal expected wealth penalized by its variance") — the same
convention as the rest of this repo's discrete algorithms. `train` here
therefore ascends (`theta.grad = -g_hat`), not descends like
`mfc.algorithms.lq.train`.

There is no running reward (`Portfolio`'s cost is purely terminal), so
unlike LQ's cost-to-go sum, `reinforce_step`'s policy score at every t is
weighted by the *same* single terminal reward — no backward accumulation
needed.
"""

from __future__ import annotations

import time

import torch

from ..progress import training_bar

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
    mean-field sensitivity term: g_hat = (1/B) sum_b [sum_t grad_theta
    log(pi_t^theta(alpha_t^(b)|X_t^(b),mu_hat_t))] * R^(b), with R^(b) :=
    X_T^(b) - chi*(X_T^(b)-mu_hat_T)^2 the per-particle terminal reward
    (`Portfolio`'s own g(x,m), using the deterministic mu_hat_T as the
    population-mean argument — see `Portfolio.rollout`). `mu_hat` is
    treated as exogenous throughout, so this never captures how theta
    influences the population mean itself — the same ablation
    `mfc.algorithms.reinforce` implements for the discrete case, and
    `mfc.algorithms.lq.reinforce_step` for LQ. `theta` must require grad.
    Returns shape (T,2).
    """
    out = env.rollout(theta, lam, B, generator=generator)
    X, alpha, mu_hat = out["X"], out["alpha"], out["mu_hat"]
    terminal_reward = X[-1] - env.config.chi * (X[-1] - mu_hat[-1]) ** 2  # (B,)

    tau = env.config.tau
    means = theta[:, 0:1] * (X[:-1] - mu_hat[:-1].unsqueeze(-1)) + theta[:, 1:2]  # (T,B)
    log_probs = -0.5 * ((alpha - means) / tau) ** 2  # additive normalizing constant doesn't depend on theta

    surrogate = (log_probs.sum(dim=0) * terminal_reward.detach()).sum()
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
    progress_desc: str | None = None,
) -> dict:
    """
    Train theta by gradient ASCENT on J^lambda(theta) (`algorithm=
    "exact_gradient"`, using the closed-form `env.exact_gradient`, no
    sampling) or on its classical-REINFORCE MC estimate (`algorithm=
    "reinforce"`, needs `B`). Validates every `validate_every` steps by
    computing the exact unperturbed objective J^0(theta) — exact and free
    (no Monte Carlo error), same as `mfc.algorithms.lq.train`. Returns
    theta_final, theta_history, validation_iterations, validation_J,
    elapsed_seconds — the same fields `mfc.algorithms.lq.train` returns.
    `progress_desc` labels a live progress bar over the training iterations
    (None disables it); see `mfc.progress`.
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
    with training_bar(n_train, desc=progress_desc) as bar:
        for m in bar:
            if algorithm == "exact_gradient":
                g_hat = env.exact_gradient(theta.detach(), lam)
            else:
                g_hat = reinforce_step(env, theta, lam, B, generator=generator)

            optimizer.zero_grad()
            theta.grad = -g_hat  # reward maximization: ascend
            optimizer.step()
            theta_history.append(theta.detach().clone())

            if m % validate_every == 0 or m == n_train - 1:
                val_iterations.append(m)
                val_J.append(env.exact_objective(theta.detach(), 0.0).item())
                bar.set_postfix_str(f"J={val_J[-1]:.4f}", refresh=False)
    elapsed = time.perf_counter() - start

    return {
        "theta_final": theta.detach().clone(),
        "theta_history": torch.stack(theta_history),
        "validation_iterations": torch.tensor(val_iterations),
        "validation_J": torch.tensor(val_J, dtype=theta.dtype),
        "elapsed_seconds": elapsed,
    }
