"""
Training loop for the continuous-state benchmarks (`mfc.environments.lq`,
`mfc.environments.portfolio`).

One loop serves both environments and all three algorithms, since the
estimators (`mfc.algorithms.continuous_simplex`,
`mfc.algorithms.continuous_reinforce`) are generic over the environment
contract documented in `mfc.algorithms._continuous`:

  - `"simplex"`: the Research_Project.tex continuous-state affine-
    perturbation MF-REINFORCE estimator — one auxiliary batch for mean and
    log-standard-deviation sensitivities, one
    main batch for the perturbed policy gradient, (n_aux+B)*T transitions
    per step.
  - `"reinforce"`: the classical-REINFORCE ablation of the same estimator
    (context.md: "Show missing mean-field term by comparison with
    reinforce"), B*T transitions per step.
  - `"exact_gradient"`: `env.exact_gradient`, the reference's closed-form
    O(T) adjoint gradient. Not a model-free method and not part of the
    algorithm comparison (see `configs/lq.py`) — it is the *oracle* these
    benchmarks exist to be measured against, kept trainable here because
    "what the exactly-optimized policy converges to" is the reference point
    for every diagnostic in `scripts/test.py`.

Sign convention. `env.MAXIMIZE` says whether the environment's J^lambda is a
reward to maximize (portfolio: E[X_T]-chi*Var(X_T)) or a cost to minimize
(LQ, matching its own reference's notation); the estimators always return
grad_theta J^lambda in that same convention, and this is the single place
where the ascend-vs-descend choice is made.
"""

from __future__ import annotations

import time

import torch

from ..progress import training_bar
from . import continuous_reinforce, continuous_simplex

__all__ = ["ALGORITHMS", "gradient_step", "train"]

ALGORITHMS = ("simplex", "reinforce", "exact_gradient")


def gradient_step(
    env,
    theta: torch.Tensor,
    lam: float,
    *,
    algorithm: str,
    B: int | None = None,
    n_aux: int | None = None,
    eta: float | None = None,
    baseline=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One gradient estimate at the given theta under `algorithm` (one of
    ALGORITHMS), dispatched to the corresponding estimator. `B` is required
    by "simplex" and "reinforce", `n_aux` by "simplex"; "exact_gradient"
    needs neither. Returns shape (T,2)."""
    if algorithm == "exact_gradient":
        return env.exact_gradient(theta.detach(), lam)
    if algorithm == "simplex":
        return continuous_simplex.gradient_step(env, theta, lam, B=B, n_aux=n_aux, eta=eta, baseline=baseline, generator=generator)
    if algorithm == "reinforce":
        return continuous_reinforce.gradient_step(env, theta, lam, B=B, baseline=baseline, generator=generator)
    raise ValueError(f"unknown algorithm {algorithm!r}; available: {', '.join(ALGORITHMS)}")


def train(
    env,
    theta0: torch.Tensor,
    *,
    algorithm: str,
    n_train: int,
    lr: float,
    lam: float,
    B: int | None = None,
    n_aux: int | None = None,
    eta: float | None = None,
    baseline=None,
    validate_every: int = 20,
    generator: torch.Generator | None = None,
    progress_desc: str | None = None,
) -> dict:
    """
    Train theta by Adam on J^lambda(theta) — ascending or descending
    according to `env.MAXIMIZE` — using `algorithm`'s gradient estimate
    (Research_Project.tex's continuous-state MF-REINFORCE construction for
    "simplex").
    Validates every `validate_every` steps by computing the exact
    unperturbed objective J^0(theta), which for both continuous benchmarks
    is closed-form and free of Monte Carlo error, unlike every discrete
    benchmark's validation. Returns theta_final, theta_history,
    validation_iterations, validation_J, elapsed_seconds — the same fields
    `scripts/train.py`'s discrete `train_run` returns, so
    `scripts/test.py`/the notebooks can treat every run uniformly.
    `progress_desc` labels a live progress bar over the training iterations
    (None disables it); see `mfc.progress`.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; available: {', '.join(ALGORITHMS)}")
    if algorithm in ("simplex", "reinforce") and B is None:
        raise ValueError(f"algorithm={algorithm!r} requires B")
    if algorithm == "simplex" and n_aux is None:
        raise ValueError("algorithm='simplex' requires n_aux")

    theta = theta0.clone().detach()
    optimizer = torch.optim.Adam([theta], lr=lr)
    sign = -1.0 if env.MAXIMIZE else 1.0  # Adam always descends on theta.grad
    theta_history = [theta.clone()]
    val_iterations, val_J = [], []

    start = time.perf_counter()
    with training_bar(n_train, desc=progress_desc) as bar:
        for m in bar:
            g_hat = gradient_step(env, theta, lam, algorithm=algorithm, B=B, n_aux=n_aux, eta=eta, baseline=baseline, generator=generator)
            optimizer.zero_grad()
            theta.grad = sign * g_hat
            optimizer.step()
            theta_history.append(theta.clone())

            if m % validate_every == 0 or m == n_train - 1:
                val_iterations.append(m)
                val_J.append(env.exact_objective(theta, 0.0).item())
                bar.set_postfix_str(f"J={val_J[-1]:.4f}", refresh=False)
    elapsed = time.perf_counter() - start

    return {
        "theta_final": theta.clone(),
        "theta_history": torch.stack(theta_history),
        "validation_iterations": torch.tensor(val_iterations),
        "validation_J": torch.tensor(val_J, dtype=theta.dtype),
        "elapsed_seconds": elapsed,
    }
