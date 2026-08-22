"""
Classical continuous-state REINFORCE, as a baseline for the "missing
mean-field term" comparison (context.md) — the continuous-state counterpart
of `mfc.algorithms.reinforce`.

The perturbed policy gradient in Research_Project.tex's "Continuous State
Space" section has two pieces: the direct policy score L_t, and the
population-law score grad_theta log q_t^{lambda,theta}. In the implemented
joint Gaussian-law chart, that law score is
D_t^theta h_t^m + K_t^theta h_t^sigma with
D_t^theta = grad_theta mu_t^theta and
K_t^theta = grad_theta log sigma_t^theta. Classical REINFORCE keeps only
the first:

    g_hat_reinforce(theta) = (1/B) sum_b sum_{t<T} L_t^(b) * G_t^(b).

It is model-free, needs no auxiliary batch (B*T transitions per step against
simplex's (n_aux+B)*T), and is a structurally biased estimator of
grad_theta J(theta) whenever the population argument influences the dynamics
or costs — by exactly the term it drops.

No perturbation scale. `gradient_step` takes `lam` like every estimator
here, but *training* runs it at lam=0, on the nominal population flow: the
randomization exists only to expose the mean-field sensitivity through a
likelihood ratio, so an algorithm that discards that term gains nothing from
injecting it and would merely optimize J^lambda instead of J with extra
variance. `scripts/train.py` therefore keeps reinforce out of
ALGORITHMS_WITH_PERTURBATION_SCALE and records `lam=None` for its runs, the
same convention the discrete `mfc.algorithms.reinforce` already followed.

At lam > 0 this is instead the *ablation* of the simplex estimator: run on
the same perturbed rollout, it is `continuous_simplex.gradient_estimate`
with the sensitivity flow set to zero, sample for sample. That is what makes
the comparison an ablation of one term rather than of two different
algorithms, and it is how `scripts/test.py`'s fixed-theta diagnostics
(`continuous_gradient_diagnostics`, `continuous_mean_field_term`) isolate
the omitted term — which only exists on a perturbed rollout in the first
place.
"""

from __future__ import annotations

import torch

from ._continuous import policy_score, policy_score_sum, returns_to_go
from .continuous_simplex import resolve_baseline

__all__ = ["gradient_estimate", "gradient_step"]


def gradient_estimate(env, theta: torch.Tensor, out: dict[str, torch.Tensor], *, baseline=None) -> torch.Tensor:
    """
    REINFORCE policy-gradient estimate (1/B) sum_b sum_t L_t^(b)(G_t^(b)-b_t)
    from one main batch `out` (an `env.rollout`, at lam=0 in training and at
    the simplex estimator's own lam when used as its ablation). Identical to
    `mfc.algorithms.continuous_simplex.gradient_estimate` with D_hat = 0, and
    written through the same helpers so that it is the same estimator,
    sample for sample, minus the population-score term. Returns shape (T,2).
    """
    G = returns_to_go(out)
    weights = G - resolve_baseline(G, baseline)
    coeff, features = policy_score(env, theta, out)
    return policy_score_sum(coeff, features, weights[:-1]) / G.shape[1]


def gradient_step(
    env,
    theta: torch.Tensor,
    lam: float,
    *,
    B: int,
    baseline=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One full REINFORCE gradient estimate at the given theta: one main
    batch of `B` trajectories, no auxiliary batch. `lam=0` (what training
    uses) simulates the nominal process; `lam>0` reproduces the simplex
    estimator's own rollout, for the fixed-theta ablation. Returns shape
    (T,2)."""
    return gradient_estimate(env, theta, env.rollout(theta.detach(), lam, B, generator=generator), baseline=baseline)
