"""
Perturbation-agnostic machinery shared by every continuous-state algorithm
(`mfc.algorithms.continuous_simplex`, `mfc.algorithms.continuous_reinforce`):
returns-to-go from a rollout, and the Gaussian policy score. The
continuous-state counterpart of `mfc.algorithms._common`.

Environment contract. Generic over any continuous-state environment exposing

  - `config.tau` (policy std) and `config.rho` (perturbation std),
  - `forward_moments(theta, lam) -> (mu, Sigma)`, from which the nominal
    coordinate flow c_t^theta = mu_t^{theta,0} is read,
  - `policy_features(t, x, M) -> phi_t(x,m)` of shape (*x.shape, 2),
  - `rollout(theta, lam, B, *, generator) -> dict` with keys
    X (T+1,B), alpha (T,B), M (T+1,B), xi (T+1,B), mu (T+1,),
    running (T,B), terminal (B,),
  - `exact_objective(theta, lam)` / `exact_gradient(theta, lam)` (used only
    by diagnostics, never by the estimators here),

i.e. `mfc.environments.lq.LQ` and `mfc.environments.portfolio.Portfolio`.
Both are scalar-state, one-dimensional-mean-field-argument models whose
policy is Gaussian and *linear in theta_t*,

    pi_t^theta(.|x,m) = N(theta_t . phi_t(x,m), tau^2),

which is the setting of the reference's own continuous-state analysis
(files/reference/continuous_state_space(2).tex; files/Research_Project.tex,
Sec. "Policy gradient for the perturbed control problem in the LQ setting").
That linearity is what lets `policy_score` below be exact and closed-form
rather than an autograd surrogate — and it removes, structurally, the trap
documented in `LQ.rollout`: alpha is *data* here, sampled under a detached
theta, and is never re-derived from the theta being differentiated.

Sign convention. `running`/`terminal` are in each environment's own
convention (LQ: costs to minimize; portfolio: rewards to maximize), so
everything here estimates grad_theta of that same J^lambda; the descend-vs-
ascend choice is made once, in `mfc.algorithms.continuous.train`, from the
environment's own `MAXIMIZE` flag.
"""

from __future__ import annotations

import torch

__all__ = ["returns_to_go", "policy_score", "policy_score_sum", "leave_one_out_baseline"]


def returns_to_go(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    G_t^{lambda,theta} = sum_{s>=t} running_s + terminal, t=0,...,T
    (eq. continuous-perturbed-return), from one `env.rollout` output.
    G_T is the terminal cost/reward alone. Returns shape (T+1,B).
    """
    running, terminal = out["running"], out["terminal"]
    tail = torch.flip(torch.cumsum(torch.flip(running, [0]), dim=0), [0])  # (T,B)
    return torch.cat([tail + terminal.unsqueeze(0), terminal.unsqueeze(0)])


def policy_score(env, theta: torch.Tensor, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The two factors of the per-sample policy score
    L_t^{(b)} = grad_theta log p_t^theta(alpha_t^(b) | X_t^(b), M_t^(b))
    (eq. continuous-policy-score): for the Gaussian policy
    N(theta_t . phi_t(x,m), tau^2), L_t^{(b)} is supported on row t alone and
    equals coeff[t,b] * features[t,b], with

        coeff[t,b] = (alpha_t^(b) - theta_t . phi_t^(b)) / tau^2,
        features[t,b] = phi_t(X_t^(b), M_t^(b)).

    Returned separately, of shapes (T,B) and (T,B,2), rather than as a dense
    (T,B,T,2) score tensor: every consumer here contracts them immediately,
    and the sparsity (one nonzero row per t) is exactly what makes that
    contraction cheap. `theta` is used as data (detached); no autograd graph
    is built or needed.
    """
    theta = theta.detach()
    X, alpha, M = out["X"][:-1], out["alpha"], out["M"][:-1]  # (T,B) each
    T = alpha.shape[0]
    features = torch.stack([env.policy_features(t, X[t], M[t]) for t in range(T)])  # (T,B,2)
    means = (theta.unsqueeze(1) * features).sum(dim=-1)  # (T,B)
    return (alpha - means) / env.config.tau**2, features


def policy_score_sum(coeff: torch.Tensor, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    sum_t sum_b w_t^(b) L_t^(b), the score-weighted sum every policy-gradient
    estimator here is built from, given `policy_score`'s (T,B) / (T,B,2)
    factors and matching (T,B) weights (typically G_t - baseline_t). Returns
    shape (T,2), i.e. theta's own shape.
    """
    return torch.einsum("tb,tbk->tk", weights * coeff, features)


def leave_one_out_baseline(G: torch.Tensor) -> torch.Tensor:
    """
    b_t^(b) = (1/(B-1)) sum_{b'!=b} G_t^(b'), the leave-one-out mean return
    (shape (T+1,B), matching `returns_to_go`'s G). Admissible as a baseline
    (Remark "Admissible baselines", continuous_state_space(2).tex): it is
    independent of trajectory b, and both the policy score and the
    population-perturbation score are centered conditionally on everything
    it depends on, so subtracting it leaves the estimator exactly unbiased
    while removing the (large) contribution of E[G_t] to its variance.
    Falls back to no baseline (all zeros) for B=1, where a leave-one-out
    mean does not exist.
    """
    B = G.shape[1]
    if B < 2:
        return torch.zeros_like(G)
    return (G.sum(dim=1, keepdim=True) - G) / (B - 1)
