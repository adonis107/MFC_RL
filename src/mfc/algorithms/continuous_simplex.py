"""
Continuous-state affine-perturbation MF-REINFORCE.

Reference: files/Research_Project.tex, Sec. "Continuous State Space":
  - "Policy gradient for the perturbed control problem in the LQ setting"
    defines the affine law perturbation
    (I + lambda f_t)#P_{X_t}, with f_t(x)=A_t x + B_t.
  - Eq. "policy_gradient_LQ_setting" decomposes the gradient into the usual
    policy score and the score of the generated random law q_t^{lambda,theta}.
  - Eq. "model_free_moments" estimates the required mean and variance
    sensitivities by the same likelihood-ratio idea.
  - The LQ and portfolio benchmark sections give the closed-form perturbed
    objectives and exact-gradient oracles used for validation.

This module deliberately does not use files/reference/continuous_state_space(2).tex.

The affine perturbation enters the simulator through the generated Gaussian
population argument

    M_t^{lambda,theta} = (1+lambda*zeta_t)*mu_t^theta + lambda*beta_t,
    Sigma_t^{lambda,theta} = (1+lambda*zeta_t)^2 Sigma_t^theta,

with zeta_t,beta_t independent N(0,rho^2). The estimator uses the joint
q_t^{lambda,theta}(M_t,Sigma_t) score from Research_Project.tex, contracted
against model-free estimates of grad_theta mu_t^theta and
grad_theta log sigma_t^theta.

Perturbation score. The generated-law density is
q_t^{lambda,theta}(m,Sigma). With c_t = 1+lambda*zeta_t, a_t=zeta_t,
b_t=beta_t, D_t=grad_theta mu_t and K_t=grad_theta log sigma_t,

    grad_theta log q_t^{lambda,theta}(M_t,Sigma_t)
      = h_t^m D_t + h_t^sigma K_t,

    h_t^m = beta_t*c_t/(lambda*rho^2),
    h_t^sigma = c_t*(zeta_t - mu_t*beta_t)/(lambda*rho^2) - 1.

These coefficients are centered under the perturbation distribution, which
is what the algorithm's causality and baseline arguments rest on.

`mfc.algorithms.continuous` owns the training loop and the descend-vs-ascend
sign; everything here returns grad_theta J^lambda(theta) in the
environment's own sign convention (see `mfc.algorithms._continuous`).
"""

from __future__ import annotations

import torch

from ._continuous import leave_one_out_baseline, policy_score, policy_score_sum, returns_to_go

__all__ = [
    "population_score_coefficients",
    "perturbation_score",
    "resolve_baseline",
    "estimate_objective",
    "estimate_sensitivity_flow",
    "gradient_estimate",
    "gradient_step",
]


def population_score_coefficients(
    zeta: torch.Tensor,
    beta: torch.Tensor,
    mu: torch.Tensor,
    lam: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scalar coefficients (h_m, h_log_sigma) of the joint generated-law score
    from `docs/continuous_simplex_joint_moments.md`.

    Shapes broadcast, so this takes a whole (T+1,B) block at once given
    `mu` of shape (T+1,1). `zeta` and `beta` are the sampled affine
    perturbation coordinates, not the standardized marginal-mean `xi`.
    """
    c = 1.0 + lam * zeta
    denom = lam * rho**2
    return beta * c / denom, c * (zeta - mu * beta) / denom - 1.0


def perturbation_score(xi: torch.Tensor, mu: torch.Tensor, lam: float, rho: float) -> torch.Tensor:
    """
    Backward-compatible marginal generated-mean score used by the old
    mean-only specialization. New code should use
    `population_score_coefficients`, which implements the joint
    `(M_t,Sigma_t)` score from `Research_Project.tex`.
    """
    scale = torch.sqrt(mu**2 + 1.0)
    return xi / (lam * rho * scale) + (xi**2 - 1.0) * mu / scale**2


def _split_sensitivity_flow(D_hat, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accept the new dict-shaped moment sensitivity flow, while keeping a bare
    tensor as a mean-only compatibility fallback for older diagnostics.
    """
    if isinstance(D_hat, dict):
        D_mu = D_hat["mean"]
        D_log_sigma = D_hat["log_sigma"]
    else:
        D_mu = D_hat
        D_log_sigma = torch.zeros_like(D_mu)
    expected = (theta.shape[0] + 1, *theta.shape)
    if tuple(D_mu.shape) != expected or tuple(D_log_sigma.shape) != expected:
        raise ValueError(f"moment sensitivity flow must have shape {expected}; got {tuple(D_mu.shape)} and {tuple(D_log_sigma.shape)}")
    return D_mu, D_log_sigma


def resolve_baseline(G: torch.Tensor, baseline) -> torch.Tensor:
    """
    Turn the `baseline` argument shared by the estimators below into the
    (T+1,B) array subtracted from the returns: None for no baseline, "loo"
    for `_continuous.leave_one_out_baseline`, or an explicit deterministic
    (T+1,) / (T+1,B) tensor (Remark "Admissible baselines").
    """
    if baseline is None:
        return torch.zeros_like(G)
    if isinstance(baseline, str):
        if baseline != "loo":
            raise ValueError(f"unknown baseline {baseline!r}; available: None, 'loo', or a (T+1,) tensor")
        return leave_one_out_baseline(G)
    return torch.as_tensor(baseline, dtype=G.dtype, device=G.device).reshape(-1, 1).expand_as(G)


def estimate_objective(
    env,
    theta: torch.Tensor,
    lam: float,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of J^lambda(theta) from `n_samples` independent
    lambda-perturbed trajectories (the same rollout the gradient estimators
    use, without any score bookkeeping). Returns shape (n_samples,) of total
    returns G_0^(b); average for the estimate, std/sqrt(n_samples) for its
    Monte Carlo standard error. Both continuous benchmarks also have
    `env.exact_objective(theta, lam)` in closed form, so this is a check on
    the simulator rather than the only route to J^lambda — unlike
    `mfc.algorithms.simplex.estimate_objective`.
    """
    return returns_to_go(env.rollout(theta.detach(), lam, n_samples, generator=generator))[0]


def estimate_sensitivity_flow(
    env,
    theta: torch.Tensor,
    n: int,
    eta: float,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    Single-batch forward estimator of the moment sensitivities
    D_t^theta = grad_theta mu_t^theta and
    K_t^theta = grad_theta log sigma_t^theta, t=0,...,T, from one reusable
    batch of `n` auxiliary eta-perturbed trajectories. This implements
    Research_Project.tex's model-free estimates for grad E[X_t^theta] and
    grad Var(X_t^theta), then converts the latter to grad log sigma_t.
    Returns a dict with two tensors, each of shape (T+1, T, 2).

    The transformed statistics are X_t - mu_t^theta and
    (X_t - mu_t^theta)^2 - Sigma_t^theta. These deterministic baselines
    leave the expectation unchanged because the running scores are centered,
    and remove large offsets from the outer products.

    The rollout is generated in full first and the recursion run over it
    afterwards, which is equivalent to interleaving them: the simulated
    state, perturbation and action trajectories do not depend on the
    sensitivity estimates at all — only the cumulative scores do. D_hat_t is still
    formed strictly before time t's perturbation and action enter any score,
    preserving the martingale structure the analysis relies on.
    """
    theta = theta.detach()
    T = theta.shape[0]
    out = env.rollout(theta, eta, n, generator=generator)
    mu, Sigma, X = out["mu"], out["Sigma_nominal"], out["X"]
    coeff, features = policy_score(env, theta, out)  # (T,n), (T,n,2)

    C = torch.zeros(n, T, 2, dtype=theta.dtype, device=theta.device)  # cumulative scores C_t^(r)
    D_mu_hat = []
    D_log_sigma_hat = []
    for t in range(T + 1):
        centered = X[t] - mu[t]  # (n,)
        D_mu_t = torch.einsum("r,rtk->tk", centered, C) / n
        D_var_t = torch.einsum("r,rtk->tk", centered**2 - Sigma[t], C) / n
        D_log_sigma_t = D_var_t / (2.0 * Sigma[t])
        D_mu_hat.append(D_mu_t)
        D_log_sigma_hat.append(D_log_sigma_t)
        if t == T:
            break
        h_mu, h_log_sigma = population_score_coefficients(out["zeta"][t], out["beta"][t], mu[t], eta, env.config.rho)  # (n,), (n,)
        C = C + h_mu.view(-1, 1, 1) * D_mu_t + h_log_sigma.view(-1, 1, 1) * D_log_sigma_t
        C[:, t, :] = C[:, t, :] + coeff[t].unsqueeze(-1) * features[t]  # policy score, row t only
    return {"mean": torch.stack(D_mu_hat), "log_sigma": torch.stack(D_log_sigma_hat)}


def gradient_estimate(
    env,
    theta: torch.Tensor,
    out: dict[str, torch.Tensor],
    D_hat: torch.Tensor,
    lam: float,
    *,
    baseline=None,
) -> torch.Tensor:
    """
    The shared-sensitivity plug-in gradient estimator
    ghat_{B,n,lambda}(theta),

        (1/B) sum_b sum_t
            (D_t h_t^m + K_t h_t^sigma + 1_{t<T} L_t^(b)) (G_t^(b)-b_t),

    from one main batch `out` (an `env.rollout` at scale `lam`) and the
    auxiliary sensitivity flow `D_hat`. Returns shape (T,2).

    Neither the (T+1,B,T,2) block of per-sample perturbation scores nor the
    (T,B,T,2) block of per-sample policy scores is ever formed: the first is
    a rank-one product in the sensitivity, so the batch is contracted against
    the scalars h_t^(b) first and the result multiplied into D_hat once per
    t; the second is supported on a single row per t (see
    `_continuous.policy_score`). `baseline` is None, "loo", or a
    deterministic (T+1,) tensor — see `resolve_baseline`.
    """
    D_mu, D_log_sigma = _split_sensitivity_flow(D_hat, theta)
    G = returns_to_go(out)  # (T+1,B)
    B = G.shape[1]
    weights = G - resolve_baseline(G, baseline)

    coeff, features = policy_score(env, theta, out)
    g = policy_score_sum(coeff, features, weights[:-1])  # policy-score term, (T,2)

    h_mu, h_log_sigma = population_score_coefficients(out["zeta"], out["beta"], out["mu"].unsqueeze(-1), lam, env.config.rho)
    g = g + torch.einsum("t,tdk->dk", (weights * h_mu).sum(dim=1), D_mu)
    g = g + torch.einsum("t,tdk->dk", (weights * h_log_sigma).sum(dim=1), D_log_sigma)
    return g / B


def gradient_step(
    env,
    theta: torch.Tensor,
    lam: float,
    *,
    B: int,
    n_aux: int,
    eta: float | None = None,
    baseline=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    One full plug-in gradient estimate at the given theta: estimate the
    moment-sensitivity flow from one auxiliary batch of `n_aux`
    eta-perturbed trajectories (`eta` defaults to `lam`, as in the
    reference's own algorithm), then form the estimate from one main batch
    of `B` lambda-perturbed trajectories. This is the per-iteration body of
    Research_Project.tex's continuous-state MF-REINFORCE gradient body —
    (n_aux+B)*T simulated transitions, linear in the horizon. Returns shape
    (T,2).
    """
    eta = lam if eta is None else eta
    D_hat = estimate_sensitivity_flow(env, theta, n_aux, eta, generator=generator)
    out = env.rollout(theta.detach(), lam, B, generator=generator)
    return gradient_estimate(env, theta, out, D_hat, lam, baseline=baseline)
