"""
Continuous-state simplex-perturbed MF-REINFORCE — the continuous-state
counterpart of `mfc.algorithms.simplex`.

Reference: files/Research_Project.tex, Sec. "Continuous State Space"
(the perturbed process P_t^lambda = (Id + lambda*f_t) # P_{X_t}, its policy
gradient in the LQ setting, and the model-free estimation of the moment
sensitivities), in the exact form of files/reference/continuous_state_space(2).tex:
  - Sec. "Finite-dimensional law coordinates and perturbation score" for the
    chart m_t, the coordinate perturbation and the score S_t^{lambda,theta}.
  - Theorem "Perturbed policy-gradient formula" for the estimator's shape.
  - Sec. "Model-free estimation of the coordinate sensitivities" and
    Algorithm "Single-batch forward estimation of the coordinate
    sensitivities" for D_hat.
  - Algorithm "Linear-time continuous-state MF-REINFORCE" for the training
    loop combining both (driven from `mfc.algorithms.continuous.train`).

What changes relative to the discrete state space is only *how the
population law is randomized*. There, the law is a probability vector and is
perturbed inside the simplex, M_t = (1-lambda)*mu_t + lambda*q_t with q_t
drawn on Delta_N. Here the law lives in P_2(R) and is perturbed by
transport: with f(x) = zeta*x + beta the affine direction and (zeta,beta) ~
N(0,rho^2) x N(0,rho^2),

    M_t^{lambda,theta} = (Id + lambda*f_t) # mu_t^theta.

Both continuous benchmarks in this repo (`mfc.environments.lq`,
`mfc.environments.portfolio`) consume the population law through its mean
alone, so the chart of Assumption "Finite-dimensional law chart" is
one-dimensional (K=1): the coordinate is the population mean,

    c_t^theta = mu_t^theta = E[X_t^theta],   m_t(c) = the state law of mean c,

and the transported law has mean

    M_t^{lambda,theta} = (1+lambda*zeta_t)*mu_t^theta + lambda*beta_t
                       = mu_t^theta + lambda*rho*sqrt((mu_t^theta)^2+1)*xi_t,

with xi_t ~ N(0,1) standard (this is `env.rollout`'s `xi`). Using the
*marginal* law of the mean coordinate, rather than the joint law of
(mean, variance) that the affine map also moves, is exact and not an
approximation: the return is conditionally independent of the transported
variance given the transported mean, so the extra conditional score
integrates to zero against it — and dropping it only removes variance.

Perturbation score. The coordinate density is
q_t^{lambda,theta}(m) = N(m; mu_t^theta, lambda^2*rho^2*((mu_t^theta)^2+1)),
so with D_t^theta := grad_theta mu_t^theta = grad_theta c_t^theta,

    S_t^{lambda,theta}(M_t) = grad_theta log q_t^{lambda,theta}(M_t)
                            = D_t^theta * h_t,
    h_t = xi_t / (lambda*rho*sqrt(mu_t^2+1)) + (xi_t^2-1)*mu_t/(mu_t^2+1).

The first term is the reference's generic transport score
lambda^{-1}(D_t)^T Xi_t (eq. continuous-measure-score-forward-algorithm);
the second is the correction carried by the *scale* of this particular
perturbation, which moves with mu_t^theta because the affine direction
multiplies the state (a pure-translation direction f(x)=beta would drop it).
Both terms are centered, E[h_t]=0, which is what the algorithm's causality
and baseline arguments rest on.

`mfc.algorithms.continuous` owns the training loop and the descend-vs-ascend
sign; everything here returns grad_theta J^lambda(theta) in the
environment's own sign convention (see `mfc.algorithms._continuous`).
"""

from __future__ import annotations

import torch

from ._continuous import leave_one_out_baseline, policy_score, policy_score_sum, returns_to_go

__all__ = [
    "perturbation_score",
    "resolve_baseline",
    "estimate_objective",
    "estimate_sensitivity_flow",
    "gradient_estimate",
    "gradient_step",
]


def perturbation_score(xi: torch.Tensor, mu: torch.Tensor, lam: float, rho: float) -> torch.Tensor:
    """
    The scalar factor h_t of the population-perturbation score
    S_t^{lambda,theta}(M_t) = D_t^theta * h_t (see the module docstring),
    evaluated at the standardized perturbation `xi` (= `env.rollout`'s xi)
    around the nominal coordinate `mu` (= its `mu`). Shapes broadcast, so
    this takes a whole (T+1,B) block at once given mu of shape (T+1,1).
    """
    scale = torch.sqrt(mu**2 + 1.0)
    return xi / (lam * rho * scale) + (xi**2 - 1.0) * mu / scale**2


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
) -> torch.Tensor:
    """
    Single-batch forward estimator of the coordinate sensitivities
    D_t^theta = grad_theta c_t^theta = grad_theta mu_t^theta, t=0,...,T, from
    one reusable batch of `n` auxiliary eta-perturbed trajectories
    (Algorithm "Single-batch forward estimation of the coordinate
    sensitivities"; eqs. continuous-forward-sensitivity-estimator and
    continuous-forward-cumulative-score-update). Returns shape (T+1, T, 2) —
    one theta-shaped sensitivity per time index, K=1 needing no extra axis.

    Both parts of Assumption "Observable moments and recoverable coordinates"
    are trivial for the mean coordinate: psi_t(x)=x and chi_t=identity, so
    the recovery Jacobian J_t^theta is 1 and the transformed statistic is
    Phi_t(x) = x - b_t. The deterministic baseline b_t is the nominal mean
    mu_t^theta itself, which leaves the expectation unchanged (the cumulative
    scores are centered, Remark "Conditional centering") and removes the
    O(mu_t) offset from the outer product.

    The rollout is generated in full first and the recursion run over it
    afterwards, which is equivalent to interleaving them: the simulated
    state, perturbation and action trajectories do not depend on the
    sensitivity estimates at all — only the cumulative scores do (Sec. "Bias
    and mean-squared error of the linear-time estimator"). D_hat_t is still
    formed strictly before time t's perturbation and action enter any score,
    preserving the martingale structure the analysis relies on.
    """
    theta = theta.detach()
    T = theta.shape[0]
    out = env.rollout(theta, eta, n, generator=generator)
    mu, X, xi = out["mu"], out["X"], out["xi"]
    coeff, features = policy_score(env, theta, out)  # (T,n), (T,n,2)

    C = torch.zeros(n, T, 2, dtype=theta.dtype, device=theta.device)  # cumulative scores C_t^(r)
    D_hat = []
    for t in range(T + 1):
        Phi = X[t] - mu[t]  # (n,)
        D_hat.append(torch.einsum("r,rtk->tk", Phi, C) / n)
        if t == T:
            break
        h = perturbation_score(xi[t], mu[t], eta, env.config.rho)  # (n,)
        C = C + h.view(-1, 1, 1) * D_hat[t]  # plug-in transport score, shared across the batch
        C[:, t, :] = C[:, t, :] + coeff[t].unsqueeze(-1) * features[t]  # policy score, row t only
    return torch.stack(D_hat)


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
    ghat_{B,n,lambda}(theta) (eqs. continuous-forward-main-replica and
    continuous-forward-shared-gradient-estimator),

        (1/B) sum_b sum_t (D_hat_t * h_t^(b) + 1_{t<T} L_t^(b)) (G_t^(b)-b_t),

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
    G = returns_to_go(out)  # (T+1,B)
    B = G.shape[1]
    weights = G - resolve_baseline(G, baseline)

    coeff, features = policy_score(env, theta, out)
    g = policy_score_sum(coeff, features, weights[:-1])  # policy-score term, (T,2)

    h = perturbation_score(out["xi"], out["mu"].unsqueeze(-1), lam, env.config.rho)  # (T+1,B)
    g = g + torch.einsum("t,tdk->dk", (weights * h).sum(dim=1), D_hat)  # population-score term
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
    coordinate-sensitivity flow from one auxiliary batch of `n_aux`
    eta-perturbed trajectories (`eta` defaults to `lam`, as in the
    reference's own algorithm), then form the estimate from one main batch
    of `B` lambda-perturbed trajectories. This is the per-iteration body of
    Algorithm "Linear-time continuous-state MF-REINFORCE" — (n_aux+B)*T
    simulated transitions, linear in the horizon. Returns shape (T,2).
    """
    eta = lam if eta is None else eta
    D_hat = estimate_sensitivity_flow(env, theta, n_aux, eta, generator=generator)
    out = env.rollout(theta.detach(), lam, B, generator=generator)
    return gradient_estimate(env, theta, out, D_hat, lam, baseline=baseline)
