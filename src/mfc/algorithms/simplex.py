"""
Simplex-perturbed discrete-state MF-REINFORCE.

Reference: Meunier, Pham & Reisinger, discrete-space theory
(files/reference/discrete_state_space(2).tex):
  - Sec. "Simplex density and perturbation score" for q_t=phi(U_t) and H_k(q).
  - Sec. "Policy-gradient formula for the perturbed objective" for the
    perturbed policy-gradient theorem.
  - Sec. "Model-free estimation of the population-flow sensitivities" for the
    single-batch forward estimator of D_t^theta(k).
  - Sec. "Discrete-state MF-REINFORCE algorithm", Algorithm 1, for the full
    training loop combining both.

What's specific to the simplex perturbation lives here: sampling q_t=phi(U_t)
and its score H_k(q), the auxiliary sensitivity-flow estimator, and the
plug-in gradient estimator built from them. Batched policy evaluation/
sampling/scoring and the nominal population-flow providers are generic
across algorithms and live in `mfc.algorithms._common` (imported below, and
re-exported here since simplex's own estimators need them directly).
"""

from __future__ import annotations

import torch

from ._common import eval_batched, exact_objective, exact_population_flow, one_hot, particle_population_flow, policy_score, sample_actions

__all__ = [
    "one_hot",
    "eval_batched",
    "sample_actions",
    "policy_score",
    "exact_population_flow",
    "particle_population_flow",
    "exact_objective",
    "sample_perturbation",
    "perturbation_score",
    "estimate_objective",
    "estimate_sensitivity_flow",
    "gradient_estimate",
    "gradient_step",
    "train",
]


def sample_perturbation(
    n_states: int,
    batch_shape: tuple[int, ...],
    sigma: float,
    *,
    dtype: torch.dtype,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample U ~ N(0, sigma^2 I_{N-1}) and q = phi(U) in the simplex interior,
    via the logistic parametrization. Since phi(u) = softmax([u, 0]),
    U = phi^{-1}(q) exactly, which is used by `perturbation_score` below.
    Returns (U, q) of shape (*batch_shape, N-1) and (*batch_shape, N).
    """
    U = sigma * torch.randn(*batch_shape, n_states - 1, dtype=dtype, device=device, generator=generator)
    logits = torch.cat([U, torch.zeros(*batch_shape, 1, dtype=dtype, device=device)], dim=-1)
    return U, torch.softmax(logits, dim=-1)


def perturbation_score(U: torch.Tensor, q: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    H_k(q), k=1,...,N-1, specialized to rho=N(0,sigma^2 I_{N-1}), for which
    a_i(z)=-z_i/sigma^2 and z=U exactly (U=phi^{-1}(q) by construction).
    Returns shape (*batch, N-1).
    """
    a = -U / sigma**2
    q_head, q_tail = q[..., :-1], q[..., -1:]
    return a / q_head + a.sum(dim=-1, keepdim=True) / q_tail - 1.0 / q_head + 1.0 / q_tail


def estimate_objective(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu_flow: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    lam: float,
    sigma: float,
    n_samples: int,
    *,
    gamma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of J^lambda(theta): n_samples independent
    lambda-perturbed trajectories (same rollout as `gradient_estimate`,
    without the score/gradient bookkeeping). `gamma=1.0` (default) recovers
    the undiscounted return; see `_common.exact_objective`. Returns shape
    (n_samples,) of returns G_0^(b); average for the estimate,
    std/sqrt(n_samples) for its Monte Carlo standard error.
    """
    theta = theta.detach()
    device, dtype = theta.device, theta.dtype
    N = env.n_states

    states = torch.multinomial(mu0.expand(n_samples, N), 1, generator=generator).reshape(n_samples)
    rewards = torch.zeros(T, n_samples, dtype=dtype, device=device)
    terminal_reward = None

    for t in range(T + 1):
        mu_t = mu_flow[t]
        _, q = sample_perturbation(N, (n_samples,), sigma, dtype=dtype, device=device, generator=generator)
        M = (1.0 - lam) * mu_t + lam * q
        if t < T:
            actions = sample_actions(action_probs_fn, theta, t, states, M, generator=generator)
            rewards[t] = env.reward(states, actions, M)
            states = env.sample_next_states(states, actions, M, generator=generator)
        else:
            terminal_reward = env.terminal_reward(states, M)

    discounts = gamma ** torch.arange(T, dtype=dtype, device=device)
    return (rewards * discounts.view(-1, 1)).sum(dim=0) + (gamma**T) * terminal_reward


def estimate_sensitivity_flow(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu_flow: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    n: int,
    eta: float,
    sigma: float,
    *,
    baseline: float | torch.Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Single-batch forward estimator of D_t^theta(k)=grad_theta mu_t^theta(k),
    k=1,...,N-1, for t=0,...,T, from one reusable batch of n auxiliary
    trajectories (eq. discrete-forward-sensitivity-estimator and
    discrete-forward-cumulative-update). Returns shape (T+1, N-1, D).
    """
    D = theta.numel()
    device, dtype = theta.device, theta.dtype
    N = env.n_states

    states = torch.multinomial(mu0.expand(n, N), 1, generator=generator).reshape(n)
    C = torch.zeros(n, D, dtype=dtype, device=device)
    D_hat = []

    for t in range(T + 1):
        indicator = one_hot(states, N, dtype)[:, : N - 1]  # (n, N-1)
        D_hat.append((indicator - baseline).T @ C / n)  # (N-1, D)
        if t == T:
            break

        mu_t = mu_flow[t]
        U, q = sample_perturbation(N, (n,), sigma, dtype=dtype, device=device, generator=generator)
        M = (1.0 - eta) * mu_t + eta * q
        actions = sample_actions(action_probs_fn, theta, t, states, M, generator=generator)
        Psi = policy_score(action_probs_fn, theta, t, states, actions, M)  # (n, D)
        H = perturbation_score(U, q, sigma)  # (n, N-1)
        C = C + Psi - ((1.0 - eta) / eta) * (H @ D_hat[t])
        states = env.sample_next_states(states, actions, M, generator=generator)

    return torch.stack(D_hat)


def gradient_estimate(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu_flow: torch.Tensor,
    mu0: torch.Tensor,
    D_hat: torch.Tensor,
    T: int,
    B: int,
    lam: float,
    sigma: float,
    *,
    gamma: float = 1.0,
    baseline: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Plug-in simplex policy-gradient estimator ĝ_{B,n,λ,η}(θ)
    (eq. discrete-plugin-gradient-estimator), using the shared auxiliary
    sensitivity estimate `D_hat`. `gamma=1.0` (default) recovers the
    undiscounted estimator; see `_common.exact_objective`. Returns shape (D,).

    Accumulated straight into one (D,) vector instead of materializing the
    per-(t,b) score tensors: written literally, the estimator needs (T,B,D)
    and (T+1,B,D) buffers for L_t and Q_t(D_t) plus same-sized temporaries
    for the weight-and-sum, which at the distribution-planning benchmark's
    D~7.7e4 and B=500 is several GB per gradient step — enough to OOM a
    24GB card shared by a few concurrent training jobs. Two exact
    rearrangements avoid them (the rollout, and hence the consumed random
    stream, is untouched, so this is the same estimator sample-for-sample):

      - Q_t is never formed. It is a rank-(N-1) product
        Q_t = -((1-λ)/λ) H_t D̂_t, so weighting it by w_t := G_t - b_t and
        summing over the batch can contract the (N-1) axis first:
        sum_b w_t^(b) Q_t^(b) = -((1-λ)/λ) (w_t^T H_t) D̂_t. Only the tiny
        (T+1,B,N-1) block of perturbation scores is kept.
      - L_t is consumed as it is produced, by swapping the two sums in
        sum_{t<T} L_t G_t (with G_t = sum_{s>=t} γ^s r_s + γ^T g_T):
            sum_{t<T} L_t G_t = sum_{t<T} (γ^t r_t) C_t + (γ^T g_T) C_{T-1},
        where C_t := sum_{s<=t} L_s is a running score sum, available at
        step t. Only C_t and the current L_t are ever live.
    """
    device, dtype = theta.device, theta.dtype
    N, D = env.n_states, theta.numel()
    baseline = torch.zeros(T + 1, dtype=dtype, device=device) if baseline is None else baseline

    states = torch.multinomial(mu0.expand(B, N), 1, generator=generator).reshape(B)
    rewards = torch.zeros(T, B, dtype=dtype, device=device)
    terminal_reward = torch.zeros(B, dtype=dtype, device=device)
    H = torch.zeros(T + 1, B, N - 1, dtype=dtype, device=device)
    C = torch.zeros(B, D, dtype=dtype, device=device)  # running score sum sum_{s<=t} L_s
    g = torch.zeros(D, dtype=dtype, device=device)

    for t in range(T + 1):
        mu_t = mu_flow[t]
        U, q = sample_perturbation(N, (B,), sigma, dtype=dtype, device=device, generator=generator)
        M = (1.0 - lam) * mu_t + lam * q
        H[t] = perturbation_score(U, q, sigma)  # (B, N-1)

        if t < T:
            actions = sample_actions(action_probs_fn, theta, t, states, M, generator=generator)
            L = policy_score(action_probs_fn, theta, t, states, actions, M)  # (B, D)
            C += L
            g -= baseline[t] * L.sum(dim=0)
            del L  # the step's largest tensor; drop it before the next one is built
            rewards[t] = env.reward(states, actions, M)
            g += (gamma**t) * (rewards[t] @ C)
            states = env.sample_next_states(states, actions, M, generator=generator)
        else:
            terminal_reward[:] = env.terminal_reward(states, M)
            g += (gamma**T) * (terminal_reward @ C)  # C is C_{T-1} here

    G = torch.zeros(T + 1, B, dtype=dtype, device=device)
    G[T] = (gamma**T) * terminal_reward
    for t in range(T - 1, -1, -1):
        G[t] = (gamma**t) * rewards[t] + G[t + 1]

    H_weighted = torch.einsum("tb,tbk->tk", G - baseline.view(-1, 1), H)  # (T+1, N-1)
    g -= ((1.0 - lam) / lam) * torch.einsum("tk,tkd->d", H_weighted, D_hat)
    return g / B


def gradient_step(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    *,
    T: int,
    n_aux: int,
    B: int,
    lam: float,
    eta: float | None = None,
    sigma: float = 1.0,
    gamma: float = 1.0,
    baseline_D: float | torch.Tensor = 0.0,
    baseline_G: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    One full plug-in gradient estimate ĝ_{B,n,λ,η}(θ) at the given θ and
    initial law: obtain the nominal flow via `population_flow_fn` (exact by
    default; pass e.g. `functools.partial(particle_population_flow,
    n_particles=...)` for the empirical-flow variant of Assumption "Access
    to the nominal population flow"), estimate the population-flow
    sensitivities from one auxiliary batch, then form the estimate from one
    main batch (Algorithm 1's per-iteration body, factored out so callers
    that need a different μ_0 each step — e.g. the reference's randomized
    training protocol — can drive the loop themselves). `gamma=1.0`
    (default) recovers the undiscounted estimator; see
    `_common.exact_objective`. Returns shape (D,).
    """
    eta = lam if eta is None else eta
    mu_flow = population_flow_fn(env, action_probs_fn, theta, mu0, T, generator=generator)
    D_hat = estimate_sensitivity_flow(
        env, action_probs_fn, theta, mu_flow, mu0, T, n_aux, eta, sigma, baseline=baseline_D, generator=generator
    )
    return gradient_estimate(
        env, action_probs_fn, theta, mu_flow, mu0, D_hat, T, B, lam, sigma, gamma=gamma, baseline=baseline_G, generator=generator
    )


def train(
    env,
    action_probs_fn,
    theta0: torch.Tensor,
    mu0: torch.Tensor,
    *,
    T: int,
    n_train: int,
    n_aux: int,
    B: int,
    lam: float,
    eta: float | None = None,
    sigma: float = 1.0,
    gamma: float = 1.0,
    lr: float = 1e-3,
    baseline_D: float | torch.Tensor = 0.0,
    baseline_G: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    project=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1 (Simplex-perturbed discrete-state MF-REINFORCE) with a fixed
    initial law μ_0, as in the formal algorithm statement (Algorithm 1's
    input list). `population_flow_fn` selects exact vs. particle nominal-flow
    estimation (see `gradient_step`). `project`, if given, implements
    Pi_Theta and is applied in-place after each step. Returns (theta_final,
    theta_history) with history shape (n_train+1, D).
    """
    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    history = [theta.detach().clone()]

    for _ in range(n_train):
        g_hat = gradient_step(
            env,
            action_probs_fn,
            theta,
            mu0,
            T=T,
            n_aux=n_aux,
            B=B,
            lam=lam,
            eta=eta,
            sigma=sigma,
            gamma=gamma,
            baseline_D=baseline_D,
            baseline_G=baseline_G,
            population_flow_fn=population_flow_fn,
            generator=generator,
        )
        optimizer.zero_grad()
        theta.grad = -g_hat
        optimizer.step()
        if project is not None:
            theta.data = project(theta.data)
        history.append(theta.detach().clone())

    return theta.detach(), torch.stack(history)
