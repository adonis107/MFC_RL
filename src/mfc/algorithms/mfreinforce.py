"""
Logit-perturbed discrete-state MF-REINFORCE (Meunier, Pham & Reisinger,
"Model-free Learning for Mean-Field Control Problems", files/Discrete RL -
Meunier, Pham, Reisinger.md).

Population laws are parametrized by (unreduced) logits l=log(mu) in R^N
(Sec. 2.1); the perturbed law is M_t^{eps,theta} = softmax(log(mu_t) +
eps*Lambda_t), Lambda_t ~ N(0, I_N) (Sec. 2.3). Because this perturbation is
a simple additive Gaussian shift in logit space, its integration-by-parts
score is just the Gaussian's own score eps^{-1}*Lambda_t (Stein's identity
for a shifted mean) — no analogue of simplex's H_k(q) is needed (Theorem
2.3, eq. 2.6):

    grad_theta V_eps(mu_0,theta)
      = E[ sum_{t=0}^T (eps^{-1} Lambda_t . D_t + 1_{t<T} L_t) G_t^eps ],

with D_t := grad_theta l_t^theta in R^{N x D} (gradient of the LOGITS, not
the probabilities) and L_t the usual policy score.

D_t is estimated by Algorithm 2 ("State Distribution Gradient Estimation"):
for each target time t=1,...,T, a *fresh* batch of n auxiliary trajectories
is resimulated from scratch over s=0,...,t-1 (reusing the already-computed
D_0,...,D_{t-1} from earlier target times, but never reusing trajectories
across target times). This is NOT the reusable single-batch trick of
discrete_state_space(2).tex's simplex estimator — see its Remark
"Comparison with stagewise logit estimation" — so one training step costs
B*T + B*n*T*(T+1)/2 simulated transitions (Algorithm 2's own O(n*T(T+1)/2)
cost, paid once per each of the B main trajectories), quadratic in T,
against simplex's linear T*(n_aux+B). This drives the equal-budget
allocation in configs/twostate.py.

Shares the generic batched policy/flow machinery in `mfc.algorithms._common`
(same environment and action_probs_fn contracts as `mfc.algorithms.simplex`
and `mfc.algorithms.reinforce`).
"""

from __future__ import annotations

import torch

from ._common import exact_population_flow, one_hot, policy_score, sample_actions, weighted_score_sum

__all__ = ["estimate_logit_sensitivity_flow", "gradient_estimate", "gradient_step", "train"]


def _perturbed_law(mu: torch.Tensor, Lambda: torch.Tensor, epsilon: float) -> torch.Tensor:
    """M = softmax(log(mu) + epsilon*Lambda)."""
    l = torch.log(mu.clamp_min(1e-12)) + epsilon * Lambda
    return torch.softmax(l, dim=-1)


def estimate_logit_sensitivity_flow(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu_flow: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    n: int,
    epsilon: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Algorithm 2: stagewise estimator of D_t := grad_theta l_t^theta
    (gradient of the LOGITS log(mu_t^theta)), t=0,...,T. D_0=0 (mu_0 is
    theta-independent). For each t=1,...,T, a fresh batch of n trajectories
    is simulated over s=0,...,t-1 using the already-computed D_0,...,D_{t-1}
    (from earlier target times) inside the running score

        C_s^(k) += eps^{-1} Lambda_s^(k) . D_s + Psi_s^(k),

    then D_t = ((1/n) sum_k 1{Y_t^(k)=x^(i)} C_t^(k))_i / mu_flow_t(i)
    (elementwise; converts the sensitivity of the probability to the
    sensitivity of its log). Returns shape (T+1, N, D).
    """
    D = theta.numel()
    dtype, device = theta.dtype, theta.device
    N = env.n_states

    D_hat = [torch.zeros(N, D, dtype=dtype, device=device)]
    for t in range(1, T + 1):
        states = torch.multinomial(mu0.expand(n, N), 1, generator=generator).reshape(n)
        C = torch.zeros(n, D, dtype=dtype, device=device)
        for s in range(t):
            mu_s = mu_flow[s]
            Lambda = torch.randn(n, N, dtype=dtype, device=device, generator=generator)
            M = _perturbed_law(mu_s, Lambda, epsilon)
            actions = sample_actions(action_probs_fn, theta, s, states, M, generator=generator)
            Psi = policy_score(action_probs_fn, theta, s, states, actions, M)  # (n, D)
            C = C + (Lambda @ D_hat[s]) / epsilon + Psi
            states = env.sample_next_states(states, actions, M, generator=generator)

        indicator = one_hot(states, N, dtype)  # (n, N)
        D_t = (indicator.T @ C) / n  # (N, D), sensitivity of the probability
        D_hat.append(D_t / mu_flow[t].unsqueeze(-1).clamp_min(1e-12))  # sensitivity of the logit

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
    epsilon: float,
    *,
    gamma: float = 1.0,
    baseline: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Plug-in logit-perturbed policy-gradient estimator (Theorem 2.3, eq.
    2.6), using the shared auxiliary sensitivity estimate `D_hat`. `gamma=1.0`
    (default) recovers the undiscounted estimator; see
    `_common.exact_objective`. Returns shape (D,).

    The (T,B,D)/(T+1,B,D) score buffers the formula suggests are never
    built, by the same two exact rearrangements `simplex.gradient_estimate`
    documents in full: eps^{-1} Lambda_t . D_t is a rank-N product, so the
    batch sum contracts the (N) axis before the (D) one; and the policy-score
    term goes through `_common.weighted_score_sum`. Same estimator, same
    random stream, a fraction of the memory (which for a large policy is the
    whole cost of a gradient step).
    """
    device, dtype = theta.device, theta.dtype
    N = env.n_states
    baseline = torch.zeros(T + 1, dtype=dtype, device=device) if baseline is None else baseline

    states = torch.multinomial(mu0.expand(B, N), 1, generator=generator).reshape(B)
    rewards = torch.zeros(T, B, dtype=dtype, device=device)
    terminal_reward = torch.zeros(B, dtype=dtype, device=device)
    Lambdas = torch.zeros(T + 1, B, N, dtype=dtype, device=device)
    steps = []

    for t in range(T + 1):
        mu_t = mu_flow[t]
        Lambda = torch.randn(B, N, dtype=dtype, device=device, generator=generator)
        Lambdas[t] = Lambda
        M = _perturbed_law(mu_t, Lambda, epsilon)

        if t < T:
            actions = sample_actions(action_probs_fn, theta, t, states, M, generator=generator)
            steps.append((t, states, actions, M))
            rewards[t] = env.reward(states, actions, M)
            states = env.sample_next_states(states, actions, M, generator=generator)
        else:
            terminal_reward[:] = env.terminal_reward(states, M)

    G = torch.zeros(T + 1, B, dtype=dtype, device=device)
    G[T] = (gamma**T) * terminal_reward
    for t in range(T - 1, -1, -1):
        G[t] = (gamma**t) * rewards[t] + G[t + 1]

    weights = G - baseline.view(-1, 1)  # (T+1, B)
    g = weighted_score_sum(action_probs_fn, theta, steps, weights[:T])
    Lambda_weighted = torch.einsum("tb,tbn->tn", weights, Lambdas)  # (T+1, N)
    g += torch.einsum("tn,tnd->d", Lambda_weighted, D_hat) / epsilon
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
    epsilon: float,
    gamma: float = 1.0,
    baseline_G: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One full plug-in gradient estimate at the given theta and initial
    law: obtain the nominal flow via `population_flow_fn` (exact by
    default; pass e.g. `functools.partial(particle_population_flow,
    n_particles=...)` for the empirical-flow variant), estimate the
    logit-sensitivity flow from Algorithm 2, then form the estimate from one
    main batch. Returns shape (D,)."""
    mu_flow = population_flow_fn(env, action_probs_fn, theta, mu0, T, generator=generator)
    D_hat = estimate_logit_sensitivity_flow(env, action_probs_fn, theta, mu_flow, mu0, T, n_aux, epsilon, generator=generator)
    return gradient_estimate(env, action_probs_fn, theta, mu_flow, mu0, D_hat, T, B, epsilon, gamma=gamma, baseline=baseline_G, generator=generator)


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
    epsilon: float,
    gamma: float = 1.0,
    lr: float = 1e-3,
    baseline_G: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    project=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1 (Mean-Field REINFORCE) with a fixed initial law mu0, mirroring
    `simplex.train`'s structure and calling convention. `project`, if given,
    implements Pi_Theta and is applied in-place after each step. Returns
    (theta_final, theta_history) with history shape (n_train+1, D).
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
            epsilon=epsilon,
            gamma=gamma,
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
