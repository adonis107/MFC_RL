"""
Perturbation-agnostic machinery shared by every discrete-state algorithm
(simplex, mfreinforce, reinforce): batched policy evaluation/sampling and
scoring via `torch.func`, and the nominal population-flow providers (exact
recursion and the particle/empirical estimate of Assumption "Access to the
nominal population flow", discrete_state_space(2).tex).

Generic over any discrete-state environment exposing `n_states`, `n_actions`,
`transition_probs(states, actions, mu)`, `sample_next_states(states, actions,
mu)`, `reward(states, actions, mu)` and `terminal_reward(states, mu)`
(see `mfc.environments.twostate`), and over any policy given as a pure
function `action_probs_fn(theta, t, state, mu) -> probs` evaluated on a
single (unbatched) sample: `state` a 0-d long tensor, `mu` a `(n_states,)`
tensor, returning a `(n_actions,)` probability vector. This function is
traced under `torch.func.vmap`/`grad`, so it must avoid `.item()` and any
data-dependent Python control flow; use `one_hot` below for indexing.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap


def one_hot(index: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    """vmap-safe one-hot encoding (`torch.nn.functional.one_hot` is not vmap-compatible)."""
    return (torch.arange(n, device=index.device) == index.unsqueeze(-1)).to(dtype)


def eval_batched(fn, theta: torch.Tensor, t: int, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Evaluate `fn(theta, t, state, mu)` over a batch via vmap. `mu` may be a
    single shared law of shape (n_states,) or already batched to match `states`."""
    flat_states = states.reshape(-1)
    flat_mu = mu.expand(*states.shape, mu.shape[-1]).reshape(-1, mu.shape[-1])
    out = vmap(lambda s, m: fn(theta, t, s, m))(flat_states, flat_mu)
    return out.reshape(*states.shape, *out.shape[1:])


def sample_actions(
    action_probs_fn,
    theta: torch.Tensor,
    t: int,
    states: torch.Tensor,
    mu: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample actions ~ pi_t^theta(.|states, mu)."""
    probs = eval_batched(action_probs_fn, theta.detach(), t, states, mu)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator)
    return flat.reshape(states.shape)


def policy_score(
    action_probs_fn,
    theta: torch.Tensor,
    t: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample policy score L = grad_theta log pi_t^theta(a|x,mu), holding
    (x,a,mu) fixed, shape (*states.shape, D). Computed functionally via
    `torch.func.grad`, so `theta` need not require grad.
    """
    flat_states = states.reshape(-1)
    flat_actions = actions.reshape(-1)
    flat_mu = mu.expand(*states.shape, mu.shape[-1]).reshape(-1, mu.shape[-1])

    def log_prob(th, s, a, m):
        probs = action_probs_fn(th, t, s, m)
        return torch.log((probs * one_hot(a, probs.shape[-1], probs.dtype)).sum().clamp_min(1e-12))

    scores = vmap(grad(log_prob), in_dims=(None, 0, 0, 0))(theta, flat_states, flat_actions, flat_mu)
    return scores.reshape(*states.shape, -1)


def exact_population_flow(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    *,
    generator: torch.Generator | None = None,
    detach: bool = True,
) -> torch.Tensor:
    """
    Propagate mu_t^theta exactly: mu_{t+1} = mu_t @ K_t(mu_t), with
    K_t(mu)[x,x'] = sum_a pi_t(a|x,mu) P(x'|x,a,mu). Returns shape
    (T+1, n_states). `generator` is accepted (unused; this flow is
    deterministic) so this and `particle_population_flow` share one
    `population_flow_fn(env, action_probs_fn, theta, mu0, T, *, generator)`
    contract.

    `detach=True` (default): theta is detached first, since within a
    training loop the nominal flow is a fixed input to the perturbation
    construction and must never be differentiated (the "Model-free
    structure" remark). `detach=False` keeps theta attached, making the
    returned flow (and objectives built from it, e.g. `exact_objective`)
    differentiable via ordinary autograd — used only for diagnostics
    (`scripts/test.py`'s gradient-bias oracle), never inside a model-free
    estimator itself.
    """
    if detach:
        theta = theta.detach()
        mu0 = mu0.detach()
    N, A = env.n_states, env.n_actions
    device = mu0.device
    states_grid = torch.arange(N, device=device).unsqueeze(-1).expand(N, A)
    actions_grid = torch.arange(A, device=device).unsqueeze(0).expand(N, A)
    states_all = torch.arange(N, device=device)

    mu_flow = [mu0]
    for t in range(T):
        mu_t = mu_flow[-1]
        pi = eval_batched(action_probs_fn, theta, t, states_all, mu_t)  # (N, A)
        P = env.transition_probs(states_grid, actions_grid, mu_t)  # (N, A, N)
        K = torch.einsum("xa,xay->xy", pi, P)
        mu_flow.append(mu_t @ K)
    return torch.stack(mu_flow)


def particle_population_flow(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    *,
    n_particles: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Estimate the nominal flow (mu_t^theta)_{t=0}^T empirically from
    n_particles independent, mean-field-coupled trajectories under the
    unperturbed dynamics (Assumption "Access to the nominal population
    flow"): mu_hat_t(i) = (1/Ntilde) sum_r 1{X_t^{(r)}=i}, with each
    particle's action and transition drawn using the *current empirical*
    mu_hat_t as the shared population argument. Same output shape as
    `exact_population_flow`: (T+1, n_states). `theta` is detached for the
    same reason as `exact_population_flow`.
    """
    theta = theta.detach()
    N = env.n_states
    dtype, device = mu0.dtype, mu0.device

    states = torch.multinomial(mu0.detach().expand(n_particles, N), 1, generator=generator).reshape(n_particles)
    mu_flow = []
    for t in range(T + 1):
        mu_hat = one_hot(states, N, dtype).mean(dim=0)
        mu_flow.append(mu_hat)
        if t == T:
            break
        actions = sample_actions(action_probs_fn, theta, t, states, mu_hat, generator=generator)
        states = env.sample_next_states(states, actions, mu_hat, generator=generator)
    return torch.stack(mu_flow)


def exact_objective(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, gamma: float = 1.0, detach: bool = True
) -> torch.Tensor:
    """
    J(theta;mu0) = sum_{t<T} gamma^t E_{x~mu_t,a~pi_t}[r_t(x,a,mu_t)] + gamma^T E_{x~mu_T}[g(x,mu_T)],
    evaluated exactly (no trajectory noise) via the exact population flow.
    `gamma=1.0` (default) recovers the undiscounted objective used by
    environments with no discount factor (e.g. `mfc.environments.twostate`);
    environments with a genuine discount (e.g. `mfc.environments.cybersecurity`)
    pass their own `config.gamma`. Environment- and policy-agnostic.
    `detach=False` keeps the result differentiable in theta (see
    `exact_population_flow`); used as the ground-truth objective for
    gradient-bias diagnostics.
    """
    if detach:
        theta = theta.detach()
    mu_flow = exact_population_flow(env, action_probs_fn, theta, mu0, T, detach=detach)
    N, A = env.n_states, env.n_actions
    device = mu0.device
    states_grid = torch.arange(N, device=device).unsqueeze(-1).expand(N, A)
    actions_grid = torch.arange(A, device=device).unsqueeze(0).expand(N, A)
    states_all = torch.arange(N, device=device)

    total = torch.zeros((), dtype=mu0.dtype, device=device)
    for t in range(T):
        mu_t = mu_flow[t]
        pi = eval_batched(action_probs_fn, theta, t, states_all, mu_t)  # (N, A)
        r = env.reward(states_grid, actions_grid, mu_t)  # (N, A)
        total = total + (gamma**t) * (mu_t * (pi * r).sum(dim=-1)).sum()
    g = env.terminal_reward(states_all, mu_flow[T])  # (N,)
    return total + (gamma**T) * (mu_flow[T] * g).sum()
