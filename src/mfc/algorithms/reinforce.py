"""
Classical discrete-state REINFORCE, as a baseline for the "missing
mean-field term" comparison (context.md).

The exact policy-gradient formula for the mean-field control objective has
two pieces (discrete_state_space(2).tex, Theorem "Perturbed policy-gradient
formula", taking lambda -> 1 / no perturbation): the direct policy score
L_t = grad_theta log pi_t^theta(a|x,mu_t), and a population-sensitivity
correction Q_t(D_t) that accounts for mu_t^theta itself depending on theta
(the agent's own influence on the population it is embedded in). Classical
REINFORCE treats mu_t as an exogenous input — it uses the actual nominal
flow mu_t^theta to drive the simulated trajectories and reward, but its
gradient estimate keeps only L_t*G_t and omits Q_t(D_t) entirely:

    g_hat_reinforce(theta) = (1/B) sum_b sum_t L_t^(b) * G_t^(b).

This is model-free and needs no perturbation or auxiliary sensitivity
batch — its only per-step cost is the main batch, T*B transitions (or
T*(particle_size+B) under the particle flow) — but it is a structurally
biased estimator of grad_theta J(theta) whenever the reward genuinely
depends on the population law, since it is missing exactly the term that
captures the mean-field feedback. Comparing this against simplex/mfreinforce
at equal budget is what isolates the value of that term.

Shares the generic batched policy/flow machinery in `mfc.algorithms._common`
(same environment and action_probs_fn contracts as `mfc.algorithms.simplex`).
"""

from __future__ import annotations

import torch

from ._common import exact_population_flow, policy_score, sample_actions

__all__ = ["gradient_estimate", "gradient_step", "train"]


def gradient_estimate(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu_flow: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    B: int,
    *,
    baseline: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Plain REINFORCE policy-gradient estimate ĝ = (1/B) sum_b sum_t L_t^(b) G_t^(b),
    from B trajectories simulated against the given nominal flow `mu_flow`
    (used directly, unperturbed, as the population argument at every step —
    no simplex/logit randomization, since there is no population-sensitivity
    term to make model-free here). Returns shape (D,).
    """
    device, dtype = theta.device, theta.dtype
    N, D = env.n_states, theta.numel()
    baseline = torch.zeros(T + 1, dtype=dtype, device=device) if baseline is None else baseline

    states = torch.multinomial(mu0.expand(B, N), 1, generator=generator).reshape(B)
    rewards = torch.zeros(T, B, dtype=dtype, device=device)
    L = torch.zeros(T, B, D, dtype=dtype, device=device)
    terminal_reward = None

    for t in range(T + 1):
        mu_t = mu_flow[t]
        if t < T:
            actions = sample_actions(action_probs_fn, theta, t, states, mu_t, generator=generator)
            L[t] = policy_score(action_probs_fn, theta, t, states, actions, mu_t)
            rewards[t] = env.reward(states, actions, mu_t)
            states = env.sample_next_states(states, actions, mu_t, generator=generator)
        else:
            terminal_reward = env.terminal_reward(states, mu_t)

    G = torch.zeros(T + 1, B, dtype=dtype, device=device)
    G[T] = terminal_reward
    for t in range(T - 1, -1, -1):
        G[t] = rewards[t] + G[t + 1]

    weighted = L * (G[:T] - baseline[:T].view(-1, 1)).unsqueeze(-1)
    return weighted.sum(dim=(0, 1)) / B


def gradient_step(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    *,
    T: int,
    B: int,
    baseline: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One full REINFORCE gradient estimate at the given theta and initial
    law: obtain the nominal flow via `population_flow_fn` (exact by
    default; pass e.g. `functools.partial(particle_population_flow,
    n_particles=...)` for the empirical-flow variant), then form the
    estimate from one main batch. Returns shape (D,)."""
    mu_flow = population_flow_fn(env, action_probs_fn, theta, mu0, T, generator=generator)
    return gradient_estimate(env, action_probs_fn, theta, mu_flow, mu0, T, B, baseline=baseline, generator=generator)


def train(
    env,
    action_probs_fn,
    theta0: torch.Tensor,
    mu0: torch.Tensor,
    *,
    T: int,
    n_train: int,
    B: int,
    lr: float = 1e-3,
    baseline: torch.Tensor | None = None,
    population_flow_fn=exact_population_flow,
    project=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classical REINFORCE with a fixed initial law mu0, mirroring
    `simplex.train`'s structure and calling convention. `project`, if given,
    implements Pi_Theta and is applied in-place after each step. Returns
    (theta_final, theta_history) with history shape (n_train+1, D).
    """
    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    history = [theta.detach().clone()]

    for _ in range(n_train):
        g_hat = gradient_step(
            env, action_probs_fn, theta, mu0, T=T, B=B, baseline=baseline, population_flow_fn=population_flow_fn, generator=generator
        )
        optimizer.zero_grad()
        theta.grad = -g_hat
        optimizer.step()
        if project is not None:
            theta.data = project(theta.data)
        history.append(theta.detach().clone())

    return theta.detach(), torch.stack(history)
