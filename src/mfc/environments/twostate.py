"""
Two-state two-action mean-field control benchmark.

Reference: Meunier, Pham & Reisinger, discrete-space benchmarks,
Section "Two-State Two-Action Toy Problem" (files/reference/discrete_benchmarks.tex).

This module defines the MDP (state/action spaces, transition kernel,
running and terminal reward) together with the restricted policy class and
closed-form optimal policy fixed by the reference for this benchmark: "we
use the same restricted policy class as in the reference experiment...
parametrized by two Bernoulli logits", shared across the compared
estimators. Generic trajectory sampling, score functions, and training
mechanics for an arbitrary policy live in `mfc.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ST, MV = 0, 1  # action encoding: stay in place / attempt to move


@dataclass(frozen=True)
class TwoStateConfig:
    """Model parameters, fixed as in the reference."""

    lambda0: float = 0.5  # P(switch | state 0, MV)
    lambda1: float = 0.8  # P(switch | state 1, MV)
    kappa: float = 10.0  # Wasserstein penalty weight
    p: float = 0.6  # target law bar_mu = (p, 1-p)
    T: int = 2  # horizon


class TwoState:
    """
    State space X = {0, 1}, action space A = {ST, MV}.

    ST leaves the state unchanged. MV attempts to switch to the other
    state and succeeds with probability lambda_x. The transition kernel
    does not depend on the population law; the mean-field interaction
    enters only through the reward.
    """

    n_states = 2
    n_actions = 2

    def __init__(
        self,
        config: TwoStateConfig = TwoStateConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device
        self._switch_prob = torch.tensor([config.lambda0, config.lambda1], dtype=dtype, device=device)
        self.target_law = torch.tensor([config.p, 1.0 - config.p], dtype=dtype, device=device)

    def transition_probs(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        """P(x' | x, a, mu), shape (*states.shape, 2). Independent of mu."""
        p_switch = torch.where(actions == MV, self._switch_prob[states], torch.zeros_like(states, dtype=self.dtype))
        prob_x0 = torch.where(states == 0, 1.0 - p_switch, p_switch)
        return torch.stack([prob_x0, 1.0 - prob_x0], dim=-1)

    def sample_next_states(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample X_{t+1} ~ P(.|x,a,mu)."""
        probs = self.transition_probs(states, actions, mu)
        flat = torch.multinomial(probs.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(states.shape)

    def reward(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """r(x, a, mu) = 1{x=1} - mu(1)^2 - kappa * |mu(1) - (1-p)|. Independent of a."""
        mu1 = mu[..., 1]
        return (states == 1).to(self.dtype) - mu1.square() - self.config.kappa * (mu1 - self.target_law[1]).abs()

    def terminal_reward(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """g(x, mu) = r(x, mu)."""
        return self.reward(states, None, mu)

    def optimal_policy(self) -> torch.Tensor:
        """
        Closed-form optimal stationary policy pi*(a|x), rows=states,
        columns=(ST, MV). Drives every initial law to bar_mu=(p,1-p)
        after one step and keeps it there. With the reference parameters
        this gives pi*(ST|0)=0.2, pi*(ST|1)=0.25.
        """
        lambda0, lambda1, p = self.config.lambda0, self.config.lambda1, self.config.p
        pi = torch.zeros(self.n_states, self.n_actions, dtype=self.dtype, device=self.device)
        pi[0, MV] = (1.0 - p) / lambda0
        pi[0, ST] = 1.0 - pi[0, MV]
        pi[1, MV] = p / lambda1
        pi[1, ST] = 1.0 - pi[1, MV]
        return pi

    def policy_probs(self, theta: torch.Tensor, t: int, state: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Stationary Bernoulli-logit policy pi^theta(MV|x) = sigmoid(theta_x),
        the restricted, population-independent policy class shared by the
        estimators compared in the reference experiment. Matches the
        `action_probs_fn(theta, t, state, mu)` contract of
        `mfc.algorithms.simplex` (single unbatched sample; vmap/grad-safe:
        avoids indexing tricks that break under `torch.func.vmap`).
        """
        onehot = (torch.arange(self.n_states, device=theta.device) == state).to(theta.dtype)
        p_move = torch.sigmoid((onehot * theta).sum())
        return torch.stack([1.0 - p_move, p_move])

    def init_theta(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """theta_0=(0,0), the reference's initialization. `generator` is
        accepted (unused: the init is deterministic) so this matches the
        `env.init_theta(generator=...)` contract shared with environments
        that need a random initialization (e.g. `mfc.environments.cybersecurity`)."""
        return torch.zeros(self.n_states, dtype=self.dtype, device=self.device)

    def optimal_theta(self) -> torch.Tensor:
        """
        theta*, the parameter of `policy_probs` realizing `optimal_policy()`
        exactly: theta*_x = logit(pi*(MV|x)) = log(pi*(MV|x)/pi*(ST|x)),
        inverting the Bernoulli-logit parametrization. Ground truth for the
        theta-bias/variance diagnostic (scripts/test.py).
        """
        pi = self.optimal_policy()
        return torch.log(pi[:, MV] / pi[:, ST])
