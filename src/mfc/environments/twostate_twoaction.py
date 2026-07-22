from __future__ import annotations
from dataclasses import dataclass

import torch


# Attributes
@dataclass
class TwoStateConfig:
    device: torch.device
    dtype: torch.dtype
    
    T: int = 2
    lam0: float = 0.5
    lam1: float = 0.8
    lam: int = 10
    target_p: float = 0.6

    low: float = 0.1
    high: float = 0.9

    q_sigma: float = 1.0
    q_clip: float = 1e-8

    N: int = 200 # Main perturbed trajectories for MF-REINFORCE
    n: int = 10 # Trajectories for the gradient of logits estimation

    lr: float = 1e-3 # Learning rate
    n_train: int = 10_000 # Number of epochs
    training_runs: int = 5 # Number of independent training runs for each epsilon value
    validate_every: int = 10 # Freeze the policy and sample a validation episode, for which we compute the population reward starting from a fixed initial distribution


# Environment
class TwoStateMFC:
    """
    Two-state, two-action mean-field control problem.
    
    State space: X = {0, 1}
    Action space: A = {ST, MV}, encoded as {0, 1}

    If a = ST, the state stays fixed.
    If a = MV, state x switches to 1-x with probability lambda_x.
    """

    def __init__(self, config: TwoStateConfig):
        self.config = config
        self.n_states = 2
        self.n_actions = 2

        self.switch_probs = torch.tensor([config.lam0, config.lam1], dtype=config.dtype, device=config.device)
        self.target_B = torch.tensor([config.target_p, 1.0 - config.target_p], dtype=config.dtype, device=config.device)
        self._state_rewards = torch.tensor([0.0, 1.0], dtype=config.dtype, device=config.device)
        self._transition_tensor = torch.zeros(
            self.n_actions,
            self.n_states,
            self.n_states,
            dtype=config.dtype,
            device=config.device,
        )
        self._transition_tensor[0] = torch.eye(self.n_states, dtype=config.dtype, device=config.device)
        self._transition_tensor[1, 0] = torch.tensor([1.0 - config.lam0, config.lam0], dtype=config.dtype, device=config.device)
        self._transition_tensor[1, 1] = torch.tensor([config.lam1, 1.0 - config.lam1], dtype=config.dtype, device=config.device)

    def policy_probs(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Static Bernoulli-logit policy.

        The reference parametrization uses one logit per state, with
        pi(MV|x)=sigmoid(theta_x) and pi(ST|x)=1-sigmoid(theta_x). A legacy
        2x2 row-softmax tensor is still accepted for saved exploratory runs.
        """
        if theta.ndim == 1:
            p_move = torch.sigmoid(theta)
            return torch.stack([1.0 - p_move, p_move], dim=-1)
        return torch.softmax(theta, dim=-1)

    def action_probabilities(self, theta: torch.Tensor, t: int, mu: torch.Tensor) -> torch.Tensor:
        pi = self.policy_probs(theta)
        if mu.ndim == 1:
            return pi
        return pi.expand(*mu.shape[:-1], self.n_states, self.n_actions)
    
    def optimal_policy(self) -> torch.Tensor:
        """Optimal static policy matrix, rows: states, columns: actions."""
        p = self.config.target_p
        lam0, lam1 = self.config.lam0, self.config.lam1

        pi = torch.zeros((self.n_states, self.n_actions), dtype=self.config.dtype, device=self.config.device)

        pi[0, 1] = (1.0 - p) / lam0 # MV | state 0
        pi[0, 0] = 1.0 - pi[0, 1]   # ST | state 0

        pi[1, 1] = p / lam1         # MV | state 1
        pi[1, 0] = 1.0 - pi[1, 1]   # ST | state 1

        return pi
    
    def transition_probs(self, state: int, action: int, mu: torch.Tensor) -> torch.Tensor:
        """P(.|x,a,mu)"""
        return self._transition_tensor[action, state]

    def transition_tensor(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.ndim == 1:
            return self._transition_tensor
        return self._transition_tensor.expand(*mu.shape[:-1], self.n_actions, self.n_states, self.n_states)
    
    def averaged_kernel(self, theta: torch.Tensor, t: int, mu: torch.Tensor) -> torch.Tensor:
        """K_theta(x'|x,mu)"""
        pi = self.action_probabilities(theta, t, mu)
        transitions = self.transition_tensor(mu)
        return torch.einsum("...xa,...axy->...xy", pi, transitions)
    
    def reward(self, state: int, mu: torch.Tensor, action: int | None = None) -> torch.Tensor:
        """
        r(x,a,mu) = 1_{x=1} - mu(1)^2 - lambda * W1(mu, B).
        On {0, 1} with the usual distance, W1(mu, B) = |mu(1) - B(1)|.
        """
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        return self.reward_batch(state_t, mu, None if action is None else torch.as_tensor(action, device=self.config.device))

    def reward_batch(self, states: torch.Tensor, mu: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        state_rewards = self._state_rewards[states]
        mu1 = mu[..., 1]
        w1 = torch.abs(mu1 - self.target_B[1])
        return state_rewards - mu1.square() - self.config.lam * w1

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.reward(state, mu)

    def terminal_reward_batch(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.reward_batch(states, mu)

    def exact_population_flow(self, theta: torch.Tensor, mu0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Computes mu_t exactly using the averaged kernel."""
        steps = self.config.T if horizon is None else horizon
        mu_flow = [mu0]

        for t in range(steps):
            K = self.averaged_kernel(theta, t, mu_flow[-1])
            mu_flow.append(mu_flow[-1] @ K)

        return torch.stack(mu_flow)
    
    def exact_value(self, theta: torch.Tensor, mu0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Exact finite-horizon population value under the static policy."""
        steps = self.config.T if horizon is None else horizon
        mu_flow = self.exact_population_flow(theta, mu0, steps)

        value = torch.tensor(0.0, dtype=self.config.dtype, device=self.config.device)

        for t in range(steps):
            mu_t = mu_flow[t]
            value += sum(mu_t[x] * self.reward(x, mu_t) for x in range(2))

        mu_T = mu_flow[steps]
        value += sum(mu_T[x] * self.reward(x, mu_T) for x in range(2))

        return value
    
    def sample_state(self, mu: torch.Tensor) -> int:
        return int(torch.multinomial(mu, num_samples=1).item())
    
    def sample_action(self, theta: torch.Tensor, t: int, state: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        return int(self.sample_actions_batch(theta, t, state_t, mu).item())

    @torch.no_grad()
    def sample_actions_batch(self, theta: torch.Tensor, t: int, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        pi = self.action_probabilities(theta, t, mu)
        if pi.ndim == 2:
            probs = pi[states.reshape(-1)]
        else:
            probs = pi.reshape(-1, self.n_states, self.n_actions)[
                torch.arange(states.numel(), device=states.device),
                states.reshape(-1),
            ]
        return torch.multinomial(probs, num_samples=1).reshape_as(states)
    
    def sample_next_state(self, state: int, action: int, mu: torch.Tensor) -> int:
        state_t = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        action_t = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return int(self.sample_next_states_batch(state_t, action_t, mu).item())

    @torch.no_grad()
    def sample_next_states_batch(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        transitions = self.transition_tensor(mu)
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        if transitions.ndim == 3:
            probs = transitions[actions_flat, states_flat]
        else:
            probs = transitions.reshape(-1, self.n_actions, self.n_states, self.n_states)[
                torch.arange(states_flat.numel(), device=states.device),
                actions_flat,
                states_flat,
            ]
        return torch.multinomial(probs, num_samples=1).reshape_as(states)
    
    def policy_score(self, theta: torch.Tensor, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        """Analytic score grad_theta log pi_theta(action | state)."""
        states = torch.as_tensor(state, dtype=torch.long, device=self.config.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.config.device)
        return self.policy_scores_batch(theta, t, mu, states, actions).reshape_as(theta)

    def policy_scores_batch(
        self,
        theta: torch.Tensor,
        t: int,
        mu: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        pi = self.policy_probs(theta)
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        rows = torch.arange(states_flat.numel(), device=theta.device)

        if theta.ndim == 1:
            scores = torch.zeros(states_flat.numel(), self.n_states, dtype=theta.dtype, device=theta.device)
            p_move = pi[:, 1]
            scores[rows, states_flat] = (actions_flat == 1).to(theta.dtype) - p_move[states_flat]
            return scores.reshape(*states.shape, self.n_states)

        scores = torch.zeros(states_flat.numel(), self.n_states, self.n_actions, dtype=theta.dtype, device=theta.device)
        scores[rows, states_flat] = -pi[states_flat]
        scores[rows, states_flat, actions_flat] += 1.0
        return scores.reshape(*states.shape, -1)
