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
    n_train: int = 5_000 # Number of epochs
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

    def policy_probs(self, theta: torch.Tensor) -> torch.Tensor:
        """Static policy pi_theta(a|x), where theta is a 2D tensor of shape (n_states, n_actions)."""
        return torch.softmax(theta, dim=-1)
    
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
        probs = torch.zeros(2, dtype=self.config.dtype, device=self.config.device)

        if action == 0: probs[state] = 1.0 # ST
        else: # MV
            probs[1 - state] = self.switch_probs[state]
            probs[state] = 1.0 - self.switch_probs[state]
        
        return probs
    
    def averaged_kernel(self, theta: torch.Tensor, t: int, mu: torch.Tensor) -> torch.Tensor:
        """K_theta(x'|x,mu)"""
        pi = self.policy_probs(theta)
        
        K = torch.zeros(2, 2, dtype=self.config.dtype, device=self.config.device)

        for x in range(2):
            for a in range(2):
                K[x] += pi[x, a] * self.transition_probs(x, a, mu)

        return K
    
    def reward(self, state: int, mu: torch.Tensor, action: int | None = None) -> torch.Tensor:
        """
        r(x,a,mu) = 1_{x=1} - mu(1)^2 - lambda * W1(mu, B).
        On {0, 1} with the usual distance, W1(mu, B) = |mu(1) - B(1)|.
        """
        x_reward = torch.tensor(1.0 if state == 1 else 0.0, dtype=self.config.dtype, device=self.config.device)

        mu1 = mu[1]
        w1 = torch.abs(mu1 - self.target_B[1])

        return x_reward - mu1**2 - self.config.lam * w1

    def terminal_reward(self, state: int, mu: torch.Tensor) -> torch.Tensor:
        return self.reward(state, mu)

    def exact_population_flow(self, theta: torch.Tensor, mu0: torch.Tensor) -> torch.Tensor:
        """Computes mu_t exactly using the averaged kernel."""
        mu_flow = [mu0]

        for t in range(self.config.T):
            K = self.averaged_kernel(theta, t, mu_flow[-1])
            mu_flow.append(mu_flow[-1] @ K)

        return torch.stack(mu_flow)
    
    def exact_value(self, theta: torch.Tensor, mu0: torch.Tensor) -> torch.Tensor:
        """Exact finite-horizon population value under the static policy."""
        mu_flow = self.exact_population_flow(theta, mu0)

        value = torch.tensor(0.0, dtype=self.config.dtype, device=self.config.device)

        for t in range(self.config.T):
            mu_t = mu_flow[t]
            value += sum(mu_t[x] * self.reward(x, mu_t) for x in range(2))

        mu_T = mu_flow[self.config.T]
        value += sum(mu_T[x] * self.reward(x, mu_T) for x in range(2))

        return value
    
    def sample_state(self, mu: torch.Tensor) -> int:
        return int(torch.multinomial(mu, num_samples=1).item())
    
    def sample_action(self, theta: torch.Tensor, t: int, state: int, mu: torch.Tensor) -> int:
        pi = self.policy_probs(theta)
        return int(torch.multinomial(pi[state], num_samples=1).item())
    
    def sample_next_state(self, state: int, action: int, mu: torch.Tensor) -> int:
        probs = self.transition_probs(state, action, mu)
        return int(torch.multinomial(probs, num_samples=1).item())
    
    def policy_score(self, theta: torch.Tensor, t: int, mu: torch.Tensor, state: int, action: int) -> torch.Tensor:
        """Analytic score grad_theta log pi_theta(action | state)."""
        pi = self.policy_probs(theta)
        score = torch.zeros_like(theta)

        score[state] = -pi[state]
        score[state, action] += 1.0

        return score
