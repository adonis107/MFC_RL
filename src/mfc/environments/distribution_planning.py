"""
Distribution-planning mean-field control benchmark: a central planner steers
a population around a discrete torus toward a fixed target law while
penalizing individual movement.

Reference: Meunier, Pham & Reisinger, discrete-space benchmarks,
Section "Distribution Planning" (files/reference/discrete_benchmarks.tex),
based on Carmona et al.'s distribution-planning mean-field control problem.

Unlike `mfc.environments.twostate`, the population state is 9-dimensional
(Delta_10) and the policy is a population-dependent MLP (as in
`mfc.environments.cybersecurity`); unlike cybersecurity, the transition
kernel is deterministic and mu-independent (a cyclic shift by the action),
so the mean-field interaction enters purely through the reward. Generic
trajectory sampling, score functions, and training mechanics for an
arbitrary policy live in `mfc.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LEFT, STAY, RIGHT = 0, 1, 2  # action encoding: move to site x-1 / stay / move to site x+1 (mod N_STATES)
N_STATES = 10


@dataclass(frozen=True)
class DistributionPlanningConfig:
    """Model parameters, fixed as in the reference."""

    c_mov: float = 0.01  # movement penalty weight (named to avoid clashing with the simplex perturbation lambda)
    target_law: tuple[float, ...] = (0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.0)
    hidden_width: int = 256  # policy MLP hidden width


def _one_hot_batched(index: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot encoding for a batched (possibly multi-dim) index, shape (*index.shape, n)."""
    return (torch.arange(n, device=index.device) == index.unsqueeze(-1)).to(dtype)


class DistributionPlanning:
    """
    State space X = Z/10Z = {0,...,9} (a discrete torus), action space
    A = {LEFT, STAY, RIGHT} moving the agent to x-1, x, x+1 (mod 10). The
    transition kernel is deterministic and independent of the population
    law; the mean-field interaction enters only through the reward's
    ||mu - mu_target||_2^2 penalty.
    """

    n_states = N_STATES
    n_actions = 3
    _delta = (-1, 0, 1)  # move-size per action index (LEFT, STAY, RIGHT)

    def __init__(
        self,
        config: DistributionPlanningConfig = DistributionPlanningConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device
        self._delta_t = torch.tensor(self._delta, dtype=torch.long, device=device)
        self.target_law = torch.tensor(config.target_law, dtype=dtype, device=device)

    def transition_probs(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        """P(x' | x, a) = 1{x' = x + delta(a) mod 10}, shape (*batch, n_states). Independent of mu."""
        next_state = (states + self._delta_t[actions]) % self.n_states
        return _one_hot_batched(next_state, self.n_states, self.dtype)

    def sample_next_states(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Deterministic: X_{t+1} = X_t + delta(A_t) (mod 10). `generator` is accepted (unused) for interface parity."""
        return (states + self._delta_t[actions]) % self.n_states

    def reward(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """r(x, a, mu) = -c_mov*|delta(a)| - ||mu - mu_target||_2^2. States-independent; broadcasts over actions.shape."""
        movement_cost = self.config.c_mov * self._delta_t[actions].abs().to(self.dtype)
        mismatch = (mu - self.target_law).square().sum()
        return -movement_cost - mismatch

    def terminal_reward(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """g(x, mu) = -||mu - mu_target||_2^2. Broadcasts to states.shape."""
        mismatch = (mu - self.target_law).square().sum()
        return -mismatch * torch.ones_like(states, dtype=self.dtype)

    def _layer_dims(self) -> list[int]:
        """(t, mu) input -> hidden -> hidden -> |X|x|A| logits."""
        H = self.config.hidden_width
        return [1 + self.n_states, H, H, self.n_states * self.n_actions]

    def _mlp_shapes(self) -> list[tuple[tuple[int, int], tuple[int]]]:
        dims = self._layer_dims()
        return [((b, a), (b,)) for a, b in zip(dims[:-1], dims[1:])]

    def unpack_theta(self, theta: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Slice the flat parameter vector `theta` into (weight, bias) pairs
        for each of the 3 layers. Slicing on static (non-data-dependent)
        bounds, so this stays vmap/grad-safe."""
        layers = []
        i = 0
        for w_shape, b_shape in self._mlp_shapes():
            n_w = w_shape[0] * w_shape[1]
            w = theta[i : i + n_w].reshape(w_shape)
            i += n_w
            n_b = b_shape[0]
            b = theta[i : i + n_b]
            i += n_b
            layers.append((w, b))
        return layers

    def policy_probs(self, theta: torch.Tensor, t: int, state: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Population-dependent policy pi_t^theta(.|x,mu): a 2-hidden-layer
        (width `hidden_width`, tanh) MLP taking (t,mu) and outputting
        |X|x|A|=10x3 logits, row-selected by x and row-wise softmaxed.
        `theta` packs every MLP weight/bias into one flat vector (see
        `unpack_theta`/`init_theta`) so this remains a pure
        `action_probs_fn(theta, t, state, mu) -> probs` compatible with
        `mfc.algorithms` (single unbatched sample; vmap/grad-safe: avoids
        indexing tricks that break under `torch.func.vmap`).
        """
        (W1, b1), (W2, b2), (W3, b3) = self.unpack_theta(theta)
        t_feat = torch.full((1,), float(t), dtype=theta.dtype, device=theta.device)
        x = torch.cat([t_feat, mu])
        h1 = torch.tanh(W1 @ x + b1)
        h2 = torch.tanh(W2 @ h1 + b2)
        logits = (W3 @ h2 + b3).reshape(self.n_states, self.n_actions)
        onehot = (torch.arange(self.n_states, device=theta.device) == state).to(theta.dtype)
        row_logits = onehot @ logits
        return torch.softmax(row_logits, dim=-1)

    def init_theta(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Flattened MLP parameters, PyTorch's default `nn.Linear`-style
        init: each layer's weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
        chunks = []
        for w_shape, b_shape in self._mlp_shapes():
            fan_in = w_shape[1]
            bound = fan_in**-0.5
            w = (torch.rand(w_shape, dtype=self.dtype, device=self.device, generator=generator) * 2 - 1) * bound
            b = (torch.rand(b_shape, dtype=self.dtype, device=self.device, generator=generator) * 2 - 1) * bound
            chunks.append(w.reshape(-1))
            chunks.append(b)
        return torch.cat(chunks)

    def sample_mu0(self, batch_shape: tuple[int, ...] = (), *, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        mu_0 ~ Dirichlet(1,...,1) (10-dim), the reference's per-iteration
        randomized initial law: full support on the interior of Delta_10, so
        log(mu_0) is well-defined almost surely (needed by the logit
        perturbation). Sampled as E_i ~ Exp(1) iid, mu_0 = E / sum(E)
        (Dirichlet(1,...,1) is exactly normalized i.i.d. exponentials), via
        `torch.rand(..., generator=generator)` since
        `torch.distributions.Dirichlet` does not accept an explicit generator.
        """
        u = torch.rand(*batch_shape, self.n_states, dtype=self.dtype, device=self.device, generator=generator)
        E = -torch.log(u.clamp_min(1e-12))
        return E / E.sum(dim=-1, keepdim=True)

    def cyclic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """d_cyc(x,y) = min(|x-y|, 10-|x-y|), the torus's natural metric
        (reference "Evaluation criteria": used by the terminal transport
        discrepancy E_T^(W))."""
        diff = (x - y).abs()
        return torch.minimum(diff, self.n_states - diff)
