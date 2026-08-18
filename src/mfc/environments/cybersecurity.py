"""
Mean-field cybersecurity benchmark: a central planner controls the
protection status of a large population of interacting computers.

Reference: Meunier, Pham & Reisinger, discrete-space benchmarks,
Section "Cybersecurity Example" (files/reference/discrete_benchmarks.tex),
first studied as a mean-field game in Kolokoltsov & Bensoussan's botnet-
defense model and adapted to mean-field control by Carmona et al.

Unlike `mfc.environments.twostate`, the transition mechanism itself depends
nonlinearly on the population law through endogenous infection intensities
(the continuous-time generator Q^{mu,a} is discretized exactly via a matrix
exponential), and the policy is population-dependent, represented by an MLP
rather than a lookup table. Generic trajectory sampling, score functions,
and training mechanics for an arbitrary policy live in `mfc.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Ordered basis (DI, DS, UI, US): defended/undefended x infected/susceptible.
DI, DS, UI, US = 0, 1, 2, 3
KEEP, SWITCH = 0, 1  # action encoding: keep current protection level / switch it


@dataclass(frozen=True)
class CyberSecurityConfig:
    """Model parameters, fixed as in the reference."""

    beta_uu: float = 0.3
    beta_ud: float = 0.4
    beta_du: float = 0.3
    beta_dd: float = 0.4
    q_rec_D: float = 0.5  # recovery rate, defended
    q_rec_U: float = 0.4  # recovery rate, undefended
    q_inf_D: float = 0.4  # hacker-driven infection rate, defended
    q_inf_U: float = 0.3  # hacker-driven infection rate, undefended
    v_H: float = 0.6  # hacker "virulence"
    lambda_sw: float = 0.8  # protection-switch intensity (named to avoid clashing with the simplex perturbation lambda)
    k_D: float = 0.3  # defense cost
    k_I: float = 0.5  # infection cost
    dt: float = 0.2  # discretization step
    gamma: float = 0.5  # discount factor
    hidden_width: int = 32  # policy MLP hidden width


def _one_hot_batched(index: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot encoding for a batched (possibly multi-dim) index, shape (*index.shape, n)."""
    return (torch.arange(n, device=index.device) == index.unsqueeze(-1)).to(dtype)


class CyberSecurity:
    """
    State space X = {DI, DS, UI, US}, action space A = {KEEP, SWITCH}.

    SWITCH toggles the protection level (D<->U) at rate lambda_sw; infection
    and recovery proceed at rates depending on protection level and, for
    infection, on the population's infected fractions. The transition kernel
    is the exact matrix exponential of this continuous-time generator over
    one step of length dt, so discretization does not sacrifice
    stochasticity (no Euler approximation).
    """

    n_states = 4
    n_actions = 2

    def __init__(
        self,
        config: CyberSecurityConfig = CyberSecurityConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device
        self._f = torch.tensor(
            [config.k_D + config.k_I, config.k_D, config.k_I, 0.0], dtype=dtype, device=device
        )  # f(x) = k_D*1{defended} + k_I*1{infected}, ordered (DI, DS, UI, US)

    def _generator_matrix(self, mu: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q^{mu,a}, shape (*batch, 4, 4); mu and actions broadcast to a
        common batch shape. Built via `torch.stack` (not in-place scatter)
        so this stays differentiable w.r.t. mu, needed when
        `exact_population_flow(detach=False)` backprops through it."""
        c = self.config
        sw = c.lambda_sw * actions.to(self.dtype)
        iota_D = c.v_H * c.q_inf_D + c.beta_dd * mu[..., DI] + c.beta_ud * mu[..., UI]
        iota_U = c.v_H * c.q_inf_U + c.beta_uu * mu[..., UI] + c.beta_du * mu[..., DI]
        zeros = torch.zeros_like(sw)
        iota_D = iota_D + zeros
        iota_U = iota_U + zeros
        q_rec_D = zeros + c.q_rec_D
        q_rec_U = zeros + c.q_rec_U

        row_DI = torch.stack([-q_rec_D - sw, q_rec_D, sw, zeros], dim=-1)
        row_DS = torch.stack([iota_D, -iota_D - sw, zeros, sw], dim=-1)
        row_UI = torch.stack([sw, zeros, -q_rec_U - sw, q_rec_U], dim=-1)
        row_US = torch.stack([zeros, sw, iota_U, -iota_U - sw], dim=-1)
        return torch.stack([row_DI, row_DS, row_UI, row_US], dim=-2)

    def transition_probs(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """P_dt^{mu,a}(x, .) = exp(dt Q^{mu,a})(x, .), shape (*batch, 4)."""
        batch_shape = torch.broadcast_shapes(states.shape, actions.shape)
        Q = self._generator_matrix(mu, actions.expand(batch_shape))
        P = torch.linalg.matrix_exp(self.config.dt * Q)  # (*batch, 4, 4)
        onehot_x = _one_hot_batched(states.expand(batch_shape), self.n_states, self.dtype)
        return (onehot_x.unsqueeze(-1) * P).sum(dim=-2)

    def sample_next_states(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample X_{t+1} ~ P_dt^{mu,a}(.|x)."""
        probs = self.transition_probs(states, actions, mu)
        flat = torch.multinomial(probs.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(states.shape)

    def reward(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """r(x, a, mu) = -dt * f(x), independent of a and mu."""
        return -self.config.dt * self._f[states]

    def terminal_reward(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """g(x, mu) = r(x, mu)."""
        return self.reward(states, None, mu)

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
        |X|x|A|=4x2 logits, row-selected by x and row-wise softmaxed.
        `theta` packs every MLP weight/bias into one flat vector (see
        `unpack_theta`/`init_theta`), keeping this a pure
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

    def aggregate_fractions(self, mu_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (I_t, D_t): aggregate infected and defended fractions from a
        population flow of shape (..., T+1, 4) (or (..., 4)), ordered
        (DI, DS, UI, US) per this environment's convention — I_t :=
        mu_t(DI)+mu_t(UI), D_t := mu_t(DI)+mu_t(DS) (reference "Evaluation
        criteria"). Each returned tensor has shape (..., T+1) (or (...,)).
        """
        infected = mu_flow[..., DI] + mu_flow[..., UI]
        defended = mu_flow[..., DI] + mu_flow[..., DS]
        return infected, defended

    def sample_mu0(self, batch_shape: tuple[int, ...] = (), *, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        mu_0 ~ Dirichlet(1,1,1,1), the reference's per-iteration randomized
        initial law: full support on the interior of Delta_4, so log(mu_0)
        is well-defined almost surely (needed by the logit perturbation).
        Sampled as E_i ~ Exp(1) iid, mu_0 = E / sum(E) (Dirichlet(1,...,1)
        is exactly normalized i.i.d. exponentials), via `torch.rand(...,
        generator=generator)` since `torch.distributions.Dirichlet` does not
        accept an explicit generator.
        """
        u = torch.rand(*batch_shape, self.n_states, dtype=self.dtype, device=self.device, generator=generator)
        E = -torch.log(u.clamp_min(1e-12))
        return E / E.sum(dim=-1, keepdim=True)
