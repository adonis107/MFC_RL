"""
Targeted-advertising mean-field control benchmark with social influence.

Reference: Meunier, Pham & Reisinger, discrete-space benchmarks,
Section "Targeted advertising with social influence" (files/reference/
discrete_benchmarks.tex), introduced by Motte & Pham (Section 3.5 of
"Mean Field Markov Decision Processes"). A company repeatedly advertises to
increase its customer proportion while controlling expenditure. Unlike the
other discrete-state benchmarks here, the mean-field interaction acts
through the *transition kernel* (a customer's own retention/conversion
probability depends on how widely adopted the product already is), not the
reward, and the optimal policy is population-dependent but *individual-state-
independent* (only the aggregate advertising rate matters), which this
module's `policy_probs` encodes directly rather than deriving. Generic
trajectory sampling, score functions, and training mechanics for an
arbitrary policy live in `mfc.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NO_AD, AD = 0, 1  # action encoding: withhold / display an advertisement
NOT_CUSTOMER, CUSTOMER = 0, 1  # state encoding


@dataclass(frozen=True)
class AdvertisingConfig:
    """Model parameters, fixed as in the reference."""

    kappa_ad: float = 0.2  # advertising efficiency
    c_ad: float = 0.15  # advertising cost
    gamma: float = 0.5  # discount factor (finite-horizon truncation of the source infinite-horizon problem)
    T: int = 5  # horizon; also normalizes the policy's time input (t/T), so kept as a genuine model constant
    hidden_width: int = 32  # policy MLP hidden width


class Advertising:
    """
    State space X = {0 (not a customer), 1 (customer)}, action space
    A = {0 (no ad), 1 (ad)}. The transition kernel is independent of the
    individual's own state x: P(1|x,a,mu) = min(mu(1) + kappa_ad*a, 1), so
    an advertisement raises everyone's conversion/retention probability by
    kappa_ad (capped at 1), with the *current* customer proportion mu(1)
    driving a positive social-influence effect.
    """

    n_states = 2
    n_actions = 2

    def __init__(
        self,
        config: AdvertisingConfig = AdvertisingConfig(),
        *,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
    ):
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.dtype = dtype
        self.device = device

    def transition_probs(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """P(1|x,a,mu) = min(mu(1) + kappa_ad*a, 1), independent of x. Shape (*batch, 2)."""
        p1 = (mu[..., CUSTOMER] + self.config.kappa_ad * actions.to(self.dtype)).clamp(max=1.0)
        return torch.stack([1.0 - p1, p1], dim=-1)

    def sample_next_states(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample X_{t+1} ~ P(.|x,a,mu)."""
        probs = self.transition_probs(states, actions, mu)
        flat = torch.multinomial(probs.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(actions.shape)

    def reward(self, states: torch.Tensor, actions: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """r(x, a, mu) = x - c_ad*a. Independent of mu directly (all mean-field dependence is in the transition kernel)."""
        return states.to(self.dtype) - self.config.c_ad * actions.to(self.dtype)

    def terminal_reward(self, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """g(x, mu) = 0 (the source infinite-horizon problem is truncated with no terminal bonus)."""
        return torch.zeros_like(states, dtype=self.dtype)

    def _layer_dims(self) -> list[int]:
        """(t/T, mu(1)) input -> hidden -> hidden -> 1 sigmoid logit (the aggregate advertising probability)."""
        H = self.config.hidden_width
        return [2, H, H, 1]

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
        pi_theta(1|t,x,mu) = q_theta(t/T, mu(1)) for every x (reference
        "Policy parametrization": only the aggregate advertising rate enters
        the population recursion and expected reward, so the optimal policy
        class is individual-state-independent — `state` is accepted for
        interface parity with `mfc.algorithms` but ignored). q_theta is a
        2-hidden-layer (width `hidden_width`, tanh) MLP with a sigmoid
        output, taking the normalized time t/T and current customer
        proportion mu(1). `theta` packs every MLP weight/bias into one flat
        vector (see `unpack_theta`/`init_theta`); vmap/grad-safe (single
        unbatched sample).
        """
        (W1, b1), (W2, b2), (W3, b3) = self.unpack_theta(theta)
        t_feat = torch.full((1,), float(t) / self.config.T, dtype=theta.dtype, device=theta.device)
        p1 = mu[CUSTOMER : CUSTOMER + 1]
        x = torch.cat([t_feat, p1])
        h1 = torch.tanh(W1 @ x + b1)
        h2 = torch.tanh(W2 @ h1 + b2)
        q = torch.sigmoid((W3 @ h2 + b3).squeeze(-1))
        return torch.stack([1.0 - q, q])

    def init_theta(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Flattened MLP parameters, PyTorch's default `nn.Linear`-style
        init: each layer's weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        Drawn in float64 regardless of `self.dtype`, then cast down: `torch.rand`
        consumes the generator's stream differently per dtype, so drawing
        directly at `self.dtype` would give a float32 run a *different* random
        theta0 than a float64 run at the same seed, not just a lower-precision
        copy of the same one — silently invalidating any float32-vs-float64
        comparison run at the same INIT_SEED (scripts/train.py)."""
        chunks = []
        for w_shape, b_shape in self._mlp_shapes():
            fan_in = w_shape[1]
            bound = fan_in**-0.5
            w = (torch.rand(w_shape, dtype=torch.float64, device=self.device, generator=generator) * 2 - 1) * bound
            b = (torch.rand(b_shape, dtype=torch.float64, device=self.device, generator=generator) * 2 - 1) * bound
            chunks.append(w.reshape(-1).to(self.dtype))
            chunks.append(b.to(self.dtype))
        return torch.cat(chunks)

    def sample_mu0(self, batch_shape: tuple[int, ...] = (), *, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        p0 ~ U([0.05, 0.95]), mu0 = (1-p0, p0), the reference's per-iteration
        randomized initial law (reference "Policy parametrization": the
        interval deliberately excludes the simplex boundary, required for
        the logit perturbation). A model constant, not RunConfig-parametrized
        (see configs/advertising.py's module docstring) — unlike two-state's
        mu0_low/mu0_high, this range is fixed by the benchmark itself.
        """
        p0 = torch.rand(batch_shape, dtype=self.dtype, device=self.device, generator=generator) * 0.9 + 0.05
        return torch.stack([1.0 - p0, p0], dim=-1)

    def reference_policy(self, p: torch.Tensor) -> torch.Tensor:
        """
        q_hat(p): the source model's explicit stationary optimal
        infinite-horizon advertising probability (reference "Infinite-
        horizon reference policy"), given generically in terms of
        `config.c_ad`, `config.kappa_ad`, `config.gamma` (not hardcoded to
        the benchmark's specific numeric case) via the three-regime
        closed form. Evaluation-only structural benchmark; never used by
        training. `p` (customer proportion) may be batched.
        """
        c, kappa, gamma = self.config.c_ad, self.config.kappa_ad, self.config.gamma
        ratio = c / kappa
        p_bar = 1.0 - c * (1.0 - gamma) / gamma

        if ratio < gamma:
            return torch.where(p < p_bar, torch.ones_like(p), torch.zeros_like(p))
        elif ratio < gamma / (1.0 - gamma):
            q = torch.zeros_like(p)
            q = torch.where(p < 1.0 - 2.0 * kappa, torch.ones_like(p), q)
            ramp_mask = (p >= 1.0 - 2.0 * kappa) & (p < 1.0 - (2.0 - gamma) * kappa)
            q = torch.where(ramp_mask, (1.0 - kappa - p) / kappa, q)
            plateau_mask = (p >= 1.0 - (2.0 - gamma) * kappa) & (p < p_bar)
            q = torch.where(plateau_mask, torch.ones_like(p), q)
            return q
        else:
            return torch.zeros_like(p)
