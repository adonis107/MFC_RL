"""
Run configurations for the cybersecurity benchmark.

Values taken directly from the reference (files/reference/discrete_benchmarks.tex,
"Cybersecurity Example", paragraphs "Compared perturbation schemes" and
"Training and validation protocol"): the simplex lambda grid, sigma=1.0
(U_t~N(0,I_3)), B=200, n_aux=1, lr=1e-3, validate_every=10, T_train=3,
T_val=50, mu0_val=(0.25,0.25,0.25,0.25), and 5 independent seeds. n_train
=20_000 is the reference's own value (used here as "main"); mid/smoke shrink
it for faster iteration. epsilon is fixed (not swept, matching this repo's
twostate convention) at 1.0, the value used for the reference's Figure 3
flow visualization; the reference itself sweeps epsilon in {0.2,0.5,1.0,2.0}.

Unlike two-state, mu0 is *not* resampled from a RunConfig-parametrized range
each training iteration: it is always mu0~Dirichlet(1,1,1,1), a model
constant (`mfc.environments.cybersecurity.CyberSecurity.sample_mu0`), so
there are no mu0_low/mu0_high fields here — see `scripts/train.py`'s
SAMPLE_MU0_FACTORIES.

Discounting lives on `CyberSecurityConfig.gamma` (an environment constant,
0.5 per the reference), not here: `scripts/train.py` reads `env.config.gamma`
and threads it through every algorithm's `gamma=` keyword (added generically
to `mfc.algorithms` for this environment; a no-op default of 1.0 elsewhere).

Equal-budget allocation. The reference itself uses identical (B, n_aux) for
simplex and mfreinforce here — no budget-matching adjustment — but this repo
adds an "equal_budget" mode as for two-state (see configs/twostate.py's
module docstring for the full derivation): the complexity formulas are
generic in (B, n_aux, T), so they are reused verbatim, evaluated at
T=T_train (mfreinforce's stagewise auxiliary batch, and hence the
equal-budget target, scales with the *training* horizon, not T_val).
Unlike two-state, `main` runs *only* budget_mode="equal_budget" (mfreinforce's
own (n_aux, B)=(1,200) as the fixed reference anchor, simplex/reinforce
adapted to match) and *only* flow="particle" (the empirical population-flow
estimate, Assumption "Access to the nominal population flow" — not the exact
recursion), rather than sweeping both options as two-state's `main` does.
Because cybersecurity's n_aux=1 (vs two-state's n_aux=10), the quadratic term
in mfreinforce's cost stays small even as T grows, so simplex's equal-budget
main batch only grows from 599 (T=3) to 899 (T=6) to 1499 (T=12) — nothing
like two-state's T=10 batch of ~11190.

Horizons. `main` sweeps T_train in {3, 6, 12}: 3 is the reference's own
(short, to limit sensitivity-estimator error accumulation — see above),
6 and 12 are this repo's own extension (doubling each step) for a
horizon-scaling comparison, as two-state's `main` does with {2, 5, 10}.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CyberSecurityRunConfig:
    name: str

    # algorithms compared, and in which regime(s) (context.md's comparison plan)
    algorithms: tuple[str, ...] = ("simplex", "mfreinforce", "reinforce")
    budget_modes: tuple[str, ...] = ("equal_parameters",)  # "equal_parameters" and/or "equal_budget"
    flows: tuple[str, ...] = ("exact",)  # "exact" and/or "particle"
    particle_size: int = 200  # N~ trajectories/step for the particle population estimate

    # perturbation grid (reference "Compared perturbation schemes")
    lambdas: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8)  # simplex perturbation scale
    epsilon: float = 1.0  # logit (MF-REINFORCE) perturbation scale
    sigma: float = 1.0  # simplex Gaussian perturbation std

    # horizon (reference "Training and validation protocol")
    horizons: tuple[int, ...] = (3,)  # T_train: kept short to limit sensitivity-estimator error accumulation
    T_val: int | None = 50  # validation horizon, longer than training

    # Monte Carlo budget and optimization (reference "Training and validation protocol")
    B: int = 200  # main trajectories per gradient step
    n_aux: int = 1  # auxiliary trajectories per step
    lr: float = 1e-3
    n_train: int = 20_000  # training iterations
    validate_every: int = 10

    # seeds and initial laws (reference "Training and validation protocol")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    mu0_val: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # uniform law, validation only

    def logit_transitions(self, T: int) -> int:
        """C_logit(T) = B*T + B*n_aux*T*(T+1)/2 (see configs/twostate.py's module docstring)."""
        return self.B * T + self.B * self.n_aux * (T * (T + 1) // 2)

    def equal_budget_target(self, T: int) -> int:
        """Per-step transition budget at horizon T that matches mfreinforce's cost: C_logit(T)/T."""
        return self.logit_transitions(T) // T

    def simplex_B_equal_budget(self, T: int) -> int:
        """B for simplex under budget_mode="equal_budget": n_aux stays fixed, the main batch absorbs the rest."""
        return self.equal_budget_target(T) - self.n_aux

    def reinforce_B_equal_budget(self, T: int) -> int:
        """B for reinforce under budget_mode="equal_budget": no auxiliary batch, so the whole allocation is B."""
        return self.equal_budget_target(T)


MAIN = CyberSecurityRunConfig(
    name="main",
    budget_modes=("equal_budget",),
    flows=("particle",),
    horizons=(3, 6, 12),
)

MID = CyberSecurityRunConfig(
    name="mid",
    n_train=5_000,
    seeds=(0,),
)

SMOKE = CyberSecurityRunConfig(
    name="smoke",
    n_train=20,
    seeds=(0,),
)
