"""
Run configurations for the targeted-advertising benchmark.

Values taken directly from the reference (files/reference/discrete_benchmarks.tex,
"Targeted advertising with social influence", paragraphs "Policy
parametrization", "Perturbation schemes"): the simplex lambda grid,
sigma=1.0, B=200, n_aux=10, validate_every=10, T=5, and 5 independent
seeds. The reference doesn't state B/n_aux/lr/n_train numerically for this
benchmark (only the *budget-matching formula* between simplex and logit,
"Training comparisons"), so — as this repo does everywhere the reference
gives a formula but not a number — they're set here to match two-state's
own values (same state/action-space scale: |X|=|A|=2), used as "main".
mid/smoke shrink n_train for faster iteration. epsilon is fixed (not swept,
matching this repo's convention elsewhere) at 1.0, the mid-point of no
reference-stated grid for this benchmark either. The lambda grid is trimmed
to {0.1, 0.2, 0.4} (the reference's own grid also carries 0.8) to keep the
paid run's job count down, as for cybersecurity and distribution planning:
0.8 is far outside the small-perturbation regime the theory covers, and
two-state's `main` still sweeps the full grid including it.

Unlike two-state, mu0 is *not* resampled from a RunConfig-parametrized
range: p0~U([0.05,0.95]) is a fixed model constant (`mfc.environments.
advertising.Advertising.sample_mu0`; the reference deliberately excludes
the simplex boundary here, unlike two-state's own U([0.1,0.9])), so there
are no mu0_low/mu0_high fields — see `scripts/train.py`'s
SAMPLE_MU0_FACTORIES. mu0_val is a single representative point (p0=0.5)
rather than the reference's full validation grid G_val subset of
[0.05,0.95] (`\bar J_val` averages over many p0) — a deliberate
simplification to this repo's single-mu0_val `train_run` convention, as
established for the other benchmarks.

Discounting lives on `AdvertisingConfig.gamma` (an environment constant,
0.5 per the reference, truncating the source infinite-horizon problem), not
here — `scripts/train.py` reads `env.config.gamma` and threads it through
every algorithm's `gamma=` keyword, as for cybersecurity.

Equal-budget allocation. As elsewhere, this repo adds an "equal_budget"
mode (see configs/twostate.py's module docstring for the full derivation):
the complexity formulas are generic in (B, n_aux, T), so they are reused
verbatim. `main` runs *only* budget_mode="equal_budget" and *only*
flow="particle" (the empirical population-flow estimate, not the exact
recursion), as cybersecurity's and distribution planning's do — two-state's
`main` is the only place the exact flow and the equal-parameters regime are
compared. At B=200, n_aux=10, T=5 the target is C_logit(5)/5 = 6200 per
step, so simplex trains at B=6190 and reinforce at B=6200.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdvertisingRunConfig:
    name: str

    # algorithms compared, and in which regime(s) (context.md's comparison plan)
    algorithms: tuple[str, ...] = ("simplex", "mfreinforce", "reinforce")
    budget_modes: tuple[str, ...] = ("equal_parameters",)  # "equal_parameters" and/or "equal_budget"
    flows: tuple[str, ...] = ("exact",)  # "exact" and/or "particle"
    particle_size: int = 200  # N~ trajectories/step for the particle population estimate (matches B)

    # perturbation grid (reference "Perturbation schemes")
    lambdas: tuple[float, ...] = (0.1, 0.2, 0.4)  # simplex perturbation scale (reference also has 0.8; see module docstring)
    epsilon: float = 1.0  # logit (MF-REINFORCE) perturbation scale
    sigma: float = 1.0  # simplex Gaussian perturbation std

    # horizon (reference "Benchmark parameters")
    horizons: tuple[int, ...] = (5,)

    # Monte Carlo budget and optimization (not numerically stated by the reference for this benchmark; see module docstring)
    B: int = 200  # main trajectories per gradient step
    n_aux: int = 10  # auxiliary trajectories per step
    lr: float = 1e-3
    n_train: int = 10_000  # training iterations
    validate_every: int = 10

    # seeds and initial laws
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    mu0_val: tuple[float, float] = (0.5, 0.5)  # a single representative point; see module docstring

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


MAIN = AdvertisingRunConfig(
    name="main",
    budget_modes=("equal_budget",),
    flows=("particle",),
)

MID = AdvertisingRunConfig(
    name="mid",
    n_train=5_000,
    seeds=(0,),
)

SMOKE = AdvertisingRunConfig(
    name="smoke",
    n_train=20,
    seeds=(0,),
)
