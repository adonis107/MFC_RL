"""
Run configurations for the distribution-planning benchmark.

Values taken directly from the reference (files/reference/discrete_benchmarks.tex,
"Distribution Planning", paragraphs "Compared perturbation schemes" and
"Training and validation protocol"): the simplex lambda grid, sigma=1.0
(U_t~N(0,I_9)), B=500, n_aux=10, lr=1e-4, validate_every=10, T=5 (both
training and validation — unlike cybersecurity, there is no T_train/T_val
split here, so no T_val field), mu0_val=uniform(1/10,...,1/10), and 5
independent seeds. n_train=100_000 is the reference's own value (used here
as "main"); mid/smoke shrink it for faster iteration. epsilon is fixed (not
swept, matching this repo's twostate/cybersecurity convention): the
reference reports epsilon in {1.0, 2.0} both giving "more stable training
and faster convergence" than smaller values, without a clear winner between
the two (Figure 5 just shows both trajectories differ qualitatively) — kept
here at the more moderate 1.0, as for cybersecurity; the reference's full
grid is epsilon in {0.5, 0.75, 1.0, 2.0}. The lambda grid is trimmed to
{0.1, 0.2, 0.4} (the reference's own grid also carries 0.8) to keep the paid
run's job count down, as for cybersecurity: 0.8 is far outside the
small-perturbation regime the theory covers, and two-state's `main` still
sweeps the full grid including it.

Unlike two-state, mu0 is *not* resampled from a RunConfig-parametrized range
each training iteration: it is always mu0~Dirichlet(1,...,1) (10-dim), a
model constant
(`mfc.environments.distribution_planning.DistributionPlanning.sample_mu0`),
so there are no mu0_low/mu0_high fields here — see `scripts/train.py`'s
SAMPLE_MU0_FACTORIES. There is also no discount here (gamma=1, the default):
unlike cybersecurity, the reference states no discount factor for this
benchmark.

Equal-budget allocation. As for cybersecurity, the reference itself uses
identical (B, n_aux) for simplex and mfreinforce here — no budget-matching
adjustment — but this repo adds an "equal_budget" mode as for two-state (see
configs/twostate.py's module docstring for the full derivation): the
complexity formulas are generic in (B, n_aux, T), so they are reused
verbatim. `main` runs *only* budget_mode="equal_budget" and *only*
flow="particle" (the empirical population-flow estimate, not the exact
recursion), as cybersecurity's and advertising's do — two-state's `main` is
the only place the exact flow and the equal-parameters regime are compared.
This is the expensive one: at B=500, n_aux=10, T=5 the target is
C_logit(5)/5 = 15500 per step, so simplex trains at B=15490 and reinforce at
B=15500, ~31x the reference's own batch, against a ~76.6k-parameter policy
MLP for n_train=100_000 iterations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionPlanningRunConfig:
    name: str

    # algorithms compared, and in which regime(s) (context.md's comparison plan)
    algorithms: tuple[str, ...] = ("simplex", "mfreinforce", "reinforce")
    budget_modes: tuple[str, ...] = ("equal_parameters",)  # "equal_parameters" and/or "equal_budget"
    flows: tuple[str, ...] = ("exact",)  # "exact" and/or "particle"
    particle_size: int = 500  # N~ trajectories/step for the particle population estimate (matches B)

    # perturbation grid (reference "Compared perturbation schemes")
    lambdas: tuple[float, ...] = (0.1, 0.2, 0.4)  # simplex perturbation scale (reference also has 0.8; see module docstring)
    epsilon: float = 1.0  # logit (MF-REINFORCE) perturbation scale
    sigma: float = 1.0  # simplex Gaussian perturbation std

    # horizon (reference "Model": T=5 for both training and validation)
    horizons: tuple[int, ...] = (5,)

    # Monte Carlo budget and optimization (reference "Training and validation protocol")
    B: int = 500  # main trajectories per gradient step
    n_aux: int = 10  # auxiliary trajectories per step
    lr: float = 1e-4
    n_train: int = 100_000  # training iterations
    validate_every: int = 10

    # seeds and initial laws (reference "Training and validation protocol")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    mu0_val: tuple[float, ...] = tuple([0.1] * 10)  # uniform law, validation only

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


MAIN = DistributionPlanningRunConfig(
    name="main",
    budget_modes=("equal_budget",),
    flows=("particle",),
)

# mid/smoke also shrink B (unlike twostate/cybersecurity's tiers, which only
# shrink n_train), to keep a dev iteration quick: the policy MLP here has
# ~76.6k parameters (width-256 hidden layers, vs. cybersecurity's ~1.5k at
# width 32), so per-step cost scales steeply in B. Speed, not fidelity —
# MAIN keeps B=500 exactly.
#
# This used to be a memory limit as well: the estimators materialized
# (T,B,D) per-sample score tensors, several GB per step at B=500. They no
# longer do (see `mfc.algorithms._common.weighted_score_sum`), and a step
# now costs ~100 MiB even at the equal-budget B=15490, so if a faster
# machine makes the wall-clock affordable there is nothing stopping these
# tiers from moving back toward B=500.
MID = DistributionPlanningRunConfig(
    name="mid",
    n_train=5_000,
    seeds=(0,),
    B=100,
)

SMOKE = DistributionPlanningRunConfig(
    name="smoke",
    n_train=20,
    seeds=(0,),
    B=20,
)
