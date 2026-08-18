"""
Run configurations for the two-state two-action benchmark.

Values taken directly from the reference (files/reference/discrete_benchmarks.tex,
"Two-State Two-Action Toy Problem", paragraphs "Compared perturbation schemes"
and "Training protocol"): the lambda grid, sigma=1.0, B=200, n=10, lr=1e-3,
validate_every=10, mu0(1)~U([0.1,0.9]), mu0_val=(0.2,0.8), and 5 independent
seeds. n_train=5000 is the reference's value (the "mid" default); epsilon=0.2
(fixed, not swept) and the three-tier main/mid/smoke structure follow
context.md's plan for this repo rather than the reference, which only runs
the exact-flow, equal-parameters case.

Axes beyond the reference, with no prescribed value, are explicit adjustable
choices:
  - MAIN's horizons=(2,5,10) (base/medium/long) and n_train=10_000 push past
    the reference's own (T=2, n_train=5000) for extra margin on the
    horizon-scaling and convergence tests context.md asks for.
  - particle_size: 200 (matches B), the N~ trajectories per step used to
    estimate the population law when flows includes "particle".

Equal-budget allocation. mfreinforce (logit-perturbed MF-REINFORCE) keeps
the reference's own (n_aux, B) = (10, 200) unchanged, per this task's
instructions — this is the anchor the other two algorithms are matched to.
Its complexity is *not* simplex's T*(n_aux+B): mfreinforce here is Meunier,
Pham & Reisinger's original Algorithm 1+2 (files/Discrete RL - Meunier, Pham,
Reisinger.md), which has no reusable auxiliary batch — Algorithm 2's
gradient-of-logits estimator (cost n_aux*T*(T+1)/2, one fresh stagewise
batch per target time) is rerun independently for *every* main trajectory,
giving
    C_logit(T) = B*T + B*n_aux*T*(T+1)/2,
quadratic in T, vs. simplex's linear T*(n_aux+B) (its reusable single-batch
forward estimator, discrete_state_space(2).tex Remark "Linear per-iteration
complexity" — a genuine algorithmic improvement, not just a relabeling).
Consequently the equal-budget target is horizon-dependent: at each T, both
adapted algorithms get the same per-step transition budget
    target(T) = C_logit(T) / T = B*(1 + n_aux*(T+1)/2).
reinforce has no auxiliary sensitivity batch, so its whole allocation is
B_reinforce(T) = target(T). simplex keeps n_aux fixed at the reference value
(so the sensitivity estimator's own noise properties stay comparable across
the horizon sweep) and lets its main batch absorb the rest:
B_simplex(T) = target(T) - n_aux. Both are the same under either flow: the
particle-flow simulator cost (particle_size*T, shared by whichever
algorithms use particle flow) appears identically on both sides of the
budget equation and cancels out of target(T).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoStateRunConfig:
    name: str

    # algorithms compared, and in which regime(s) (context.md's comparison plan)
    algorithms: tuple[str, ...] = ("simplex", "mfreinforce", "reinforce")
    budget_modes: tuple[str, ...] = ("equal_parameters",)  # "equal_parameters" and/or "equal_budget"
    flows: tuple[str, ...] = ("exact",)  # "exact" and/or "particle"
    particle_size: int = 200  # N~ trajectories/step for the particle population estimate

    # perturbation grid (reference "Compared perturbation schemes")
    lambdas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)  # simplex perturbation scale
    epsilon: float = 0.2  # logit (MF-REINFORCE) perturbation scale
    sigma: float = 1.0  # simplex Gaussian perturbation std

    # horizon
    horizons: tuple[int, ...] = (2,)  # T values to run; main sweeps base/medium/long

    # Monte Carlo budget and optimization (reference "Training protocol")
    B: int = 200  # main trajectories per gradient step
    n_aux: int = 10  # auxiliary trajectories per step
    lr: float = 1e-3
    n_train: int = 10_000  # training iterations
    validate_every: int = 10

    # seeds and initial laws (reference "Training protocol")
    seeds: tuple[int, ...] = (0,)
    mu0_low: float = 0.1
    mu0_high: float = 0.9
    mu0_val: tuple[float, float] = (0.2, 0.8)

    def logit_transitions(self, T: int) -> int:
        """
        C_logit(T) = B*T + B*n_aux*T*(T+1)/2: mfreinforce's own simulated-
        transition cost at horizon T under its fixed (n_aux, B) (Meunier,
        Pham & Reisinger's original stagewise Algorithm 1+2 — see the module
        docstring). The equal-budget anchor.
        """
        return self.B * T + self.B * self.n_aux * (T * (T + 1) // 2)

    def equal_budget_target(self, T: int) -> int:
        """Per-step transition budget at horizon T that matches mfreinforce's
        cost: C_logit(T)/T. Shared by simplex's (n_aux+B) and reinforce's B
        under budget_mode="equal_budget"."""
        return self.logit_transitions(T) // T

    def simplex_B_equal_budget(self, T: int) -> int:
        """B for simplex under budget_mode="equal_budget" at horizon T:
        n_aux stays at the reference value; the main batch absorbs the rest
        of the equal-budget target (see module docstring)."""
        return self.equal_budget_target(T) - self.n_aux

    def reinforce_B_equal_budget(self, T: int) -> int:
        """B for reinforce under budget_mode="equal_budget" at horizon T: no
        auxiliary sensitivity batch, so its whole allocation is the
        equal-budget target."""
        return self.equal_budget_target(T)


MAIN = TwoStateRunConfig(
    name="main",
    budget_modes=("equal_parameters", "equal_budget"),
    flows=("exact", "particle"),
    horizons=(2, 5, 10),
    n_train=10_000,
    seeds=(0, 1, 2, 3, 4),
)

MID = TwoStateRunConfig(
    name="mid",
    horizons=(2,),
    n_train=5_000,
    seeds=(0,),
)

SMOKE = TwoStateRunConfig(
    name="smoke",
    horizons=(2,),
    n_train=20,
    seeds=(0,),
)
