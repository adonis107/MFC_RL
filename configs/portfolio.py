"""
Run configurations for the mean-variance portfolio benchmark.

Mirrors `configs/lq.py`'s structure and rationale (see its module
docstring: `mfc.algorithms.portfolio.train`'s `"exact_gradient"` uses the
reference's closed-form gradient directly — no sampling, no budget to
match — and `"reinforce"` is the classical-REINFORCE ablation, needing only
a Monte Carlo batch size `B`).

`lambdas=(0.025,0.05,0.1,0.2,0.4)` is the reference's own grid (Sec.
"Training and evaluation": "we repeat the gradient experiment for
lambda in {0.025,0.05,0.1,0.2,0.4}") — unlike LQ, which has no reference
grid and reuses this repo's canonical (0.05,0.1,0.2,0.4,0.8), this
benchmark's own reference value is used as-is.

`lr`/`n_train`/`B` are picked empirically (see the exact-gradient vs.
optimal-theta check and the reinforce convergence check run while building
this module): exact_gradient converges to within ~1% of J^0(theta*) in a
few hundred iterations; reinforce needs a much larger batch (B=2000, not
this repo's usual 200) and many more iterations to make visible progress at
all — the reference's own tau=0.02 (baseline exploration std) makes the
policy score's magnitude (proportional to 1/tau^2) very large, so the naive
REINFORCE estimator here is extremely high-variance; this is itself a
notebook finding (context.md's "Show missing mean-field term by comparison
with reinforce"), not a bug to hide by shrinking tau.

Training and validation both use the environment's own fixed
`PortfolioConfig.mu0`/`Sigma0` (no randomized-initial-law training protocol
is described for this benchmark either — see `mfc.environments.portfolio`'s
module docstring), and theta is horizon-specific (see `mfc.environments.lq`
for why), so there is no `mu0_val`/`T_val` field here either.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioRunConfig:
    name: str

    algorithms: tuple[str, ...] = ("exact_gradient", "reinforce")

    lambdas: tuple[float, ...] = (0.025, 0.05, 0.1, 0.2, 0.4)  # reference's own grid
    horizons: tuple[int, ...] = (10,)  # T=10 is the reference's own baseline

    B: int = 2000  # reinforce Monte Carlo batch size (exact_gradient needs none) -- large: see module docstring
    lr: float = 0.02
    n_train: int = 6_000
    validate_every: int = 100

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


MAIN = PortfolioRunConfig(name="main", horizons=(5, 10, 20))  # 5/20: this repo's own horizon-scaling extension around the reference's T=10

MID = PortfolioRunConfig(name="mid", n_train=2_000, seeds=(0,))

SMOKE = PortfolioRunConfig(name="smoke", n_train=20, seeds=(0,))
