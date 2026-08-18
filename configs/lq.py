"""
Run configurations for the linear-quadratic (LQ) benchmark.

Unlike every other benchmark's RunConfig, there is no `budget_modes`/`flows`
sweep here: `algorithm="exact_gradient"` (`mfc.algorithms.lq.train`) uses the
reference's closed-form gradient directly (files/reference/LQ_framework.tex,
"Exact Gradient Algorithm") — no sampling, no budget to match — and
`algorithm="reinforce"` is the classical-REINFORCE ablation (context.md's
"Show missing mean-field term by comparison with reinforce"), which only
needs a Monte Carlo batch size `B`.

`lambdas` reuses this repo's canonical grid (`(0.05,0.1,0.2,0.4,0.8)`, as in
`configs/cybersecurity.py`/`configs/distribution_planning.py`): LQ's own
lambda (LQ_framework.tex, Sec. "Randomized Perturbation") plays the same
role as the discrete benchmarks' simplex perturbation scale, so both
algorithms are trained at every lambda in the grid (reinforce's rollout is
itself lambda-perturbed — see `mfc.environments.lq.LQ.rollout` — unlike the
discrete `reinforce.py`, which has no perturbation to sweep at all).

`lr`/`n_train` are picked empirically (see the exact-gradient vs. Riccati-
optimum check and the reinforce convergence check run while building this
module) so that *both* algorithms reach within ~1% of J^0(theta*) by
n_train=8000 at every horizon in `MAIN.horizons`, including the least
favorable case, T=10 (the a+c=1.7 uncontrolled mean is unstable, so the
longest horizon needs the most iterations to fully converge).

Training and validation both use the environment's own fixed `LQConfig.mu0`/
`Sigma0` (see `mfc.environments.lq`'s module docstring: LQ_framework.tex
describes no randomized-initial-law training protocol, unlike the discrete
benchmarks), so there is no `mu0_val`/`mu0_low`/`mu0_high` field here.
Likewise theta is genuinely time-indexed and horizon-specific (shape
(T,2)), so there is no `T_val` distinct from the training horizon.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LQRunConfig:
    name: str

    algorithms: tuple[str, ...] = ("exact_gradient", "reinforce")

    lambdas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    horizons: tuple[int, ...] = (5,)

    B: int = 200  # reinforce Monte Carlo batch size (exact_gradient needs none)
    lr: float = 0.05
    n_train: int = 8_000
    validate_every: int = 100

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


MAIN = LQRunConfig(name="main", horizons=(3, 5, 10))

MID = LQRunConfig(name="mid", n_train=2_000, seeds=(0,))

SMOKE = LQRunConfig(name="smoke", n_train=20, seeds=(0,))
