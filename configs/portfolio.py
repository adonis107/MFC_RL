"""
Run configurations for the mean-variance portfolio benchmark.

Mirrors `configs/lq.py`'s structure and rationale (see its module docstring
for the full argument): the compared algorithms are the continuous-state
simplex-perturbed MF-REINFORCE estimator (`mfc.algorithms.continuous_simplex`)
and its classical-REINFORCE ablation, both model-free; the closed-form
`"exact_gradient"` is the oracle `scripts/test.py` measures against, not a
competitor. The budget is equal by construction — simplex spends
(`n_aux`+`B`)*T transitions per step, reinforce the same total in one main
batch.

`lambdas=(0.025,0.05,0.1,0.2,0.4)` is the reference's own grid (Sec.
"Training and evaluation": "we repeat the gradient experiment for
lambda in {0.025,0.05,0.1,0.2,0.4}") — unlike LQ, which has no reference
grid and reuses this repo's canonical (0.05,0.1,0.2,0.4,0.8), this
benchmark's own reference value is used as-is. Note that `rho=1` here (also
the reference's own), so a given lambda perturbs the population mean about
three times as hard as the same lambda does in LQ. As in `configs/lq.py`,
the grid is swept for simplex only: reinforce has no perturbation scale and
is trained once per (T, seed) on the nominal process.

Budget and convergence. This is the hard benchmark of the two, and the
reason is structural, not a tuning failure: the reference's own tau=0.02
(baseline exploration std) makes every score-function estimator's magnitude
proportional to 1/tau^2 = 2500, while the gradient itself is O(10^-2). Both
model-free algorithms therefore work at a very low signal-to-noise ratio and
plateau on a stochastic-gradient noise floor rather than converging to
theta*: at the settings below they recover a visible fraction of the
available improvement over zero theta, with simplex ahead of the REINFORCE
baseline. Measured while building
this module, neither more iterations (20_000 at lr=0.01 oscillates around
the same plateau and ends no better) nor a larger step (lr=0.03 destabilizes)
moves that floor; a 4x larger batch buys only ~0.02 of J for 2x the runtime.
This is worth reporting in `notebooks/portfolio.ipynb`: it is the
MSE-vs-budget trade-off of the Research_Project.tex score-function estimator
made visible, not something to hide by shrinking the reference's tau.

Training and validation both use the environment's own fixed
`PortfolioConfig.mu0`/`Sigma0` (no randomized-initial-law training protocol
is described for this benchmark — see `mfc.environments.portfolio`'s module
docstring), and theta is horizon-specific (see `mfc.environments.lq` for
why), so there is no `mu0_val`/`T_val` field here either.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioRunConfig:
    name: str

    algorithms: tuple[str, ...] = ("simplex", "reinforce")

    lambdas: tuple[float, ...] = (0.025, 0.05, 0.1, 0.2, 0.4)  # reference's own grid
    horizons: tuple[int, ...] = (10,)  # T=10 is the reference's own baseline

    # Monte Carlo budget: simplex's own (n_aux, B) is the anchor, reinforce matches its total.
    # Both are 5x this repo's usual sizes -- see the module docstring on tau=0.02.
    B: int = 1000  # main trajectories per gradient step
    n_aux: int = 500  # auxiliary trajectories per step (moment-sensitivity flow)
    baseline: str | None = "loo"  # leave-one-out return baseline; see configs/lq.py

    lr: float = 0.01
    n_train: int = 6_000
    validate_every: int = 100

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def transitions_per_step(self, T: int) -> int:
        """Simulated transitions one simplex gradient step costs at horizon T."""
        return (self.n_aux + self.B) * T

    def reinforce_B_equal_budget(self) -> int:
        """B for reinforce under this repo's equal-budget rule (see `configs/lq.py`)."""
        return self.n_aux + self.B


MAIN = PortfolioRunConfig(name="main", horizons=(5, 10, 20))  # 5/20: this repo's own horizon-scaling extension around the reference's T=10

MID = PortfolioRunConfig(name="mid", n_train=2_000, seeds=(0,))

SMOKE = PortfolioRunConfig(name="smoke", n_train=20, seeds=(0,))
