"""
Run configurations for the linear-quadratic (LQ) benchmark.

Compared algorithms (context.md's comparison plan, in its continuous-state
form): `"simplex"` — the continuous-state simplex-perturbed MF-REINFORCE
estimator of `mfc.algorithms.continuous_simplex` — against `"reinforce"`,
the classical-REINFORCE ablation of that same estimator with the
population-perturbation score dropped ("Show missing mean-field term by
comparison with reinforce"). Both are model-free, and run on a common
rollout they differ by exactly one term — which is how `scripts/test.py`'s
fixed-theta diagnostics compare them; in *training* each runs on the process
it actually needs (see `lambdas` below). The
closed-form `"exact_gradient"` (LQ_framework.tex, "Exact Gradient
Algorithm") is deliberately *not* in this list: it is the oracle every
diagnostic in `scripts/test.py` measures against, not a competitor — it uses
the model, which is what the whole construction exists to avoid.

Budget. Unlike the discrete benchmarks there is no `budget_modes` sweep,
because there is only one allocation worth running: equal budget
(context.md, "For all subsequent environments and tests, we must use equal
budget"). Simplex spends (`n_aux`+`B`)*T transitions per step — one
auxiliary batch for the coordinate sensitivities, one main batch for the
gradient (Remark "Linear per-iteration complexity",
continuous_state_space(2).tex) — and reinforce gets the same total in a
single main batch, `reinforce_B_equal_budget()`. There is no exact-vs-
particle flow sweep either: the nominal coordinate flow c_t^theta =
mu_t^theta is the exact forward moment recursion for both continuous
benchmarks (Assumption "Access to the nominal population coordinates").

`lambdas` reuses this repo's canonical grid (`(0.05,0.1,0.2,0.4,0.8)`, as in
`configs/cybersecurity.py`/`configs/distribution_planning.py`): LQ's own
lambda (LQ_framework.tex, Sec. "Randomized Perturbation") plays exactly the
role the discrete benchmarks' simplex perturbation scale does. It is swept
for simplex only. reinforce has no perturbation scale — the randomization
exists solely to expose the mean-field sensitivity through a likelihood
ratio, so an algorithm that drops that term has no reason to inject it, and
would only be trading variance for a shifted objective J^lambda — so it is
trained once per (T, seed) on the nominal process, exactly as the discrete
`reinforce.py` is (see `scripts/train.py`'s
ALGORITHMS_WITH_PERTURBATION_SCALE). The comparison is therefore
simplex-at-each-lambda against one honest REINFORCE baseline, not against a
REINFORCE handicapped by a perturbation it cannot use.

`baseline="loo"` subtracts the leave-one-out mean return from the returns
multiplying both score terms. It is admissible exactly (Remark "Admissible
baselines": the leave-one-out mean is independent of the trajectory it
weights, and both scores are conditionally centered), so it changes no
estimate's expectation. It matters a lot in practice here — LQ's costs are
large and strictly positive, so E[G_t] contributes most of the raw
estimator's variance: measured at a fixed theta with the exact sensitivity
flow, dropping it multiplies the gradient's standard deviation by ~6, and in
training it is the difference between converging to ~1.01x J^0(theta*) and
to ~2x.

`lr`/`n_train`/`B`/`n_aux` are picked empirically (see the convergence
sweeps run while building this module) so that both algorithms converge at
every lambda in the grid at T=3 and T=5: at `main`'s n_train and T=5,
simplex lands within 0.2% of J^0(theta*) across the whole lambda grid and
reinforce within 0.2-0.5%, consistently behind it (T=3 is easier, and was
already converging at half these batch sizes). The batch sizes are what the *smallest*
lambda costs: the estimator's variance grows as lambda^-2 (Theorem "Bias and
MSE of the gradient estimator"), and at lambda=0.05, T=5, halving them turns
a run that settles at 1.002x J^0(theta*) into one that wanders between 1.0x
and 1.4x without ever settling.

T=10 is deliberately kept in `MAIN.horizons` even though simplex does *not*
converge there at this budget (it lands around 1.3-2.9x J^0(theta*), behind
reinforce's 1.05-1.11x). With a+c=1.9 the uncontrolled population mean is
unstable, so the coordinate sensitivities D_t^theta grow geometrically with
t and the perturbation score's variance grows with them — the "curse of
time" of Remark "Horizon dependence" (continuous_state_space(2).tex), which
is a statistical property of the estimator and not of its O((n_aux+B)*T)
cost. Doubling the batch again does not fix it. That degradation *is* the
horizon-scaling result context.md asks for, so it is reported rather than
hidden by dropping the horizon.

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

    algorithms: tuple[str, ...] = ("simplex", "reinforce")

    lambdas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    horizons: tuple[int, ...] = (5,)

    # Monte Carlo budget: simplex's own (n_aux, B) is the anchor, reinforce matches its total
    B: int = 400  # main trajectories per gradient step
    n_aux: int = 200  # auxiliary trajectories per step (coordinate-sensitivity flow)
    baseline: str | None = "loo"  # leave-one-out return baseline; see module docstring

    lr: float = 0.02
    n_train: int = 6_000
    validate_every: int = 100

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def transitions_per_step(self, T: int) -> int:
        """Simulated transitions one simplex gradient step costs at horizon
        T: (n_aux+B)*T, linear in the horizon."""
        return (self.n_aux + self.B) * T

    def reinforce_B_equal_budget(self) -> int:
        """B for reinforce under this repo's equal-budget rule: it has no
        auxiliary batch, so the whole per-step allocation goes to the main
        batch. Horizon-independent, unlike the discrete benchmarks'
        `reinforce_B_equal_budget(T)` (there the anchor is mfreinforce, whose
        stagewise auxiliary batch costs O(T^2); here the anchor is simplex,
        which is linear in T, so the same B matches at every horizon)."""
        return self.n_aux + self.B


MAIN = LQRunConfig(name="main", horizons=(3, 5, 10))

MID = LQRunConfig(name="mid", n_train=2_000, seeds=(0,))

SMOKE = LQRunConfig(name="smoke", n_train=20, seeds=(0,))
