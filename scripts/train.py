"""
Train (env, alg) over a run config (main/mid/smoke) and save everything.

Usage:
    uv run python scripts/train.py --env twostate --alg simplex --config smoke

    # one independent job (skips training if its .pt already exists):
    uv run python scripts/train.py --env twostate --alg simplex --config main \\
        --budget-mode equal_budget --flow particle --T 10 --lam 0.1 --seed 3 --dtype float32

    # print every job in a config as a runnable command, one per line, instead
    # of training (Python is the source of truth for the grid; `train_all.sh`
    # just schedules the printed commands in parallel):
    uv run python scripts/train.py --env twostate --alg simplex --config main --list

`--budget-mode`/`--flow`/`--T`/`--lam`/`--seed` each pin one axis of the sweep
to a single value (`--budget-mode`/`--flow` are validated against
SUPPORTED_BUDGET_MODES/SUPPORTED_FLOWS by argparse's `choices=`; `--T`/`--lam`/
`--seed` accept any value, e.g. an extra seed beyond the config's own — see
`list_experiments`); omitted axes still sweep the full config grid, so
`run_all`'s old whole-config behavior is every one of these omitted.
`--progress auto|on|off` controls the per-run progress bar over training
iterations (annotated with the latest validation objective). "auto" draws one
for a terminal or notebook but not into a redirected log; it cannot tell that
sibling workers are sharing a terminal, so `train_all.sh` passes `--progress
off` whenever it schedules more than one worker — see `mfc.progress`. Either
way each job prints one self-contained `... started` / `... done in ...` line,
so a parallel run's output stays readable.

`--dtype float32` switches training precision (every `mfc.environments.*`
class defaults to float64); always compare a `smoke` run in both dtypes
(objectives, policy error, theta) before trusting float32 for a paid `main`
run — training is stochastic, so exact trajectory-by-trajectory equality
isn't the bar, but the final numbers and conclusions should match closely.

Every individual job is independent (a distinct `(alg, budget_mode, flow, T,
lam, seed)` combination) and already saved under its own unique filename, so
`train_all.sh` runs many single-job invocations concurrently via `--list`'s
output — resumability comes for free: a job whose `.pt` already exists is
skipped (pass `--overwrite` to force a retrain).

A config's `budget_modes x flows x horizons` cartesian product is the
default grid, but a config can instead set `groups` (an explicit tuple of
(budget_mode, flow, T) triples — e.g. configs/twostate.py's MAIN, which
only wants 5 of its 12 cartesian-product combinations); `list_experiments`
reads `groups` when set and ignores the cartesian-product fields entirely.
mfreinforce ignores budget_mode (fixed n_aux/B), so whenever a grid (from
either source) requests the same (flow, T, lam, seed) under more than one
budget_mode for mfreinforce, only the first is trained for real — `--list`
doesn't even print the rest, and `run_all` copies them in-process; a
`--list`-driven parallel run needs a separate `--materialize-duplicates`
pass afterward (once the primaries have actually finished training) to
create those copies — `train_all.sh` runs it automatically.

For each (horizon, perturbation scale, seed) in the config, runs Algorithm 1
with the reference's per-iteration randomized initial law
(mu0(1)~U([mu0_low,mu0_high]) each step; files/reference/discrete_benchmarks.tex,
"Training protocol" — note this differs from `mfc.algorithms.simplex.train`,
which takes a single fixed mu0 per the formal algorithm statement), logs the
exact validation objective every `validate_every` steps, and saves the full
theta trajectory plus validation history to `runs/<env>/<config>/...pt` so
that downstream diagnostics (scripts/test.py, notebooks) never need to
retrain.

The continuous-state benchmarks (`lq`, `portfolio`) take a different path
through this module: `run_continuous` drives `mfc.algorithms.continuous.train`
over a (T, lam, seed) grid with no budget_mode/flow axis, since their
allocation is equal-budget by construction and their nominal population
coordinate flow is exact (see `configs/lq.py`). Their algorithms are
`mfc.algorithms.continuous.ALGORITHMS`: "simplex" (the continuous-state
simplex-perturbed MF-REINFORCE estimator), "reinforce" (its ablation), and
"exact_gradient" (the closed-form oracle, trainable but not part of the
comparison).

All three discrete algorithms are implemented, for both flows and both budget modes.
Under budget_mode="equal_budget", simplex's and reinforce's main batch B is
horizon-dependent (`cfg.simplex_B_equal_budget(T)` / `cfg.reinforce_B_equal_budget(T)`)
— see configs/twostate.py's module docstring: mfreinforce's true per-step
cost is quadratic in T (Algorithm 2's stagewise, non-reusable auxiliary
batch), unlike simplex's linear T*(n_aux+B) or reinforce's T*B, so matching
it takes a growing main batch as T grows. mfreinforce is itself the
equal-budget anchor, so its own (n_aux, B) never change. Neither reinforce
nor mfreinforce has a lambda-like perturbation scale swept by the config
(mfreinforce's is the single fixed `cfg.epsilon`), so both ignore the
config's lambda grid entirely rather than retraining once per (redundant)
lambda value.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from configs.advertising import MAIN as ADVERTISING_MAIN
from configs.advertising import MID as ADVERTISING_MID
from configs.advertising import SMOKE as ADVERTISING_SMOKE
from configs.cybersecurity import MAIN as CYBERSECURITY_MAIN
from configs.cybersecurity import MID as CYBERSECURITY_MID
from configs.cybersecurity import SMOKE as CYBERSECURITY_SMOKE
from configs.distribution_planning import MAIN as DISTRIBUTION_PLANNING_MAIN
from configs.distribution_planning import MID as DISTRIBUTION_PLANNING_MID
from configs.distribution_planning import SMOKE as DISTRIBUTION_PLANNING_SMOKE
from configs.lq import MAIN as LQ_MAIN
from configs.lq import MID as LQ_MID
from configs.lq import SMOKE as LQ_SMOKE
from configs.portfolio import MAIN as PORTFOLIO_MAIN
from configs.portfolio import MID as PORTFOLIO_MID
from configs.portfolio import SMOKE as PORTFOLIO_SMOKE
from configs.twostate import MAIN as TWOSTATE_MAIN
from configs.twostate import MID as TWOSTATE_MID
from configs.twostate import SMOKE as TWOSTATE_SMOKE
from mfc.algorithms import _common, continuous, mfreinforce, reinforce, simplex
from mfc.environments.advertising import Advertising
from mfc.environments.cybersecurity import CyberSecurity
from mfc.environments.distribution_planning import DistributionPlanning
from mfc.environments.lq import LQ
from mfc.environments.portfolio import Portfolio
from mfc.environments.twostate import TwoState
from mfc.progress import PROGRESS_MODES, progress_enabled, training_bar

ENVIRONMENTS = {"twostate": TwoState, "cybersecurity": CyberSecurity, "distribution_planning": DistributionPlanning, "advertising": Advertising}
CONFIGS = {
    "twostate": {"main": TWOSTATE_MAIN, "mid": TWOSTATE_MID, "smoke": TWOSTATE_SMOKE},
    "cybersecurity": {"main": CYBERSECURITY_MAIN, "mid": CYBERSECURITY_MID, "smoke": CYBERSECURITY_SMOKE},
    "distribution_planning": {"main": DISTRIBUTION_PLANNING_MAIN, "mid": DISTRIBUTION_PLANNING_MID, "smoke": DISTRIBUTION_PLANNING_SMOKE},
    "advertising": {"main": ADVERTISING_MAIN, "mid": ADVERTISING_MID, "smoke": ADVERTISING_SMOKE},
    "lq": {"main": LQ_MAIN, "mid": LQ_MID, "smoke": LQ_SMOKE},
    "portfolio": {"main": PORTFOLIO_MAIN, "mid": PORTFOLIO_MID, "smoke": PORTFOLIO_SMOKE},
}

# Continuous-state environments (Gaussian/non-Gaussian scalar state — see
# mfc.environments.lq's module docstring for why these can't share the
# discrete ENVIRONMENTS/STEP_FACTORIES machinery above). All of them are
# driven by the single `mfc.algorithms.continuous.train` loop, whose
# algorithms are `continuous.ALGORITHMS`.
CONTINUOUS_ENVIRONMENTS = {"lq": LQ, "portfolio": Portfolio}

SUPPORTED_BUDGET_MODES = {"equal_parameters", "equal_budget"}
SUPPORTED_FLOWS = {"exact", "particle"}
# Which algorithms are actually trained once per lambda. simplex's whole
# construction is parametrized by it; the continuous benchmarks'
# exact_gradient genuinely descends on grad J^lambda, so its optimum
# theta^{lambda,*} moves with lambda too. reinforce is not on this list in
# either state space: the perturbation exists only to make the mean-field
# sensitivity term available through a likelihood ratio, and an algorithm
# that drops that term gains nothing from injecting it — it would just be
# optimizing J^lambda instead of J, with extra variance, at 5x the jobs. It
# is therefore trained on the *nominal* process (lambda=0) and recorded with
# lam=None, exactly as the discrete `reinforce.py` already was.
ALGORITHMS_WITH_PERTURBATION_SCALE = {"simplex", "exact_gradient"}

# Environment-specific per-iteration initial-law sampling ("Training
# protocol" for each benchmark): two-state resamples mu0(1)~U([mu0_low,
# mu0_high]) from RunConfig-level bounds; cybersecurity, distribution_planning,
# and advertising each expose their own `env.sample_mu0` (a model constant,
# not RunConfig-parametrized — cybersecurity/distribution_planning sample
# Dirichlet(1,...,1), advertising samples p0~U([0.05,0.95]), but the
# call site is identical for all three). Each factory closes over (cfg, env)
# and returns `sample(generator) -> mu0`.
INIT_SEED = 0  # fixed, reused across all training seeds within a run, so paired runs share one policy initialization

# Cap on the number of theta snapshots `train_run` keeps (evenly spaced over
# training, always including iteration 0 and the final iterate). Storing every
# iterate is fine for a small policy but not for a large one: distribution
# planning's MLP has D~7.7e4 parameters, so at the reference's n_train=100_000
# the full history is ~28.5 GiB in float32 — which, left on the training
# device, fills a 24GB GPU on its own partway through a run. Snapshots are
# also moved to CPU: nothing downstream (scripts/test.py, the notebooks) reads
# `theta_history` at all — they use `theta_final`/`validation_J` — so it has no
# business occupying accelerator memory, and its accompanying iteration indices
# are saved as `theta_history_iterations`.
MAX_THETA_SNAPSHOTS = 256


def _twostate_mu0_sampler(cfg, env):
    def sample(generator):
        m1 = torch.rand((), dtype=env.dtype, device=env.device, generator=generator) * (cfg.mu0_high - cfg.mu0_low) + cfg.mu0_low
        return torch.stack([1.0 - m1, m1])

    return sample


def _env_sample_mu0_sampler(cfg, env):
    def sample(generator):
        return env.sample_mu0(generator=generator)

    return sample


SAMPLE_MU0_FACTORIES = {
    "twostate": _twostate_mu0_sampler,
    "cybersecurity": _env_sample_mu0_sampler,
    "distribution_planning": _env_sample_mu0_sampler,
    "advertising": _env_sample_mu0_sampler,
}


def make_simplex_step(cfg, flow: str, budget_mode: str):
    """Adapts `simplex.gradient_step` to the `step_fn(env, action_probs_fn,
    theta, mu0, *, T, lam, generator) -> g_hat` contract `train_run` expects,
    closing over the config's perturbation std and choice of nominal-flow
    estimator (exact recursion vs. `particle_size` interacting particles,
    per Assumption "Access to the nominal population flow"). n_aux is always
    the reference value; B is `cfg.B` under "equal_parameters" or
    `cfg.simplex_B_equal_budget(T)` under "equal_budget" (horizon-dependent
    — see configs/twostate.py's module docstring), resolved per T since the
    step function only learns T at call time."""
    if flow == "exact":
        population_flow_fn = simplex.exact_population_flow
    elif flow == "particle":
        population_flow_fn = functools.partial(simplex.particle_population_flow, n_particles=cfg.particle_size)
    else:
        raise ValueError(f"unknown flow {flow!r}; available: {sorted(SUPPORTED_FLOWS)}")

    if budget_mode == "equal_parameters":
        batch_size = lambda T: cfg.B
    elif budget_mode == "equal_budget":
        batch_size = cfg.simplex_B_equal_budget
    else:
        raise ValueError(f"unknown budget_mode {budget_mode!r}; available: {sorted(SUPPORTED_BUDGET_MODES)}")

    def step(env, action_probs_fn, theta, mu0, *, T, lam, gamma=1.0, generator):
        return simplex.gradient_step(
            env,
            action_probs_fn,
            theta,
            mu0,
            T=T,
            n_aux=cfg.n_aux,
            B=batch_size(T),
            lam=lam,
            sigma=cfg.sigma,
            gamma=gamma,
            population_flow_fn=population_flow_fn,
            generator=generator,
        )

    return step


def make_reinforce_step(cfg, flow: str, budget_mode: str):
    """Adapts `reinforce.gradient_step` to the same `step_fn(env,
    action_probs_fn, theta, mu0, *, T, lam, generator) -> g_hat` contract as
    `make_simplex_step` (accepting but ignoring `lam`: REINFORCE has no
    perturbation scale). B is `cfg.B` under "equal_parameters" or
    `cfg.reinforce_B_equal_budget(T)` under "equal_budget" (no auxiliary
    batch, so REINFORCE's whole equal-budget allocation goes to B — see
    configs/twostate.py's module docstring), resolved per T."""
    if flow == "exact":
        population_flow_fn = _common.exact_population_flow
    elif flow == "particle":
        population_flow_fn = functools.partial(_common.particle_population_flow, n_particles=cfg.particle_size)
    else:
        raise ValueError(f"unknown flow {flow!r}; available: {sorted(SUPPORTED_FLOWS)}")

    if budget_mode == "equal_parameters":
        batch_size = lambda T: cfg.B
    elif budget_mode == "equal_budget":
        batch_size = cfg.reinforce_B_equal_budget
    else:
        raise ValueError(f"unknown budget_mode {budget_mode!r}; available: {sorted(SUPPORTED_BUDGET_MODES)}")

    def step(env, action_probs_fn, theta, mu0, *, T, lam, gamma=1.0, generator):
        del lam
        return reinforce.gradient_step(env, action_probs_fn, theta, mu0, T=T, B=batch_size(T), gamma=gamma, population_flow_fn=population_flow_fn, generator=generator)

    return step


def make_mfreinforce_step(cfg, flow: str, budget_mode: str):
    """Adapts `mfreinforce.gradient_step` to the same step_fn contract
    (accepting but ignoring `lam`: mfreinforce's perturbation scale is the
    single fixed `cfg.epsilon`, not swept like simplex's lambda grid).
    `budget_mode` is accepted for interface parity but unused: (n_aux, B)
    are always the reference values `cfg.n_aux`/`cfg.B` — mfreinforce is
    itself the equal-budget anchor (configs/twostate.py's module
    docstring), so there is nothing to rebalance."""
    if flow == "exact":
        population_flow_fn = _common.exact_population_flow
    elif flow == "particle":
        population_flow_fn = functools.partial(_common.particle_population_flow, n_particles=cfg.particle_size)
    else:
        raise ValueError(f"unknown flow {flow!r}; available: {sorted(SUPPORTED_FLOWS)}")
    if budget_mode not in SUPPORTED_BUDGET_MODES:
        raise ValueError(f"unknown budget_mode {budget_mode!r}; available: {sorted(SUPPORTED_BUDGET_MODES)}")

    def step(env, action_probs_fn, theta, mu0, *, T, lam, gamma=1.0, generator):
        del lam
        return mfreinforce.gradient_step(
            env, action_probs_fn, theta, mu0, T=T, n_aux=cfg.n_aux, B=cfg.B, epsilon=cfg.epsilon, gamma=gamma, population_flow_fn=population_flow_fn, generator=generator
        )

    return step


STEP_FACTORIES = {"simplex": make_simplex_step, "reinforce": make_reinforce_step, "mfreinforce": make_mfreinforce_step}

# J(theta;mu0), exact and environment/policy-agnostic (no trajectory noise);
# used for validation logging here and for diagnostics in scripts/test.py.
exact_objective = simplex.exact_objective


def experiment_tag(alg_name: str, budget_mode: str, flow: str, T: int, lam: float | None, seed: int, *, dtype: str = "float64") -> str:
    """The exact `<...>.pt` basename (sans extension) `run_all` saves a
    discrete-env job under, factored out so any other caller that needs to
    predict/parse it (e.g. `materialize_duplicates`'s primary/copy paths)
    can't drift from what `run_all` actually writes. `dtype` only appears
    when non-default (float64 keeps the original, pre-dtype-flag filenames,
    so existing `runs/` on disk stay valid) — without it, a float32 job
    would silently "already exist" against a float64 one saved at the same
    tag."""
    lam_tag = f"_lam{lam}" if lam is not None else ""
    dtype_tag = "" if dtype == "float64" else f"_dtype{dtype}"
    return f"{alg_name}_{budget_mode}_{flow}_T{T}{lam_tag}_seed{seed}{dtype_tag}"


def train_run(
    env,
    action_probs_fn,
    theta0,
    cfg,
    *,
    step_fn,
    env_name: str,
    alg_name: str,
    budget_mode: str,
    flow: str,
    T: int,
    lam: float | None,
    seed: int,
    progress_desc: str | None = None,
) -> dict:
    """Run one (T, lam, seed) — lam is None for algorithms with no
    perturbation scale (see ALGORITHMS_WITH_PERTURBATION_SCALE) — randomizing
    mu0 every step per the reference's training protocol (per-environment
    sampler, see SAMPLE_MU0_FACTORIES). Validates at `cfg.T_val` when the
    config sets it (cybersecurity: a longer horizon than training, per its
    "Training and validation protocol"), else at the same `T` used for
    training (twostate). Discounts by `env.config.gamma` when the
    environment defines one (else undiscounted, gamma=1.0). Returns
    everything needed for downstream diagnostics: the theta trajectory
    (subsampled to MAX_THETA_SNAPSHOTS CPU-resident snapshots, indexed by
    `theta_history_iterations`) and validation history.

    `progress_desc` labels a live progress bar over the training iterations,
    annotated with the most recent validation objective; None (the default)
    disables it. Callers resolve whether to pass one via
    `mfc.progress.progress_enabled` — see that module for why it is a
    decision and not just an isatty check."""
    dtype, device = theta0.dtype, theta0.device
    generator = torch.Generator(device=device).manual_seed(seed)
    gamma = getattr(env.config, "gamma", 1.0)
    T_val = getattr(cfg, "T_val", None) or T

    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cfg.lr)
    mu0_val = torch.tensor(cfg.mu0_val, dtype=dtype, device=device)
    sample_mu0 = SAMPLE_MU0_FACTORIES[env_name](cfg, env)

    history_stride = max(1, -(-cfg.n_train // MAX_THETA_SNAPSHOTS))  # ceil, so at most MAX_THETA_SNAPSHOTS+1 rows
    theta_history = [theta.detach().to("cpu", copy=True)]
    history_iterations = [0]
    val_iterations, val_J = [], []

    start = time.perf_counter()
    with training_bar(cfg.n_train, desc=progress_desc) as bar:
        for m in bar:
            g_hat = step_fn(env, action_probs_fn, theta, sample_mu0(generator), T=T, lam=lam, gamma=gamma, generator=generator)
            optimizer.zero_grad()
            theta.grad = -g_hat
            optimizer.step()

            if (m + 1) % history_stride == 0 or m == cfg.n_train - 1:
                theta_history.append(theta.detach().to("cpu", copy=True))
                history_iterations.append(m + 1)

            if m % cfg.validate_every == 0 or m == cfg.n_train - 1:
                val_iterations.append(m)
                val_J.append(exact_objective(env, action_probs_fn, theta, mu0_val, T_val, gamma=gamma).item())
                bar.set_postfix_str(f"J={val_J[-1]:.4f}", refresh=False)
    elapsed = time.perf_counter() - start

    return {
        "env": env_name,
        "alg": alg_name,
        "budget_mode": budget_mode,
        "flow": flow,
        "T": T,
        "lam": lam,
        "seed": seed,
        "dtype": str(dtype),
        "config": cfg,
        "theta0": theta0,
        "theta_final": theta.detach().clone(),
        "theta_history": torch.stack(theta_history),
        "theta_history_iterations": torch.tensor(history_iterations),
        "validation_iterations": torch.tensor(val_iterations),
        "validation_J": torch.tensor(val_J, dtype=dtype),
        "elapsed_seconds": elapsed,
    }


# mfreinforce's step ignores budget_mode entirely (fixed n_aux/B — see
# configs/twostate.py's module docstring) and its only randomness is seeded
# by `seed` alone, so training the same (flow, T, lam, seed) again under a
# different budget_mode reproduces a byte-identical result — wasted compute
# on a paid run. `list_experiments` drops these as `duplicates` instead of
# `experiments` (real training jobs); `run_all`/`--materialize-duplicates`
# copy each one from its primary after the primary has actually trained.
ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE = {"mfreinforce"}


def list_experiments(
    cfg,
    alg_name: str,
    *,
    budget_mode: str | None = None,
    flow: str | None = None,
    T: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
) -> tuple[list[tuple[str, str, int, float | None, int]], list[tuple[str, str, int, float | None, int, str]], list[str], list[str]]:
    """
    The full experiment grid for (cfg, alg_name), split into `experiments`
    (jobs that need real training) and `duplicates` (budget-mode-invariant
    jobs that only need a copy of another job's result — see
    ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE); together these are the
    single source of truth `run_all` acts on and `--list`/
    `--materialize-duplicates` print/materialize. Each `experiments` entry
    is (budget_mode, flow, T, lam, seed); each `duplicates` entry is
    (budget_mode, flow, T, lam, seed, primary_budget_mode) — the budget_mode
    to copy the (flow, T, lam, seed, primary_budget_mode) job's result into.

    Any axis given here is pinned to that one value instead of swept (not
    required to be a member of the config's own grid — e.g. `seed=` an
    extra seed beyond `cfg.seeds`); every omitted axis sweeps its full
    config range, so all-omitted reproduces the original whole-config
    behavior. If `cfg.groups` is set (an explicit tuple of (budget_mode,
    flow, T) triples — see configs/twostate.py), it replaces the
    budget_modes x flows x horizons cartesian product entirely, filtered by
    any given `budget_mode`/`flow`/`T` override; `cfg.budget_modes`/
    `cfg.flows`/`cfg.horizons` are then not read at all. Also returns the
    budget_modes/flows the config asked for but this repo doesn't implement
    (only meaningful without `cfg.groups` and when `budget_mode`/`flow`
    themselves are omitted, since a given override or an explicit group is
    trusted as-is).
    """
    seeds = [seed] if seed is not None else list(cfg.seeds)
    if alg_name in ALGORITHMS_WITH_PERTURBATION_SCALE:
        lambdas = [lam] if lam is not None else list(cfg.lambdas)
    else:
        if lam is not None:
            raise ValueError(f"--lam given but algorithm {alg_name!r} has no perturbation scale to sweep")
        lambdas = [None]

    groups = getattr(cfg, "groups", None)
    if groups is not None:
        skipped_budget, skipped_flows = [], []
        bmf_triples = [g for g in groups if (budget_mode is None or g[0] == budget_mode) and (flow is None or g[1] == flow) and (T is None or g[2] == T)]
    else:
        budget_modes = [budget_mode] if budget_mode is not None else sorted(set(cfg.budget_modes) & SUPPORTED_BUDGET_MODES)
        skipped_budget = [] if budget_mode is not None else sorted(set(cfg.budget_modes) - SUPPORTED_BUDGET_MODES)
        flows = [flow] if flow is not None else sorted(set(cfg.flows) & SUPPORTED_FLOWS)
        skipped_flows = [] if flow is not None else sorted(set(cfg.flows) - SUPPORTED_FLOWS)
        horizons = [T] if T is not None else list(cfg.horizons)
        bmf_triples = [(bm, fl, T_i) for bm in budget_modes for fl in flows for T_i in horizons]

    all_jobs = [(bm, fl, T_i, lam_i, s) for (bm, fl, T_i) in bmf_triples for lam_i in lambdas for s in seeds]

    if alg_name not in ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE:
        return all_jobs, [], skipped_budget, skipped_flows

    experiments, duplicates = [], []
    primary_budget_mode: dict[tuple[str, int, float | None, int], str] = {}
    for bm, fl, T_i, lam_i, s in all_jobs:
        key = (fl, T_i, lam_i, s)
        if key not in primary_budget_mode:
            primary_budget_mode[key] = bm
            experiments.append((bm, fl, T_i, lam_i, s))
        else:
            duplicates.append((bm, fl, T_i, lam_i, s, primary_budget_mode[key]))
    return experiments, duplicates, skipped_budget, skipped_flows


def materialize_duplicates(
    env_name: str,
    alg_name: str,
    config_name: str,
    *,
    output_dir: str = str(ROOT / "runs"),
    dtype: str = "float64",
    budget_mode: str | None = None,
    flow: str | None = None,
    T: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Copy each `list_experiments`-reported duplicate job's result from its
    primary (see ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE), relabeling the
    copy's own `budget_mode` field so `scripts.test`'s `group_by` still
    groups it correctly. A primary that hasn't finished training yet is
    reported and skipped (rerun this after the main `run_all`/`--list`
    training pass has actually completed — `train_all.sh` does both, in
    that order). Returns the list of newly-copied result dicts (skips a
    duplicate whose output already exists unless `overwrite=True`, same
    resumability contract as `run_all`)."""
    cfg = CONFIGS[env_name][config_name]
    _, duplicates, _, _ = list_experiments(cfg, alg_name, budget_mode=budget_mode, flow=flow, T=T, lam=lam, seed=seed)
    out_root = Path(output_dir) / env_name / config_name

    copied = []
    for budget_mode_i, flow_i, T_i, lam_i, seed_i, primary_budget_mode in duplicates:
        dup_tag = experiment_tag(alg_name, budget_mode_i, flow_i, T_i, lam_i, seed_i, dtype=dtype)
        dup_path = out_root / f"{dup_tag}.pt"
        if dup_path.exists() and not overwrite:
            print(f"[{env_name}/{config_name}] {dup_tag} ... skipped (already exists)")
            continue

        primary_tag = experiment_tag(alg_name, primary_budget_mode, flow_i, T_i, lam_i, seed_i, dtype=dtype)
        primary_path = out_root / f"{primary_tag}.pt"
        if not primary_path.exists():
            print(f"[{env_name}/{config_name}] {dup_tag} ... skipped (primary {primary_tag} not trained yet)")
            continue

        # Deliberately no map_location: the copy must match its primary in
        # every respect but `budget_mode`, device included, so that the two
        # tags are interchangeable downstream. (Cheap regardless — `train_run`
        # already keeps the bulky `theta_history` on the CPU.)
        result = dict(torch.load(primary_path, weights_only=False), budget_mode=budget_mode_i)
        torch.save(result, dup_path)
        print(f"[{env_name}/{config_name}] {dup_tag} ... copied from {primary_tag}")
        copied.append(result)
    return copied


def run_all(
    env_name: str,
    alg_name: str,
    config_name: str,
    *,
    output_dir: str = str(ROOT / "runs"),
    dtype: str = "float64",
    budget_mode: str | None = None,
    flow: str | None = None,
    T: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    progress: str = "auto",
) -> list[dict]:
    """Train every job `list_experiments` returns for this (env, alg,
    config[, axis overrides]). A job whose output `.pt` already exists is
    skipped unless `overwrite=True` — the resumability a long/parallel/
    paid run needs (`train_all.sh` relies on this, not its own bookkeeping).
    Budget-mode-invariant duplicates (see
    ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE) are materialized as copies
    afterward, in-process — `materialize_duplicates` only needs to be called
    separately when jobs were trained out-of-process (`--list` + a parallel
    launcher, e.g. `train_all.sh`). `progress` is one of
    `mfc.progress.PROGRESS_MODES` and controls the per-run progress bar."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(f"unknown env {env_name!r}; available: {sorted(ENVIRONMENTS)}")
    if alg_name not in STEP_FACTORIES:
        raise ValueError(f"algorithm {alg_name!r} not implemented yet; available: {sorted(STEP_FACTORIES)}")
    cfg = CONFIGS[env_name][config_name]

    experiments, duplicates, skipped_budget, skipped_flows = list_experiments(cfg, alg_name, budget_mode=budget_mode, flow=flow, T=T, lam=lam, seed=seed)
    if skipped_budget:
        print(f"note: budget_modes {skipped_budget} not implemented yet, skipping")
    if skipped_flows:
        print(f"note: flows {skipped_flows} not implemented yet, skipping")
    if not experiments and not duplicates:
        print("nothing to run: no supported budget_mode/flow combination in this config")
        return []

    dtype_t = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    env = ENVIRONMENTS[env_name](dtype=dtype_t)
    theta0 = env.init_theta(generator=torch.Generator(device=env.device).manual_seed(INIT_SEED))
    action_probs_fn = env.policy_probs

    out_root = Path(output_dir) / env_name / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    show_progress = progress_enabled(progress)
    step_fns: dict[tuple[str, str], object] = {}
    results = []
    for budget_mode_i, flow_i, T_i, lam_i, seed_i in experiments:
        tag = experiment_tag(alg_name, budget_mode_i, flow_i, T_i, lam_i, seed_i, dtype=dtype)
        out_path = out_root / f"{tag}.pt"
        if out_path.exists() and not overwrite:
            print(f"[{env_name}/{config_name}] {tag} ... skipped (already exists)")
            continue

        step_key = (budget_mode_i, flow_i)
        if step_key not in step_fns:
            step_fns[step_key] = STEP_FACTORIES[alg_name](cfg, flow_i, budget_mode_i)

        # Every status line is self-contained (`<label> ... <status>`, as for
        # "skipped" above) rather than a dangling "<label> ... " prefix that a
        # later write completes: a bar would redraw over such a prefix, and
        # under train_all.sh's concurrent workers the prefixes and their
        # summaries interleave across processes into unreadable output. With a
        # bar there is no need to announce the start at all — the bar is the
        # announcement — so that line is only for the bar-less path.
        label = f"[{env_name}/{config_name}] {tag}"
        if not show_progress:
            print(f"{label} ... started", flush=True)
        result = train_run(
            env,
            action_probs_fn,
            theta0,
            cfg,
            step_fn=step_fns[step_key],
            env_name=env_name,
            alg_name=alg_name,
            budget_mode=budget_mode_i,
            flow=flow_i,
            T=T_i,
            lam=lam_i,
            seed=seed_i,
            progress_desc=tag if show_progress else None,
        )
        print(f"{label} ... done in {result['elapsed_seconds']:.1f}s, final validation J={result['validation_J'][-1].item():.4f}")
        torch.save(result, out_path)
        results.append(result)

    if duplicates:
        results += materialize_duplicates(
            env_name, alg_name, config_name, output_dir=output_dir, dtype=dtype,
            budget_mode=budget_mode, flow=flow, T=T, lam=lam, seed=seed, overwrite=overwrite,
        )
    return results


def list_continuous_experiments(
    cfg, alg_name: str, *, T: int | None = None, lam: float | None = None, seed: int | None = None
) -> list[tuple[int, float | None, int]]:
    """The full experiment grid for a CONTINUOUS_ENVIRONMENTS config, as
    (T, lam, seed) tuples, mirroring `list_experiments`'s override semantics
    (any axis given here is pinned; every omitted axis sweeps its full config
    range) and its treatment of the perturbation scale: `lam` is None for an
    algorithm outside ALGORITHMS_WITH_PERTURBATION_SCALE, so reinforce gets
    one job per (T, seed) rather than one per lambda.

    No budget_mode/flow axis: the continuous configs have a single,
    always-equal-budget allocation (`cfg.n_aux`/`cfg.B` for simplex,
    `cfg.reinforce_B_equal_budget()` for reinforce — see `configs/lq.py`),
    and the population coordinate flow is exact for both benchmarks, so
    there is no exact-vs-particle flow choice either."""
    horizons = [T] if T is not None else list(cfg.horizons)
    seeds = [seed] if seed is not None else list(cfg.seeds)
    if alg_name in ALGORITHMS_WITH_PERTURBATION_SCALE:
        lambdas = [lam] if lam is not None else list(cfg.lambdas)
    else:
        if lam is not None:
            raise ValueError(f"--lam given but algorithm {alg_name!r} has no perturbation scale to sweep")
        lambdas = [None]
    return [(T_i, lam_i, s) for T_i in horizons for lam_i in lambdas for s in seeds]


def run_continuous(
    env_name: str,
    alg_name: str,
    config_name: str,
    *,
    output_dir: str = str(ROOT / "runs"),
    dtype: str = "float64",
    T: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    progress: str = "auto",
) -> list[dict]:
    """
    The runner shared by every CONTINUOUS_ENVIRONMENTS entry (`lq`,
    `portfolio`) — parallel to `run_all`/`train_run` but driving
    `mfc.algorithms.continuous.train` instead of the discrete
    STEP_FACTORIES machinery (see `mfc.environments.lq`'s module docstring
    for why: continuous Gaussian/non-Gaussian scalar state, no
    `n_states`/`action_probs_fn`). The per-algorithm sampling budget comes
    from the config and is equal-budget by construction: simplex spends
    (`cfg.n_aux`+`cfg.B`)*T transitions per step and reinforce the same
    total in one main batch, `cfg.reinforce_B_equal_budget()`*T (context.md:
    "For all subsequent environments and tests, we must use equal budget").
    Trained on CPU regardless of CUDA
    availability — their per-step work is all small scalar/(T,B) ops, for
    which GPU kernel-launch overhead dominates (this repo's twostate
    profiling found the same: GPU ~2.3x slower than CPU for tiny-tensor
    workloads), confirmed directly while tuning `configs/lq.py`.
    `T`/`lam`/`seed` pin one axis each to a single value (as in `run_all`);
    a job whose output `.pt` already exists is skipped unless `overwrite=True`.
    `progress` is one of `mfc.progress.PROGRESS_MODES` and controls the
    per-run progress bar, as in `run_all`.
    """
    if env_name not in CONTINUOUS_ENVIRONMENTS:
        raise ValueError(f"unknown continuous env {env_name!r}; available: {sorted(CONTINUOUS_ENVIRONMENTS)}")
    if alg_name not in continuous.ALGORITHMS:
        raise ValueError(f"unknown {env_name} algorithm {alg_name!r}; available: {', '.join(continuous.ALGORITHMS)}")
    env_cls = CONTINUOUS_ENVIRONMENTS[env_name]
    cfg = CONFIGS[env_name][config_name]
    budget = {
        "simplex": {"B": cfg.B, "n_aux": cfg.n_aux},
        "reinforce": {"B": cfg.reinforce_B_equal_budget()},
        "exact_gradient": {},  # closed-form: no sampling at all
    }[alg_name]
    dtype_t = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    env = env_cls(device="cpu", dtype=dtype_t)
    out_root = Path(output_dir) / env_name / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    dtype_tag = "" if dtype == "float64" else f"_dtype{dtype}"  # see run_all's identical dtype_tag: without this, a float32 job
    # would silently no-op as "already exists" against a float64 one saved at the same tag
    show_progress = progress_enabled(progress)
    results = []
    for T_i, lam_i, seed_i in list_continuous_experiments(cfg, alg_name, T=T, lam=lam, seed=seed):
        # lam_i is None for an algorithm with no perturbation scale (reinforce): it is
        # trained on the nominal process, and its tag carries no lambda -- same convention
        # as `experiment_tag` for the discrete benchmarks.
        tag = f"{alg_name}_T{T_i}{'' if lam_i is None else f'_lam{lam_i}'}_seed{seed_i}{dtype_tag}"
        out_path = out_root / f"{tag}.pt"
        if out_path.exists() and not overwrite:
            print(f"[{env_name}/{config_name}] {tag} ... skipped (already exists)")
            continue

        theta0 = env.init_theta(T=T_i)
        label = f"[{env_name}/{config_name}] {tag}"
        if not show_progress:  # see run_all: self-contained status lines, the bar replaces the start announcement
            print(f"{label} ... started", flush=True)
        generator = torch.Generator(device="cpu").manual_seed(seed_i)
        result = continuous.train(
            env, theta0, algorithm=alg_name, n_train=cfg.n_train, lr=cfg.lr, lam=lam_i if lam_i is not None else 0.0,
            baseline=cfg.baseline, validate_every=cfg.validate_every, generator=generator,
            progress_desc=tag if show_progress else None, **budget,
        )
        result.update({"env": env_name, "alg": alg_name, "T": T_i, "lam": lam_i, "seed": seed_i, "dtype": dtype, "config": cfg, "theta0": theta0})
        print(f"{label} ... done in {result['elapsed_seconds']:.1f}s, final validation J={result['validation_J'][-1].item():.4f}")
        torch.save(result, out_path)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="twostate", choices=sorted(set(ENVIRONMENTS) | set(CONTINUOUS_ENVIRONMENTS)))
    parser.add_argument("--alg", default="simplex", choices=sorted(set(STEP_FACTORIES) | set(continuous.ALGORITHMS)))
    parser.add_argument("--config", default="smoke", choices=["main", "mid", "smoke"])
    parser.add_argument("--output-dir", default=str(ROOT / "runs"))
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--budget-mode", default=None, choices=sorted(SUPPORTED_BUDGET_MODES), help="discrete envs only; pins one axis instead of sweeping the config's full grid")
    parser.add_argument("--flow", default=None, choices=sorted(SUPPORTED_FLOWS), help="discrete envs only; pins one axis instead of sweeping the config's full grid")
    parser.add_argument("--T", type=int, default=None, help="pins the horizon instead of sweeping the config's full grid")
    parser.add_argument("--lam", type=float, default=None, help="pins the perturbation scale instead of sweeping the config's full grid")
    parser.add_argument("--seed", type=int, default=None, help="pins the seed instead of sweeping the config's full grid")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a job's output .pt already exists (default: skip it)")
    parser.add_argument(
        "--progress", default="auto", choices=list(PROGRESS_MODES),
        help="per-run progress bar over training iterations: 'auto' (a terminal or notebook, but not a log file), or force 'on'/'off'. "
             "Use 'off' when several jobs share one terminal — train_all.sh passes it for you (see mfc.progress)",
    )
    parser.add_argument("--list", action="store_true", help="print every job needing real training as a runnable `uv run python scripts/train.py ...` command, one per line, instead of training (see train_all.sh)")
    parser.add_argument(
        "--materialize-duplicates", action="store_true",
        help="copy budget-mode-invariant duplicate jobs (see ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE) from their already-trained primary, instead of training; run this after a --list-driven parallel launcher finishes (train_all.sh does)",
    )
    args = parser.parse_args()

    default_output_dir = str(ROOT / "runs")
    output_dir_flag = "" if args.output_dir == default_output_dir else f" --output-dir {args.output_dir}"

    if args.materialize_duplicates:
        if args.env in CONTINUOUS_ENVIRONMENTS:
            return  # no budget_mode axis; nothing to materialize
        materialize_duplicates(
            args.env, args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype,
            budget_mode=args.budget_mode, flow=args.flow, T=args.T, lam=args.lam, seed=args.seed, overwrite=args.overwrite,
        )
        return

    if args.list:
        cfg = CONFIGS[args.env][args.config]
        if args.env in CONTINUOUS_ENVIRONMENTS:
            experiments = [(None, None, T_i, lam_i, seed_i) for T_i, lam_i, seed_i in list_continuous_experiments(cfg, args.alg, T=args.T, lam=args.lam, seed=args.seed)]
        else:
            experiments, _duplicates, skipped_budget, skipped_flows = list_experiments(cfg, args.alg, budget_mode=args.budget_mode, flow=args.flow, T=args.T, lam=args.lam, seed=args.seed)
            if skipped_budget:
                print(f"# note: budget_modes {skipped_budget} not implemented yet, skipping", file=sys.stderr)
            if skipped_flows:
                print(f"# note: flows {skipped_flows} not implemented yet, skipping", file=sys.stderr)
        for budget_mode_i, flow_i, T_i, lam_i, seed_i in experiments:
            axis_flags = f" --T {T_i} --seed {seed_i}"
            if budget_mode_i is not None:
                axis_flags += f" --budget-mode {budget_mode_i} --flow {flow_i}"
            if lam_i is not None:
                axis_flags += f" --lam {lam_i}"
            print(f"uv run python scripts/train.py --env {args.env} --alg {args.alg} --config {args.config} --dtype {args.dtype}{axis_flags}{output_dir_flag}")
        return

    if args.env in CONTINUOUS_ENVIRONMENTS:
        run_continuous(
            args.env, args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype,
            T=args.T, lam=args.lam, seed=args.seed, overwrite=args.overwrite, progress=args.progress,
        )
        return
    run_all(
        args.env, args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype,
        budget_mode=args.budget_mode, flow=args.flow, T=args.T, lam=args.lam, seed=args.seed,
        overwrite=args.overwrite, progress=args.progress,
    )


if __name__ == "__main__":
    main()
