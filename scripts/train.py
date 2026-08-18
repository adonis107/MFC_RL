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

All three algorithms are implemented, for both flows and both budget modes.
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
from configs.twostate import MAIN as TWOSTATE_MAIN
from configs.twostate import MID as TWOSTATE_MID
from configs.twostate import SMOKE as TWOSTATE_SMOKE
from mfc.algorithms import _common, lq as lq_algorithms, mfreinforce, reinforce, simplex
from mfc.environments.advertising import Advertising
from mfc.environments.cybersecurity import CyberSecurity
from mfc.environments.distribution_planning import DistributionPlanning
from mfc.environments.lq import LQ
from mfc.environments.twostate import TwoState

ENVIRONMENTS = {"twostate": TwoState, "cybersecurity": CyberSecurity, "distribution_planning": DistributionPlanning, "advertising": Advertising}
CONFIGS = {
    "twostate": {"main": TWOSTATE_MAIN, "mid": TWOSTATE_MID, "smoke": TWOSTATE_SMOKE},
    "cybersecurity": {"main": CYBERSECURITY_MAIN, "mid": CYBERSECURITY_MID, "smoke": CYBERSECURITY_SMOKE},
    "distribution_planning": {"main": DISTRIBUTION_PLANNING_MAIN, "mid": DISTRIBUTION_PLANNING_MID, "smoke": DISTRIBUTION_PLANNING_SMOKE},
    "advertising": {"main": ADVERTISING_MAIN, "mid": ADVERTISING_MID, "smoke": ADVERTISING_SMOKE},
    "lq": {"main": LQ_MAIN, "mid": LQ_MID, "smoke": LQ_SMOKE},
}

SUPPORTED_BUDGET_MODES = {"equal_parameters", "equal_budget"}
SUPPORTED_FLOWS = {"exact", "particle"}
ALGORITHMS_WITH_PERTURBATION_SCALE = {"simplex"}  # reinforce has no lambda; sweeping it would just retrain redundantly

# Environment-specific per-iteration initial-law sampling ("Training
# protocol" for each benchmark): two-state resamples mu0(1)~U([mu0_low,
# mu0_high]) from RunConfig-level bounds; cybersecurity, distribution_planning,
# and advertising each expose their own `env.sample_mu0` (a model constant,
# not RunConfig-parametrized — cybersecurity/distribution_planning sample
# Dirichlet(1,...,1), advertising samples p0~U([0.05,0.95]), but the
# call site is identical for all three). Each factory closes over (cfg, env)
# and returns `sample(generator) -> mu0`.
INIT_SEED = 0  # fixed, reused across all training seeds within a run, so paired runs share one policy initialization


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
) -> dict:
    """Run one (T, lam, seed) — lam is None for algorithms with no
    perturbation scale (see ALGORITHMS_WITH_PERTURBATION_SCALE) — randomizing
    mu0 every step per the reference's training protocol (per-environment
    sampler, see SAMPLE_MU0_FACTORIES). Validates at `cfg.T_val` when the
    config sets it (cybersecurity: a longer horizon than training, per its
    "Training and validation protocol"), else at the same `T` used for
    training (twostate). Discounts by `env.config.gamma` when the
    environment defines one (else undiscounted, gamma=1.0). Returns
    everything needed for downstream diagnostics: full theta trajectory and
    validation history."""
    dtype, device = theta0.dtype, theta0.device
    generator = torch.Generator(device=device).manual_seed(seed)
    gamma = getattr(env.config, "gamma", 1.0)
    T_val = getattr(cfg, "T_val", None) or T

    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cfg.lr)
    mu0_val = torch.tensor(cfg.mu0_val, dtype=dtype, device=device)
    sample_mu0 = SAMPLE_MU0_FACTORIES[env_name](cfg, env)

    theta_history = [theta.detach().clone()]
    val_iterations, val_J = [], []

    start = time.perf_counter()
    for m in range(cfg.n_train):
        g_hat = step_fn(env, action_probs_fn, theta, sample_mu0(generator), T=T, lam=lam, gamma=gamma, generator=generator)
        optimizer.zero_grad()
        theta.grad = -g_hat
        optimizer.step()
        theta_history.append(theta.detach().clone())

        if m % cfg.validate_every == 0 or m == cfg.n_train - 1:
            val_iterations.append(m)
            val_J.append(exact_objective(env, action_probs_fn, theta, mu0_val, T_val, gamma=gamma).item())
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
) -> list[dict]:
    """Train every job `list_experiments` returns for this (env, alg,
    config[, axis overrides]). A job whose output `.pt` already exists is
    skipped unless `overwrite=True` — the resumability a long/parallel/
    paid run needs (`train_all.sh` relies on this, not its own bookkeeping).
    Budget-mode-invariant duplicates (see
    ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE) are materialized as copies
    afterward, in-process — `materialize_duplicates` only needs to be called
    separately when jobs were trained out-of-process (`--list` + a parallel
    launcher, e.g. `train_all.sh`)."""
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

        print(f"[{env_name}/{config_name}] {tag} ...", end=" ", flush=True)
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
        )
        print(f"done in {result['elapsed_seconds']:.1f}s, final validation J={result['validation_J'][-1].item():.4f}")
        torch.save(result, out_path)
        results.append(result)

    if duplicates:
        results += materialize_duplicates(
            env_name, alg_name, config_name, output_dir=output_dir, dtype=dtype,
            budget_mode=budget_mode, flow=flow, T=T, lam=lam, seed=seed, overwrite=overwrite,
        )
    return results


def list_lq_experiments(cfg, *, T: int | None = None, lam: float | None = None, seed: int | None = None) -> list[tuple[int, float, int]]:
    """The full LQ experiment grid as (T, lam, seed) tuples, mirroring
    `list_experiments`'s override semantics (any axis given here is pinned;
    every omitted axis sweeps its full config range)."""
    horizons = [T] if T is not None else list(cfg.horizons)
    lambdas = [lam] if lam is not None else list(cfg.lambdas)
    seeds = [seed] if seed is not None else list(cfg.seeds)
    return [(T_i, lam_i, s) for T_i in horizons for lam_i in lambdas for s in seeds]


def run_lq(
    alg_name: str,
    config_name: str,
    *,
    output_dir: str = str(ROOT / "runs"),
    dtype: str = "float64",
    T: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """
    LQ's own runner, parallel to `run_all`/`train_run` but for
    `mfc.algorithms.lq.train`'s closed-form/Monte-Carlo training instead of
    the discrete STEP_FACTORIES machinery (see `mfc.environments.lq`'s
    module docstring for why LQ cannot share that machinery: continuous
    Gaussian state/policy, no `n_states`/`action_probs_fn`). Trained on CPU
    regardless of CUDA availability — LQ's per-step work is all small
    scalar/(T,B) ops, for which GPU kernel-launch overhead dominates (this
    repo's twostate profiling found the same: GPU ~2.3x slower than CPU for
    tiny-tensor workloads), confirmed directly while tuning `configs/lq.py`.
    `T`/`lam`/`seed` pin one axis each to a single value (as in `run_all`);
    a job whose output `.pt` already exists is skipped unless `overwrite=True`.
    """
    if alg_name not in ("exact_gradient", "reinforce"):
        raise ValueError(f"unknown LQ algorithm {alg_name!r}; available: exact_gradient, reinforce")
    cfg = CONFIGS["lq"][config_name]
    dtype_t = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    env = LQ(device="cpu", dtype=dtype_t)
    out_root = Path(output_dir) / "lq" / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    dtype_tag = "" if dtype == "float64" else f"_dtype{dtype}"  # see run_all's identical dtype_tag: without this, a float32 job
    # would silently no-op as "already exists" against a float64 one saved at the same tag
    results = []
    for T_i, lam_i, seed_i in list_lq_experiments(cfg, T=T, lam=lam, seed=seed):
        tag = f"{alg_name}_T{T_i}_lam{lam_i}_seed{seed_i}{dtype_tag}"
        out_path = out_root / f"{tag}.pt"
        if out_path.exists() and not overwrite:
            print(f"[lq/{config_name}] {tag} ... skipped (already exists)")
            continue

        theta0 = env.init_theta(T=T_i)
        print(f"[lq/{config_name}] {tag} ...", end=" ", flush=True)
        generator = torch.Generator(device="cpu").manual_seed(seed_i)
        result = lq_algorithms.train(
            env, theta0, algorithm=alg_name, n_train=cfg.n_train, lr=cfg.lr, lam=lam_i,
            B=cfg.B if alg_name == "reinforce" else None, validate_every=cfg.validate_every, generator=generator,
        )
        result.update({"env": "lq", "alg": alg_name, "T": T_i, "lam": lam_i, "seed": seed_i, "dtype": dtype, "config": cfg, "theta0": theta0})
        print(f"done in {result['elapsed_seconds']:.1f}s, final validation J={result['validation_J'][-1].item():.4f}")
        torch.save(result, out_path)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="twostate", choices=sorted(set(ENVIRONMENTS) | {"lq"}))
    parser.add_argument("--alg", default="simplex", choices=sorted(set(STEP_FACTORIES) | {"exact_gradient"}))
    parser.add_argument("--config", default="smoke", choices=["main", "mid", "smoke"])
    parser.add_argument("--output-dir", default=str(ROOT / "runs"))
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--budget-mode", default=None, choices=sorted(SUPPORTED_BUDGET_MODES), help="discrete envs only; pins one axis instead of sweeping the config's full grid")
    parser.add_argument("--flow", default=None, choices=sorted(SUPPORTED_FLOWS), help="discrete envs only; pins one axis instead of sweeping the config's full grid")
    parser.add_argument("--T", type=int, default=None, help="pins the horizon instead of sweeping the config's full grid")
    parser.add_argument("--lam", type=float, default=None, help="pins the perturbation scale instead of sweeping the config's full grid")
    parser.add_argument("--seed", type=int, default=None, help="pins the seed instead of sweeping the config's full grid")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a job's output .pt already exists (default: skip it)")
    parser.add_argument("--list", action="store_true", help="print every job needing real training as a runnable `uv run python scripts/train.py ...` command, one per line, instead of training (see train_all.sh)")
    parser.add_argument(
        "--materialize-duplicates", action="store_true",
        help="copy budget-mode-invariant duplicate jobs (see ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE) from their already-trained primary, instead of training; run this after a --list-driven parallel launcher finishes (train_all.sh does)",
    )
    args = parser.parse_args()

    default_output_dir = str(ROOT / "runs")
    output_dir_flag = "" if args.output_dir == default_output_dir else f" --output-dir {args.output_dir}"

    if args.materialize_duplicates:
        if args.env == "lq":
            return  # LQ has no budget_mode axis; nothing to materialize
        materialize_duplicates(
            args.env, args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype,
            budget_mode=args.budget_mode, flow=args.flow, T=args.T, lam=args.lam, seed=args.seed, overwrite=args.overwrite,
        )
        return

    if args.list:
        cfg = CONFIGS[args.env][args.config]
        if args.env == "lq":
            experiments = [(None, None, T_i, lam_i, seed_i) for T_i, lam_i, seed_i in list_lq_experiments(cfg, T=args.T, lam=args.lam, seed=args.seed)]
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

    if args.env == "lq":
        run_lq(args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype, T=args.T, lam=args.lam, seed=args.seed, overwrite=args.overwrite)
        return
    run_all(
        args.env, args.alg, args.config, output_dir=args.output_dir, dtype=args.dtype,
        budget_mode=args.budget_mode, flow=args.flow, T=args.T, lam=args.lam, seed=args.seed, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
