"""
Train (env, alg) over a run config (main/mid/smoke) and save everything.

Usage:
    uv run python scripts/train.py --env twostate --alg simplex --config smoke

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

from configs.twostate import MAIN as TWOSTATE_MAIN
from configs.twostate import MID as TWOSTATE_MID
from configs.twostate import SMOKE as TWOSTATE_SMOKE
from mfc.algorithms import _common, mfreinforce, reinforce, simplex
from mfc.environments.twostate import TwoState

ENVIRONMENTS = {"twostate": TwoState}
CONFIGS = {"twostate": {"main": TWOSTATE_MAIN, "mid": TWOSTATE_MID, "smoke": TWOSTATE_SMOKE}}

SUPPORTED_BUDGET_MODES = {"equal_parameters", "equal_budget"}
SUPPORTED_FLOWS = {"exact", "particle"}
ALGORITHMS_WITH_PERTURBATION_SCALE = {"simplex"}  # reinforce has no lambda; sweeping it would just retrain redundantly


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

    def step(env, action_probs_fn, theta, mu0, *, T, lam, generator):
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

    def step(env, action_probs_fn, theta, mu0, *, T, lam, generator):
        del lam
        return reinforce.gradient_step(env, action_probs_fn, theta, mu0, T=T, B=batch_size(T), population_flow_fn=population_flow_fn, generator=generator)

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

    def step(env, action_probs_fn, theta, mu0, *, T, lam, generator):
        del lam
        return mfreinforce.gradient_step(
            env, action_probs_fn, theta, mu0, T=T, n_aux=cfg.n_aux, B=cfg.B, epsilon=cfg.epsilon, population_flow_fn=population_flow_fn, generator=generator
        )

    return step


STEP_FACTORIES = {"simplex": make_simplex_step, "reinforce": make_reinforce_step, "mfreinforce": make_mfreinforce_step}

# J(theta;mu0), exact and environment/policy-agnostic (no trajectory noise);
# used for validation logging here and for diagnostics in scripts/test.py.
exact_objective = simplex.exact_objective


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
    mu0 every step per the reference's training protocol. Returns everything
    needed for downstream diagnostics: full theta trajectory and validation
    history."""
    dtype, device = theta0.dtype, theta0.device
    generator = torch.Generator(device=device).manual_seed(seed)

    theta = theta0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cfg.lr)
    mu0_val = torch.tensor(cfg.mu0_val, dtype=dtype, device=device)

    theta_history = [theta.detach().clone()]
    val_iterations, val_J = [], []

    def sample_mu0():
        m1 = torch.rand((), dtype=dtype, device=device, generator=generator) * (cfg.mu0_high - cfg.mu0_low) + cfg.mu0_low
        return torch.stack([1.0 - m1, m1])

    start = time.perf_counter()
    for m in range(cfg.n_train):
        g_hat = step_fn(env, action_probs_fn, theta, sample_mu0(), T=T, lam=lam, generator=generator)
        optimizer.zero_grad()
        theta.grad = -g_hat
        optimizer.step()
        theta_history.append(theta.detach().clone())

        if m % cfg.validate_every == 0 or m == cfg.n_train - 1:
            val_iterations.append(m)
            val_J.append(exact_objective(env, action_probs_fn, theta, mu0_val, T).item())
    elapsed = time.perf_counter() - start

    return {
        "env": env_name,
        "alg": alg_name,
        "budget_mode": budget_mode,
        "flow": flow,
        "T": T,
        "lam": lam,
        "seed": seed,
        "config": cfg,
        "theta0": theta0,
        "theta_final": theta.detach().clone(),
        "theta_history": torch.stack(theta_history),
        "validation_iterations": torch.tensor(val_iterations),
        "validation_J": torch.tensor(val_J, dtype=dtype),
        "elapsed_seconds": elapsed,
    }


def run_all(env_name: str, alg_name: str, config_name: str, *, output_dir: str = str(ROOT / "runs")) -> list[dict]:
    if env_name not in ENVIRONMENTS:
        raise ValueError(f"unknown env {env_name!r}; available: {sorted(ENVIRONMENTS)}")
    if alg_name not in STEP_FACTORIES:
        raise ValueError(f"algorithm {alg_name!r} not implemented yet; available: {sorted(STEP_FACTORIES)}")
    cfg = CONFIGS[env_name][config_name]

    budget_modes = sorted(set(cfg.budget_modes) & SUPPORTED_BUDGET_MODES)
    skipped_budget = sorted(set(cfg.budget_modes) - SUPPORTED_BUDGET_MODES)
    flows = sorted(set(cfg.flows) & SUPPORTED_FLOWS)
    skipped_flows = sorted(set(cfg.flows) - SUPPORTED_FLOWS)
    if skipped_budget:
        print(f"note: budget_modes {skipped_budget} not implemented yet, skipping")
    if skipped_flows:
        print(f"note: flows {skipped_flows} not implemented yet, skipping")
    if not budget_modes or not flows:
        print("nothing to run: no supported budget_mode/flow combination in this config")
        return []

    env = ENVIRONMENTS[env_name]()
    theta0 = env.init_theta()
    action_probs_fn = env.policy_probs

    out_root = Path(output_dir) / env_name / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    lambdas = cfg.lambdas if alg_name in ALGORITHMS_WITH_PERTURBATION_SCALE else (None,)

    results = []
    for budget_mode in budget_modes:
        for flow in flows:
            step_fn = STEP_FACTORIES[alg_name](cfg, flow, budget_mode)
            for T in cfg.horizons:
                for lam in lambdas:
                    for seed in cfg.seeds:
                        lam_tag = f"_lam{lam}" if lam is not None else ""
                        tag = f"{alg_name}_{budget_mode}_{flow}_T{T}{lam_tag}_seed{seed}"
                        print(f"[{env_name}/{config_name}] {tag} ...", end=" ", flush=True)
                        result = train_run(
                            env,
                            action_probs_fn,
                            theta0,
                            cfg,
                            step_fn=step_fn,
                            env_name=env_name,
                            alg_name=alg_name,
                            budget_mode=budget_mode,
                            flow=flow,
                            T=T,
                            lam=lam,
                            seed=seed,
                        )
                        print(f"done in {result['elapsed_seconds']:.1f}s, final validation J={result['validation_J'][-1].item():.4f}")
                        torch.save(result, out_root / f"{tag}.pt")
                        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="twostate", choices=sorted(ENVIRONMENTS))
    parser.add_argument("--alg", default="simplex", choices=sorted(STEP_FACTORIES))
    parser.add_argument("--config", default="smoke", choices=["main", "mid", "smoke"])
    parser.add_argument("--output-dir", default=str(ROOT / "runs"))
    args = parser.parse_args()

    run_all(args.env, args.alg, args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
