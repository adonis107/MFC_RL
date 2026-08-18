"""
Compute diagnostics beyond training scope from saved scripts/train.py runs.

Usage:
    uv run python scripts/test.py --env twostate --alg simplex --config smoke

Loads every runs/<env>/<config>/<alg>_*.pt produced by scripts/train.py,
computes the context.md diagnostics that are genuinely beyond training scope
for each (budget_mode, flow, T, lam) group (theta bias/variance across
seeds; state distribution vs. the optimal benchmark; policy error;
population-tracking error; J^lambda vs J; gradient bias/variance) plus the
simplex-perturbation coverage check, and saves the result to
runs/<env>/<config>/<alg>_diagnostics.pt for notebooks to load. The building
blocks below (policy_error, objective_gap, gradient_diagnostics, rollout,
generalization_eval, ...) are also meant to be called directly, e.g. from a
notebook, on a single loaded run.

`objective_gap` and `gradient_diagnostics` are simplex-specific (they call
`simplex.estimate_objective`/`simplex.gradient_step` directly, since J^lambda
and the plug-in gradient are only defined for the simplex perturbation);
`run_diagnostics` will raise if called with alg="reinforce" or
"mfreinforce" for that reason. Everything else here (state_distribution,
policy_error, population_tracking_error, generalization_eval, rollout,
exact_gradient) only uses the algorithm-agnostic exact-flow/objective
machinery in `mfc.algorithms._common` and works for any algorithm's saved
runs. Horizon-scaling aggregation across the saved T values is left to
notebooks, where the plotting is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from mfc.algorithms import simplex
from mfc.environments.twostate import TwoState

ENVIRONMENTS = {"twostate": TwoState}


# --------------------------------------------------------------------------
# Loading saved runs
# --------------------------------------------------------------------------


def load_runs(env_name: str, alg_name: str, config_name: str, *, output_dir: str = str(ROOT / "runs")) -> list[dict]:
    """Load every individual run file scripts.train.run_all saved for this
    (env, alg, config). Matches on "..._seed<N>.pt" specifically (not just
    "{alg}_*.pt") so this never picks up run_diagnostics' own
    "{alg}_diagnostics.pt" summary file sitting in the same directory."""
    root = Path(output_dir) / env_name / config_name
    return [torch.load(f, weights_only=False) for f in sorted(root.glob(f"{alg_name}_*_seed*.pt"))]


def group_by(runs: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    """Group run dicts by a tuple of keys, e.g. ('budget_mode','flow','T','lam')."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


# --------------------------------------------------------------------------
# Ground-truth adapter
# --------------------------------------------------------------------------


def constant_policy_fn(pi: torch.Tensor):
    """Wraps a fixed (n_states, n_actions) policy matrix as an
    `action_probs_fn(theta, t, state, mu)` (theta, t, mu ignored), so a
    known-optimal policy can be fed through the same flow/rollout machinery
    as a learned one."""

    def action_probs_fn(theta, t, state, mu):
        return pi[state]

    return action_probs_fn


# --------------------------------------------------------------------------
# Policy / flow diagnostics
# --------------------------------------------------------------------------


def state_distribution(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """mu_t^theta, t=0..T: state distribution over time under the given policy."""
    return simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)


def policy_error(env, action_probs_fn, theta: torch.Tensor) -> torch.Tensor:
    """
    Per-state L1 policy error vs. env.optimal_policy(): sum_a
    |pi^theta(a|x,mu)-pi*(a|x)|, evaluated at mu=env.target_law. Generalizes
    the reference's E_0, E_1 (|pi^theta(ST|0)-0.2|, |pi^theta(ST|1)-0.25|)
    to any environment exposing `optimal_policy()`; most meaningful when
    that optimal policy is stationary, as for two-state. Requires
    `env.optimal_policy()` and `env.target_law`. Returns shape (n_states,).
    """
    pi_star = env.optimal_policy()
    mu = env.target_law
    theta = theta.detach()
    errors = torch.zeros(env.n_states, dtype=theta.dtype, device=theta.device)
    for x in range(env.n_states):
        probs = action_probs_fn(theta, 0, torch.tensor(x, device=theta.device), mu)
        errors[x] = (probs - pi_star[x]).abs().sum()
    return errors


def population_tracking_error(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """
    E_flow(theta) = (1/T) sum_{t=1}^T d_TV(mu_t^theta, target_law): average
    total-variation distance of the learned flow from the environment's
    target law, from mu0. Requires `env.target_law`.
    """
    mu_flow = state_distribution(env, action_probs_fn, theta.detach(), mu0, T)
    tv = 0.5 * (mu_flow[1:] - env.target_law).abs().sum(dim=-1)
    return tv.mean()


# --------------------------------------------------------------------------
# Gradient bias / variance
# --------------------------------------------------------------------------


def exact_gradient(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """
    grad_theta J(theta;mu0), the true unperturbed gradient, via ordinary
    autograd through the differentiable exact population flow. Ground truth
    for `gradient_diagnostics`; never used inside the model-free estimator
    (differentiating the dynamics is exactly what it's built to avoid).
    """
    theta = theta.detach().requires_grad_(True)
    J = simplex.exact_objective(env, action_probs_fn, theta, mu0, T, detach=False)
    return torch.autograd.grad(J, theta)[0]


def gradient_diagnostics(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, lam: float, n_aux: int, B: int, sigma: float, reps: int, generator=None
) -> dict:
    """
    Empirical bias and variance of the simplex plug-in gradient estimator at
    a fixed theta: `reps` independent draws of `simplex.gradient_step`,
    compared to `exact_gradient`. The reported bias mixes the theorem's
    O(lambda) perturbation bias with any residual plug-in bias from the
    auxiliary sensitivity estimate; the two aren't separable from samples
    alone.
    """
    theta = theta.detach()
    oracle = exact_gradient(env, action_probs_fn, theta, mu0, T)
    samples = torch.stack(
        [simplex.gradient_step(env, action_probs_fn, theta, mu0, T=T, n_aux=n_aux, B=B, lam=lam, sigma=sigma, generator=generator) for _ in range(reps)]
    )
    mean = samples.mean(dim=0)
    return {
        "oracle_gradient": oracle,
        "mean_estimate": mean,
        "bias": mean - oracle,
        "std": samples.std(dim=0),
        "mse": ((samples - oracle) ** 2).mean(dim=0),
    }


# --------------------------------------------------------------------------
# Theta bias / variance across seeds
# --------------------------------------------------------------------------


def theta_diagnostics(theta_finals: torch.Tensor, optimal_theta: torch.Tensor | None = None) -> dict:
    """Bias and variance of the learned theta across independent seeds
    (theta_finals: shape (n_seeds, D)), relative to the environment's known
    optimal theta if given."""
    mean = theta_finals.mean(dim=0)
    result = {"mean": mean, "std": theta_finals.std(dim=0)}
    if optimal_theta is not None:
        result["bias"] = mean - optimal_theta
        result["mse"] = ((theta_finals - optimal_theta) ** 2).mean(dim=0)
    return result


# --------------------------------------------------------------------------
# J^lambda vs J
# --------------------------------------------------------------------------


def objective_gap(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, lam: float, sigma: float, n_samples: int, generator=None) -> dict:
    """J^lambda(theta) (Monte Carlo) vs. J(theta) (exact), at the given theta."""
    theta = theta.detach()
    J = simplex.exact_objective(env, action_probs_fn, theta, mu0, T)
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)
    samples = simplex.estimate_objective(env, action_probs_fn, theta, mu_flow, mu0, T, lam, sigma, n_samples, generator=generator)
    return {"J": J, "J_lambda_mean": samples.mean(), "J_lambda_se": samples.std() / n_samples**0.5, "gap": samples.mean() - J}


# --------------------------------------------------------------------------
# Simplex-perturbation coverage: d_TV(M^lambda, mu) <= lambda
# --------------------------------------------------------------------------


def perturbation_coverage(mu_samples: torch.Tensor, lam: float, sigma: float, n_samples: int, *, generator=None) -> list[dict]:
    """
    For each mu in mu_samples (shape (K, n_states)), sample n_samples draws
    of M^lambda=(1-lambda)*mu+lambda*q and report d_TV(M^lambda,mu) stats.
    By construction (discrete_state_space(2).tex, Theorem "Perturbation
    estimate"), d_TV<=lambda always for the simplex perturbation; this
    checks the implementation honors that bound exactly, not just on
    average.
    """
    N = mu_samples.shape[-1]
    dtype, device = mu_samples.dtype, mu_samples.device
    results = []
    for mu in mu_samples:
        _, q = simplex.sample_perturbation(N, (n_samples,), sigma, dtype=dtype, device=device, generator=generator)
        M = (1.0 - lam) * mu + lam * q
        tv = 0.5 * (M - mu).abs().sum(dim=-1)
        results.append({"mu": mu, "mean_dTV": tv.mean(), "max_dTV": tv.max(), "within_bound": bool((tv <= lam + 1e-9).all())})
    return results


# --------------------------------------------------------------------------
# Generalization without retraining
# --------------------------------------------------------------------------


def generalization_eval(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, scenarios: list[dict]) -> list[dict]:
    """
    Evaluate the trained (env, theta) exactly under each scenario without
    retraining. Each scenario is a dict with a `name` and optional
    overrides `env` (a substitute env instance, e.g. with a different
    kappa/lambda0/lambda1 for model misspecification or interaction
    strength), `mu0`, `T` (default: the given ones). Covers context.md's
    "generalization" axis.
    """
    results = []
    for sc in scenarios:
        sc_env = sc.get("env", env)
        sc_mu0 = sc.get("mu0", mu0)
        sc_T = sc.get("T", T)
        J = simplex.exact_objective(sc_env, action_probs_fn, theta.detach(), sc_mu0, sc_T)
        results.append({"name": sc["name"], "J": J})
    return results


# --------------------------------------------------------------------------
# Learned vs optimal trajectories
# --------------------------------------------------------------------------


def rollout(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, generator=None) -> torch.Tensor:
    """Sample one state trajectory X_0,...,X_T under the given policy, using
    the exact nominal flow mu_t^theta as the population argument at each
    step (unperturbed, matching X_t^theta's own law). Returns shape (T+1,)."""
    theta = theta.detach()
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)
    states = [torch.multinomial(mu0, 1, generator=generator).squeeze(-1)]
    for t in range(T):
        action = simplex.sample_actions(action_probs_fn, theta, t, states[-1], mu_flow[t], generator=generator)
        states.append(env.sample_next_states(states[-1], action, mu_flow[t], generator=generator))
    return torch.stack(states)


# --------------------------------------------------------------------------
# Aggregator + CLI
# --------------------------------------------------------------------------


def run_diagnostics(env_name: str, alg_name: str, config_name: str, *, output_dir: str = str(ROOT / "runs")) -> dict:
    runs = load_runs(env_name, alg_name, config_name, output_dir=output_dir)
    if not runs:
        print(f"no saved runs found under {output_dir}/{env_name}/{config_name}/{alg_name}_*.pt; run scripts/train.py first")
        return {}

    env = ENVIRONMENTS[env_name]()
    action_probs_fn = env.policy_probs
    cfg = runs[0]["config"]
    mu0_val = torch.tensor(cfg.mu0_val, dtype=env.dtype)
    optimal_theta = env.optimal_theta() if hasattr(env, "optimal_theta") else None

    diagnostics = {}
    for key, group in group_by(runs, "budget_mode", "flow", "T", "lam").items():
        budget_mode, flow, T, lam = key
        theta_finals = torch.stack([r["theta_final"] for r in group])
        theta = group[0]["theta_final"]

        entry = {
            "budget_mode": budget_mode,
            "flow": flow,
            "T": T,
            "lam": lam,
            "n_seeds": len(group),
            "theta": theta_diagnostics(theta_finals, optimal_theta),
            "state_distribution": state_distribution(env, action_probs_fn, theta, mu0_val, T),
        }
        if hasattr(env, "optimal_policy"):
            entry["policy_error"] = policy_error(env, action_probs_fn, theta)
        if hasattr(env, "target_law"):
            entry["population_tracking_error"] = population_tracking_error(env, action_probs_fn, theta, mu0_val, T)
        # J^lambda vs J is only defined for the simplex perturbation.
        if alg_name == "simplex":
            entry["objective_gap"] = objective_gap(env, action_probs_fn, theta, mu0_val, T, lam=lam, sigma=cfg.sigma, n_samples=2000)

        diagnostics[key] = entry
        summary = f"theta_std={entry['theta']['std'].tolist()}"
        if "objective_gap" in entry:
            summary = f"J^lambda-J={entry['objective_gap']['gap'].item():+.4f}, " + summary
        print(f"[{env_name}/{config_name}] budget={budget_mode} flow={flow} T={T} lam={lam}: {summary}")

    # The d_TV<=lambda bound (Theorem "Perturbation estimate") is specific to
    # the simplex perturbation; mfreinforce's logit perturbation satisfies a
    # different bound (Lemma 2.2: E[d_TV]<=epsilon/2) that this doesn't check.
    if alg_name == "simplex":
        diagnostics["perturbation_coverage"] = perturbation_coverage(
            torch.stack([mu0_val, env.target_law]) if hasattr(env, "target_law") else mu0_val.unsqueeze(0),
            lam=cfg.lambdas[0],
            sigma=cfg.sigma,
            n_samples=5000,
        )

    out_path = Path(output_dir) / env_name / config_name / f"{alg_name}_diagnostics.pt"
    torch.save(diagnostics, out_path)
    print(f"saved diagnostics to {out_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="twostate", choices=sorted(ENVIRONMENTS))
    parser.add_argument("--alg", default="simplex")
    parser.add_argument("--config", default="smoke", choices=["main", "mid", "smoke"])
    parser.add_argument("--output-dir", default=str(ROOT / "runs"))
    args = parser.parse_args()

    run_diagnostics(args.env, args.alg, args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
