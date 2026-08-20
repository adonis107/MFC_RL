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

from mfc.algorithms import mfreinforce, simplex
from mfc.environments.advertising import Advertising
from mfc.environments.cybersecurity import CyberSecurity
from mfc.environments.distribution_planning import DistributionPlanning
from mfc.environments.twostate import TwoState

ENVIRONMENTS = {"twostate": TwoState, "cybersecurity": CyberSecurity, "distribution_planning": DistributionPlanning, "advertising": Advertising}


# --------------------------------------------------------------------------
# Loading saved runs
# --------------------------------------------------------------------------


def load_runs(env_name: str, alg_name: str, config_name: str, *, output_dir: str = str(ROOT / "runs"), device: str | None = None) -> list[dict]:
    """Load every individual run file scripts.train.run_all saved for this
    (env, alg, config). Matches on "..._seed<N>.pt" specifically (not just
    "{alg}_*.pt") so this never picks up run_diagnostics' own
    "{alg}_diagnostics.pt" summary file sitting in the same directory.
    Tensors in each loaded run are moved to `device` (default: the ambient
    `torch.get_default_device()` — only correct if a caller has actually set
    one, e.g. a notebook's setup cell or tests/conftest.py; pass the target
    env's own `.device` explicitly otherwise, since environments auto-detect
    CUDA regardless of the torch-global default), so checkpoints cached from
    a different device (e.g. CPU-trained results reloaded in a CUDA session)
    work without retraining."""
    root = Path(output_dir) / env_name / config_name
    device = device if device is not None else torch.get_default_device()
    runs = []
    for f in sorted(root.glob(f"{alg_name}_*_seed*.pt")):
        run = torch.load(f, weights_only=False)
        runs.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in run.items()})
    return runs


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


def reference_policy_fn(env, *, action: int = 1):
    """
    Wraps `env.reference_policy(p)` (e.g. `mfc.environments.advertising
    .Advertising`'s closed-form infinite-horizon optimal advertising rate,
    a function of the population law alone) as an
    `action_probs_fn(theta, t, state, mu)` (theta, t, state ignored) so it
    can be fed through the same flow/rollout machinery as a learned policy.
    `action` is the index the reference policy's returned probability
    applies to (default 1: "display an advertisement" for advertising).
    Assumes a 2-action environment (`1 - action` is the complement).
    """
    assert env.n_actions == 2, "reference_policy_fn assumes a 2-action environment"

    def action_probs_fn(theta, t, state, mu):
        q = env.reference_policy(mu[action])
        # built via stack (not in-place indexing) so this stays vmap-safe,
        # since `eval_batched`/`state_distribution` trace this under vmap
        return torch.stack([q, 1.0 - q]) if action == 0 else torch.stack([1.0 - q, q])

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


def intervention_probability(env, action_probs_fn, theta: torch.Tensor, mu_flow: torch.Tensor, *, action: int = 1) -> torch.Tensor:
    """
    A_t := sum_x mu_t(x) pi_t(action|x,mu_t), the population-averaged
    probability of taking `action` (default 1: cybersecurity's "switch
    protection level") under the given policy, at each t of the given flow
    (reference "Evaluation criteria"). `mu_flow`: shape (T+1, n_states),
    e.g. from `state_distribution`. Environment-agnostic. Returns shape (T+1,).
    """
    theta = theta.detach()
    T1, N = mu_flow.shape
    states_all = torch.arange(N, device=theta.device)
    out = torch.zeros(T1, dtype=theta.dtype, device=theta.device)
    for t in range(T1):
        mu_t = mu_flow[t]
        pi = simplex.eval_batched(action_probs_fn, theta, t, states_all, mu_t)  # (N, A)
        out[t] = (mu_t * pi[:, action]).sum()
    return out


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
# Distribution-planning-specific diagnostics (reference "Evaluation criteria")
# --------------------------------------------------------------------------


def terminal_mismatch_l2(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """E_T^(2)(theta) = ||mu_T^theta - mu_target||_2. Requires `env.target_law`."""
    mu_flow = state_distribution(env, action_probs_fn, theta.detach(), mu0, T)
    return (mu_flow[T] - env.target_law).norm()


def average_mismatch_l2(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """E_bar^(2)(theta) = (1/(T+1)) sum_{t=0}^T ||mu_t^theta - mu_target||_2. Requires `env.target_law`."""
    mu_flow = state_distribution(env, action_probs_fn, theta.detach(), mu0, T)
    return (mu_flow - env.target_law).norm(dim=-1).mean()


def cyclic_wasserstein(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    W_{1,d_cyc}(mu, target): the reference's terminal transport discrepancy
    E_T^(W), computed exactly for a discrete law on a cycle of length N via
    the closed-form circular-EMD reduction: with F_mu, F_target the
    cumulative laws (cutting the circle at index 0), d = F_mu - F_target,
    W_1 = sum_i |d_i - median(d)| (any median-minimizing c gives the same
    optimal L1 cost; `torch.quantile` matches). Verified against known
    point-mass ground truth (d_cyc(0,k) for all k) before use here.
    """
    F_mu = torch.cumsum(mu, dim=-1)
    F_target = torch.cumsum(target, dim=-1)
    d = F_mu - F_target
    c = torch.quantile(d, 0.5)
    return (d - c).abs().sum()


def cumulative_movement(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, stay_action: int) -> torch.Tensor:
    """
    C_mov(theta) = sum_{t=0}^{T-1} sum_x mu_t(x) [1 - pi_t(stay_action|x,mu_t)]:
    expected cumulative movement (probability mass that moves at each step,
    summed over the episode). `stay_action` is the "remain in place" action
    index (e.g. `distribution_planning.STAY`); environment-agnostic otherwise.
    """
    theta = theta.detach()
    mu_flow = state_distribution(env, action_probs_fn, theta, mu0, T)
    N = env.n_states
    states_all = torch.arange(N, device=theta.device)
    total = torch.zeros((), dtype=theta.dtype, device=theta.device)
    for t in range(T):
        mu_t = mu_flow[t]
        pi = simplex.eval_batched(action_probs_fn, theta, t, states_all, mu_t)  # (N, A)
        total = total + (mu_t * (1.0 - pi[:, stay_action])).sum()
    return total


# --------------------------------------------------------------------------
# Gradient bias / variance
# --------------------------------------------------------------------------


def exact_gradient(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, gamma: float = 1.0) -> torch.Tensor:
    """
    grad_theta J(theta;mu0), the true unperturbed gradient, via ordinary
    autograd through the differentiable exact population flow. Ground truth
    for `gradient_diagnostics`; never used inside the model-free estimator
    (differentiating the dynamics is exactly what it's built to avoid).
    `gamma=1.0` (default) recovers the undiscounted objective; pass the
    environment's own discount (e.g. `env.config.gamma`) when it has one.
    """
    theta = theta.detach().requires_grad_(True)
    J = simplex.exact_objective(env, action_probs_fn, theta, mu0, T, gamma=gamma, detach=False)
    return torch.autograd.grad(J, theta)[0]


def gradient_diagnostics(
    env,
    action_probs_fn,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    T: int,
    *,
    lam: float,
    n_aux: int,
    B: int,
    sigma: float,
    reps: int,
    gamma: float = 1.0,
    generator=None,
) -> dict:
    """
    Empirical bias and variance of the simplex plug-in gradient estimator at
    a fixed theta: `reps` independent draws of `simplex.gradient_step`,
    compared to `exact_gradient`. The reported bias mixes the theorem's
    O(lambda) perturbation bias with any residual plug-in bias from the
    auxiliary sensitivity estimate; the two aren't separable from samples
    alone. `gamma=1.0` (default) recovers the undiscounted objective/estimator.
    """
    theta = theta.detach()
    oracle = exact_gradient(env, action_probs_fn, theta, mu0, T, gamma=gamma)
    samples = torch.stack(
        [
            simplex.gradient_step(env, action_probs_fn, theta, mu0, T=T, n_aux=n_aux, B=B, lam=lam, sigma=sigma, gamma=gamma, generator=generator)
            for _ in range(reps)
        ]
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


def objective_gap(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, lam: float, sigma: float, n_samples: int, gamma: float = 1.0, generator=None
) -> dict:
    """J^lambda(theta) (Monte Carlo) vs. J(theta) (exact), at the given
    theta. `gamma=1.0` (default) recovers the undiscounted objective; pass
    the environment's own discount (e.g. `env.config.gamma`) when it has one."""
    theta = theta.detach()
    J = simplex.exact_objective(env, action_probs_fn, theta, mu0, T, gamma=gamma)
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)
    samples = simplex.estimate_objective(env, action_probs_fn, theta, mu_flow, mu0, T, lam, sigma, n_samples, gamma=gamma, generator=generator)
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


def logit_perturbation_coverage(mu_samples: torch.Tensor, epsilon: float, n_samples: int, *, generator=None) -> list[dict]:
    """
    For each mu in mu_samples (shape (K, n_states)), sample n_samples draws
    of M^epsilon=softmax(log(mu)+epsilon*Lambda), Lambda~N(0,I_N)
    (`mfc.algorithms.mfreinforce._perturbed_law`) and report d_TV(M^epsilon,mu)
    stats. Unlike the simplex perturbation's almost-sure d_TV<=lambda bound,
    mfreinforce's logit perturbation only satisfies a bound *in expectation*
    (files/Discrete RL - Meunier, Pham, Reisinger.md, Lemma 2.2:
    E[d_TV(mu,mu_epsilon)]<=epsilon/2, via Pinsker's inequality against the
    perturbation's KL divergence) — individual draws can and do exceed
    epsilon/2, so `within_bound` here checks the *sampled mean*, not (as for
    `perturbation_coverage`'s almost-sure bound) every single draw.
    """
    N = mu_samples.shape[-1]
    dtype, device = mu_samples.dtype, mu_samples.device
    results = []
    for mu in mu_samples:
        Lambda = torch.randn(n_samples, N, dtype=dtype, device=device, generator=generator)
        M = mfreinforce._perturbed_law(mu, Lambda, epsilon)
        tv = 0.5 * (M - mu).abs().sum(dim=-1)
        results.append({"mu": mu, "mean_dTV": tv.mean(), "max_dTV": tv.max(), "within_bound": bool(tv.mean() <= epsilon / 2 + 1e-9)})
    return results


# --------------------------------------------------------------------------
# Population-flow sensitivity D_t^theta(k) = grad_theta mu_t^theta(k):
# exact value, model-free estimator, and the resulting oracle-D gradient
# --------------------------------------------------------------------------


def exact_sensitivity_flow(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int) -> torch.Tensor:
    """
    D_t^theta(k) = grad_theta mu_t^theta(k), t=0,...,T, k=1,...,N-1, via
    ordinary autograd through the differentiable exact population flow
    (`simplex.exact_population_flow(..., detach=False)`). Ground truth for
    `simplex.estimate_sensitivity_flow`'s single-batch forward estimator
    (discrete_state_space(2).tex, "Model-free estimation of the
    population-flow sensitivities"). Returns shape (T+1, N-1, D).
    """
    theta = theta.detach().requires_grad_(True)
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T, detach=False)
    N = env.n_states
    rows = []
    for t in range(T + 1):
        cols = [torch.autograd.grad(mu_flow[t, k], theta, retain_graph=True)[0] for k in range(N - 1)]
        rows.append(torch.stack(cols))
    return torch.stack(rows).detach()


def sensitivity_estimation_error(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, eta: float, n: int, sigma: float, reps: int, generator=None
) -> dict:
    """
    Empirical bias/variance of `simplex.estimate_sensitivity_flow`'s
    single-batch forward estimator D_hat_t(k) against the exact D_t^theta(k)
    (`exact_sensitivity_flow`), from `reps` independent auxiliary batches at
    fixed (eta, n). Validates the sensitivity-estimation assumption
    (discrete_state_space(2).tex, "Bias and mean-squared error of the
    gradient estimator"): A_eta := sup|D_t^eta,theta - D_t^theta| should
    shrink with eta, and V_eta/n := the residual Monte Carlo variance should
    shrink with n. Returns per-(t,k) `bias_norm` and `variance` (each shape
    (T+1, N-1)) plus a scalar `mse` averaged over reps.

    Accumulates a running sum/sum-of-squares across reps rather than
    `torch.stack`-ing every draw: each draw has shape (T+1, N-1, D), and for
    a large policy network (D in the tens of thousands, e.g.
    distribution_planning's ~76.6k-parameter MLP) stacking hundreds of reps
    materializes a multi-GB tensor and reliably OOMs a modest GPU, even
    though the final statistics only need reps-many scalars' worth of
    aggregation.
    """
    theta = theta.detach()
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)
    exact = exact_sensitivity_flow(env, action_probs_fn, theta, mu0, T)

    total = torch.zeros_like(exact)
    total_sq = torch.zeros_like(exact)
    for _ in range(reps):
        sample = simplex.estimate_sensitivity_flow(env, action_probs_fn, theta, mu_flow, mu0, T, n, eta, sigma, generator=generator)
        total += sample
        total_sq += sample**2

    mean = total / reps
    bias_norm = (mean - exact).norm(dim=-1)
    # Population (ddof=0) variance from the running moments, so that
    # mse = bias_norm**2 + variance holds exactly (E[(X-c)^2] = Var(X) + (E[X]-c)^2).
    variance = (total_sq / reps - mean**2).clamp_min(0.0).sum(dim=-1)
    mse = bias_norm**2 + variance
    # Delta-method standard error of bias_norm itself (a norm of an
    # approximately-Gaussian reps-averaged mean): SE(||mean-exact||) ~=
    # sqrt(sum_i Var(mean_i)) = sqrt(variance/reps), reported so callers can
    # tell a genuine A_eta signal from residual finite-reps noise in the mean.
    bias_se = (variance / reps).sqrt()
    return {"bias_norm": bias_norm, "bias_se": bias_se, "variance": variance, "mse": mse}


def oracle_gradient_estimate(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, lam: float, sigma: float, B: int, gamma: float = 1.0, generator=None
) -> torch.Tensor:
    """
    One sample of the oracle-D plug-in gradient estimator
    ghat_{B,lambda}^orc(theta) (discrete_state_space(2).tex, Proposition
    "Mean of the main-batch estimator"): `simplex.gradient_estimate` fed the
    *exact* sensitivity flow (`exact_sensitivity_flow`) instead of the
    auxiliary plug-in estimate. Since E[ghat^orc] = grad J^lambda(theta)
    exactly, averaging many reps and comparing to `exact_gradient` (= grad
    J(theta)) isolates the perturbation-bias term (III) of the estimator's
    bias decomposition from the sensitivity-estimation term (II), which
    `gradient_diagnostics`'s ordinary plug-in samples mix together.
    """
    theta = theta.detach()
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)
    D_exact = exact_sensitivity_flow(env, action_probs_fn, theta, mu0, T)
    return simplex.gradient_estimate(env, action_probs_fn, theta, mu_flow, mu0, D_exact, T, B, lam, sigma, gamma=gamma, generator=generator)


# --------------------------------------------------------------------------
# Stability of the perturbed state marginal: d_TV(nu_t^{lambda,theta}, mu_t^theta)
# --------------------------------------------------------------------------


def state_marginal_stability(
    env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, *, lam: float, sigma: float, n_samples: int, generator=None
) -> torch.Tensor:
    """
    Empirical d_TV(nu_t^{lambda,theta}, mu_t^theta), t=0,...,T, where
    nu_t^{lambda,theta} := Law(X_t^{lambda,theta}) is estimated from
    n_samples independent lambda-perturbed trajectories (fresh q_t at every
    step, as in `simplex.estimate_objective`) and mu_t^theta is the exact
    nominal flow. Validates the Lemma "Stability of the state marginal"
    (discrete_state_space(2).tex): d_TV(nu_t,mu_t) <= L_K*lambda*t. Returns
    shape (T+1,).
    """
    theta = theta.detach()
    device, dtype = theta.device, theta.dtype
    N = env.n_states
    mu_flow = simplex.exact_population_flow(env, action_probs_fn, theta, mu0, T)

    states = torch.multinomial(mu0.expand(n_samples, N), 1, generator=generator).reshape(n_samples)
    tv = torch.zeros(T + 1, dtype=dtype, device=device)
    counts0 = torch.bincount(states, minlength=N).to(dtype) / n_samples
    tv[0] = 0.5 * (counts0 - mu_flow[0]).abs().sum()
    for t in range(T):
        mu_t = mu_flow[t]
        _, q = simplex.sample_perturbation(N, (n_samples,), sigma, dtype=dtype, device=device, generator=generator)
        M = (1.0 - lam) * mu_t + lam * q
        actions = simplex.sample_actions(action_probs_fn, theta, t, states, M, generator=generator)
        states = env.sample_next_states(states, actions, M, generator=generator)
        counts = torch.bincount(states, minlength=N).to(dtype) / n_samples
        tv[t + 1] = 0.5 * (counts - mu_flow[t + 1]).abs().sum()
    return tv


# --------------------------------------------------------------------------
# Generalization without retraining
# --------------------------------------------------------------------------


def generalization_eval(env, action_probs_fn, theta: torch.Tensor, mu0: torch.Tensor, T: int, scenarios: list[dict], *, gamma: float = 1.0) -> list[dict]:
    """
    Evaluate the trained (env, theta) exactly under each scenario without
    retraining. Each scenario is a dict with a `name` and optional
    overrides `env` (a substitute env instance, e.g. with a different
    kappa/lambda0/lambda1 for model misspecification or interaction
    strength), `mu0`, `T` (default: the given ones). Covers context.md's
    "generalization" axis. `gamma=1.0` (default) recovers the undiscounted
    objective; pass the environment's own discount (e.g. `env.config.gamma`)
    when it has one — a scenario's own substitute `env` can also override it
    via a `gamma` key.
    """
    results = []
    for sc in scenarios:
        sc_env = sc.get("env", env)
        sc_mu0 = sc.get("mu0", mu0)
        sc_T = sc.get("T", T)
        sc_gamma = sc.get("gamma", gamma)
        J = simplex.exact_objective(sc_env, action_probs_fn, theta.detach(), sc_mu0, sc_T, gamma=sc_gamma)
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
    env = ENVIRONMENTS[env_name]()
    runs = load_runs(env_name, alg_name, config_name, output_dir=output_dir, device=env.device)
    if not runs:
        print(f"no saved runs found under {output_dir}/{env_name}/{config_name}/{alg_name}_*.pt; run scripts/train.py first")
        return {}

    action_probs_fn = env.policy_probs
    cfg = runs[0]["config"]
    mu0_val = torch.tensor(cfg.mu0_val, dtype=env.dtype, device=env.device)
    optimal_theta = env.optimal_theta() if hasattr(env, "optimal_theta") else None
    gamma = getattr(env.config, "gamma", 1.0)

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
            entry["objective_gap"] = objective_gap(env, action_probs_fn, theta, mu0_val, T, lam=lam, sigma=cfg.sigma, n_samples=2000, gamma=gamma)

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
