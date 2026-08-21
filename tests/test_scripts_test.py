from __future__ import annotations

import pytest
import torch

from configs.twostate import TwoStateRunConfig
from mfc.algorithms import simplex
from mfc.environments.lq import LQ, LQConfig
from mfc.environments.twostate import TwoState, TwoStateConfig
from scripts.test import (
    constant_policy_fn,
    continuous_generalization_eval,
    continuous_gradient_diagnostics,
    continuous_mean_field_term,
    continuous_objective_gap,
    continuous_oracle_gradient_estimate,
    continuous_perturbation_coverage,
    continuous_sensitivity_error,
    continuous_state_marginal_stability,
    exact_coordinate_sensitivity,
    exact_gradient,
    exact_sensitivity_flow,
    generalization_eval,
    gradient_diagnostics,
    group_by,
    intervention_probability,
    load_runs,
    logit_perturbation_coverage,
    objective_gap,
    oracle_gradient_estimate,
    perturbation_coverage,
    policy_error,
    population_tracking_error,
    rollout,
    run_continuous_diagnostics,
    run_diagnostics,
    sensitivity_estimation_error,
    state_distribution,
    state_marginal_stability,
    theta_diagnostics,
)
from scripts.train import run_all, run_continuous

torch.set_default_dtype(torch.float64)


def test_policy_error_is_zero_at_optimal_theta():
    env = TwoState()
    assert torch.allclose(policy_error(env, env.policy_probs, env.optimal_theta()), torch.zeros(2), atol=1e-10)


def test_population_tracking_error_is_zero_at_optimal_theta():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    err = population_tracking_error(env, env.policy_probs, env.optimal_theta(), mu0, T=3)
    assert err.item() == 0.0 or err.item() < 1e-10


def test_state_distribution_matches_optimal_policy_fixed_point():
    env = TwoState()
    mu0 = torch.tensor([0.3, 0.7])
    flow = state_distribution(env, constant_policy_fn(env.optimal_policy()), env.optimal_theta(), mu0, T=3)
    assert torch.allclose(flow[1], env.target_law, atol=1e-10)
    assert torch.allclose(flow[3], env.target_law, atol=1e-10)


def test_intervention_probability_matches_hand_computed_average():
    env = TwoState()
    theta = env.optimal_theta()
    pi = env.optimal_policy()  # pi[:,1]=pi(MV|.) = [0.8, 0.75]
    mu_flow = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=env.dtype)
    A = intervention_probability(env, env.policy_probs, theta, mu_flow, action=1)
    expected = mu_flow @ pi[:, 1]
    assert torch.allclose(A, expected, atol=1e-10)


def test_theta_diagnostics_matches_hand_computed_statistics():
    thetas = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    optimal = torch.tensor([2.0, 2.0])
    result = theta_diagnostics(thetas, optimal)
    assert torch.allclose(result["mean"], torch.tensor([3.0, 2.0]))
    assert torch.allclose(result["bias"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result["std"], thetas.std(dim=0))
    assert torch.allclose(result["mse"], ((thetas - optimal) ** 2).mean(dim=0))


def test_objective_gap_matches_exact_objective_and_shrinks_with_lambda():
    torch.manual_seed(0)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta = env.optimal_theta()

    small = objective_gap(env, env.policy_probs, theta, mu0, T=2, lam=0.05, sigma=1.0, n_samples=20_000)
    large = objective_gap(env, env.policy_probs, theta, mu0, T=2, lam=0.8, sigma=1.0, n_samples=20_000)

    from mfc.algorithms import simplex

    assert torch.allclose(small["J"], simplex.exact_objective(env, env.policy_probs, theta, mu0, 2))
    # both gaps should be resolvable well above their own Monte Carlo noise
    assert small["gap"].abs() > 3 * small["J_lambda_se"] or small["gap"].abs() < 1e-6
    assert large["gap"].abs() > small["gap"].abs()


def test_perturbation_coverage_respects_the_dTV_bound():
    torch.manual_seed(1)
    mu_samples = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.05, 0.95]])
    for lam in (0.05, 0.2, 0.8):
        results = perturbation_coverage(mu_samples, lam=lam, sigma=1.0, n_samples=5000)
        for r in results:
            assert r["within_bound"]
            assert r["max_dTV"] <= lam + 1e-9


def test_logit_perturbation_coverage_respects_the_expected_dTV_bound():
    """Lemma 2.2 (files/Discrete RL - Meunier, Pham, Reisinger.md):
    E[d_TV(mu,mu_epsilon)]<=epsilon/2 -- unlike simplex's almost-sure bound,
    this only holds for the sampled mean, not every draw."""
    torch.manual_seed(1)
    mu_samples = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.05, 0.95]])
    for epsilon in (0.2, 0.5, 1.0, 2.0):
        results = logit_perturbation_coverage(mu_samples, epsilon=epsilon, n_samples=5000)
        for r in results:
            assert r["within_bound"]
            assert r["mean_dTV"] <= epsilon / 2 + 1e-9


def test_exact_gradient_matches_finite_differences():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    g = exact_gradient(env, env.policy_probs, theta0, mu0, T=2)

    from mfc.algorithms import simplex

    eps = 1e-5
    fd = torch.zeros(2)
    for i in range(2):
        tp, tm = theta0.clone(), theta0.clone()
        tp[i] += eps
        tm[i] -= eps
        Jp = simplex.exact_objective(env, env.policy_probs, tp, mu0, 2)
        Jm = simplex.exact_objective(env, env.policy_probs, tm, mu0, 2)
        fd[i] = (Jp - Jm) / (2 * eps)
    assert torch.allclose(g, fd, atol=1e-4)


def test_gradient_diagnostics_shapes_and_zero_bias_reference():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta = torch.tensor([0.3, -0.4])
    result = gradient_diagnostics(env, env.policy_probs, theta, mu0, T=2, lam=0.2, n_aux=10, B=50, sigma=1.0, reps=5)
    for key in ("oracle_gradient", "mean_estimate", "bias", "std", "mse"):
        assert result[key].shape == (2,)
    assert torch.equal(result["bias"], result["mean_estimate"] - result["oracle_gradient"])


def test_rollout_has_correct_length_and_valid_states():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    traj = rollout(env, env.policy_probs, env.init_theta(), mu0, T=6)
    assert traj.shape == (7,)
    assert torch.all((traj == 0) | (traj == 1))


def test_generalization_eval_matches_exact_objective_and_reacts_to_overrides():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta = torch.tensor([0.3, -0.4])

    from mfc.algorithms import simplex

    scenarios = [
        {"name": "baseline"},
        {"name": "other_mu0", "mu0": torch.tensor([0.5, 0.5])},
        {"name": "longer_T", "T": 5},
        {"name": "higher_kappa", "env": TwoState(TwoStateConfig(kappa=50.0))},
    ]
    results = generalization_eval(env, env.policy_probs, theta, mu0, 2, scenarios)
    by_name = {r["name"]: r["J"] for r in results}

    assert torch.allclose(by_name["baseline"], simplex.exact_objective(env, env.policy_probs, theta, mu0, 2))
    assert by_name["other_mu0"] != by_name["baseline"]
    assert by_name["longer_T"] != by_name["baseline"]
    assert by_name["higher_kappa"] < by_name["baseline"]  # a bigger interaction penalty can only reduce J here


def test_load_runs_and_run_diagnostics_end_to_end(tmp_path):
    cfg = TwoStateRunConfig(name="tiny", horizons=(2,), lambdas=(0.2,), n_train=10, seeds=(0, 1), flows=("exact",), budget_modes=("equal_parameters",))
    import scripts.train as train_mod

    train_mod.CONFIGS.setdefault("twostate", {})["tiny_test"] = cfg

    run_all("twostate", "simplex", "tiny_test", output_dir=str(tmp_path))
    runs = load_runs("twostate", "simplex", "tiny_test", output_dir=str(tmp_path))
    assert len(runs) == 2

    groups = group_by(runs, "budget_mode", "flow", "T", "lam")
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2

    diagnostics = run_diagnostics("twostate", "simplex", "tiny_test", output_dir=str(tmp_path))
    assert (tmp_path / "twostate" / "tiny_test" / "simplex_diagnostics.pt").exists()
    entry = next(v for k, v in diagnostics.items() if k != "perturbation_coverage")
    assert entry["n_seeds"] == 2
    assert "policy_error" in entry
    assert "objective_gap" in entry

    # regression: load_runs must ignore run_diagnostics' own "{alg}_diagnostics.pt"
    # summary file sitting in the same directory (it matched the naive
    # "{alg}_*.pt" glob and got loaded as if it were a run, corrupting the list)
    reloaded = load_runs("twostate", "simplex", "tiny_test", output_dir=str(tmp_path))
    assert len(reloaded) == 2
    assert all("theta_final" in r for r in reloaded)


def test_exact_sensitivity_flow_matches_finite_differences():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    T = 2

    D = exact_sensitivity_flow(env, env.policy_probs, theta0, mu0, T)
    assert D.shape == (T + 1, env.n_states - 1, theta0.numel())

    eps = 1e-5
    for t in range(T + 1):
        for i in range(theta0.numel()):
            tp, tm = theta0.clone(), theta0.clone()
            tp[i] += eps
            tm[i] -= eps
            mu_p = simplex.exact_population_flow(env, env.policy_probs, tp, mu0, T)
            mu_m = simplex.exact_population_flow(env, env.policy_probs, tm, mu0, T)
            fd = (mu_p[t, 0] - mu_m[t, 0]) / (2 * eps)
            assert torch.allclose(D[t, 0, i], fd, atol=1e-4)


def test_exact_sensitivity_flow_is_zero_at_t0_since_mu0_is_theta_independent():
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    D = exact_sensitivity_flow(env, env.policy_probs, theta0, mu0, T=2)
    assert torch.allclose(D[0], torch.zeros_like(D[0]))


def test_sensitivity_estimation_error_shrinks_with_more_auxiliary_samples():
    torch.manual_seed(0)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])

    small_n = sensitivity_estimation_error(env, env.policy_probs, theta0, mu0, T=2, eta=0.2, n=20, sigma=1.0, reps=200)
    large_n = sensitivity_estimation_error(env, env.policy_probs, theta0, mu0, T=2, eta=0.2, n=400, sigma=1.0, reps=200)
    assert large_n["variance"].sum() < small_n["variance"].sum()
    assert large_n["mse"].sum() < small_n["mse"].sum()


def test_oracle_gradient_estimate_matches_gradient_estimate_called_directly():
    torch.manual_seed(0)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    T, lam, sigma, B = 2, 0.2, 1.0, 50

    g = torch.Generator(device=env.device).manual_seed(1)
    sample = oracle_gradient_estimate(env, env.policy_probs, theta0, mu0, T, lam=lam, sigma=sigma, B=B, generator=g)

    g2 = torch.Generator(device=env.device).manual_seed(1)
    mu_flow = simplex.exact_population_flow(env, env.policy_probs, theta0, mu0, T)
    D_exact = exact_sensitivity_flow(env, env.policy_probs, theta0, mu0, T)
    expected = simplex.gradient_estimate(env, env.policy_probs, theta0, mu_flow, mu0, D_exact, T, B, lam, sigma, generator=g2)
    assert torch.allclose(sample, expected)


def test_oracle_gradient_estimate_bias_shrinks_as_lambda_shrinks():
    """Proposition "Mean of the main-batch estimator": E[ghat^orc] = grad
    J^lambda(theta) exactly, and the Gradient-level convergence theorem says
    ||grad J^lambda - grad J|| = O(lambda). So the oracle-D estimator's
    empirical bias against the exact unperturbed gradient should shrink as
    lambda shrinks."""
    torch.manual_seed(0)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    T, sigma, B, reps = 2, 1.0, 200, 100

    oracle = exact_gradient(env, env.policy_probs, theta0, mu0, T)

    def bias_norm(lam):
        samples = torch.stack([oracle_gradient_estimate(env, env.policy_probs, theta0, mu0, T, lam=lam, sigma=sigma, B=B) for _ in range(reps)])
        return (samples.mean(dim=0) - oracle).norm()

    assert bias_norm(0.05) < bias_norm(0.8)


def test_state_marginal_stability_returns_valid_tv_distances():
    """Lemma "Stability of the state marginal": d_TV(nu_t,mu_t) <=
    L_K*lambda*t. Checks basic validity (t=0 close to zero up to
    finite-sample MC noise, all values valid TV distances in [0,1]) rather
    than monotonicity in lambda: twostate's small, strongly mean-reverting
    dynamics make this O(lambda*t) upper bound loose, without a clean
    empirical growth trend at reasonable sample sizes."""
    torch.manual_seed(0)
    env = TwoState()
    mu0 = torch.tensor([0.8, 0.2])
    theta0 = torch.tensor([0.3, -0.4])
    T = 3

    tv = state_marginal_stability(env, env.policy_probs, theta0, mu0, T, lam=0.5, sigma=1.0, n_samples=20_000)
    assert tv.shape == (T + 1,)
    assert tv[0].item() < 0.02
    assert bool((tv >= 0).all())
    assert bool((tv <= 1.0).all())


def test_run_diagnostics_skips_simplex_only_checks_for_other_algorithms(tmp_path):
    """objective_gap (J^lambda vs J) and perturbation_coverage (the
    simplex d_TV<=lambda bound) are only meaningful for the simplex
    perturbation; run_diagnostics must not attempt them — and must not
    crash — for an algorithm with no such perturbation (lam=None)."""
    cfg = TwoStateRunConfig(name="tiny_rf", horizons=(2,), n_train=5, seeds=(0,), flows=("exact",), budget_modes=("equal_parameters",))
    import scripts.train as train_mod

    train_mod.CONFIGS.setdefault("twostate", {})["tiny_rf_test"] = cfg
    run_all("twostate", "reinforce", "tiny_rf_test", output_dir=str(tmp_path))

    diagnostics = run_diagnostics("twostate", "reinforce", "tiny_rf_test", output_dir=str(tmp_path))
    assert "perturbation_coverage" not in diagnostics
    entry = next(iter(diagnostics.values()))
    assert "objective_gap" not in entry
    assert "policy_error" in entry  # the algorithm-agnostic diagnostics still run


# --------------------------------------------------------------------------
# Continuous-state diagnostics (`lq`, `portfolio`)
# --------------------------------------------------------------------------


def test_exact_coordinate_sensitivity_matches_finite_differences():
    env = LQ(device="cpu")
    T = 3
    theta = 0.2 * torch.randn(T, 2, dtype=env.dtype, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0))
    D = exact_coordinate_sensitivity(env, theta)
    assert D.shape == (T + 1, T, 2)
    assert D[0].abs().max().item() == 0.0  # mu_0 does not depend on theta

    eps = 1e-6
    for t in (1, T):
        for i in range(T):
            for j in range(2):
                bumped = theta.clone()
                bumped[i, j] += eps
                fd = (env.forward_moments(bumped, 0.0)[0][t] - env.forward_moments(theta, 0.0)[0][t]) / eps
                assert D[t, i, j].item() == pytest.approx(fd.item(), abs=1e-5)


def test_continuous_objective_gap_matches_the_closed_forms_and_scales_as_lambda_squared():
    env = LQ(device="cpu")
    theta = env.riccati_optimal(3)
    generator = torch.Generator(device="cpu").manual_seed(0)
    gaps = {}
    for lam in (0.1, 0.2, 0.4):
        out = continuous_objective_gap(env, theta, lam=lam, n_samples=20_000, generator=generator)
        assert out["J"].item() == pytest.approx(env.exact_objective(theta, 0.0).item())
        # the Monte Carlo column is a check on the simulator against the closed form
        assert abs(out["J_lambda_mean"].item() - out["J_lambda"].item()) < 4.0 * out["J_lambda_se"].item()
        gaps[lam] = out["gap"].item()
    for lam in (0.2, 0.4):
        assert gaps[lam] / gaps[lam / 2] == pytest.approx(4.0, rel=1e-6)  # exact O(lambda^2), not just a fitted rate


def test_continuous_gradient_diagnostics_bias_decomposition_is_exact():
    env = LQ(device="cpu")
    theta = 0.2 * torch.randn(2, 2, dtype=env.dtype, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0))
    gd = continuous_gradient_diagnostics(
        env, theta, lam=0.2, B=64, n_aux=32, reps=8, baseline="loo", generator=torch.Generator(device="cpu").manual_seed(0)
    )
    assert torch.allclose(gd["oracle_gradient"], env.exact_gradient(theta, 0.0))
    assert torch.allclose(gd["perturbed_gradient"], env.exact_gradient(theta, 0.2))
    assert torch.allclose(gd["bias"], gd["perturbation_bias"] + gd["estimation_bias"], atol=1e-10)


def test_continuous_oracle_gradient_estimate_bias_shrinks_as_lambda_shrinks():
    """With the exact sensitivity flow, E[ghat] = grad J^lambda exactly, so
    its distance to grad J^0 is the perturbation bias alone — O(lambda^2) for
    this centered perturbation."""
    env = LQ(device="cpu")
    theta = 0.2 * torch.randn(2, 2, dtype=env.dtype, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0))
    target = env.exact_gradient(theta, 0.0)
    biases = {}
    for lam in (0.1, 0.4):
        generator = torch.Generator(device="cpu").manual_seed(0)
        samples = torch.stack(
            [continuous_oracle_gradient_estimate(env, theta, lam=lam, B=20_000, baseline="loo", generator=generator) for _ in range(20)]
        )
        biases[lam] = (samples.mean(dim=0) - target).norm().item()
    assert biases[0.1] < biases[0.4]


def test_continuous_sensitivity_error_shrinks_with_more_auxiliary_samples():
    env = LQ(device="cpu")
    theta = 0.2 * torch.randn(3, 2, dtype=env.dtype, device="cpu", generator=torch.Generator(device="cpu").manual_seed(0))
    errs = {}
    for n in (50, 5_000):
        out = continuous_sensitivity_error(env, theta, eta=0.2, n=n, reps=10, generator=torch.Generator(device="cpu").manual_seed(0))
        assert out["bias_norm"].shape == (theta.shape[0] + 1,)
        assert torch.allclose(out["mse"], out["bias_norm"] ** 2 + out["variance"])
        errs[n] = out["total_mse"].item()
    assert errs[5_000] < errs[50]


def test_continuous_perturbation_coverage_respects_the_W2_bound():
    """The transport analogue of d_TV(M^lambda,mu) <= lambda: every single
    draw must satisfy W_2 <= lambda*|Z|*sqrt(1+mu^2+Sigma), and the mean
    squared distance must match its closed form."""
    env = LQ(device="cpu")
    theta = env.riccati_optimal(3)
    for lam in (0.1, 0.5):
        results = continuous_perturbation_coverage(env, theta, lam=lam, n_samples=20_000, generator=torch.Generator(device="cpu").manual_seed(0))
        assert len(results) == 4
        for r in results:
            assert r["within_bound"]
            assert r["mean_W2_sq"].item() == pytest.approx(r["predicted_mean_W2_sq"].item(), rel=0.05)


def test_continuous_state_marginal_stability_matches_the_exact_moment_recursions():
    env = LQ(device="cpu")
    theta = env.riccati_optimal(3)
    out = continuous_state_marginal_stability(env, theta, lam=0.4, n_samples=200_000, generator=torch.Generator(device="cpu").manual_seed(0))
    assert torch.allclose(out["exact_mean"], out["nominal_mean"])  # the perturbation is centered: mu is exactly lambda-free
    assert (out["variance_inflation"] >= 0).all()
    assert torch.allclose(out["empirical_mean"], out["exact_mean"], atol=0.05)
    assert torch.allclose(out["empirical_variance"], out["exact_variance"], rtol=0.05)


def test_continuous_generalization_eval_matches_exact_objective_and_reacts_to_overrides():
    env = LQ(device="cpu")
    theta = env.riccati_optimal(3)
    results = continuous_generalization_eval(
        env,
        theta,
        [{"name": "baseline"}, {"name": "mu0 x 2", "env": LQ(LQConfig(mu0=2 * env.config.mu0), device="cpu")}, {"name": "lam=0.8", "lam": 0.8}],
        lam=0.0,
    )
    assert results[0]["J"].item() == pytest.approx(env.exact_objective(theta, 0.0).item())
    assert results[1]["J"].item() > results[0]["J"].item()  # a larger population mean costs more
    assert results[2]["J"].item() > results[0]["J"].item()  # so does a stronger perturbation


def test_run_continuous_diagnostics_end_to_end(tmp_path):
    from configs.lq import LQRunConfig
    import scripts.train as train_mod

    cfg = LQRunConfig(name="tiny", horizons=(2,), lambdas=(0.2,), n_train=5, seeds=(0, 1), B=16, n_aux=8, validate_every=2)
    train_mod.CONFIGS.setdefault("lq", {})["tiny_test"] = cfg

    run_continuous("lq", "simplex", "tiny_test", output_dir=str(tmp_path))
    runs = load_runs("lq", "simplex", "tiny_test", output_dir=str(tmp_path), device="cpu")
    assert len(runs) == 2

    diagnostics = run_continuous_diagnostics("lq", "simplex", "tiny_test", output_dir=str(tmp_path))
    assert (tmp_path / "lq" / "tiny_test" / "simplex_diagnostics.pt").exists()
    entry = diagnostics[(2, 0.2)]
    assert entry["n_seeds"] == 2
    assert entry["theta"]["mean"].shape == (4,)  # theta is (T,2), flattened for the across-seed statistics
    for key in ("objective_gap", "gradient", "sensitivity", "perturbation_coverage", "state_marginal"):
        assert key in entry


def test_continuous_mean_field_term_matches_reinforces_own_bias():
    """The paired difference measures exactly what REINFORCE omits, so it
    must agree (up to sign and the O(lambda^p) perturbation term) with
    REINFORCE's directly-measured bias — while being far less noisy."""
    env = LQ(device="cpu")
    T, lam, reps, B = 5, 0.2, 60, 2_000
    theta_star = env.riccati_optimal(T)

    mf = continuous_mean_field_term(env, theta_star, lam=lam, B=B, reps=reps, baseline="loo", generator=torch.Generator(device="cpu").manual_seed(0))
    gd = continuous_gradient_diagnostics(
        env, theta_star, lam=lam, B=B, reps=reps, algorithm="reinforce", baseline="loo", generator=torch.Generator(device="cpu").manual_seed(1)
    )

    assert mf["mean"].norm().item() > 4.0 * mf["se"].norm().item()  # resolved
    assert mf["se"].norm().item() < gd["bias_se"].norm().item()  # and less noisy than the direct measurement
    combined_se = (mf["se"] ** 2 + gd["bias_se"] ** 2).sum().sqrt().item()
    assert (mf["mean"] + gd["bias"]).norm().item() < 4.0 * combined_se  # bias = -term
