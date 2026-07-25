import pytest

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mfc.algorithms import (
    AdaptiveSimplexControllerConfig,
    ConsistentAdaptiveSimplexMFREINFORCE,
    FiniteBudgetAdaptiveSimplexMFREINFORCE,
    LogitsPerturbedMFREINFORCE,
    SimplexPerturbedMFREINFORCE,
)
from mfc.environments import (
    CybersecurityConfig,
    CybersecurityMFC,
    CybersecurityPolicy,
    DistributionPlanningConfig,
    DistributionPlanningMFC,
    DistributionPlanningPolicy,
    TwoStateConfig,
    TwoStateMFC,
)
from mfc.experiments.twostate import reference_metrics, simulator_transitions_per_update
from mfc.experiments.twostate_adaptive_comparison import (
    TwoStateAdaptiveComparisonConfig,
    run_twostate_adaptive_comparison,
    tiny_twostate_adaptive_comparison_config,
)


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def test_twostate_batch_helpers_match_scalar_formulas():
    config = TwoStateConfig(device=DEVICE, dtype=DTYPE)
    env = TwoStateMFC(config)
    theta = torch.tensor([0.2, -0.3], dtype=DTYPE)
    mu = torch.tensor([0.35, 0.65], dtype=DTYPE)
    mus = torch.stack([mu, torch.flip(mu, dims=[0])])

    assert torch.allclose(env.action_probabilities(theta, 0, mu), env.policy_probs(theta))
    assert torch.allclose(env.action_probabilities(theta, 0, mus), env.policy_probs(theta).expand(2, 2, 2))
    for action in range(env.n_actions):
        for state in range(env.n_states):
            assert torch.allclose(env.transition_tensor(mu)[action, state], env.transition_probs(state, action, mu))

    states = torch.tensor([0, 1, 1])
    actions = torch.tensor([1, 0, 1])
    scores = env.policy_scores_batch(theta, 0, mu, states, actions).reshape(3, 2)
    scalar_scores = torch.stack([env.policy_score(theta, 0, mu, int(s), int(a)) for s, a in zip(states, actions)])
    assert torch.allclose(scores, scalar_scores)


def test_cybersecurity_batch_helpers_match_scalar_calls():
    config = CybersecurityConfig(device=DEVICE, dtype=DTYPE, hidden_units=8)
    env = CybersecurityMFC(config)
    policy = CybersecurityPolicy(config)
    mus = torch.stack(
        [
            torch.full((config.n_states,), 1.0 / config.n_states, dtype=DTYPE),
            torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DTYPE),
        ]
    )

    probs = env.action_probabilities(policy, 1, mus)
    scalar_probs = torch.stack([env.action_probabilities(policy, 1, mu) for mu in mus])
    assert torch.allclose(probs, scalar_probs)

    transitions = env.transition_tensor(mus)
    scalar_transitions = torch.stack(
        [torch.stack([env.transition_matrix(mu, action) for action in range(env.n_actions)]) for mu in mus]
    )
    assert torch.allclose(transitions, scalar_transitions)

    states = torch.tensor([0, 3])
    actions = torch.tensor([1, 0])
    scores = env.policy_scores_batch(policy, 1, mus, states, actions, chunk_size=1)
    scalar_scores = torch.stack([env.policy_score(policy, 1, mus[i], int(states[i]), int(actions[i])) for i in range(2)])
    assert torch.allclose(scores, scalar_scores)


def test_distribution_batch_helpers_match_scalar_calls_and_deterministic_next_state():
    config = DistributionPlanningConfig(device=DEVICE, dtype=DTYPE, hidden_units=8)
    env = DistributionPlanningMFC(config)
    policy = DistributionPlanningPolicy(config)
    mus = torch.stack(
        [
            torch.full((config.n_states,), 1.0 / config.n_states, dtype=DTYPE),
            torch.linspace(1.0, 2.0, config.n_states, dtype=DTYPE),
        ]
    )
    mus[1] = mus[1] / mus[1].sum()

    probs = env.action_probabilities(policy, 2, mus)
    scalar_probs = torch.stack([env.action_probabilities(policy, 2, mu) for mu in mus])
    assert torch.allclose(probs, scalar_probs)

    for action in range(env.n_actions):
        for state in range(env.n_states):
            assert torch.allclose(env.transition_tensor(mus)[0, action, state], env.transition_probs(state, action, mus[0]))

    states = torch.tensor([0, 5, 9])
    actions = torch.tensor([0, 1, 2])
    expected_next = torch.tensor([9, 5, 0])
    assert torch.equal(env.sample_next_states_batch(states, actions, mus[0]), expected_next)

    scores = env.policy_scores_batch(policy, 2, mus[:2], states[:2], actions[:2], chunk_size=1)
    scalar_scores = torch.stack([env.policy_score(policy, 2, mus[i], int(states[i]), int(actions[i])) for i in range(2)])
    assert torch.allclose(scores, scalar_scores)


def test_exact_distribution_value_matches_manual_loop():
    config = DistributionPlanningConfig(device=DEVICE, dtype=DTYPE, hidden_units=8, T=3)
    env = DistributionPlanningMFC(config)
    policy = DistributionPlanningPolicy(config)
    mu = torch.full((config.n_states,), 1.0 / config.n_states, dtype=DTYPE)

    value = env.exact_value(policy, mu, config.T)
    manual_value = torch.zeros((), dtype=DTYPE)
    manual_mu = mu
    for t in range(config.T):
        pi = policy.probs(t, manual_mu)
        step_reward = torch.zeros((), dtype=DTYPE)
        for state in range(config.n_states):
            for action in range(config.n_actions):
                step_reward = step_reward + manual_mu[state] * pi[state, action] * env.reward(state, manual_mu, action)
        manual_value = manual_value + step_reward
        manual_mu = manual_mu @ env.averaged_kernel(policy, t, manual_mu)
    manual_value = manual_value + env.terminal_reward(0, manual_mu)

    assert torch.allclose(value, manual_value, atol=1e-10, rtol=1e-10)


def test_algorithm_smoke_outputs_are_finite_for_notebook02_environments():
    cases = [
        (TwoStateConfig(device=DEVICE, dtype=DTYPE, T=2), TwoStateMFC, None, 4, 2),
        (CybersecurityConfig(device=DEVICE, dtype=DTYPE, hidden_units=8, T_train=2), CybersecurityMFC, CybersecurityPolicy, 4, 1),
        (DistributionPlanningConfig(device=DEVICE, dtype=DTYPE, hidden_units=8, T=2), DistributionPlanningMFC, DistributionPlanningPolicy, 4, 1),
    ]

    for config, env_cls, policy_cls, flow_particles, n_inner in cases:
        torch.manual_seed(123)
        env = env_cls(config)
        control = (
            torch.zeros(env.n_states, dtype=DTYPE)
            if policy_cls is None
            else policy_cls(config)
        )
        horizon = getattr(config, "T_train", getattr(config, "T", None))
        mu0 = torch.full((env.n_states,), 1.0 / env.n_states, dtype=DTYPE)

        simplex = SimplexPerturbedMFREINFORCE(env)
        flow = env.exact_population_flow(control, mu0, horizon=horizon)
        sensitivity = simplex.estimate_sensitivity(control, flow, eta=0.2, n_aux=2)
        simplex_grad, simplex_diag = simplex.gradient_estimate(control, flow, sensitivity, eps_law=0.2, B=2)
        assert torch.isfinite(simplex_grad).all()
        assert torch.isfinite(simplex_diag["mean_return"])
        assert torch.isfinite(simplex_diag["grad_norm"])

        logits = LogitsPerturbedMFREINFORCE(env)
        logits_grad, logits_diag = logits.gradient_estimate(
            control,
            mu0,
            epsilon=0.2,
            N=2,
            n=n_inner,
            flow_particles=flow_particles,
            horizon=horizon,
            mu_flow=flow,
        )
        assert torch.isfinite(logits_grad).all()
        assert torch.isfinite(logits_diag["mean_return"])
        assert torch.isfinite(logits_diag["grad_norm"])


def test_batched_population_flow_is_close_to_exact_twostate_flow():
    torch.manual_seed(0)
    config = TwoStateConfig(device=DEVICE, dtype=DTYPE)
    env = TwoStateMFC(config)
    theta = torch.tensor([0.2, -0.1], dtype=DTYPE)
    mu0 = torch.tensor([0.25, 0.75], dtype=DTYPE)
    flow = LogitsPerturbedMFREINFORCE(env).estimate_population_flow(theta, mu0, n_particles=10_000)
    exact_flow = env.exact_population_flow(theta, mu0)

    assert torch.max(torch.abs(flow - exact_flow)) < 0.03


def test_notebook_twostate_flow_error_matches_reference_definition():
    config = TwoStateConfig(device=DEVICE, dtype=DTYPE)
    env = TwoStateMFC(config)
    theta = torch.zeros(env.n_states, dtype=DTYPE)
    mu0 = torch.tensor([0.2, 0.8], dtype=DTYPE)

    metrics = reference_metrics(env, theta, mu0, config.T)
    flow = env.exact_population_flow(theta, mu0, config.T)
    expected = (flow[1:, 1] - env.target_B[1]).abs().mean()
    old_notebook_metric = (flow[:, 1] - env.target_B[1]).abs().mean()

    assert metrics["flow_error"] == pytest.approx(float(expected))
    assert metrics["flow_error"] != pytest.approx(float(old_notebook_metric))


def test_notebook_simulator_cost_formulas_expose_equal_n_n_cost_gap():
    assert simulator_transitions_per_update("Simplex", horizon=2, B=200, n_aux_or_inner=10) == 420
    assert simulator_transitions_per_update("Logits", horizon=2, B=200, n_aux_or_inner=10) == 6400
    assert simulator_transitions_per_update("Simplex", horizon=2, B=2800, n_aux_or_inner=400) == 6400


def test_twostate_adaptive_comparison_uses_equal_simulator_budgets_by_default():
    config = TwoStateAdaptiveComparisonConfig(device=DEVICE, dtype=DTYPE)

    simplex_budget = config.budget_for_method("fixed_simplex")
    finite_budget = config.budget_for_method("finite_adaptive")
    consistent_budget = config.budget_for_method("consistent_adaptive")
    logit_budget = config.budget_for_method("logits")

    assert simplex_budget == {"B": 2800, "n": 400}
    assert finite_budget == simplex_budget
    assert consistent_budget == simplex_budget
    assert logit_budget == {"B": 200, "n": 10}
    assert simulator_transitions_per_update("Simplex", horizon=2, **{"B": simplex_budget["B"], "n_aux_or_inner": simplex_budget["n"]}) == (
        simulator_transitions_per_update("Logits", horizon=2, B=logit_budget["B"], n_aux_or_inner=logit_budget["n"])
    )


def test_twostate_adaptive_comparison_can_sweep_adaptive_initial_lambdas():
    config = tiny_twostate_adaptive_comparison_config(DEVICE, DTYPE)
    config.adaptive_initial_lambdas = (0.05, 0.1)

    results = run_twostate_adaptive_comparison(config, show_progress=False)

    assert set(results["Finite Adaptive Simplex"]) == {0.05, 0.1}
    assert set(results["Consistent Adaptive Simplex"]) == {0.05, 0.1}
    assert results["Finite Adaptive Simplex"][0.05][0]["history"]["lambda"][0] == pytest.approx(0.05)
    assert results["Consistent Adaptive Simplex"][0.1][0]["history"]["lambda"][0] == pytest.approx(0.1)


def test_finite_budget_adaptive_simplex_smoke_has_finite_diagnostics_and_bounds():
    torch.manual_seed(321)
    config = TwoStateConfig(device=DEVICE, dtype=DTYPE, T=2)
    env = TwoStateMFC(config)
    theta = torch.tensor([0.1, -0.2], dtype=DTYPE)
    mu0 = torch.tensor([0.3, 0.7], dtype=DTYPE)
    flow = env.exact_population_flow(theta, mu0, config.T)
    controller_config = AdaptiveSimplexControllerConfig(
        initial_lambda=0.2,
        lambda_min=0.05,
        lambda_max=0.6,
        checkpoint_interval=1,
        diagnostic_replications=2,
        controller_lr=0.02,
    )
    algorithm = FiniteBudgetAdaptiveSimplexMFREINFORCE(env, controller_config)

    grad, diag = algorithm.gradient_estimate(theta, flow, iteration=0, B=3, n_aux=2)

    assert torch.isfinite(grad).all()
    assert torch.isfinite(diag["grad_norm"])
    assert controller_config.lambda_min <= algorithm.lambda_ctrl <= controller_config.lambda_max
    assert diag["checkpoint"] is True
    assert diag["controller"] is not None
    assert torch.isfinite(torch.tensor(diag["controller"]["signed_pressure"], dtype=DTYPE))
    assert diag["controller"]["bias_proxy_sq"] >= 0.0
    assert diag["controller"]["variance_proxy"] >= 0.0


def test_consistent_adaptive_simplex_envelope_schedules_and_gradient_are_finite():
    torch.manual_seed(654)
    config = TwoStateConfig(device=DEVICE, dtype=DTYPE, T=2)
    env = TwoStateMFC(config)
    theta = torch.tensor([-0.1, 0.25], dtype=DTYPE)
    mu0 = torch.tensor([0.4, 0.6], dtype=DTYPE)
    flow = env.exact_population_flow(theta, mu0, config.T)
    controller_config = AdaptiveSimplexControllerConfig(
        initial_lambda=0.4,
        lambda_min=0.05,
        lambda_max=0.8,
        checkpoint_interval=1,
        diagnostic_replications=2,
        eta_power=1.5,
        main_sample_growth_power=0.25,
        aux_sample_growth_power=0.5,
        sample_growth_interval=10.0,
        controller_lr=0.02,
    )
    algorithm = ConsistentAdaptiveSimplexMFREINFORCE(env, controller_config)

    lambda_0 = algorithm.current_lambda(0)
    lambda_later = algorithm.current_lambda(30)
    B_0, n_0 = algorithm.sample_sizes(0, B=3, n_aux=2)
    B_later, n_later = algorithm.sample_sizes(30, B=3, n_aux=2)
    grad, diag = algorithm.gradient_estimate(theta, flow, iteration=30, B=3, n_aux=2)

    assert lambda_later < lambda_0
    assert B_later >= B_0
    assert n_later >= n_0
    assert float(diag["eta"].item()) == pytest.approx(float(diag["lambda"].item()) ** 1.5)
    assert torch.isfinite(grad).all()
    assert torch.isfinite(diag["grad_norm"])
    assert diag["controller"] is not None


def test_distribution_planning_uses_reference_target_distribution():
    config = DistributionPlanningConfig(device=DEVICE, dtype=DTYPE, hidden_units=8)
    env = DistributionPlanningMFC(config)
    expected = torch.tensor(
        [0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.0],
        dtype=DTYPE,
    )

    assert torch.allclose(env.target, expected)
