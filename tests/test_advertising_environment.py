import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mfc.algorithms import LogitsPerturbedMFREINFORCE, SimplexPerturbedMFREINFORCE
from mfc.environments import AdvertisingConfig, AdvertisingMFC, AdvertisingPolicy


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _constant_half_policy(config: AdvertisingConfig) -> AdvertisingPolicy:
    policy = AdvertisingPolicy(config)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    return policy


def test_advertising_transition_and_exact_value_match_scalar_recursion():
    config = AdvertisingConfig(device=DEVICE, dtype=DTYPE, hidden_units=4, T=3)
    env = AdvertisingMFC(config)
    policy = _constant_half_policy(config)
    mu = torch.tensor([0.3, 0.7], dtype=DTYPE)

    transitions = env.transition_tensor(mu)
    assert torch.allclose(transitions[0], torch.tensor([[0.3, 0.7], [0.3, 0.7]], dtype=DTYPE))
    assert torch.allclose(transitions[1], torch.tensor([[0.1, 0.9], [0.1, 0.9]], dtype=DTYPE))

    flow = env.exact_population_flow(policy, mu, config.T)
    manual_p = torch.tensor(0.7, dtype=DTYPE)
    manual_flow = [torch.stack([1.0 - manual_p, manual_p])]
    manual_value = torch.zeros((), dtype=DTYPE)
    discount = 1.0
    for _ in range(config.T):
        q_ad = torch.tensor(0.5, dtype=DTYPE)
        manual_value = manual_value + discount * (manual_p - config.c_ad * q_ad)
        manual_p = manual_p + q_ad * min(config.kappa_ad, 1.0 - float(manual_p))
        manual_flow.append(torch.stack([1.0 - manual_p, manual_p]))
        discount *= config.gamma

    assert torch.allclose(flow, torch.stack(manual_flow))
    assert torch.allclose(env.exact_value(policy, mu, config.T), manual_value)


def test_advertising_batch_helpers_and_weighted_score_reducer_match_scalar_calls():
    torch.manual_seed(42)
    config = AdvertisingConfig(device=DEVICE, dtype=DTYPE, hidden_units=4, T=2)
    env = AdvertisingMFC(config)
    policy = AdvertisingPolicy(config)
    mus = torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]], dtype=DTYPE)
    states = torch.tensor([0, 1, 0])
    actions = torch.tensor([1, 0, 1])

    probs = env.action_probabilities(policy, 1, mus)
    scalar_probs = torch.stack([env.action_probabilities(policy, 1, mu) for mu in mus])
    assert torch.allclose(probs, scalar_probs)
    assert torch.allclose(probs[:, 0], probs[:, 1])

    scores = env.policy_scores_batch(policy, 1, mus, states, actions, chunk_size=1).reshape(states.numel(), -1)
    scalar_scores = torch.stack(
        [env.policy_score(policy, 1, mus[i], int(states[i]), int(actions[i])) for i in range(states.numel())]
    )
    assert torch.allclose(scores, scalar_scores)

    weights = torch.linspace(-0.2, 0.4, states.numel(), dtype=DTYPE)
    expected = weights @ scores
    actual = env.weighted_policy_score_sums(policy, 1, mus, states, actions, weights, chunk_size=2)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)

    grouped_weights = torch.stack([weights, weights.square() + 0.1])
    grouped_expected = grouped_weights @ scores
    grouped_actual = env.weighted_policy_score_sums(policy, 1, mus, states, actions, grouped_weights, chunk_size=2)
    assert torch.allclose(grouped_actual, grouped_expected, atol=1e-10, rtol=1e-10)


def test_advertising_reference_policy_uses_benchmark_piecewise_case():
    config = AdvertisingConfig(device=DEVICE, dtype=DTYPE)
    env = AdvertisingMFC(config)
    p = torch.tensor([0.55, 0.65, 0.75, 0.90], dtype=DTYPE)
    expected = torch.tensor([1.0, 0.75, 1.0, 0.0], dtype=DTYPE)

    assert torch.allclose(env.infinite_horizon_reference_policy(p), expected)


def test_advertising_one_step_dp_oracle_has_zero_advertising_policy():
    config = AdvertisingConfig(device=DEVICE, dtype=DTYPE, hidden_units=4, T=1)
    env = AdvertisingMFC(config)

    oracle = env.finite_horizon_dp_oracle(grid_size=21, action_grid_size=21)

    assert torch.allclose(oracle["values"][1], torch.zeros_like(oracle["values"][1]))
    assert torch.allclose(oracle["values"][0], oracle["p_grid"])
    assert torch.allclose(oracle["policy"][0], torch.zeros_like(oracle["policy"][0]))


def test_advertising_mfreinforce_algorithm_smoke_outputs_are_finite():
    torch.manual_seed(123)
    config = AdvertisingConfig(device=DEVICE, dtype=DTYPE, hidden_units=4, T=2)
    env = AdvertisingMFC(config)
    policy = AdvertisingPolicy(config)
    mu0 = torch.tensor([0.4, 0.6], dtype=DTYPE)
    flow = env.exact_population_flow(policy, mu0, config.T)

    simplex = SimplexPerturbedMFREINFORCE(env)
    sensitivity = simplex.estimate_sensitivity(policy, flow, eta=0.2, n_aux=2)
    simplex_grad, simplex_diag = simplex.gradient_estimate(policy, flow, sensitivity, eps_law=0.2, B=2)
    assert torch.isfinite(simplex_grad).all()
    assert torch.isfinite(simplex_diag["mean_return"])

    logits = LogitsPerturbedMFREINFORCE(env)
    logits_grad, logits_diag = logits.gradient_estimate(
        policy,
        mu0,
        epsilon=0.2,
        N=2,
        n=1,
        flow_particles=4,
        horizon=config.T,
        mu_flow=flow,
    )
    assert torch.isfinite(logits_grad).all()
    assert torch.isfinite(logits_diag["mean_return"])
