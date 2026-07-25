from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mfc.environments import TwoStateConfig, TwoStateMFC


def set_seed(seed: int, device: Optional[torch.device] = None) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


def fixed_validation_law(config: TwoStateConfig) -> torch.Tensor:
    return torch.tensor([0.2, 0.8], dtype=config.dtype, device=config.device)


def sample_twostate_initial_laws(config: TwoStateConfig, count: int) -> torch.Tensor:
    mu1 = config.low + (config.high - config.low) * torch.rand(
        count,
        dtype=config.dtype,
        device=config.device,
    )
    return torch.stack([1.0 - mu1, mu1], dim=-1).detach().cpu()


def prepare_twostate_run_plans(
    config: TwoStateConfig,
    seed_base: int = 11_000,
    training_runs: Optional[int] = None,
) -> List[Dict[str, object]]:
    n_runs = config.training_runs if training_runs is None else int(training_runs)
    plans: List[Dict[str, object]] = []
    for run_idx in range(n_runs):
        seed = seed_base + run_idx
        set_seed(seed, config.device)
        plans.append(
            {
                "run_idx": run_idx,
                "seed": seed,
                "initial_control": {"theta": torch.zeros(2, dtype=config.dtype, device=config.device).cpu()},
                "initial_laws": sample_twostate_initial_laws(config, config.n_train),
            }
        )
    return plans


def theta_for_estimator(theta):
    return theta.detach() if isinstance(theta, torch.nn.Parameter) else theta


def assign_ascent_gradient(theta: torch.nn.Parameter, grad_hat: torch.Tensor) -> None:
    theta.grad = -grad_hat.detach().clone().reshape_as(theta)


def reference_metrics(env: TwoStateMFC, theta, mu0: torch.Tensor, horizon: int) -> Dict[str, object]:
    ctrl = theta_for_estimator(theta)
    with torch.no_grad():
        flow = env.exact_population_flow(ctrl, mu0, horizon).detach()
        value = env.exact_value(ctrl, mu0, horizon).detach()
        policy = env.policy_probs(ctrl).detach()
        optimal = env.optimal_policy().detach()
        st_errors = (policy[:, 0] - optimal[:, 0]).abs()
        flow_error = (flow[1:, 1] - env.target_B[1]).abs().mean()
    return {
        "value": float(value.item()),
        "policy": policy.cpu(),
        "optimal_policy": optimal.cpu(),
        "final_distribution": flow[-1].cpu(),
        "flow": flow.cpu(),
        "policy_error_st_0": float(st_errors[0].item()),
        "policy_error_st_1": float(st_errors[1].item()),
        "policy_error_mean": float(st_errors.mean().item()),
        "flow_error": float(flow_error.item()),
        "err_pi_st_0": float(st_errors[0].item()),
        "err_pi_st_1": float(st_errors[1].item()),
    }


def exact_gradient(
    env: TwoStateMFC,
    theta: torch.Tensor,
    mu0: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta_var = theta.detach().clone().requires_grad_(True)
    value = env.exact_value(theta_var, mu0, horizon)
    (grad,) = torch.autograd.grad(value, theta_var)
    return value.detach(), grad.detach().reshape(-1)


def cosine_similarity_flat(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((x.flatten() @ y.flatten() / denom).item())


def simulator_transitions_per_update(
    algorithm_name: str,
    horizon: int,
    B: int,
    n_aux_or_inner: int,
    flow_mode: str = "exact",
    flow_particles: int = 0,
    diagnostic_replications: int = 0,
    checkpoint_interval: int = 0,
) -> int:
    normalized = algorithm_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"simplex", "fixedsimplex"}:
        core = (int(B) + int(n_aux_or_inner)) * int(horizon)
    elif normalized in {"finitebudgetadaptivesimplex", "consistentadaptivesimplex", "adaptivesimplex"}:
        core = (int(B) + int(n_aux_or_inner)) * int(horizon)
        if diagnostic_replications > 0 and checkpoint_interval > 0:
            core = core * (1.0 + 2.0 * int(diagnostic_replications) / int(checkpoint_interval))
    elif normalized == "logits":
        core = int(B) * int(horizon) + int(B) * int(n_aux_or_inner) * int(horizon) * (int(horizon) + 1) // 2
    else:
        raise ValueError(f"Unknown algorithm_name={algorithm_name!r}.")

    flow_cost = 0 if flow_mode == "exact" else int(flow_particles) * int(horizon)
    return int(math.ceil(core + flow_cost))


def training_population_flow(
    env: TwoStateMFC,
    algorithm,
    theta,
    mu0: torch.Tensor,
    horizon: int,
    flow_mode: str,
    flow_particles: int,
) -> torch.Tensor:
    ctrl = theta_for_estimator(theta)
    if flow_mode == "exact":
        with torch.no_grad():
            return env.exact_population_flow(ctrl, mu0, horizon).detach()
    if flow_mode == "particle":
        return algorithm.estimate_population_flow(ctrl, mu0, flow_particles, horizon=horizon).detach()
    raise ValueError(f"Unknown flow_mode={flow_mode!r}.")
