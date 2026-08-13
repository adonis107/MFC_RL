from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from ..core.evaluation import _scalar
from .common import _snapshot_times


def _cucker_smale_outputs(env: Any, control: Any, evaluation_config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    particles = int(evaluation_config.get("particles", getattr(env.config, "N_val", 256)))
    horizon = evaluation_config.get("horizon")
    lambda_value = float(evaluation_config.get("lambda", 0.0))
    rollout = env.sample_trajectories(
        control,
        particles,
        seed=seed,
        lambda_=lambda_value,
        horizon=None if horizon is None else int(horizon),
        exploration=False,
    )
    initial = rollout["state_flow"][0]
    free = env.sample_trajectories(
        env.free_controller(),
        particles,
        seed=seed + 1,
        lambda_=0.0,
        initial_states=initial,
        horizon=None if horizon is None else int(horizon),
        exploration=False,
    )
    time_rows = _cucker_time_rows("controlled", rollout) + _cucker_time_rows("free", free)
    grid_rows: List[Dict[str, Any]] = []
    heuristic_metrics: Dict[str, Any] = {}
    if "heuristic_kappa_grid" in evaluation_config:
        heuristic_particles = int(evaluation_config.get("heuristic_particles", min(particles, 512)))
        kappa_grid = torch.as_tensor(evaluation_config["heuristic_kappa_grid"], dtype=env.config.dtype, device=env.config.device)
        search = env.grid_search_alignment_controller(
            kappa_grid,
            n_particles=max(1, heuristic_particles),
            seed=seed + 3,
            horizon=None if horizon is None else int(horizon),
        )
        best_kappa = float(search["best_kappa"].item())
        heuristic = env.sample_trajectories(
            env.global_alignment_controller(best_kappa),
            particles,
            seed=seed + 4,
            lambda_=0.0,
            initial_states=initial,
            horizon=None if horizon is None else int(horizon),
            exploration=False,
        )
        time_rows.extend(_cucker_time_rows("heuristic", heuristic))
        grid_rows = [
            {
                "method": "global_alignment_grid_search",
                "kappa": float(kappa.item()),
                "objective": float(objective.item()),
            }
            for kappa, objective in zip(search["kappa_grid"], search["objectives"])
        ]
        heuristic_metrics = {
            "heuristic_objective": _scalar(heuristic["objective"]),
            "heuristic_best_kappa": best_kappa,
            "heuristic_particles": max(1, heuristic_particles),
        }
    post_horizon = int(evaluation_config.get("post_control_horizon", max(1, min(10, rollout["state_flow"].shape[0] - 1))))
    post = env.continue_uncontrolled(rollout["state_flow"][-1], post_horizon, seed=seed + 2)
    metrics = {
        "objective": _scalar(rollout["objective"]),
        "free_objective": _scalar(free["objective"]),
        "alignment_time": env.alignment_time(rollout["velocity_dispersion"], threshold=float(evaluation_config.get("alignment_threshold", 0.1))),
        "cumulative_control_energy": _scalar(rollout["cumulative_control_energy"]),
    }
    metrics.update(heuristic_metrics)
    return {
        "time_metrics": time_rows,
        "snapshots": _state_snapshot_rows("cucker-smale", rollout["state_flow"]),
        "post_control": _cucker_time_rows("post_control_free", post),
        "heuristic_grid": grid_rows,
        "metrics": metrics,
    }



def _cucker_time_rows(label: str, rollout: Mapping[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows = []
    velocity = rollout["velocity_dispersion"].detach().cpu()
    diameter = rollout["spatial_diameter"].detach().cpu()
    control = rollout.get("control_energy")
    for t in range(velocity.numel()):
        row = {
            "method": label,
            "time": t,
            "velocity_dispersion": float(velocity[t].item()),
            "spatial_diameter": float(diameter[t].item()),
        }
        if control is not None and t < control.numel():
            row["control_energy"] = float(control[t].detach().cpu().item())
        rows.append(row)
    return rows



def _kuramoto_outputs(env: Any, control: Any, evaluation_config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    particles = int(evaluation_config.get("particles", getattr(env.config, "N_val", 256)))
    horizon = evaluation_config.get("horizon")
    lambda_value = float(evaluation_config.get("lambda", 0.0))
    rollout = env.sample_trajectories(
        control,
        particles,
        seed=seed,
        lambda_=lambda_value,
        horizon=None if horizon is None else int(horizon),
        exploration=False,
    )
    initial = rollout["lifted_phase_flow"][0]
    free = env.sample_trajectories(
        env.free_controller(),
        particles,
        seed=seed + 1,
        lambda_=0.0,
        initial_phases=initial,
        frequencies=rollout["frequencies"],
        horizon=None if horizon is None else int(horizon),
        exploration=False,
    )
    time_rows = _kuramoto_time_rows("controlled", rollout) + _kuramoto_time_rows("free", free)
    grid_rows: List[Dict[str, Any]] = []
    heuristic_metrics: Dict[str, Any] = {}
    if "heuristic_kappa_grid" in evaluation_config and "heuristic_nu_grid" in evaluation_config:
        heuristic_particles = int(evaluation_config.get("heuristic_particles", min(particles, 512)))
        kappa_grid = torch.as_tensor(evaluation_config["heuristic_kappa_grid"], dtype=env.config.dtype, device=env.config.device)
        nu_grid = torch.as_tensor(evaluation_config["heuristic_nu_grid"], dtype=env.config.dtype, device=env.config.device)
        search = env.grid_search_base_controller(
            kappa_grid,
            nu_grid,
            n_particles=max(1, heuristic_particles),
            seed=seed + 3,
            horizon=None if horizon is None else int(horizon),
        )
        best_kappa = float(search["best_kappa"].item())
        best_nu = float(search["best_nu"].item())
        heuristic = env.sample_trajectories(
            env.base_controller(best_kappa, best_nu),
            particles,
            seed=seed + 4,
            lambda_=0.0,
            initial_phases=initial,
            frequencies=rollout["frequencies"],
            horizon=None if horizon is None else int(horizon),
            exploration=False,
        )
        time_rows.extend(_kuramoto_time_rows("heuristic", heuristic))
        objectives = search["objectives"]
        grid_rows = [
            {
                "method": "base_controller_grid_search",
                "kappa": float(search["kappa_grid"][i].item()),
                "nu": float(search["nu_grid"][j].item()),
                "objective": float(objectives[i, j].item()),
            }
            for i in range(search["kappa_grid"].numel())
            for j in range(search["nu_grid"].numel())
        ]
        heuristic_metrics = {
            "heuristic_objective": _scalar(heuristic["objective"]),
            "heuristic_best_kappa": best_kappa,
            "heuristic_best_nu": best_nu,
            "heuristic_particles": max(1, heuristic_particles),
        }
    post_horizon = int(evaluation_config.get("post_control_horizon", max(1, min(10, rollout["phase_flow"].shape[0] - 1))))
    post = env.continue_uncontrolled(
        rollout["lifted_phase_flow"][-1],
        post_horizon,
        seed=seed + 2,
        frequencies=rollout["frequencies"],
    )
    metrics = {
        "objective": _scalar(rollout["objective"]),
        "free_objective": _scalar(free["objective"]),
        "synchronization_time": env.synchronization_time(
            rollout["order_parameter"], threshold=float(evaluation_config.get("synchronization_threshold", 0.9))
        ),
        "phase_locking_time": env.phase_locking_time(
            rollout["target_aligned_order"], threshold=float(evaluation_config.get("phase_locking_threshold", 0.85))
        ),
        "cumulative_control_energy": _scalar(rollout["cumulative_control_energy"]),
    }
    metrics.update(heuristic_metrics)
    return {
        "time_metrics": time_rows,
        "snapshots": _phase_snapshot_rows("kuramoto", rollout["phase_flow"]),
        "post_control": _kuramoto_time_rows("post_control_free", post),
        "heuristic_grid": grid_rows,
        "metrics": metrics,
    }



def _kuramoto_time_rows(label: str, rollout: Mapping[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows = []
    order = rollout["order_parameter"].detach().cpu()
    aligned = rollout["target_aligned_order"].detach().cpu()
    sync_cost = rollout["synchronization_cost"].detach().cpu()
    control = rollout.get("control_energy")
    for t in range(order.numel()):
        row = {
            "method": label,
            "time": t,
            "order_parameter": float(order[t].item()),
            "target_aligned_order": float(aligned[t].item()),
            "synchronization_cost": float(sync_cost[t].item()),
        }
        if control is not None and t < control.numel():
            row["control_energy"] = float(control[t].detach().cpu().item())
        rows.append(row)
    return rows



def _state_snapshot_rows(env_name: str, state_flow: torch.Tensor) -> List[Dict[str, Any]]:
    rows = []
    times = _snapshot_times(state_flow.shape[0])
    flow = state_flow.detach().cpu()
    for t in times:
        for particle in range(flow.shape[1]):
            rows.append(
                {
                    "env": env_name,
                    "time": t,
                    "particle": particle,
                    "position": float(flow[t, particle, 0].item()),
                    "velocity": float(flow[t, particle, 1].item()),
                }
            )
    return rows



def _phase_snapshot_rows(env_name: str, phase_flow: torch.Tensor) -> List[Dict[str, Any]]:
    rows = []
    times = _snapshot_times(phase_flow.shape[0])
    flow = phase_flow.detach().cpu()
    for t in times:
        for particle in range(flow.shape[1]):
            phase = float(flow[t, particle].item())
            rows.append(
                {
                    "env": env_name,
                    "time": t,
                    "particle": particle,
                    "phase": phase,
                    "cos_phase": float(torch.cos(torch.tensor(phase)).item()),
                    "sin_phase": float(torch.sin(torch.tensor(phase)).item()),
                }
            )
    return rows


__all__: list[str] = []
