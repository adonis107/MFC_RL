from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mfc.algorithms import (
    AdaptiveSimplexControllerConfig,
    ConsistentAdaptiveSimplexMFREINFORCE,
    FiniteBudgetAdaptiveSimplexMFREINFORCE,
    LogitsPerturbedMFREINFORCE,
    SimplexPerturbedMFREINFORCE,
)
from mfc.environments import TwoStateConfig, TwoStateMFC
from mfc.experiments.twostate import (
    assign_ascent_gradient,
    fixed_validation_law,
    prepare_twostate_run_plans,
    reference_metrics,
    set_seed,
    simulator_transitions_per_update,
    theta_for_estimator,
    training_population_flow,
)


@dataclass
class TwoStateAdaptiveComparisonConfig:
    device: torch.device
    dtype: torch.dtype
    n_train: int = 5_000
    training_runs: int = 5
    validate_every: int = 10
    B: int = 2800
    n_aux: int = 400
    logit_B: int = 200
    logit_n: int = 10
    lr: float = 1e-3
    train_horizon: int = 2
    validation_horizon: int = 2
    flow_mode: str = "exact"
    flow_particles: int = 0
    seed_base: int = 31_000
    simplex_lambdas: Sequence[float] = (0.1, 0.2, 0.4, 0.8)
    logit_epsilons: Sequence[float] = (0.2, 0.5, 1.0, 2.0)
    adaptive_initial_lambdas: Sequence[float] = (0.4,)
    finite_controller: AdaptiveSimplexControllerConfig = field(
        default_factory=lambda: AdaptiveSimplexControllerConfig(
            initial_lambda=0.4,
            lambda_min=0.01,
            lambda_max=0.8,
            checkpoint_interval=100,
            diagnostic_replications=4,
            contraction=0.5,
        )
    )
    consistent_controller: AdaptiveSimplexControllerConfig = field(
        default_factory=lambda: AdaptiveSimplexControllerConfig(
            initial_lambda=0.4,
            lambda_min=0.01,
            lambda_max=0.8,
            checkpoint_interval=100,
            diagnostic_replications=4,
            contraction=0.5,
            envelope_lambda0=0.4,
            envelope_m0=1000.0,
            envelope_zeta=0.25,
            eta_power=1.5,
            main_sample_growth_power=0.25,
            aux_sample_growth_power=0.5,
            sample_growth_interval=1000.0,
        )
    )

    def env_config(self) -> TwoStateConfig:
        return TwoStateConfig(
            device=self.device,
            dtype=self.dtype,
            T=self.train_horizon,
            N=self.B,
            n=self.n_aux,
            lr=self.lr,
            n_train=self.n_train,
            training_runs=self.training_runs,
            validate_every=self.validate_every,
        )

    def budget_for_method(self, method: str) -> Dict[str, int]:
        if method == "logits":
            return {"B": int(self.logit_B), "n": int(self.logit_n)}
        return {"B": int(self.B), "n": int(self.n_aux)}


def tiny_twostate_adaptive_comparison_config(
    device: torch.device,
    dtype: torch.dtype,
) -> TwoStateAdaptiveComparisonConfig:
    return TwoStateAdaptiveComparisonConfig(
        device=device,
        dtype=dtype,
        n_train=2,
        training_runs=1,
        validate_every=1,
        B=3,
        n_aux=2,
        logit_B=3,
        logit_n=2,
        simplex_lambdas=(0.2,),
        logit_epsilons=(0.2,),
        adaptive_initial_lambdas=(0.2,),
        finite_controller=AdaptiveSimplexControllerConfig(
            initial_lambda=0.2,
            lambda_min=0.05,
            lambda_max=0.6,
            checkpoint_interval=1,
            diagnostic_replications=1,
        ),
        consistent_controller=AdaptiveSimplexControllerConfig(
            initial_lambda=0.2,
            lambda_min=0.05,
            lambda_max=0.6,
            checkpoint_interval=1,
            diagnostic_replications=1,
            envelope_lambda0=0.2,
            envelope_m0=10.0,
            envelope_zeta=0.25,
            eta_power=1.5,
            sample_growth_interval=10.0,
        ),
    )


def _load_theta(config: TwoStateConfig, payload: Dict[str, torch.Tensor], trainable: bool = True):
    theta = payload["theta"].to(dtype=config.dtype, device=config.device).detach().clone()
    return torch.nn.Parameter(theta) if trainable else theta


def _record_numeric_history(history: Dict[str, List[float]], metrics: Dict[str, object]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            history.setdefault(key, []).append(float(value))


def _tensor_float(value, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _algorithm_label(method: str) -> str:
    labels = {
        "fixed_simplex": "Fixed Simplex",
        "finite_adaptive": "Finite Adaptive Simplex",
        "consistent_adaptive": "Consistent Adaptive Simplex",
        "logits": "Logits",
    }
    return labels[method]


def _new_algorithm(method: str, env: TwoStateMFC, comparison_config: TwoStateAdaptiveComparisonConfig):
    if method == "fixed_simplex":
        return SimplexPerturbedMFREINFORCE(env)
    if method == "finite_adaptive":
        return FiniteBudgetAdaptiveSimplexMFREINFORCE(env, replace(comparison_config.finite_controller))
    if method == "consistent_adaptive":
        return ConsistentAdaptiveSimplexMFREINFORCE(env, replace(comparison_config.consistent_controller))
    if method == "logits":
        return LogitsPerturbedMFREINFORCE(env)
    raise ValueError(f"Unknown method={method!r}.")


def _with_adaptive_initial_lambda(
    comparison_config: TwoStateAdaptiveComparisonConfig,
    initial_lambda: float,
) -> TwoStateAdaptiveComparisonConfig:
    return replace(
        comparison_config,
        finite_controller=replace(comparison_config.finite_controller, initial_lambda=float(initial_lambda)),
        consistent_controller=replace(
            comparison_config.consistent_controller,
            initial_lambda=float(initial_lambda),
            envelope_lambda0=float(initial_lambda),
        ),
    )


def _gradient_for_method(
    method: str,
    algorithm,
    theta,
    mu0: torch.Tensor,
    mu_flow: torch.Tensor,
    episode: int,
    B: int,
    n_aux: int,
    perturbation: Optional[float],
):
    ctrl = theta_for_estimator(theta)
    if method == "fixed_simplex":
        value = float(perturbation)
        return algorithm.complete_gradient_estimate(ctrl, mu_flow, value, B, n_aux, eta=value, baseline="batch_mean")
    if method == "logits":
        value = float(perturbation)
        return algorithm.gradient_estimate(
            ctrl,
            mu0,
            value,
            B,
            n_aux,
            flow_particles=1,
            horizon=mu_flow.shape[0] - 1,
            mu_flow=mu_flow,
        )
    if method in {"finite_adaptive", "consistent_adaptive"}:
        return algorithm.gradient_estimate(ctrl, mu_flow, episode, B, n_aux, baseline="batch_mean")
    raise ValueError(f"Unknown method={method!r}.")


def _base_transition_cost(
    method: str,
    horizon: int,
    B: int,
    n_aux: int,
    flow_mode: str,
    flow_particles: int,
) -> int:
    if method == "logits":
        return simulator_transitions_per_update("Logits", horizon, B, n_aux, flow_mode, flow_particles)
    return simulator_transitions_per_update("Simplex", horizon, B, n_aux, flow_mode, flow_particles)


def run_twostate_method(
    method: str,
    comparison_config: TwoStateAdaptiveComparisonConfig,
    perturbations: Sequence[Optional[float]],
    run_plans: Optional[List[Dict[str, object]]] = None,
    show_progress: bool = True,
    max_runtime_seconds: Optional[float] = None,
) -> Dict[object, List[Dict[str, object]]]:
    config = comparison_config.env_config()
    fixed_mu0 = fixed_validation_law(config)
    plans = run_plans or prepare_twostate_run_plans(
        config,
        seed_base=comparison_config.seed_base,
        training_runs=comparison_config.training_runs,
    )
    label = _algorithm_label(method)
    results: Dict[object, List[Dict[str, object]]] = {}

    for perturbation in perturbations:
        parameter_key = "adaptive" if perturbation is None else float(perturbation)
        results[parameter_key] = []
        for run_idx, plan in enumerate(plans):
            set_seed(int(plan["seed"]), config.device)
            env = TwoStateMFC(config)
            theta = _load_theta(config, plan["initial_control"], trainable=True)
            optimizer = torch.optim.Adam([theta], lr=comparison_config.lr)
            algorithm = _new_algorithm(method, env, comparison_config)
            history: Dict[str, List[float]] = {
                "episode": [],
                "validation_value": [],
                "train_return_mean": [],
                "grad_norm": [],
                "lambda": [],
                "eta": [],
                "lambda_ctrl": [],
                "cumulative_simulator_transitions": [],
                "elapsed_seconds": [],
            }
            controller_records: List[Dict[str, object]] = []
            cumulative_transitions = 0
            run_start = time.perf_counter()
            stop_reason = "completed"
            episodes_completed = 0
            budget = comparison_config.budget_for_method(method)
            B = budget["B"]
            n_aux = budget["n"]
            iterator = range(comparison_config.n_train)
            if show_progress:
                iterator = tqdm(iterator, desc=f"{label} {parameter_key} run={run_idx}")

            for episode in iterator:
                if max_runtime_seconds is not None and time.perf_counter() - run_start >= max_runtime_seconds:
                    stop_reason = f"max_runtime_{max_runtime_seconds:.1f}s"
                    break

                mu0 = plan["initial_laws"][episode].to(dtype=config.dtype, device=config.device)
                mu_flow = training_population_flow(
                    env,
                    algorithm,
                    theta,
                    mu0,
                    comparison_config.train_horizon,
                    comparison_config.flow_mode,
                    comparison_config.flow_particles,
                )
                grad_hat, diag = _gradient_for_method(
                    method,
                    algorithm,
                    theta,
                    mu0,
                    mu_flow,
                    episode,
                    B,
                    n_aux,
                    None if perturbation is None else float(perturbation),
                )

                transition_value = _tensor_float(diag.get("simulator_transitions"), default=float("nan"))
                if math.isnan(transition_value):
                    transitions = _base_transition_cost(
                        method,
                        comparison_config.train_horizon,
                        B,
                        n_aux,
                        comparison_config.flow_mode,
                        comparison_config.flow_particles,
                    )
                else:
                    transitions = int(transition_value)
                cumulative_transitions += transitions

                controller_diag = diag.get("controller")
                if controller_diag is not None:
                    controller_records.append({"episode": episode, **controller_diag})

                optimizer.zero_grad(set_to_none=True)
                assign_ascent_gradient(theta, grad_hat)
                optimizer.step()
                episodes_completed = episode + 1

                if episode % comparison_config.validate_every == 0 or episode == comparison_config.n_train - 1:
                    metrics = reference_metrics(env, theta, fixed_mu0, comparison_config.validation_horizon)
                    history["episode"].append(float(episode))
                    history["validation_value"].append(metrics["value"])
                    history["train_return_mean"].append(_tensor_float(diag.get("mean_return")))
                    history["grad_norm"].append(_tensor_float(diag.get("grad_norm")))
                    history["lambda"].append(_tensor_float(diag.get("lambda"), default=float(perturbation or 0.0)))
                    history["eta"].append(_tensor_float(diag.get("eta"), default=float(perturbation or 0.0)))
                    history["lambda_ctrl"].append(_tensor_float(diag.get("lambda_ctrl")))
                    history["cumulative_simulator_transitions"].append(float(cumulative_transitions))
                    history["elapsed_seconds"].append(time.perf_counter() - run_start)
                    _record_numeric_history(history, metrics)
                    if show_progress:
                        iterator.set_postfix(value=f"{metrics['value']:.4g}", grad=f"{history['grad_norm'][-1]:.3g}")

            final_metrics = reference_metrics(env, theta, fixed_mu0, comparison_config.validation_horizon)
            results[parameter_key].append(
                {
                    "algorithm": label,
                    "method": method,
                    "parameter": parameter_key,
                    "run_idx": run_idx,
                    "seed": int(plan["seed"]),
                    "theta": theta.detach().cpu().clone(),
                    "history": history,
                    "controller_diagnostics": controller_records,
                    "final_value": final_metrics["value"],
                    "reference_metrics": final_metrics,
                    "runtime_seconds": time.perf_counter() - run_start,
                    "episodes_completed": episodes_completed,
                    "main_trajectories": B,
                    "auxiliary_trajectories": n_aux,
                    "total_simulator_transitions": cumulative_transitions,
                    "stop_reason": stop_reason,
                }
            )
    return results


def run_twostate_adaptive_comparison(
    comparison_config: TwoStateAdaptiveComparisonConfig,
    show_progress: bool = True,
    max_runtime_seconds: Optional[float] = None,
) -> Dict[str, Dict[object, List[Dict[str, object]]]]:
    env_config = comparison_config.env_config()
    run_plans = prepare_twostate_run_plans(
        env_config,
        seed_base=comparison_config.seed_base,
        training_runs=comparison_config.training_runs,
    )
    finite_results: Dict[object, List[Dict[str, object]]] = {}
    consistent_results: Dict[object, List[Dict[str, object]]] = {}
    for initial_lambda in comparison_config.adaptive_initial_lambdas:
        sweep_config = _with_adaptive_initial_lambda(comparison_config, float(initial_lambda))
        finite_results.update(
            run_twostate_method(
                "finite_adaptive",
                sweep_config,
                (float(initial_lambda),),
                run_plans=run_plans,
                show_progress=show_progress,
                max_runtime_seconds=max_runtime_seconds,
            )
        )
        consistent_results.update(
            run_twostate_method(
                "consistent_adaptive",
                sweep_config,
                (float(initial_lambda),),
                run_plans=run_plans,
                show_progress=show_progress,
                max_runtime_seconds=max_runtime_seconds,
            )
        )
    return {
        "Fixed Simplex": run_twostate_method(
            "fixed_simplex",
            comparison_config,
            comparison_config.simplex_lambdas,
            run_plans=run_plans,
            show_progress=show_progress,
            max_runtime_seconds=max_runtime_seconds,
        ),
        "Finite Adaptive Simplex": finite_results,
        "Consistent Adaptive Simplex": consistent_results,
        "Logits": run_twostate_method(
            "logits",
            comparison_config,
            comparison_config.logit_epsilons,
            run_plans=run_plans,
            show_progress=show_progress,
            max_runtime_seconds=max_runtime_seconds,
        ),
    }


def summarize_comparison_results(results: Dict[str, Dict[object, List[Dict[str, object]]]]) -> pd.DataFrame:
    rows = []
    for algorithm_name, parameter_groups in results.items():
        for parameter, runs in parameter_groups.items():
            values = np.asarray([run["final_value"] for run in runs], dtype=float)
            policy_errors = np.asarray(
                [run["reference_metrics"]["policy_error_mean"] for run in runs],
                dtype=float,
            )
            flow_errors = np.asarray([run["reference_metrics"]["flow_error"] for run in runs], dtype=float)
            transitions = np.asarray([run["total_simulator_transitions"] for run in runs], dtype=float)
            rows.append(
                {
                    "algorithm": algorithm_name,
                    "parameter": parameter,
                    "runs": len(runs),
                    "B": int(runs[0]["main_trajectories"]),
                    "n_aux_or_inner": int(runs[0]["auxiliary_trajectories"]),
                    "value_mean": float(values.mean()),
                    "value_std": float(values.std(ddof=1 if len(values) > 1 else 0)),
                    "policy_error_mean": float(policy_errors.mean()),
                    "flow_error_mean": float(flow_errors.mean()),
                    "total_simulator_transitions_mean": float(transitions.mean()),
                }
            )
    return pd.DataFrame(rows)


def controller_diagnostics_frame(results: Dict[str, Dict[object, List[Dict[str, object]]]]) -> pd.DataFrame:
    rows = []
    for algorithm_name, parameter_groups in results.items():
        for parameter, runs in parameter_groups.items():
            for run in runs:
                for record in run.get("controller_diagnostics", []):
                    row = {
                        "algorithm": algorithm_name,
                        "parameter": parameter,
                        "run_idx": run["run_idx"],
                    }
                    row.update(record)
                    rows.append(row)
    return pd.DataFrame(rows)
