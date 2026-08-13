from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

import torch

from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.controls import initialize_control, load_control
from ..core.evaluation import evaluate_control
from ..core.registry import build_environment, require_env_name
from ..core.session import RunResult, load_checkpoint, normalize_experiment_config
from ..diagnostics.common import _float_list
from ..diagnostics.gradient import run_gradient_diagnostic
from ..runner import run_train
from .common import _control_distance, _policy_output_distance, _read_csv, _trajectory_distance


def run_perturbation_bias_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    env_name = require_env_name(config)
    spec, env = build_environment(config)
    algorithm_config = dict(config.get("algorithm_config", {}))
    diagnostic = dict(config.get("diagnostic", {}))
    train_config = dict(config.get("train", {}))
    lambdas = _float_list(diagnostic.get("lambdas", [0.0, 0.01, 0.03, 0.1, 0.3]))
    control = initialize_control(spec, env)
    rows: List[Dict[str, Any]] = []

    if env_name == "portfolio":
        base_objective = env.exact_objective(control, lambda_=0.0)
        base_grad = env.exact_gradient(control, lambda_=0.0)[1].reshape(-1)
        for lambda_value in lambdas:
            objective, grad = env.exact_gradient(control, lambda_=lambda_value)
            rows.append(
                {
                    "lambda": lambda_value,
                    "objective": float(objective.item()),
                    "objective_bias": float((objective - base_objective).abs().item()),
                    "gradient_bias_norm": float(torch.linalg.norm(grad.reshape(-1) - base_grad).item()),
                }
            )
    else:
        gradient_config = copy.deepcopy(dict(config))
        gradient_config.setdefault("diagnostic", {})["lambdas"] = lambdas
        gradient_config.setdefault("train", {})["run_name"] = "gradient_bias_inner"
        result = run_gradient_diagnostic(gradient_config)
        rows = _read_csv(result.diagnostics_path) if result.diagnostics_path else []

    run_dir = _make_run_dir("perturbation-bias", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("perturbation-bias", config))
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"rows": len(rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def run_optimizer_bias_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    diagnostic = dict(config.get("diagnostic", {}))
    lambdas = _float_list(study_config.get("lambdas", diagnostic.get("lambdas", [0.01, 0.03, 0.1, 0.3])))
    if not lambdas:
        raise ValueError("optimizer-bias requires at least one lambda.")

    run_dir = _make_run_dir("optimizer-bias", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("optimizer-bias", config))

    training_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    checkpoint_payloads: List[Dict[str, Any]] = []
    for index, lambda_value in enumerate(lambdas):
        child = copy.deepcopy(dict(config))
        child.setdefault("algorithm_config", {})["lambda"] = lambda_value
        child["algorithm_config"]["eta"] = child["algorithm_config"].get("eta", lambda_value)
        child.setdefault("train", {})["output_dir"] = str(run_dir)
        child["train"]["run_name"] = f"lambda_{index:04d}_{lambda_value:g}".replace(".", "p")
        child["train"].setdefault("overwrite", True)
        result = run_train(child)
        payload = load_checkpoint(result.checkpoint_path)["payload"] if result.checkpoint_path else {}
        checkpoint_payloads.append({"lambda": lambda_value, "run": result, "payload": payload})
        for history_row in result.history:
            training_rows.append({"lambda": lambda_value, "run_dir": str(result.run_dir), **history_row})

    reference_index = int(study_config.get("reference_index", 0))
    reference = checkpoint_payloads[reference_index]
    ref_config = {
        "env": reference["payload"]["env"],
        "algorithm": reference["payload"]["algorithm"],
        "env_config": reference["payload"]["env_config"],
        "algorithm_config": reference["payload"].get("algorithm_config", {}),
        "train": reference["payload"].get("train_config", {}),
        "evaluation": reference["payload"].get("evaluation_config", {}),
    }
    ref_spec, ref_env = build_environment(ref_config)
    ref_control = load_control(ref_spec, ref_env, reference["payload"]["control"], trainable=False)
    ref_metrics = evaluate_control(
        ref_spec,
        ref_env,
        ref_control,
        ref_config.get("train", {}),
        {**dict(ref_config.get("evaluation", {})), **dict(config.get("evaluation", {}))},
        int(ref_config.get("train", {}).get("seed", 0)) + 77_000,
    )

    for item in checkpoint_payloads:
        payload = item["payload"]
        child_config = {
            "env": payload["env"],
            "algorithm": payload["algorithm"],
            "env_config": payload["env_config"],
            "algorithm_config": payload.get("algorithm_config", {}),
            "train": payload.get("train_config", {}),
            "evaluation": payload.get("evaluation_config", {}),
        }
        spec, env = build_environment(child_config)
        control = load_control(spec, env, payload["control"], trainable=False)
        metrics = evaluate_control(
            spec,
            env,
            control,
            child_config.get("train", {}),
            {**dict(child_config.get("evaluation", {})), **dict(config.get("evaluation", {}))},
            int(child_config.get("train", {}).get("seed", 0)) + 77_000,
        )
        row = {
            "lambda": item["lambda"],
            "run_dir": str(item["run"].run_dir),
            "control_distance": _control_distance(control, ref_control),
            "policy_output_distance": _policy_output_distance(spec, env, control, ref_control, child_config),
            "trajectory_distance": _trajectory_distance(spec, env, control, ref_control, child_config),
            "reference_lambda": reference["lambda"],
        }
        objective = metrics.get("objective", metrics.get("value", metrics.get("cost")))
        reference_objective = ref_metrics.get("objective", ref_metrics.get("value", ref_metrics.get("cost")))
        if isinstance(objective, (int, float)) and isinstance(reference_objective, (int, float)):
            row["unperturbed_objective"] = float(objective)
            row["reference_objective"] = float(reference_objective)
            row["optimal_value_bias_proxy"] = float(objective) - float(reference_objective)
        row.update({f"metric_{key}": value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
        rows.append(row)

    _write_csv(run_dir / "optimization_history.csv", training_rows)
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"lambdas": len(rows), "history_rows": len(training_rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, training_rows, diagnostics_path=run_dir / "diagnostics.csv")


__all__ = ["run_optimizer_bias_study", "run_perturbation_bias_study"]
