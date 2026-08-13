from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.controls import load_control
from ..core.evaluation import evaluate_control
from ..core.registry import ENVIRONMENTS, build_environment
from ..core.session import RunResult, load_checkpoint, normalize_experiment_config
from ..diagnostics.common import _float_list
from ..runner import run_train
from .common import _write_adaptive_lambda_trace
from .grids import run_variant_grid


def run_signature_ablation_study(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    modes = study_config.get("modes", ["full", "reduced", "underspecified"])
    variants = []
    for mode in modes:
        label = str(mode)
        if isinstance(mode, Mapping):
            variants.append(mode)
        else:
            variants.append({"label": label, "diagnostic.signature_mode": label})
    dimensions = study_config.get("dimensions", [])
    for dim in dimensions:
        variants.append({"label": f"dim_{int(dim)}", "diagnostic.signature_dim": int(dim)})
    return run_variant_grid(config, "signature-ablation", variants, default_command=str(study_config.get("command", "diagnose-functional-law")))



def run_adaptive_lambda_study(config: Mapping[str, Any]) -> RunResult:
    study_config = dict(config.get("study", {}))
    variants = list(study_config.get("variants", []))
    if not variants:
        env_name = str(config.get("env", ""))
        fixed = _float_list(study_config.get("fixed_lambdas", [0.05, 0.1, 0.2]))
        if env_name in ENVIRONMENTS and ENVIRONMENTS[env_name].family == "finite":
            variants = [{"label": f"fixed_{value:g}", "algorithm": "simplex", "algorithm_config.lambda": value, "algorithm_config.eta": value} for value in fixed]
            variants.extend(
                [
                    {"label": "finite_adaptive", "algorithm": "finite-adaptive-simplex"},
                    {"label": "consistent_adaptive", "algorithm": "consistent-adaptive-simplex"},
                ]
            )
        else:
            variants = [
                {
                    "label": f"fixed_{value:g}",
                    "algorithm": "continuous-mfreinforce",
                    "algorithm_config.lambda": value,
                    "algorithm_config.eta": value,
                }
                for value in fixed
            ]
    result = run_variant_grid(config, "adaptive-lambda", variants, default_command=str(study_config.get("command", "train")))
    _write_adaptive_lambda_trace(result.run_dir)
    return result



def run_particle_transfer_study(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    study_config = dict(config.get("study", {}))
    train_particles = [int(value) for value in study_config.get("train_particles", [32, 64])]
    eval_particles = [int(value) for value in study_config.get("eval_particles", train_particles)]
    run_dir = _make_run_dir("particle-transfer", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("particle-transfer", config))

    checkpoints: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for n_train in train_particles:
        child = copy.deepcopy(dict(config))
        child.setdefault("env_config", {})["N_pop"] = n_train
        child["env_config"]["N_val"] = max(n_train, int(child["env_config"].get("N_val", n_train)))
        child.setdefault("train", {})["population_particles"] = n_train
        child["train"]["particles"] = n_train
        child["train"]["output_dir"] = str(run_dir)
        child["train"]["run_name"] = f"trained_at_{n_train}"
        child["train"].setdefault("overwrite", True)
        result = run_train(child)
        payload = load_checkpoint(result.checkpoint_path)["payload"] if result.checkpoint_path else {}
        checkpoints.append({"train_particles": n_train, "payload": payload, "run_dir": str(result.run_dir)})
        for row in result.history:
            history_rows.append({"train_particles": n_train, **row})

    rows: List[Dict[str, Any]] = []
    for item in checkpoints:
        payload = item["payload"]
        for n_eval in eval_particles:
            child_config = {
                "env": payload["env"],
                "algorithm": payload["algorithm"],
                "env_config": {**dict(payload["env_config"]), "N_val": n_eval, "N_pop": n_eval},
                "algorithm_config": payload.get("algorithm_config", {}),
                "train": {**dict(payload.get("train_config", {})), "particles": n_eval, "population_particles": n_eval},
                "evaluation": {**dict(payload.get("evaluation_config", {})), "particles": n_eval},
            }
            spec, env = build_environment(child_config)
            control = load_control(spec, env, payload["control"], trainable=False)
            metrics = evaluate_control(spec, env, control, child_config["train"], child_config["evaluation"], int(child_config["train"].get("seed", 0)))
            row = {"train_particles": item["train_particles"], "eval_particles": n_eval, "run_dir": item["run_dir"]}
            row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
            rows.append(row)

    _write_csv(run_dir / "diagnostics.csv", rows)
    _write_csv(run_dir / "optimization_history.csv", history_rows)
    metrics = {"rows": len(rows), "history_rows": len(history_rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, history_rows, diagnostics_path=run_dir / "diagnostics.csv")


__all__ = [
    "run_adaptive_lambda_study",
    "run_particle_transfer_study",
    "run_signature_ablation_study",
]
