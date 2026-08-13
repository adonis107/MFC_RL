from __future__ import annotations

import copy
import itertools
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from mfc.experiments.core.artifacts import (
    _checkpoint_rng_state,
    _json_default,
    _make_run_dir,
    _metadata,
    _set_dotted,
    _write_csv,
    _write_json,
    apply_overrides,
)
from mfc.experiments.core.controls import (
    assign_gradient,
    control_parameters,
    control_payload,
    control_vector,
    initialize_control,
)
from mfc.experiments.core.evaluation import (
    _scalar,
    evaluate_control,
    finite_population_flow,
)
from mfc.experiments.diagnostics.common import _as_list
from mfc.experiments.diagnostics.functional_law import run_functional_law_diagnostic
from mfc.experiments.diagnostics.gradient import run_gradient_diagnostic
from mfc.experiments.diagnostics.perturbation import run_perturbation_diagnostic
from mfc.experiments.diagnostics.sensitivity import run_sensitivity_diagnostic
from mfc.experiments.core.gradient_steps import (
    continuous_mfreinforce_gradient_step,
    exact_gradient_step,
    finite_gradient,
    make_algorithm,
    pathwise_gradient_step,
)
from mfc.experiments.core.registry import (
    CONTINUOUS_ALGORITHMS,
    DEFAULT_DEVICE,
    ENVIRONMENTS,
    EXACT_ALGORITHMS,
    FINITE_ALGORITHMS,
    PATHWISE_ALGORITHMS,
    build_env_config,
    build_environment,
    require_algorithm_name,
    require_env_name,
    validate_compatibility,
)
from mfc.experiments.core.runtime import (
    _aux_batch,
    _main_batch,
    _training_horizon,
    sample_initial_laws,
)
from mfc.experiments.core.session import RunResult, load_checkpoint, normalize_experiment_config, set_seed


def run_train(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    env_name = require_env_name(config)
    algorithm_name = require_algorithm_name(config)
    validate_compatibility(env_name, algorithm_name)
    spec, env = build_environment(config)
    train_config = dict(config.get("train", {}))
    algorithm_config = dict(config.get("algorithm_config", {}))
    evaluation_config = dict(config.get("evaluation", {}))

    seed = int(train_config.get("seed", 0))
    set_seed(seed, env.config.device)
    steps = int(train_config.get("steps", getattr(env.config, "n_train", 1)))
    if steps <= 0:
        raise ValueError("train.steps must be positive.")
    lr = float(train_config.get("lr", getattr(env.config, "lr", 1e-3)))
    validate_every = int(train_config.get("validate_every", getattr(env.config, "validate_every", max(1, steps))))
    validate_every = max(1, validate_every)
    horizon = _training_horizon(env, train_config)
    B = _main_batch(env, train_config, algorithm_config)
    n_aux = _aux_batch(env, train_config, algorithm_config)
    flow_mode = str(train_config.get("flow_mode", "exact"))
    flow_particles = int(train_config.get("flow_particles", max(1, B)))
    history_control_coordinates = int(train_config.get("history_control_coordinates", 8))

    control = initialize_control(spec, env)
    optimizer = torch.optim.Adam(control_parameters(control), lr=lr)
    algorithm = make_algorithm(algorithm_name, env, algorithm_config)
    initial_laws = sample_initial_laws(spec, env, steps, train_config)

    run_dir = _make_run_dir("train", config)
    resolved_config = _resolved_config(config, env, steps, lr, validate_every, horizon, B, n_aux, flow_mode, flow_particles)
    _write_json(run_dir / "config.json", resolved_config)
    _write_json(run_dir / "metadata.json", _metadata("train", resolved_config))

    history: List[Dict[str, Any]] = []
    start = time.perf_counter()
    final_metrics: Dict[str, Any] = {}
    for episode in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if spec.family == "finite":
            if initial_laws is None:
                raise ValueError("Finite training requires sampled initial laws.")
            mu0 = initial_laws[episode].to(dtype=env.config.dtype, device=env.config.device)
            mu_flow = finite_population_flow(env, algorithm, control, mu0, horizon, flow_mode, flow_particles)
            grad, diag = finite_gradient(
                algorithm_name,
                algorithm,
                control,
                mu0,
                mu_flow,
                episode,
                B,
                n_aux,
                algorithm_config,
                train_config,
            )
            assign_gradient(control, grad, spec.objective)
        elif algorithm_name == "exact-gradient":
            objective, grad, diag = exact_gradient_step(spec, env, control, algorithm_config)  # type: ignore[arg-type]
            assign_gradient(control, grad, spec.objective)
        elif algorithm_name == "pathwise-gradient":
            objective, grad, diag = pathwise_gradient_step(env, control, algorithm_config, train_config, episode)  # type: ignore[arg-type]
            assign_gradient(control, grad, spec.objective)
        elif algorithm_name == "continuous-mfreinforce":
            objective, grad, diag = continuous_mfreinforce_gradient_step(
                env,
                algorithm,
                control,
                algorithm_config,
                train_config,
                episode,
            )
            assign_gradient(control, grad, spec.objective)
        else:
            raise ValueError(f"Unsupported training mode for {algorithm_name!r}.")
        optimizer.step()

        if episode % validate_every == 0 or episode == steps - 1:
            final_metrics = evaluate_control(spec, env, control, train_config, evaluation_config, seed + 10_000 + episode)
            row = {
                "episode": episode,
                "elapsed_seconds": time.perf_counter() - start,
                "grad_norm": _scalar(diag.get("grad_norm")),
                "train_mean_return": _scalar(diag.get("mean_return")),
                "lambda": _scalar(diag.get("lambda")),
                "eta": _scalar(diag.get("eta")),
                "objective": final_metrics.get("objective", final_metrics.get("value")),
                "value": final_metrics.get("value"),
                "cost": final_metrics.get("cost"),
            }
            if history_control_coordinates > 0:
                for coordinate, value in enumerate(control_vector(control)[:history_control_coordinates].detach().cpu().tolist()):
                    row[f"control_{coordinate}"] = value
            history.append(row)

    final_metrics = evaluate_control(spec, env, control, train_config, evaluation_config, seed + 20_000)
    final_metrics["runtime_seconds"] = time.perf_counter() - start
    final_metrics["steps"] = steps
    _write_csv(run_dir / "history.csv", history)
    _write_json(run_dir / "metrics.json", final_metrics)
    checkpoint = {
        "schema_version": 1,
        "env": env_name,
        "algorithm": algorithm_name,
        "env_config": _json_default(env.config),
        "algorithm_config": _json_default(algorithm_config),
        "train_config": _json_default(train_config),
        "evaluation_config": _json_default(evaluation_config),
        "control": control_payload(control),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _checkpoint_rng_state(),
        "history": _json_default(history),
        "final_metrics": _json_default(final_metrics),
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return RunResult(run_dir, resolved_config, final_metrics, history, checkpoint_path=checkpoint_path)


def _resolved_config(
    config: Mapping[str, Any],
    env: Any,
    steps: int,
    lr: float,
    validate_every: int,
    horizon: int,
    B: int,
    n_aux: int,
    flow_mode: str,
    flow_particles: int,
) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    resolved["env_config"] = _json_default(env.config)
    train = resolved.setdefault("train", {})
    train.update(
        {
            "steps": steps,
            "lr": lr,
            "validate_every": validate_every,
            "horizon": horizon,
            "B": B,
            "n": n_aux,
            "flow_mode": flow_mode,
            "flow_particles": flow_particles,
        }
    )
    return _json_default(resolved)


def run_sweep(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    sweep = dict(config.get("sweep", {}))
    parameters = sweep.get("parameters", {})
    if not parameters:
        raise ValueError("sweep.parameters must contain at least one dotted-path list.")
    run_dir = _make_run_dir("sweep", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("sweep", config))
    keys = list(parameters)
    rows = []
    for values in itertools.product(*[_as_list(parameters[key]) for key in keys]):
        child_config = copy.deepcopy(dict(config))
        for key, value in zip(keys, values):
            _set_dotted(child_config, key, value)
        child_config.setdefault("train", {})["output_dir"] = str(run_dir)
        child_config["train"]["run_name"] = "_".join(f"{key.replace('.', '-')}_{value}" for key, value in zip(keys, values))
        result = run_train(child_config)
        row = {"run_dir": str(result.run_dir)}
        row.update({key: value for key, value in zip(keys, values)})
        row.update({key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str))})
        rows.append(row)
    _write_csv(run_dir / "diagnostics.csv", rows)
    metrics = {"runs": len(rows)}
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, config, metrics, [], diagnostics_path=run_dir / "diagnostics.csv")


def _merged_cli_config(args: Any) -> Dict[str, Any]:
    from .cli import merged_cli_config

    return merged_cli_config(args)


def build_parser() -> Any:
    from .cli import build_parser as _build_parser

    return _build_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .cli import main as _main

    return _main(argv)


__all__ = [
    "ENVIRONMENTS",
    "FINITE_ALGORITHMS",
    "EXACT_ALGORITHMS",
    "PATHWISE_ALGORITHMS",
    "CONTINUOUS_ALGORITHMS",
    "DEFAULT_DEVICE",
    "RunResult",
    "apply_overrides",
    "build_env_config",
    "build_environment",
    "build_parser",
    "load_checkpoint",
    "main",
    "run_functional_law_diagnostic",
    "run_gradient_diagnostic",
    "run_perturbation_diagnostic",
    "run_sensitivity_diagnostic",
    "run_sweep",
    "run_train",
    "validate_compatibility",
]
