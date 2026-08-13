from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

from ..core.artifacts import _make_run_dir, _metadata, _write_csv, _write_json
from ..core.controls import control_vector, initialize_control
from ..core.evaluation import finite_population_flow
from ..core.gradient_steps import finite_gradient, make_algorithm
from ..core.registry import build_environment, require_algorithm_name, require_env_name, validate_compatibility
from ..core.runtime import _aux_batch, _lambda_value, _main_batch, _training_horizon, sample_initial_laws
from ..core.session import RunResult, normalize_experiment_config, set_seed
from ..diagnostics.common import _float_list, _matrix_rows, _safe_covariance


def run_score_validation(config: Mapping[str, Any]) -> RunResult:
    config = normalize_experiment_config(config)
    env_name = require_env_name(config)
    algorithm_name = require_algorithm_name(config)
    validate_compatibility(env_name, algorithm_name)
    spec, env = build_environment(config)

    algorithm_config = dict(config.get("algorithm_config", {}))
    train_config = dict(config.get("train", {}))
    diagnostic = dict(config.get("diagnostic", {}))
    seed = int(train_config.get("seed", 0))
    set_seed(seed, env.config.device)
    control = initialize_control(spec, env)
    algorithm = make_algorithm(algorithm_name, env, algorithm_config)
    horizon = _training_horizon(env, train_config)
    B = _main_batch(env, train_config, algorithm_config)
    n_aux = _aux_batch(env, train_config, algorithm_config)
    replications = int(diagnostic.get("replications", 16))
    lambdas = _float_list(diagnostic.get("lambdas", [_lambda_value(algorithm_config, train_config, 0.1)]))
    summary_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    covariance_rows: List[Dict[str, Any]] = []

    if spec.family != "finite":
        if algorithm_name != "continuous-mfreinforce":
            raise ValueError("Continuous score validation requires algorithm='continuous-mfreinforce'.")
        population_particles = int(
            train_config.get(
                "population_particles",
                algorithm_config.get("population_particles", train_config.get("particles", getattr(env.config, "N_pop", B))),
            )
        )
        for lambda_value in lambdas:
            samples = []
            for rep in range(replications):
                set_seed(seed + rep, env.config.device)
                local_config = dict(algorithm_config)
                local_config["lambda"] = lambda_value
                local_config["keep_score_diagnostics"] = True
                local_algorithm = make_algorithm(algorithm_name, env, local_config)
                _, diag = local_algorithm.complete_gradient_estimate(
                    control,
                    lambda_value,
                    B,
                    n_aux,
                    eta=float(local_config.get("eta", lambda_value)),
                    horizon=horizon,
                    population_particles=population_particles,
                    seed=seed + rep,
                    baseline=local_config.get("baseline", "batch_mean"),
                    sensitivity_baseline=local_config.get("sensitivity_baseline", "nominal"),
                    keep_score_diagnostics=True,
                )
                score = diag.get("scores")
                if score is None:
                    raise ValueError("continuous-mfreinforce did not return score diagnostics.")
                samples.append(score.detach().reshape(-1, control_vector(control).numel()))
            score_samples = torch.cat(samples, dim=0)
            _append_score_rows(summary_rows, coordinate_rows, lambda_value, score_samples)
            sample_rows.extend(_score_sample_rows(lambda_value, score_samples))
            covariance_rows.extend(_matrix_rows(lambda_value, _safe_covariance(score_samples), "covariance"))

        run_dir = _make_run_dir("score-validation", config)
        _write_json(run_dir / "config.json", config)
        _write_json(run_dir / "metadata.json", _metadata("score-validation", config))
        _write_csv(run_dir / "diagnostics.csv", summary_rows)
        _write_csv(run_dir / "score_coordinates.csv", coordinate_rows)
        _write_csv(run_dir / "score_samples.csv", sample_rows)
        _write_csv(run_dir / "score_covariance.csv", covariance_rows)
        metrics = {
            "rows": len(summary_rows),
            "coordinate_rows": len(coordinate_rows),
            "sample_rows": len(sample_rows),
            "covariance_rows": len(covariance_rows),
        }
        _write_json(run_dir / "metrics.json", metrics)
        return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")

    mu0 = sample_initial_laws(spec, env, 1, train_config)[0]
    mu_flow = finite_population_flow(env, algorithm, control, mu0, horizon, str(train_config.get("flow_mode", "exact")), B)
    for lambda_value in lambdas:
        samples = []
        for rep in range(replications):
            local_config = dict(algorithm_config)
            local_config["lambda"] = lambda_value
            local_config["epsilon"] = lambda_value
            local_config["keep_score_diagnostics"] = True
            set_seed(seed + rep, env.config.device)
            if algorithm_name == "simplex":
                _, diag = algorithm.complete_gradient_estimate(
                    control,
                    mu_flow,
                    lambda_value,
                    B,
                    n_aux,
                    eta=float(local_config.get("eta", lambda_value)),
                    baseline=local_config.get("baseline", "batch_mean"),
                    keep_score_diagnostics=True,
                )
                score = diag.get("scores")
            elif algorithm_name == "logits":
                _, diag = algorithm.gradient_estimate(
                    control,
                    mu0,
                    lambda_value,
                    B,
                    n_aux,
                    max(1, int(local_config.get("flow_particles", 1))),
                    horizon=horizon,
                    mu_flow=mu_flow,
                    keep_score_diagnostics=True,
                )
                score = diag.get("samples")
            else:
                grad, _ = finite_gradient(
                    algorithm_name,
                    algorithm,
                    control,
                    mu0,
                    mu_flow,
                    rep,
                    B,
                    n_aux,
                    local_config,
                    train_config,
                )
                score = grad.detach().reshape(1, -1)
            if score is None:
                raise ValueError(f"{algorithm_name!r} did not return score diagnostics.")
            samples.append(score.detach().reshape(-1, control_vector(control).numel()))

        score_samples = torch.cat(samples, dim=0)
        _append_score_rows(summary_rows, coordinate_rows, lambda_value, score_samples)
        sample_rows.extend(_score_sample_rows(lambda_value, score_samples))
        covariance_rows.extend(_matrix_rows(lambda_value, _safe_covariance(score_samples), "covariance"))

    run_dir = _make_run_dir("score-validation", config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metadata.json", _metadata("score-validation", config))
    _write_csv(run_dir / "diagnostics.csv", summary_rows)
    _write_csv(run_dir / "score_coordinates.csv", coordinate_rows)
    _write_csv(run_dir / "score_samples.csv", sample_rows)
    _write_csv(run_dir / "score_covariance.csv", covariance_rows)
    metrics = {
        "rows": len(summary_rows),
        "coordinate_rows": len(coordinate_rows),
        "sample_rows": len(sample_rows),
        "covariance_rows": len(covariance_rows),
    }
    _write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir, dict(config), metrics, [], diagnostics_path=run_dir / "diagnostics.csv")



def _append_score_rows(
    summary_rows: List[Dict[str, Any]],
    coordinate_rows: List[Dict[str, Any]],
    lambda_value: float,
    score_samples: torch.Tensor,
) -> None:
    mean = score_samples.mean(dim=0)
    variance = score_samples.var(dim=0, unbiased=score_samples.shape[0] > 1) if score_samples.shape[0] > 1 else torch.zeros_like(mean)
    second = score_samples.square().mean(dim=0)
    summary_rows.append(
        {
            "lambda": lambda_value,
            "samples": int(score_samples.shape[0]),
            "mean_norm": float(torch.linalg.norm(mean).item()),
            "variance_trace": float(variance.sum().item()),
            "lambda2_variance_trace": float((lambda_value**2) * variance.sum().item()),
            "second_moment_trace": float(second.sum().item()),
            "max_abs_mean": float(mean.abs().max().item()),
        }
    )
    for coordinate in range(mean.numel()):
        coordinate_rows.append(
            {
                "lambda": lambda_value,
                "coordinate": coordinate,
                "mean": float(mean[coordinate].item()),
                "variance": float(variance[coordinate].item()),
                "second_moment": float(second[coordinate].item()),
            }
        )



def _score_sample_rows(lambda_value: float, score_samples: torch.Tensor, *, max_rows: int = 1_000_000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    score_samples = score_samples.detach().reshape(score_samples.shape[0], -1).cpu()
    written = 0
    for sample in range(score_samples.shape[0]):
        for coordinate in range(score_samples.shape[1]):
            if written >= max_rows:
                return rows
            rows.append(
                {
                    "lambda": lambda_value,
                    "sample": sample,
                    "coordinate": coordinate,
                    "score": float(score_samples[sample, coordinate].item()),
                }
            )
            written += 1
    return rows


__all__ = ["run_score_validation"]
