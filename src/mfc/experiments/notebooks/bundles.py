from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from .. import presets as experiment_presets
from ..application import run_application_diagnostics
from ..core.memory import release_memory
from ..diagnostics.functional_law import run_functional_law_diagnostic
from ..diagnostics.gradient import run_gradient_diagnostic
from ..diagnostics.perturbation import run_perturbation_diagnostic
from ..diagnostics.sensitivity import run_sensitivity_diagnostic
from ..runner import run_train
from ..studies import (
    run_budget_allocation,
    run_optimization_summary,
    run_score_validation,
    run_study,
    run_variant_grid,
)
from .configs import (
    ALGORITHMS,
    CONTINUOUS_ALGORITHM,
    CONTINUOUS_ALGORITHMS,
    DISCRETE_BENCHMARKS,
    benchmark_config,
    continuous_benchmark_config,
)

ProgressCallback = Callable[[str, str, Path | None], None]


def ensure_discrete_benchmark_bundle(
    env_name: str,
    base_dir: Path | str,
    *,
    quick: bool = True,
    force: bool = False,
    extended: bool = False,
    seed: int = 0,
    preset: str | None = None,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    preset_name = experiment_presets.resolve_preset(preset, quick=quick)
    bundle: Dict[str, Any] = {"env": env_name, "base_dir": base_dir, "train": {}, "application": {}, "diagnostics": {}, "studies": {}}
    primary_config: Dict[str, Any] | None = None

    for algorithm in ALGORITHMS:
        train_dir = base_dir / "train" / f"{env_name}_{algorithm}"
        config = benchmark_config(env_name, algorithm, base_dir / "train", f"{env_name}_{algorithm}", seed=seed, quick=quick, preset=preset_name)
        if algorithm == "simplex":
            primary_config = config
        if force or not (train_dir / "checkpoint.pt").exists():
            _report(progress, "run", f"train/{algorithm}", train_dir)
            run_train(config)
            release_memory()
            _report(progress, "done", f"train/{algorithm}", train_dir)
        else:
            _report(progress, "skip", f"train/{algorithm}", train_dir)
        bundle["train"][algorithm] = train_dir

        app_dir = base_dir / "application" / f"{env_name}_{algorithm}"
        app_config = {
            "env": env_name,
            "algorithm": algorithm,
            "checkpoint": str(train_dir / "checkpoint.pt"),
            "train": {"output_dir": str(base_dir / "application"), "run_name": f"{env_name}_{algorithm}", "overwrite": True, "seed": 0},
            "evaluation": dict(config.get("evaluation", {})),
        }
        if force or not (app_dir / "metrics.json").exists():
            _report(progress, "run", f"application/{algorithm}", app_dir)
            run_application_diagnostics(app_config)
            release_memory()
            _report(progress, "done", f"application/{algorithm}", app_dir)
        else:
            _report(progress, "skip", f"application/{algorithm}", app_dir)
        bundle["application"][algorithm] = app_dir

        bundle["diagnostics"][algorithm] = _ensure_algorithm_diagnostics(env_name, algorithm, base_dir, config, force, progress)
        release_memory()

    bundle["studies"]["budget"] = _ensure_budget_study(
        env_name, base_dir, force=force, quick=quick, seed=seed, preset=preset_name, progress=progress
    )
    release_memory()
    bundle["studies"]["horizon"] = _ensure_horizon_study(
        env_name, base_dir, force=force, quick=quick, seed=seed, preset=preset_name, progress=progress
    )
    release_memory()
    bundle["studies"]["optimization"] = _ensure_optimization_summary(env_name, base_dir, bundle, force=force, progress=progress)
    release_memory()
    if extended:
        if primary_config is None:
            raise ValueError("Discrete extended studies require a simplex base config.")
        bundle["studies"].update(
            _ensure_extended_studies(
                env_name, base_dir, primary_config, bundle, force=force, quick=quick, preset=preset_name, progress=progress
            )
        )
        release_memory()
    (base_dir / "bundle.json").write_text(json.dumps(_jsonable_bundle(bundle), indent=2) + "\n")
    return bundle



def ensure_continuous_benchmark_bundle(
    env_name: str,
    base_dir: Path | str,
    *,
    quick: bool = True,
    force: bool = False,
    extended: bool = False,
    seed: int = 0,
    preset: str | None = None,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    preset_name = experiment_presets.resolve_preset(preset, quick=quick)
    bundle: Dict[str, Any] = {"env": env_name, "base_dir": base_dir, "train": {}, "application": {}, "diagnostics": {}, "studies": {}}

    primary_config: Dict[str, Any] | None = None
    for algorithm in CONTINUOUS_ALGORITHMS:
        train_dir = base_dir / "train" / f"{env_name}_{algorithm}"
        config = continuous_benchmark_config(
            env_name,
            base_dir / "train",
            f"{env_name}_{algorithm}",
            algorithm=algorithm,
            seed=seed,
            quick=quick,
            preset=preset_name,
        )
        if algorithm == CONTINUOUS_ALGORITHM:
            primary_config = config
        if force or not (train_dir / "checkpoint.pt").exists():
            _report(progress, "run", f"train/{algorithm}", train_dir)
            run_train(config)
            release_memory()
            _report(progress, "done", f"train/{algorithm}", train_dir)
        else:
            _report(progress, "skip", f"train/{algorithm}", train_dir)
        bundle["train"][algorithm] = train_dir

        app_dir = base_dir / "application" / f"{env_name}_{algorithm}"
        app_config = {
            "env": env_name,
            "algorithm": algorithm,
            "checkpoint": str(train_dir / "checkpoint.pt"),
            "train": {"output_dir": str(base_dir / "application"), "run_name": f"{env_name}_{algorithm}", "overwrite": True, "seed": 0},
            "evaluation": dict(config.get("evaluation", {})),
        }
        if force or not (app_dir / "metrics.json").exists():
            _report(progress, "run", f"application/{algorithm}", app_dir)
            run_application_diagnostics(app_config)
            release_memory()
            _report(progress, "done", f"application/{algorithm}", app_dir)
        else:
            _report(progress, "skip", f"application/{algorithm}", app_dir)
        bundle["application"][algorithm] = app_dir

        bundle["diagnostics"][algorithm] = _ensure_continuous_diagnostics(env_name, algorithm, base_dir, config, force, progress)
        release_memory()

    if primary_config is None:
        raise ValueError("Continuous bundles require a continuous-mfreinforce base config.")
    bundle["studies"]["budget"] = _ensure_continuous_budget_study(
        env_name, base_dir, force=force, quick=quick, seed=seed, preset=preset_name, progress=progress
    )
    release_memory()
    bundle["studies"]["horizon"] = _ensure_continuous_horizon_study(
        env_name, base_dir, force=force, quick=quick, seed=seed, preset=preset_name, progress=progress
    )
    release_memory()
    bundle["studies"]["optimization"] = _ensure_continuous_optimization_summary(
        env_name, base_dir, bundle, force=force, progress=progress
    )
    release_memory()
    if extended:
        bundle["studies"].update(
            _ensure_extended_studies(env_name, base_dir, primary_config, bundle, force=force, quick=quick, preset=preset_name, progress=progress)
        )
        release_memory()
    (base_dir / "bundle.json").write_text(json.dumps(_jsonable_bundle(bundle), indent=2) + "\n")
    return bundle



def bundle_paths(env_name: str, base_dir: Path | str) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    return {
        "env": env_name,
        "base_dir": base_dir,
        "train": {algorithm: base_dir / "train" / f"{env_name}_{algorithm}" for algorithm in ALGORITHMS},
        "application": {algorithm: base_dir / "application" / f"{env_name}_{algorithm}" for algorithm in ALGORITHMS},
        "diagnostics": {
            algorithm: {
                "perturbation": base_dir / "diagnostics" / f"{env_name}_{algorithm}_perturbation",
                "functional_law": base_dir / "diagnostics" / f"{env_name}_{algorithm}_functional_law",
                "gradient": base_dir / "diagnostics" / f"{env_name}_{algorithm}_gradient",
                "score": base_dir / "diagnostics" / f"{env_name}_{algorithm}_score",
                "sensitivity": base_dir / "diagnostics" / f"{env_name}_{algorithm}_sensitivity",
            }
            for algorithm in ALGORITHMS
        },
        "studies": {
            "budget": base_dir / "studies" / "budget_allocation",
            "horizon": base_dir / "studies" / "horizon_scaling",
            "optimization": base_dir / "studies" / "optimization_summary",
            **_extended_study_paths(base_dir, env_name),
        },
    }



def continuous_bundle_paths(env_name: str, base_dir: Path | str) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    return {
        "env": env_name,
        "base_dir": base_dir,
        "train": {algorithm: base_dir / "train" / f"{env_name}_{algorithm}" for algorithm in CONTINUOUS_ALGORITHMS},
        "application": {algorithm: base_dir / "application" / f"{env_name}_{algorithm}" for algorithm in CONTINUOUS_ALGORITHMS},
        "diagnostics": {
            algorithm: _continuous_diagnostic_paths(base_dir, env_name, algorithm)
            for algorithm in CONTINUOUS_ALGORITHMS
        },
        "studies": {
            "budget": base_dir / "studies" / "budget_allocation",
            "horizon": base_dir / "studies" / "horizon_scaling",
            "optimization": base_dir / "studies" / "optimization_summary",
            **_extended_study_paths(base_dir, env_name),
        },
    }



def _continuous_diagnostic_paths(base_dir: Path, env_name: str, algorithm: str) -> Dict[str, Path]:
    if algorithm == "reinforce":
        return {"gradient": base_dir / "diagnostics" / f"{env_name}_{algorithm}_gradient"}
    return {
        "perturbation": base_dir / "diagnostics" / f"{env_name}_{algorithm}_perturbation",
        "functional_law": base_dir / "diagnostics" / f"{env_name}_{algorithm}_functional_law",
        "gradient": base_dir / "diagnostics" / f"{env_name}_{algorithm}_gradient",
        "score": base_dir / "diagnostics" / f"{env_name}_{algorithm}_score",
        "sensitivity": base_dir / "diagnostics" / f"{env_name}_{algorithm}_sensitivity",
    }



def _ensure_algorithm_diagnostics(
    env_name: str,
    algorithm: str,
    base_dir: Path,
    config: Mapping[str, Any],
    force: bool,
    progress: ProgressCallback | None = None,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if algorithm == "reinforce":
        runners = {"gradient": run_gradient_diagnostic}
    else:
        runners = {
            "perturbation": run_perturbation_diagnostic,
            "functional_law": run_functional_law_diagnostic,
            "gradient": run_gradient_diagnostic,
            "score": run_score_validation,
        }
        if algorithm == "simplex":
            runners["sensitivity"] = run_sensitivity_diagnostic
    for name, runner in runners.items():
        run_name = f"{env_name}_{algorithm}_{name}"
        run_dir = base_dir / "diagnostics" / run_name
        diag_config = _with_output(config, base_dir / "diagnostics", run_name)
        if force or not (run_dir / "diagnostics.csv").exists():
            _report(progress, "run", f"diagnostics/{algorithm}/{name}", run_dir)
            runner(diag_config)
            release_memory()
            _report(progress, "done", f"diagnostics/{algorithm}/{name}", run_dir)
        else:
            _report(progress, "skip", f"diagnostics/{algorithm}/{name}", run_dir)
        out[name] = run_dir
    return out



def _ensure_continuous_diagnostics(
    env_name: str,
    algorithm: str,
    base_dir: Path,
    config: Mapping[str, Any],
    force: bool,
    progress: ProgressCallback | None = None,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if algorithm == "reinforce":
        runners = {"gradient": run_gradient_diagnostic}
    else:
        runners = {
            "perturbation": run_perturbation_diagnostic,
            "functional_law": run_functional_law_diagnostic,
            "gradient": run_gradient_diagnostic,
            "score": run_score_validation,
            "sensitivity": run_sensitivity_diagnostic,
        }
    for name, runner in runners.items():
        run_name = f"{env_name}_{algorithm}_{name}"
        run_dir = base_dir / "diagnostics" / run_name
        diag_config = _with_output(config, base_dir / "diagnostics", run_name)
        if force or not (run_dir / "diagnostics.csv").exists():
            _report(progress, "run", f"diagnostics/{algorithm}/{name}", run_dir)
            runner(diag_config)
            release_memory()
            _report(progress, "done", f"diagnostics/{algorithm}/{name}", run_dir)
        else:
            _report(progress, "skip", f"diagnostics/{algorithm}/{name}", run_dir)
        out[name] = run_dir
    return out



def _ensure_budget_study(
    env_name: str,
    base_dir: Path,
    *,
    force: bool,
    quick: bool,
    seed: int,
    preset: str,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "budget_allocation"
    config = benchmark_config(env_name, "simplex", base_dir / "studies", "budget_allocation", seed=seed, quick=quick, preset=preset)
    config["study"] = {
        "name": "budget-allocation",
        "command": "diagnose-gradient",
        "budgets": experiment_presets.budget_variants(preset),
    }
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/budget", run_dir)
        run_budget_allocation(config)
        release_memory()
        _report(progress, "done", "studies/budget", run_dir)
    else:
        _report(progress, "skip", "studies/budget", run_dir)
    return run_dir



def _ensure_continuous_budget_study(
    env_name: str,
    base_dir: Path,
    *,
    force: bool,
    quick: bool,
    seed: int,
    preset: str,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "budget_allocation"
    config = continuous_benchmark_config(env_name, base_dir / "studies", "budget_allocation", seed=seed, quick=quick, preset=preset)
    config["study"] = {
        "name": "budget-allocation",
        "command": "diagnose-gradient",
        "budgets": experiment_presets.budget_variants(preset),
    }
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/budget", run_dir)
        run_budget_allocation(config)
        release_memory()
        _report(progress, "done", "studies/budget", run_dir)
    else:
        _report(progress, "skip", "studies/budget", run_dir)
    return run_dir



def _ensure_horizon_study(
    env_name: str,
    base_dir: Path,
    *,
    force: bool,
    quick: bool,
    seed: int,
    preset: str,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "horizon_scaling"
    variants = []
    key = "env_config.T_train" if env_name == "cybersecurity" else "env_config.T"
    for horizon in experiment_presets.horizons(env_name, preset):
        variants.append({"label": f"T{horizon}", key: horizon, "train.horizon": horizon, "evaluation.horizon": horizon})
    config = benchmark_config(env_name, "simplex", base_dir / "studies", "horizon_scaling", seed=seed, quick=quick, preset=preset)
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/horizon", run_dir)
        run_variant_grid(config, "horizon-scaling", variants, default_command="diagnose-gradient")
        release_memory()
        _report(progress, "done", "studies/horizon", run_dir)
        generated = base_dir / "studies" / "horizon_scaling"
        if generated != run_dir and generated.exists():
            pass
    else:
        _report(progress, "skip", "studies/horizon", run_dir)
    return run_dir



def _ensure_continuous_horizon_study(
    env_name: str,
    base_dir: Path,
    *,
    force: bool,
    quick: bool,
    seed: int,
    preset: str,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "horizon_scaling"
    variants = []
    for horizon in experiment_presets.horizons(env_name, preset):
        variants.append({"label": f"T{horizon}", "env_config.T": horizon, "train.horizon": horizon, "evaluation.horizon": horizon})
    config = continuous_benchmark_config(env_name, base_dir / "studies", "horizon_scaling", seed=seed, quick=quick, preset=preset)
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/horizon", run_dir)
        run_variant_grid(config, "horizon-scaling", variants, default_command="diagnose-gradient")
        release_memory()
        _report(progress, "done", "studies/horizon", run_dir)
    else:
        _report(progress, "skip", "studies/horizon", run_dir)
    return run_dir



def _ensure_optimization_summary(
    env_name: str,
    base_dir: Path,
    bundle: Mapping[str, Any],
    *,
    force: bool,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "optimization_summary"
    config = {
        "env": env_name,
        "algorithm": "simplex",
        "train": {"output_dir": str(base_dir / "studies"), "run_name": "optimization_summary", "overwrite": True},
        "study": {
            "name": "optimization-summary",
            "input_dirs": [str(bundle["train"][algorithm]) for algorithm in ALGORITHMS],
        },
    }
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/optimization", run_dir)
        run_optimization_summary(config)
        release_memory()
        _report(progress, "done", "studies/optimization", run_dir)
    else:
        _report(progress, "skip", "studies/optimization", run_dir)
    return run_dir



def _ensure_continuous_optimization_summary(
    env_name: str,
    base_dir: Path,
    bundle: Mapping[str, Any],
    *,
    force: bool,
    progress: ProgressCallback | None = None,
) -> Path:
    run_dir = base_dir / "studies" / "optimization_summary"
    config = {
        "env": env_name,
        "algorithm": CONTINUOUS_ALGORITHM,
        "train": {"output_dir": str(base_dir / "studies"), "run_name": "optimization_summary", "overwrite": True},
        "study": {
            "name": "optimization-summary",
            "input_dirs": [str(path) for path in bundle["train"].values()],
        },
    }
    if force or not (run_dir / "diagnostics.csv").exists():
        _report(progress, "run", "studies/optimization", run_dir)
        run_optimization_summary(config)
        release_memory()
        _report(progress, "done", "studies/optimization", run_dir)
    else:
        _report(progress, "skip", "studies/optimization", run_dir)
    return run_dir



def _extended_study_paths(base_dir: Path, env_name: str) -> Dict[str, Path]:
    paths = {
        "optimizer_bias": base_dir / "studies" / "optimizer_bias",
        "robustness": base_dir / "studies" / "robustness",
        "ablation": base_dir / "studies" / "ablation",
        "signature": base_dir / "studies" / "signature_ablation",
        "adaptive": base_dir / "studies" / "adaptive_lambda",
        "particle": base_dir / "studies" / "particle_approximation",
        "scaling": base_dir / "studies" / "scaling",
    }
    if env_name in {"cucker-smale", "kuramoto"}:
        paths["particle_transfer"] = base_dir / "studies" / "particle_transfer"
    return paths



def _ensure_extended_studies(
    env_name: str,
    base_dir: Path,
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    force: bool,
    quick: bool,
    preset: str,
    progress: ProgressCallback | None = None,
) -> Dict[str, Path]:
    paths = _extended_study_paths(base_dir, env_name)
    train_paths = dict(bundle.get("train", {}))
    primary_algorithm = "simplex" if env_name in DISCRETE_BENCHMARKS else CONTINUOUS_ALGORITHM
    checkpoint = train_paths.get(primary_algorithm)
    checkpoint_path = Path(checkpoint) / "checkpoint.pt" if checkpoint is not None else None
    out: Dict[str, Path] = {}

    study_configs = {
        "optimizer_bias": _extended_study_config(config, base_dir, "optimizer_bias", "optimizer-bias", quick=quick, preset=preset),
        "ablation": _extended_study_config(config, base_dir, "ablation", "ablation", quick=quick, preset=preset),
        "signature": _extended_study_config(config, base_dir, "signature_ablation", "signature-ablation", quick=quick, preset=preset),
        "adaptive": _extended_study_config(config, base_dir, "adaptive_lambda", "adaptive-lambda", quick=quick, preset=preset),
        "particle": _extended_study_config(config, base_dir, "particle_approximation", "particle-approximation", quick=quick, preset=preset),
        "scaling": _extended_study_config(config, base_dir, "scaling", "scaling", quick=quick, preset=preset),
    }
    if checkpoint_path is not None:
        robustness = _extended_study_config(config, base_dir, "robustness", "robustness", quick=quick, preset=preset)
        robustness["checkpoint"] = str(checkpoint_path)
        study_configs["robustness"] = robustness
    if env_name in {"cucker-smale", "kuramoto"}:
        study_configs["particle_transfer"] = _extended_study_config(config, base_dir, "particle_transfer", "particle-transfer", quick=quick, preset=preset)

    for name, study_config in study_configs.items():
        run_dir = paths[name]
        if force or not (run_dir / "diagnostics.csv").exists():
            _report(progress, "run", f"studies/{name}", run_dir)
            run_study(study_config)
            release_memory()
            _report(progress, "done", f"studies/{name}", run_dir)
        else:
            _report(progress, "skip", f"studies/{name}", run_dir)
        out[name] = run_dir
    return out



def _extended_study_config(
    config: Mapping[str, Any],
    base_dir: Path,
    run_name: str,
    study_name: str,
    *,
    quick: bool,
    preset: str,
) -> Dict[str, Any]:
    child = json.loads(json.dumps(config))
    child.setdefault("train", {})
    child["train"]["output_dir"] = str(base_dir / "studies")
    child["train"]["run_name"] = run_name
    child["train"]["overwrite"] = True
    child.setdefault("diagnostic", {})
    diagnostic_defaults = experiment_presets.diagnostic_config(preset)
    child["diagnostic"].setdefault("replications", diagnostic_defaults["replications"])
    child["diagnostic"].setdefault("samples", diagnostic_defaults["samples"])
    child["diagnostic"].setdefault("lambdas", diagnostic_defaults["lambdas"])
    child["diagnostic"].setdefault("etas", diagnostic_defaults["etas"])
    child["study"] = _default_study_payload(study_name, child, quick=quick, preset=preset)
    return child



def _default_study_payload(study_name: str, config: Mapping[str, Any], *, quick: bool, preset: str) -> Dict[str, Any]:
    env_name = str(config.get("env"))
    if study_name == "optimizer-bias":
        return {"name": study_name, "lambdas": experiment_presets.lambda_grid(preset), "reference_index": 0}
    if study_name == "ablation":
        return {"name": study_name, "command": "diagnose-gradient"}
    if study_name == "signature-ablation":
        return {
            "name": study_name,
            "command": "diagnose-functional-law",
            "modes": ["full", "reduced", "underspecified"],
            "dimensions": experiment_presets.signature_dims(preset),
        }
    if study_name == "adaptive-lambda":
        return {"name": study_name, "command": "train", "fixed_lambdas": experiment_presets.fixed_lambda_variants(preset)}
    if study_name == "particle-approximation":
        return {"name": study_name, "particles": experiment_presets.particle_grid(preset)}
    if study_name == "particle-transfer":
        return {
            "name": study_name,
            "train_particles": experiment_presets.particle_grid(preset)[:3],
            "eval_particles": experiment_presets.particle_grid(preset),
        }
    if study_name == "scaling":
        return {"name": study_name, "command": "diagnose-gradient", "parameters": experiment_presets.parameter_grid(env_name, preset)}
    if study_name == "robustness":
        return {"name": study_name}
    return {"name": study_name}



def _with_output(config: Mapping[str, Any], output_dir: Path, run_name: str) -> Dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied.setdefault("train", {})
    copied["train"]["output_dir"] = str(output_dir)
    copied["train"]["run_name"] = run_name
    copied["train"]["overwrite"] = True
    return copied



def _report(progress: ProgressCallback | None, status: str, label: str, path: Path | None = None) -> None:
    if progress is not None:
        progress(status, label, path)



def _jsonable_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, Mapping):
            result[key] = _jsonable_bundle(value)
        else:
            result[key] = value
    return result


__all__ = [
    "bundle_paths",
    "continuous_bundle_paths",
    "ensure_continuous_benchmark_bundle",
    "ensure_discrete_benchmark_bundle",
]
