from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from .controls import load_control
from .gradient_steps import make_algorithm
from .registry import build_environment, default_algorithm_for_env, require_env_name


@dataclass
class RunResult:
    run_dir: Path
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    checkpoint_path: Optional[Path] = None
    diagnostics_path: Optional[Path] = None


def set_seed(seed: int, device: Optional[torch.device] = None) -> None:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    if (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available()):
        torch.cuda.manual_seed_all(int(seed))


def normalize_experiment_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(dict(config))
    env_name = require_env_name(config)
    config["algorithm"] = str(config.get("algorithm") or default_algorithm_for_env(env_name))
    config.setdefault("env_config", {})
    config.setdefault("algorithm_config", {})
    config.setdefault("train", {})
    config.setdefault("evaluation", {})
    return config


def load_checkpoint(path: str | os.PathLike[str], map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    config = {
        "env": payload["env"],
        "algorithm": payload["algorithm"],
        "env_config": payload["env_config"],
        "algorithm_config": payload.get("algorithm_config", {}),
        "train": payload.get("train_config", {}),
        "evaluation": payload.get("evaluation_config", {}),
    }
    spec, env = build_environment(config)
    control = load_control(spec, env, payload["control"], trainable=True)
    algorithm = make_algorithm(payload["algorithm"], env, payload.get("algorithm_config", {}))
    return {"env": env, "control": control, "algorithm": algorithm, "payload": payload, "config": config}


__all__ = ["RunResult", "load_checkpoint", "normalize_experiment_config", "set_seed"]
