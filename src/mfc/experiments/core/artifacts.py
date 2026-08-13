from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .registry import DEFAULT_DEVICE, ENVIRONMENTS, default_algorithm_for_env


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if is_dataclass(value):
        return {field.name: json_default(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_default(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_default(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def csv_value(value: Any) -> Any:
    value = json_default(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def parse_json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if not keys or any(not part for part in keys):
        raise ValueError(f"Invalid override key {dotted_key!r}.")
    cursor = config
    for key in keys[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {dotted_key!r}; {key!r} is not a mapping.")
        cursor = next_value
    cursor[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must have the form dotted.path=json_value.")
        key, raw_value = override.split("=", 1)
        set_dotted(config, key, parse_json_value(raw_value))
    return config


def load_json_config(path: str | os.PathLike[str] | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object.")
    return payload


def checkpoint_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def metadata(command: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        package_version = importlib.metadata.version("mfc-rl")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "package": {"name": "mfc-rl", "version": package_version},
        "config": json_default(config),
        "git": git_metadata(),
    }


def git_metadata() -> Dict[str, Any]:
    root = Path.cwd()
    result: Dict[str, Any] = {}
    for key, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }.items():
        try:
            completed = subprocess.run(command, cwd=root, check=False, text=True, capture_output=True)
            if completed.returncode == 0:
                result[key] = completed.stdout.strip()
        except OSError:
            pass
    return result


def make_run_dir(command: str, config: Mapping[str, Any]) -> Path:
    train_config = dict(config.get("train", {}))
    output_dir = Path(train_config.get("output_dir", config.get("output_dir", "runs")))
    run_name = train_config.get("run_name", config.get("run_name"))
    env_name = str(config.get("env", "env"))
    algorithm_name = str(config.get("algorithm", default_algorithm_for_env(env_name) if env_name in ENVIRONMENTS else "algorithm"))
    seed = int(train_config.get("seed", 0))
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{command}_{env_name}_{algorithm_name}_seed{seed}"
    run_dir = output_dir / str(run_name)
    if run_dir.exists() and not bool(train_config.get("overwrite", False)):
        suffix = 1
        base = run_dir
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=bool(train_config.get("overwrite", False)))
    return run_dir


_json_default = json_default
_write_json = write_json
_write_csv = write_csv
_csv_value = csv_value
_parse_json_value = parse_json_value
_set_dotted = set_dotted
_checkpoint_rng_state = checkpoint_rng_state
_metadata = metadata
_git_metadata = git_metadata
_make_run_dir = make_run_dir


__all__ = [
    "_checkpoint_rng_state",
    "_csv_value",
    "_git_metadata",
    "_json_default",
    "_make_run_dir",
    "_metadata",
    "_parse_json_value",
    "_set_dotted",
    "_write_csv",
    "_write_json",
    "apply_overrides",
    "checkpoint_rng_state",
    "csv_value",
    "dtype_name",
    "git_metadata",
    "json_default",
    "load_json_config",
    "make_run_dir",
    "metadata",
    "parse_json_value",
    "set_dotted",
    "write_csv",
    "write_json",
]
