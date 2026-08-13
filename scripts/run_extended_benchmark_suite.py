from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfc.experiments import notebook_helpers as nh  # noqa: E402
from mfc.experiments import presets  # noqa: E402
from mfc.experiments.core.memory import release_memory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark bundles, including extended paper studies.")
    parser.add_argument(
        "--env",
        action="append",
        choices=["all", *nh.DISCRETE_BENCHMARKS, *nh.CONTINUOUS_BENCHMARKS],
        default=None,
        help="Benchmark to run. Can be passed multiple times. Defaults to all.",
    )
    parser.add_argument("--output-dir", default="runs/extended_benchmark_bundles", help="Directory for generated bundles.")
    parser.add_argument(
        "--preset",
        choices=presets.PRESET_NAMES,
        default="smoke",
        help=(
            "Budget/grid preset. 'smoke' is a tiny structural check, 'mid' is a laptop-scale full-repo check, "
            "and 'main' uses the five-seed paper defaults."
        ),
    )
    parser.add_argument("--full", action="store_true", help="Alias for --preset main.")
    parser.add_argument("--seeds", help="Comma-separated seed list. Defaults come from the selected preset.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing artifacts.")
    parser.add_argument("--core-only", action="store_true", help="Skip extended studies and generate only the core bundle.")
    parser.add_argument("--device", help="Override the helper default device, e.g. cuda or cpu.")
    parser.add_argument(
        "--memory-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Python GC and clear unused CUDA cache between benchmark bundles. Enabled by default.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a progress list for every env/seed bundle and its sub-runs. Enabled by default.",
    )
    args = parser.parse_args(argv)

    preset = "main" if args.full else args.preset
    if args.device:
        nh.set_default_device(args.device)
    seed_values = _parse_seeds(args.seeds) if args.seeds else presets.seeds(preset)
    quick = preset == "smoke"

    selected = args.env or ["all"]
    if "all" in selected:
        envs = nh.DISCRETE_BENCHMARKS + nh.CONTINUOUS_BENCHMARKS
    else:
        envs = selected

    base = Path(args.output_dir).resolve()
    total_runs = len(envs) * len(seed_values)
    progress = ConsoleProgress(enabled=args.progress, total_runs=total_runs)
    run_index = 0
    for env_name in envs:
        for seed in seed_values:
            run_index += 1
            env_dir = base / env_name if len(seed_values) == 1 else base / env_name / f"seed_{seed:03d}"
            progress.start_run(run_index, env_name, seed, preset, env_dir)
            callback: Callable[[str, str, Path | None], None] | None = progress.step if args.progress else None
            try:
                if env_name in nh.DISCRETE_BENCHMARKS:
                    bundle = nh.ensure_discrete_benchmark_bundle(
                        env_name,
                        env_dir,
                        quick=quick,
                        force=args.force,
                        extended=not args.core_only,
                        seed=seed,
                        preset=preset,
                        progress=callback,
                    )
                else:
                    bundle = nh.ensure_continuous_benchmark_bundle(
                        env_name,
                        env_dir,
                        quick=quick,
                        force=args.force,
                        extended=not args.core_only,
                        seed=seed,
                        preset=preset,
                        progress=callback,
                    )
            except Exception as exc:
                progress.fail_run(run_index, env_name, seed, exc)
                raise
            if args.memory_cleanup:
                release_memory()
            progress.finish_run(run_index, env_name, seed, bundle["base_dir"])
    return 0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


class ConsoleProgress:
    def __init__(self, *, enabled: bool, total_runs: int) -> None:
        self.enabled = enabled
        self.total_runs = max(1, int(total_runs))
        self.suite_start = time.monotonic()
        self.run_start = self.suite_start
        self._active: dict[str, float] = {}

    def start_run(self, index: int, env_name: str, seed: int, preset: str, run_dir: Path) -> None:
        self.run_start = time.monotonic()
        if not self.enabled:
            return
        print(
            f"\n[{index}/{self.total_runs}] {self._bar(index - 1)} {env_name} seed={seed} preset={preset}",
            flush=True,
        )
        print(f"    dir  {run_dir}", flush=True)

    def step(self, status: str, label: str, path: Path | None = None) -> None:
        if not self.enabled:
            return
        if status == "run":
            self._active[label] = time.monotonic()
            print(f"    RUN  {label}", flush=True)
            return
        if status == "done":
            started = self._active.pop(label, None)
            suffix = f" ({_format_elapsed(time.monotonic() - started)})" if started is not None else ""
            print(f"    OK   {label}{suffix}", flush=True)
            return
        if status == "skip":
            print(f"    SKIP {label}", flush=True)
            return
        detail = f" -> {path}" if path is not None else ""
        print(f"    {status.upper():<4} {label}{detail}", flush=True)

    def finish_run(self, index: int, env_name: str, seed: int, run_dir: Path | str) -> None:
        if not self.enabled:
            print(f"{env_name} seed={seed}: {run_dir}")
            return
        run_elapsed = _format_elapsed(time.monotonic() - self.run_start)
        suite_elapsed = _format_elapsed(time.monotonic() - self.suite_start)
        print(f"[{index}/{self.total_runs}] {self._bar(index)} DONE {env_name} seed={seed} in {run_elapsed}", flush=True)
        print(f"    total elapsed {suite_elapsed}; output {run_dir}", flush=True)

    def fail_run(self, index: int, env_name: str, seed: int, exc: Exception) -> None:
        if not self.enabled:
            return
        run_elapsed = _format_elapsed(time.monotonic() - self.run_start)
        suite_elapsed = _format_elapsed(time.monotonic() - self.suite_start)
        print(f"[{index}/{self.total_runs}] {self._bar(index - 1)} FAIL {env_name} seed={seed} after {run_elapsed}", flush=True)
        print(f"    total elapsed {suite_elapsed}; error {type(exc).__name__}: {exc}", flush=True)

    def _bar(self, completed: int, *, width: int = 20) -> str:
        completed = max(0, min(self.total_runs, completed))
        filled = round(width * completed / self.total_runs)
        return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{remainder:02d}s"


if __name__ == "__main__":
    raise SystemExit(main())
