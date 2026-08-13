from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfc.experiments import notebook_helpers as nh  # noqa: E402
from mfc.experiments import presets  # noqa: E402


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
    for env_name in envs:
        for seed in seed_values:
            env_dir = base / env_name if len(seed_values) == 1 else base / env_name / f"seed_{seed:03d}"
            if env_name in nh.DISCRETE_BENCHMARKS:
                bundle = nh.ensure_discrete_benchmark_bundle(
                    env_name,
                    env_dir,
                    quick=quick,
                    force=args.force,
                    extended=not args.core_only,
                    seed=seed,
                    preset=preset,
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
                )
            print(f"{env_name} seed={seed}: {bundle['base_dir']}")
    return 0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
