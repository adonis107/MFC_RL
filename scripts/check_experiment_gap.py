from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfc.experiments import notebook_helpers as nh  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report figure/result coverage gaps across benchmark notebooks.")
    parser.add_argument("--all-statuses", action="store_true", help="Print the full coverage matrix, not just missing rows.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Return exit code 1 when future/study-needed rows remain.")
    args = parser.parse_args(argv)

    envs = nh.DISCRETE_BENCHMARKS + nh.CONTINUOUS_BENCHMARKS
    tables = []
    for env_name in envs:
        table = nh.figure_coverage_matrix(env_name).copy()
        table.insert(0, "env", env_name)
        tables.append(table)
    coverage = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    missing = nh.global_figure_gap_table()

    if args.all_statuses and not coverage.empty:
        print(coverage.to_string(index=False))
    elif missing.empty:
        print("No future/study-needed figure rows remain.")
    else:
        print(missing.to_string(index=False))

    if not coverage.empty and "status" in coverage:
        print("\nStatus counts:")
        print(coverage["status"].value_counts().to_string())

    return 1 if args.fail_on_missing and not missing.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
