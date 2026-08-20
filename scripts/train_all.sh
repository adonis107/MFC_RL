#!/usr/bin/env bash
# Parallel launcher for scripts/train.py. Python remains the source of truth
# for what the experiment grid is (scripts/train.py --list enumerates it,
# one runnable command per line, honoring the same axis-override flags as a
# single job, and any config's own `groups` restriction if it sets one —
# e.g. configs/twostate.py's MAIN — instead of the full cartesian product);
# this script's only job is scheduling those commands concurrently. Each
# individual `scripts/train.py` invocation already skips its own job if the
# output .pt exists (pass --overwrite to force a retrain), so a
# killed/resumed run just re-issues the same command list and picks up
# where it left off — no bookkeeping here.
#
# After the parallel pass, runs --materialize-duplicates once (sequential,
# fast): copies any budget-mode-invariant duplicate jobs (mfreinforce ignores
# budget_mode entirely — see ALGORITHMS_WITHOUT_BUDGET_MODE_DEPENDENCE in
# scripts/train.py) from their now-trained primaries. A no-op for any
# env/alg/config with no such duplicates, so this is always safe to run.
#
# Usage:
#   scripts/train_all.sh <workers> <scripts/train.py args...>
#
# Examples:
#   scripts/train_all.sh 4 --env twostate --alg simplex --config main --dtype float32
#   scripts/train_all.sh 4 --env twostate --alg mfreinforce --config main --dtype float32
#   scripts/train_all.sh 1 --env lq --alg exact_gradient --config main
#
# Run once per algorithm (the full main sweep is `algorithms x everything
# --list enumerates`); each invocation's --list output is independent, so
# separate scripts/train_all.sh calls for different algorithms are safe to
# run one after another or concurrently.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <workers> <scripts/train.py args...>" >&2
  echo "example: $0 4 --env twostate --alg simplex --config main --dtype float32" >&2
  exit 1
fi

WORKERS="$1"
shift

cd "$(dirname "$0")/.."

JOB_COUNT="$(uv run python scripts/train.py "$@" --list | wc -l)"
echo "scheduling $JOB_COUNT job(s) across $WORKERS worker(s): $*" >&2

# Every worker inherits this terminal, so with more than one of them the
# per-run progress bars would redraw over each other into noise (and each
# job's stderr is still a tty, so train.py's own "auto" cannot detect it —
# see src/mfc/progress.py). Workers fall back to their one-line-per-job
# "done in ...s" output, which interleaves cleanly.
PROGRESS_ARG=""
if [ "$WORKERS" -gt 1 ]; then
  PROGRESS_ARG=" --progress off"
fi

uv run python scripts/train.py "$@" --list | xargs -d '\n' -P "$WORKERS" -I CMD bash -c "CMD$PROGRESS_ARG"

echo "materializing any budget-mode-invariant duplicates..." >&2
uv run python scripts/train.py "$@" --materialize-duplicates
