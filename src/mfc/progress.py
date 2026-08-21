"""
Progress bars for training loops, and the single place that decides whether
to draw one.

Every training loop in this repo is a plain `for m in range(n_train)`
(`scripts/train.py`'s `train_run` for the discrete benchmarks,
`mfc.algorithms.continuous.train` for the continuous
ones), so `training_bar` just wraps that range and each loop is written the
same way.

What this module really exists for is the *policy*, which is not obvious:
a paid run gets launched in three quite different ways, and only one of them
wants a live bar.

  - One job in a terminal. A bar is exactly right: these runs are long
    (distribution planning's `main` is 100_000 iterations) and otherwise
    print nothing at all between "starting" and "done".
  - Output redirected to a log file. A bar would append thousands of
    carriage-returned frames to the log.
  - Many jobs at once under `scripts/train_all.sh`'s `xargs -P`. Every worker
    inherits the same terminal, so N bars overwrite each other into noise —
    and stderr is still a tty, so no amount of auto-detection sees it. This
    is why `train_all.sh` passes `--progress off` for multi-worker runs and
    why `progress_enabled` takes an explicit override at all.

Bars go to stderr (tqdm's default), leaving stdout to the per-job summary
lines `scripts/train.py` prints — so `2>/dev/null` drops the bars and keeps
the results.
"""

from __future__ import annotations

import sys

from tqdm import tqdm  # not tqdm.auto: that wants ipywidgets for its notebook bar and, without it, warns once per cell before falling back to exactly this one

__all__ = ["PROGRESS_MODES", "progress_enabled", "training_bar"]

PROGRESS_MODES = ("auto", "on", "off")


def progress_enabled(mode: str = "auto") -> bool:
    """
    Resolve a `--progress` mode to a yes/no. "auto" draws a bar when there is
    plausibly a human watching: an interactive terminal, or a Jupyter kernel
    (the notebooks call `run_all`/`run_continuous` directly, and a kernel's
    stderr is never a tty, so `isatty` alone would wrongly suppress bars
    there). "auto" cannot detect sibling workers sharing one terminal — see
    this module's docstring; that case needs an explicit "off".
    """
    if mode not in PROGRESS_MODES:
        raise ValueError(f"unknown progress mode {mode!r}; available: {list(PROGRESS_MODES)}")
    if mode != "auto":
        return mode == "on"
    return sys.stderr.isatty() or "ipykernel" in sys.modules


def training_bar(n_train: int, *, desc: str | None = None) -> tqdm:
    """
    A tqdm over `range(n_train)` to drive a training loop, as
    `with training_bar(...) as bar: for m in bar: ...`. `desc=None` returns a
    disabled bar whose `update`/`set_postfix_str` are no-ops, so a loop can
    wrap itself unconditionally instead of branching on whether progress is
    on. `leave=False` erases the bar when the run finishes, leaving only the
    caller's own summary line in the scrollback.
    """
    return tqdm(
        range(n_train),
        desc=desc,
        disable=desc is None,
        leave=False,
        unit="it",
        dynamic_ncols=True,  # survives a terminal resize mid-run
        mininterval=0.5,  # these runs go for hours; 10 redraws/s buys nothing
        smoothing=0.05,  # near-global rate average: the per-iteration cost is steady, so this gives a far calmer ETA
    )
