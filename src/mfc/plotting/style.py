"""
Shared chart style: a validated categorical palette and consistent chrome,
so every figure in every notebook reads as one system. Values are the
dataviz skill's reference light-mode palette (references/palette.md) —
color is assigned by fixed order (series identity), never cycled by rank;
this module only supplies the visual language, not the plots themselves
(see `diagnostics.py`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Fixed-order categorical hues. Validated for both the "adjacent" pairlist
# (overlaid lines/bars, any number of series) and, for the first three slots
# only, the "all-pairs" pairlist (scatter/small-multiples) — see palette.md.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue sequential ramp (blue, light->dark), for magnitude/order rather than identity.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Warm/cool poles for signed quantities (e.g. bias), neutral midpoint gray.
DIVERGING = {"low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def color_for(index: int) -> str:
    """Fixed categorical color for series `index` (0-based). Assign by the
    series' identity (e.g. its position in a sorted lambda/state list) and
    keep that assignment stable across a figure — never reassign by rank."""
    return CATEGORICAL[index % len(CATEGORICAL)]


def new_figure(*, figsize: tuple[float, float] = (6.0, 4.0)):
    """A figure/axes pair on the chart surface, ready for `apply_style`."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=INK["surface"])
    return fig, ax


def apply_style(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    """Common chart chrome: recessive gridlines, muted ticks, no top/right
    spines, primary-ink title, secondary-ink axis labels. Call last, after
    the marks are drawn, so `ax.legend()` (if any) picks up correct labels."""
    ax.set_facecolor(INK["surface"])
    ax.set_axisbelow(True)
    ax.grid(True, color=INK["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK["axis"])
    ax.spines["bottom"].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=9)
    if title:
        ax.set_title(title, color=INK["primary"], fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK["secondary"], fontsize=9)
    return ax


def style_legend(ax, **kwargs):
    """A legend consistent with the chart chrome: no frame, secondary ink."""
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"], **kwargs)
    return legend
