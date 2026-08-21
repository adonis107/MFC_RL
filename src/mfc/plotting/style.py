"""Publication-oriented Matplotlib helper wrappers.

The notebooks in this repository are used as research artifacts: figures
should have readable typography, clear legends, restrained grids, and stable
paper-friendly dimensions without every cell repeating the same rcParams.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Restrained, colorblind-safe categorical colors derived from Okabe-Ito.
CATEGORICAL = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#111111",  # black
]
SEQUENTIAL = ["#EFF3F8", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
DIVERGING = {"low": "#2166AC", "mid": "#F7F7F7", "high": "#B2182B"}
INK = {
    "primary": "#1A1A1A",
    "secondary": "#4D4D4D",
    "muted": "#737373",
    "grid": "#D9D9D9",
    "axis": "#737373",
    "surface": "#FFFFFF",
}
STATUS = {
    "good": "#16835D",
    "warning": "#C58A00",
    "serious": "#C65D21",
    "critical": "#B2182B",
}
FIGURE_WIDTHS = {"single": 3.35, "medium": 5.25, "double": 7.0}
GOLDEN_RATIO = (5**0.5 - 1) / 2


def set_style() -> None:
    """Install compact defaults suitable for notebooks and paper exports."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.facecolor": INK["surface"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": INK["surface"],
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK["primary"],
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.7,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.55,
            "lines.linewidth": 1.9,
            "lines.markersize": 5,
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 2.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


@contextmanager
def style_context() -> Iterator[None]:
    """Temporarily apply the repository's publication plotting defaults."""
    with mpl.rc_context():
        set_style()
        yield


def color_for(index: int) -> str:
    """Return the stable categorical color assigned to a zero-based index."""
    return CATEGORICAL[index % len(CATEGORICAL)]


def new_figure(
    *,
    width: Literal["single", "medium", "double"] | float = "medium",
    height: float | None = None,
    figsize: tuple[float, float] | None = None,
    constrained_layout: bool = True,
) -> tuple[Figure, Axes]:
    """Create a Matplotlib figure and axes with publication defaults."""
    set_style()
    if figsize is None:
        figure_width = FIGURE_WIDTHS[width] if isinstance(width, str) else float(width)
        figure_height = height if height is not None else figure_width * GOLDEN_RATIO
        figsize = (figure_width, figure_height)
    return plt.subplots(figsize=figsize, facecolor=INK["surface"], constrained_layout=constrained_layout)


def apply_style(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: Literal["x", "y", "both"] | bool | None = "y",
    legend: bool = False,
    despine: bool = False,
) -> Axes:
    """Apply common labels, grid, and legend styling to an axes."""
    del despine
    ax.set_facecolor(INK["surface"])
    ax.tick_params(colors=INK["secondary"])
    if title is not None:
        ax.set_title(title, color=INK["primary"], loc="center", pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK["primary"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK["primary"])
    if grid:
        ax.grid(True, axis=grid if isinstance(grid, str) else "both")
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.7)
    ax.margins(x=0.02)
    if legend:
        style_legend(ax)
    return ax


def style_legend(ax: Axes, **kwargs):
    """Draw a compact legend when labeled artists are present."""
    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, label) for h, label in zip(handles, labels) if label and not label.startswith("_")]
    if not visible:
        return None
    handles, labels = zip(*visible)
    defaults = {"fontsize": 9, "frameon": True}
    defaults.update(kwargs)
    legend = ax.legend(handles, labels, **defaults)
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_edgecolor("0.85")
    return legend


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    **kwargs,
) -> Path:
    """Save a figure and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, transparent=transparent, **kwargs)
    return output


__all__ = [
    "CATEGORICAL",
    "DIVERGING",
    "FIGURE_WIDTHS",
    "INK",
    "SEQUENTIAL",
    "STATUS",
    "apply_style",
    "color_for",
    "new_figure",
    "save_figure",
    "set_style",
    "style_context",
    "style_legend",
]
