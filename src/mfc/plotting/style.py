"""Publication-quality Matplotlib styling.

The module provides a quiet, print-friendly visual system for papers, reports,
and research notebooks. It deliberately avoids a LaTeX dependency so figures
render consistently in local environments, CI, and Google Colab.

Typical use
-----------
>>> from style import new_figure, apply_style, save_figure
>>> fig, ax = new_figure(width="single")
>>> ax.plot(x, y, label="Estimate")
>>> apply_style(ax, xlabel=r"Time $t$", ylabel="Value", legend=True)
>>> save_figure(fig, "figure.pdf")

For an entire notebook, call ``set_style()`` once or use ``style_context()``
to limit the settings to a single block.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Restrained, colourblind-safe categorical colours derived from Okabe-Ito.
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

# Common single- and double-column figure widths, in inches.
FIGURE_WIDTHS = {"single": 3.35, "medium": 5.25, "double": 7.0}
GOLDEN_RATIO = (5**0.5 - 1) / 2


def _rc_params() -> dict[str, object]:
    """Return the style settings without mutating global Matplotlib state."""
    return {
        "figure.facecolor": INK["surface"],
        "figure.dpi": 140,
        "savefig.facecolor": INK["surface"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK["primary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.65,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "axes.axisbelow": True,
        "axes.grid": False,
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def set_style() -> None:
    """Apply the publication style globally for the current Python session."""
    mpl.rcParams.update(_rc_params())


@contextmanager
def style_context() -> Iterator[None]:
    """Temporarily apply the publication style inside a ``with`` block."""
    with mpl.rc_context(_rc_params()):
        yield


def color_for(index: int) -> str:
    """Return the stable categorical colour assigned to a zero-based index."""
    return CATEGORICAL[index % len(CATEGORICAL)]


def new_figure(
    *,
    width: Literal["single", "medium", "double"] | float = "medium",
    height: float | None = None,
    figsize: tuple[float, float] | None = None,
    constrained_layout: bool = True,
) -> tuple[Figure, Axes]:
    """Create a paper-sized figure and axes.

    ``figsize`` is retained for backwards compatibility. Otherwise, ``width``
    accepts a standard journal width or an explicit width in inches; height
    defaults to the golden-ratio proportion.
    """
    set_style()
    if figsize is None:
        figure_width = FIGURE_WIDTHS[width] if isinstance(width, str) else float(width)
        figure_height = height if height is not None else figure_width * GOLDEN_RATIO
        figsize = (figure_width, figure_height)

    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor=INK["surface"],
        constrained_layout=constrained_layout,
    )
    return fig, ax


def apply_style(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: Literal["x", "y", "both"] | None = "y",
    legend: bool = False,
    despine: bool = True,
) -> Axes:
    """Apply clean publication chrome after plotting the data."""
    ax.set_facecolor(INK["surface"])
    ax.set_axisbelow(True)

    ax.grid(False)
    if grid is not None:
        ax.grid(
            True,
            axis=grid,
            color=INK["grid"],
            linewidth=0.5,
            linestyle="-",
            alpha=0.65,
        )

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.65)

    ax.tick_params(colors=INK["secondary"], labelsize=8, pad=3)
    if title is not None:
        ax.set_title(title, color=INK["primary"], loc="center", pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK["primary"], labelpad=5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK["primary"], labelpad=5)
    if legend:
        style_legend(ax)
    return ax


def style_legend(ax: Axes, **kwargs):
    """Draw an unobtrusive legend consistent with the figure typography."""
    defaults = {
        "frameon": False,
        "fontsize": 8,
        "labelcolor": INK["secondary"],
        "borderaxespad": 0.3,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    **kwargs,
) -> Path:
    """Export a tightly cropped, publication-ready figure and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=transparent,
        **kwargs,
    )
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
