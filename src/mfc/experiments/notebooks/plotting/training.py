from __future__ import annotations

from typing import Mapping

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfc_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_comparison(histories: Mapping[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for algorithm, history in histories.items():
        if history.empty:
            continue
        x = history["episode"] if "episode" in history else range(len(history))
        y_col = "value" if "value" in history and history["value"].notna().any() else "objective"
        if y_col in history:
            axes[0].plot(x, history[y_col], marker="o", label=algorithm)
        if "grad_norm" in history:
            axes[1].plot(x, history["grad_norm"], marker="o", label=algorithm)
    axes[0].set_title("Training value/objective")
    axes[0].set_xlabel("episode")
    axes[0].legend()
    axes[1].set_title("Gradient norm")
    axes[1].set_xlabel("episode")
    axes[1].legend()
    fig.tight_layout()


__all__ = ["plot_training_comparison"]
