from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window during tests

import torch

# Harmonizes device usage across the whole suite: any tensor created without
# an explicit device (test literals, TwoState()'s own auto-detected default,
# etc.) lands on the same device. torch.Generator() does NOT follow this —
# construct those with an explicit device= matching the tensors they're used with.
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
