from __future__ import annotations

from . import notebooks as _notebooks
from .notebooks import *  # noqa: F401,F403
from .notebooks import __all__  # noqa: F401


def set_default_device(device: str) -> None:
    _notebooks.set_default_device(device)
    globals()["DEFAULT_DEVICE"] = _notebooks.DEFAULT_DEVICE


if "set_default_device" not in __all__:
    __all__ = [*__all__, "set_default_device"]
