from __future__ import annotations

import gc

import torch


def release_memory() -> None:
    """Release Python objects and unused CUDA cache between independent runs."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


__all__ = ["release_memory"]
