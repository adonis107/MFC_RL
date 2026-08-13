from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _spaced_indices(length: int, count: int) -> List[int]:
    if length <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted({int(round(idx * (length - 1) / max(1, count - 1))) for idx in range(count)})



def _numeric_sorted(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=[column]).sort_values(column)



def _state_names(env_name: str) -> Dict[int, str]:
    return {
        "twostate": {0: "0", 1: "1"},
        "advertising": {0: "noncustomer", 1: "customer"},
        "cybersecurity": {0: "DI", 1: "DS", 2: "UI", 3: "US"},
        "distribution-planning": {idx: str(idx) for idx in range(10)},
    }[env_name]



def _time_metric_columns(env_name: str) -> List[str]:
    return {
        "twostate": ["mass_state_1", "target_abs_error"],
        "advertising": ["customer_fraction", "advertising_rate", "advertising_cost", "population_gain"],
        "cybersecurity": ["infected_fraction", "defended_fraction", "running_reward", "update_rate"],
        "distribution-planning": ["target_l1", "target_l2", "target_w1_ring_proxy", "movement_cost"],
        "lq": ["mean", "optimal_mean", "variance", "optimal_variance", "mean_error", "variance_error"],
        "portfolio": ["mean", "optimal_mean", "variance", "optimal_variance", "mean_error", "variance_error"],
        "cucker-smale": ["velocity_dispersion", "spatial_diameter", "control_energy"],
        "kuramoto": ["order_parameter", "target_aligned_order", "synchronization_cost", "control_energy"],
    }[env_name]


__all__: list[str] = []
