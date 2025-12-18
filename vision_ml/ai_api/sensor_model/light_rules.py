# light_rules.py
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class PlantLightProfile:
    name: str
    soft_max: float
    hard_max: float
    max_bright_hours: float
    spike_factor: float = 2.0


FITTONIA_PROFILE = PlantLightProfile(
    name="fittonia",
    soft_max=1000.0,
    hard_max=2000.0,
    max_bright_hours=1.5,
    spike_factor=2.0
)


def _safe_div(a: float, b: float, eps: float = 1e-6):
    return a / (b if abs(b) > eps else eps)


def compute_too_bright_now(light, rolling_avg_light_6h, light_hours_today, profile=FITTONIA_PROFILE):
    if light >= profile.hard_max:
        return 1

    if light >= profile.soft_max and light_hours_today >= profile.max_bright_hours:
        return 1

    rel_spike = _safe_div(light, rolling_avg_light_6h)
    if rolling_avg_light_6h > 0 and rel_spike >= profile.spike_factor and light >= profile.soft_max * 0.8:
        return 1

    return 0


def compute_too_bright_batch(X: np.ndarray, feature_index: Dict[str, int], profile=FITTONIA_PROFILE):
    light_idx = feature_index["light"]
    roll_idx = feature_index["rolling_avg_light_6h"]
    hours_idx = feature_index["light_hours_today"]

    out = []
    for row in X:
        out.append(compute_too_bright_now(
            light=row[light_idx],
            rolling_avg_light_6h=row[roll_idx],
            light_hours_today=row[hours_idx],
            profile=profile
        ))
    return np.array(out, dtype=int)
