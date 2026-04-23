"""Raw-unit ↔ scaled-unit conversion for model inputs and gap outputs.

The model was trained on z-score scaled features (see ``scaler.json``).
Live AVL pipelines typically have raw-unit measurements (m/s, metres),
so the helpers here close the gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def load_scaler(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _channel_scaler(scaler: Mapping, offset: int) -> tuple[float, float]:
    """Return (mean, std) for a given within-bus channel offset (0,1,2)."""
    if offset == 0:   # speed
        return scaler['speed_mean'], scaler['speed_std']
    if offset == 1:   # gap
        return scaler['gap_mean'], scaler['gap_std']
    return 0.0, 1.0   # aux is passed through unchanged


def scale_window(raw: np.ndarray, scaler: Mapping) -> np.ndarray:
    """Convert raw units to scaled units for a (seq_len, n_channels) window.

    ``n_channels`` must be a multiple of 3. Channels are interpreted as
    repeated (speed, gap, aux) triples, one per bus (target + upstream).
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] % 3 != 0:
        raise ValueError(
            f'Expected 2D (seq_len, 3k) window, got {raw.shape}'
        )
    scaled = np.empty_like(raw)
    for i in range(raw.shape[1]):
        m, s = _channel_scaler(scaler, i % 3)
        scaled[:, i] = (raw[:, i] - m) / s if s else raw[:, i]
    return scaled


def unscale_gap(scaled_gap: np.ndarray, scaler: Mapping) -> np.ndarray:
    """Inverse-scale a gap-only tensor back to metres."""
    return np.asarray(scaled_gap, dtype=np.float32) * scaler['gap_std'] + scaler['gap_mean']
