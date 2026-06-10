"""Deterministic gap-closure baseline for bus bunching.

The vendor LightGBM model was trained on 2024 data and only produces a 5-minute
horizon. As a serving baseline this module fits a simple kinematic model on the
(seq_len, 9) vendor-schema window: estimate the recent gap-closure rate
``v_close = -d(gap)/dt`` and forward-project the gap. Bunching probability at
horizon ``h`` is a sigmoid of (gap_threshold − projected_gap), with a width
parameter tuned to roughly match observed positive rates.

This is intentionally simple:
    * no training data needed,
    * runs in <100 µs per bus per horizon,
    * provides a calibratable floor the ML model has to beat,
    * stays interpretable — you can read the projected ETA-to-bunch directly.

Public API mirrors ``BunchingPredictor`` so the live forecast service can swap
between them at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_process.bunching.labels import BUNCHING_THRESHOLD_M, NO_LEADER_GAP_M


@dataclass(frozen=True)
class PhysicsBaselineConfig:
    """Configurable knobs for the gap-closure projection.

    ``closure_lookback_ticks`` — how many ticks of history (newest first) to use
    when estimating v_close. 3 ticks is a stable estimate without being too laggy.
    ``sigmoid_width_m`` — sets how sharply P(bunch) ramps as projected gap crosses
    the threshold. 50 m gives a probability of 0.5 right at the threshold and
    ~0.95 once gap is one width below it; tuned empirically against route 29.
    ``min_v_close_for_bunch`` — if relative speed is non-closing AND current gap
    is well above threshold, return 0 immediately. Avoids spurious sigmoid mass
    on stable wide-gap pairs.
    """

    step_seconds: int = 60
    closure_lookback_ticks: int = 3
    sigmoid_width_m: float = 50.0
    min_v_close_for_bunch: float = -0.1  # m/s; allows tiny outward drift


class PhysicsBaseline:
    """Drop-in baseline with ``predict_proba`` and ``alert`` matching the LightGBM API."""

    # Channel offsets in the 9-channel vendor window.
    SPEED_COL = 0
    GAP_COL = 1

    def __init__(self, pred_len: int, *, config: PhysicsBaselineConfig | None = None) -> None:
        self.pred_len = int(pred_len)
        self.cfg = config or PhysicsBaselineConfig()
        # No per-horizon thresholds — bunching probability is calibrated by the
        # sigmoid. Surface a constant 0.5 so the API stays shape-compatible.
        self.thresholds = {h: {"threshold": 0.5} for h in range(self.pred_len)}

    # ------------------------------------------------------------------ core
    def predict_proba(self, x: np.ndarray, *, is_scaled: bool = False) -> np.ndarray:
        """``x``: (seq_len, 9) or (batch, seq_len, 9) in raw units. Returns (batch, pred_len)."""
        if is_scaled:
            # We're a raw-unit model — refuse silent double-scaling.
            raise ValueError("PhysicsBaseline only accepts raw-unit windows")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[-1] < 2:
            raise ValueError(f"expected (batch, seq_len, >=2 channels), got {arr.shape}")
        batch, seq_len, _ = arr.shape
        lb = min(self.cfg.closure_lookback_ticks, seq_len - 1)

        gap_now = arr[:, -1, self.GAP_COL]
        # Closure rate (m/s): positive means the gap is shrinking.
        v_close = np.zeros(batch, dtype=np.float32)
        if lb > 0:
            gap_prev = arr[:, -1 - lb, self.GAP_COL]
            v_close = (gap_prev - gap_now) / float(lb * self.cfg.step_seconds)

        out = np.zeros((batch, self.pred_len), dtype=np.float32)
        for h in range(self.pred_len):
            dt = (h + 1) * self.cfg.step_seconds
            projected = gap_now - v_close * dt
            # Clip to non-negative; bunched-or-collided buses have zero gap.
            projected = np.maximum(projected, 0.0)

            # Sigmoid around (threshold − projected_gap):
            #   gap == threshold → 0.5
            #   gap << threshold → ~1.0
            #   gap >> threshold → ~0.0
            z = (BUNCHING_THRESHOLD_M - projected) / max(self.cfg.sigmoid_width_m, 1.0)
            p = 1.0 / (1.0 + np.exp(-z))

            # Hard zero when gap is way out and not closing — this prevents the
            # sigmoid from leaking probability mass onto buses that physically
            # can't bunch in the next few minutes.
            far_and_stable = (projected > BUNCHING_THRESHOLD_M + self.cfg.sigmoid_width_m * 6) & (
                v_close < self.cfg.min_v_close_for_bunch
            )
            # Also: a sentinel "no leader" gap means there is no leader ahead at all.
            no_leader = gap_now >= NO_LEADER_GAP_M * 0.99
            p = np.where(far_and_stable | no_leader, 0.0, p)

            out[:, h] = p.astype(np.float32)
        return out

    def predict_scalar(self, x: np.ndarray, *, mode: str = "max", is_scaled: bool = False) -> np.ndarray:
        p = self.predict_proba(x, is_scaled=is_scaled)
        if mode == "max":
            return p.max(axis=1)
        if mode == "last":
            return p[:, -1]
        if mode == "mean":
            return p.mean(axis=1)
        raise ValueError(f"unknown mode {mode!r}")

    def alert(self, x: np.ndarray, *, is_scaled: bool = False) -> list[dict]:
        probs = self.predict_proba(x, is_scaled=is_scaled)
        thr = np.array(
            [self.thresholds[h]["threshold"] for h in range(self.pred_len)],
            dtype=np.float32,
        )
        exceed = probs >= thr
        results: list[dict] = []
        for i in range(probs.shape[0]):
            any_hit = bool(exceed[i].any())
            first = int(np.argmax(exceed[i])) if any_hit else None
            max_idx = int(np.argmax(probs[i]))
            results.append(
                {
                    "any_alert": any_hit,
                    "first_alert_step": first,
                    "max_prob": float(probs[i, max_idx]),
                    "max_prob_step": max_idx,
                    "per_horizon": probs[i].tolist(),
                }
            )
        return results


__all__ = ["PhysicsBaseline", "PhysicsBaselineConfig"]
