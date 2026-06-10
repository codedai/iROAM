"""Shadow-mode logging + lightweight drift summary for the forecast service.

When ``FORECAST_SHADOW_MODE=1`` is set in the environment, every call to
``run_forecast`` appends one JSON-line per eligible bus to
``out/shadow/<date>.jsonl``. The line carries the per-bus probabilities
*before truncation*, the post-truncation summary, current gap/closure, and a
hash of the feature window — enough to back-compute precision/recall once the
ground-truth outcome is known (i.e., once that bus's later trajectory has been
ingested), without ever storing PII.

A second file ``out/shadow/<date>_drift.csv`` is appended hourly with the
distribution summary (mean / std / p01 / p99) of each input channel across
the buses scored that hour. Drift in those statistics is the cheapest early
warning that the training distribution no longer matches what we're serving.

This module is intentionally side-effect-only. The caller dispatches to it
explicitly from ``run_forecast``; if shadow mode is off, nothing happens.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SHADOW_ENV_VAR = "FORECAST_SHADOW_MODE"
SHADOW_DIR_ENV_VAR = "FORECAST_SHADOW_DIR"
_DEFAULT_DIR = Path("out/shadow")


@dataclass(frozen=True)
class ShadowConfig:
    enabled: bool
    out_dir: Path


def load_config() -> ShadowConfig:
    enabled = os.environ.get(SHADOW_ENV_VAR, "").lower() in ("1", "true", "yes", "on")
    out_dir = Path(os.environ.get(SHADOW_DIR_ENV_VAR, str(_DEFAULT_DIR)))
    return ShadowConfig(enabled=enabled, out_dir=out_dir)


def _hash_window(window_flat: np.ndarray) -> str:
    """Short hash so we can dedupe identical inputs without storing them."""
    return hashlib.blake2b(window_flat.tobytes(), digest_size=8).hexdigest()


def log_predictions(
    cfg: ShadowConfig,
    *,
    bundle_label: str,
    route_id: str,
    direction_id: int,
    service_date: str,
    t_ref_min: float,
    eligible_rows: list[dict[str, Any]],
    raw_probs: np.ndarray | None,
    feature_batch: np.ndarray | None,
) -> None:
    """Append one JSON line per eligible bus to today's shadow log."""
    if not cfg.enabled or not eligible_rows:
        return
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{service_date}.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()

    with out_path.open("a") as f:
        for i, row in enumerate(eligible_rows):
            entry = {
                "ts": now_iso,
                "bundle": bundle_label,
                "route_id": route_id,
                "direction_id": direction_id,
                "service_date": service_date,
                "t_ref_min": float(t_ref_min),
                "bus_id": row.get("bus_id"),
                "vehicle_id": row.get("vehicle_id"),
                "trip_id": row.get("trip_id"),
                "stop_idx": row.get("stop_idx"),
                "forward_gap_m": row.get("forward_gap_m"),
                "gap_closure_m_s": row.get("gap_closure_m_s"),
                # post-truncation summary (what the UI shows)
                "max_prob": row.get("max_prob"),
                "max_prob_step": row.get("max_prob_step"),
                "first_alert_step": row.get("first_alert_step"),
                "useful_horizon_steps": row.get("useful_horizon_steps"),
                "per_horizon_truncated": row.get("per_horizon"),
                # pre-truncation probabilities — useful for re-evaluating
                # different truncation strategies offline.
                "per_horizon_raw": None if raw_probs is None else [
                    float(x) for x in raw_probs[i].tolist()
                ],
            }
            if feature_batch is not None and i < feature_batch.shape[0]:
                entry["feature_hash"] = _hash_window(feature_batch[i].reshape(-1))
            f.write(json.dumps(entry) + "\n")


def log_drift_summary(
    cfg: ShadowConfig,
    *,
    bundle_label: str,
    route_id: str,
    direction_id: int,
    feature_batch: np.ndarray,
) -> None:
    """Append a per-channel distribution summary to <date>_drift.csv.

    Fires once per ``run_forecast`` call when shadow mode is on. We don't
    aggregate over time inside this process (multiple workers, no shared
    state) — each line is one snapshot, and downstream tooling rolls them up.
    """
    if not cfg.enabled or feature_batch.size == 0:
        return
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    out_path = cfg.out_dir / f"{today}_drift.csv"
    is_new = not out_path.exists()
    flat = feature_batch.reshape(feature_batch.shape[0], -1)  # (B, F)
    means = flat.mean(axis=0); stds = flat.std(axis=0)
    p01 = np.percentile(flat, 1, axis=0); p99 = np.percentile(flat, 99, axis=0)
    with out_path.open("a") as f:
        if is_new:
            f.write("ts,bundle,route_id,direction_id,feature_idx,mean,std,p01,p99\n")
        ts = datetime.now(timezone.utc).isoformat()
        for j in range(flat.shape[1]):
            f.write(
                f"{ts},{bundle_label},{route_id},{direction_id},{j},"
                f"{means[j]:.6f},{stds[j]:.6f},{p01[j]:.6f},{p99[j]:.6f}\n"
            )


__all__ = ["ShadowConfig", "load_config", "log_predictions", "log_drift_summary"]
