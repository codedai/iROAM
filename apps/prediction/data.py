"""Materialise the bunching dataset into numpy arrays for any trainer.

The parquet shards on disk are split into per-(date, direction) files holding
opaque float32 byte-blobs of (seq_len, N_CHANNELS), (seq_len, n_extra), and
(pred_len,) per row. This module loads them once and caches the resulting
arrays so the six trainers downstream don't re-parse the parquet each time.

Returned splits have BOTH the vendor (9-channel) block and the rich extras
(7-channel) so each trainer can pick which schema it consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data_process.bunching.labels import N_CHANNELS, N_EXTRA


@dataclass
class Split:
    """Materialised arrays for one chronological split."""

    X_vendor: np.ndarray   # (N, seq_len, 9)   float32, raw units
    X_extras: np.ndarray   # (N, seq_len, n_extra) float32, raw units (passthrough)
    Y: np.ndarray          # (N, pred_len) float32; NaN where label missing
    dates: list[str]       # service_date per example (for sanity / debugging)

    @property
    def n(self) -> int:
        return self.X_vendor.shape[0]

    @property
    def seq_len(self) -> int:
        return self.X_vendor.shape[1]

    @property
    def pred_len(self) -> int:
        return self.Y.shape[1]


@dataclass
class Dataset:
    """All three chronological splits + metadata."""

    train: Split
    val: Split
    test: Split
    manifest: dict

    @property
    def seq_len(self) -> int:
        return int(self.manifest["seq_len"])

    @property
    def pred_len(self) -> int:
        return int(self.manifest["pred_len"])

    @property
    def n_extra(self) -> int:
        return int(self.manifest.get("n_extra", N_EXTRA))


def _load_shards(
    dataset_dir: Path, manifest: dict, split_dates: list[str], seq_len: int, n_extra: int,
) -> Split:
    # Use the dataset's *own* n_channels (from its manifest) rather than the
    # imported N_CHANNELS constant — the constant tracks the current default
    # schema, but old datasets on disk may use a different one.
    n_channels = int(manifest.get("n_channels", N_CHANNELS))
    shards = [s for s in manifest["shards"] if s["service_date"] in set(split_dates)]
    frames = [pd.read_parquet(dataset_dir / s["path"]) for s in shards]
    if not frames:
        return Split(
            X_vendor=np.zeros((0, seq_len, n_channels), dtype=np.float32),
            X_extras=np.zeros((0, seq_len, n_extra), dtype=np.float32),
            Y=np.zeros((0, manifest["pred_len"]), dtype=np.float32),
            dates=[],
        )
    df = pd.concat(frames, ignore_index=True)
    X_v = np.stack(
        [np.frombuffer(b, dtype=np.float32).reshape(seq_len, n_channels) for b in df["window"]]
    )
    X_e = np.stack(
        [np.frombuffer(b, dtype=np.float32).reshape(seq_len, n_extra) for b in df["extras"]]
    )
    Y = np.stack(
        [np.frombuffer(b, dtype=np.float32) for b in df["labels"]]
    )
    return Split(
        X_vendor=X_v.astype(np.float32),
        X_extras=X_e.astype(np.float32),
        Y=Y.astype(np.float32),
        dates=df["service_date"].astype(str).tolist(),
    )


@lru_cache(maxsize=4)
def load_dataset(dataset_dir_str: str) -> Dataset:
    """Cached loader — call once per process and reuse."""
    dataset_dir = Path(dataset_dir_str)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    seq_len = int(manifest["seq_len"])
    n_extra = int(manifest.get("n_extra", N_EXTRA))
    split = manifest["split"]
    return Dataset(
        train=_load_shards(dataset_dir, manifest, split["train_dates"], seq_len, n_extra),
        val=_load_shards(dataset_dir, manifest, split["val_dates"], seq_len, n_extra),
        test=_load_shards(dataset_dir, manifest, split["test_dates"], seq_len, n_extra),
        manifest=manifest,
    )


def _speed_gap_offsets(n_ch: int) -> tuple[list[int], list[int]]:
    """Map vendor-block channel offsets to (speed_cols, gap_cols) for whichever
    schema produced this block.

    Schema v1 = 9 channels = 3 buses × (speed, gap, aux) → speed at 0,3,6 ; gap at 1,4,7.
    Schema v2 = 6 channels = 3 buses × (speed, fwd_gap)  → speed at 0,2,4 ; gap at 1,3,5.

    We dispatch by ``n_ch % 3 == 0`` (legacy) vs anything else (v2). The
    common cases (6 or 9) cover all current bundles.
    """
    if n_ch % 3 == 0 and n_ch >= 9:
        # Legacy: 3-channel triples (speed, gap, aux).
        speed = [j for j in range(n_ch) if j % 3 == 0]
        gap = [j for j in range(n_ch) if j % 3 == 1]
    else:
        # Schema v2: 2-channel pairs (speed, fwd_gap).
        speed = [j for j in range(n_ch) if j % 2 == 0]
        gap = [j for j in range(n_ch) if j % 2 == 1]
    return speed, gap


def compute_scaler(X_vendor_train: np.ndarray) -> dict:
    """Z-score stats over speed and gap channels of the vendor block."""
    n_ch = X_vendor_train.shape[2]
    speed_cols, gap_cols = _speed_gap_offsets(n_ch)
    speed_vals = X_vendor_train[:, :, speed_cols].reshape(-1)
    gap_vals = X_vendor_train[:, :, gap_cols].reshape(-1)
    speed_mean = float(np.mean(speed_vals))
    speed_std = float(np.std(speed_vals) or 1.0)
    gap_mean = float(np.mean(gap_vals))
    gap_std = float(np.std(gap_vals) or 1.0)
    # ``channel_layout`` documents the schema for downstream readers. Detected
    # from n_ch so both v1 (with aux) and v2 (no aux) bundles get accurate
    # metadata without an extra arg.
    if n_ch % 3 == 0 and n_ch >= 9:
        layout = [
            {"offset": 0, "name": "speed", "scale": "speed_mean/std"},
            {"offset": 1, "name": "gap", "scale": "gap_mean/std"},
            {"offset": 2, "name": "aux", "scale": "passthrough"},
        ]
    else:
        layout = [
            {"offset": 0, "name": "speed", "scale": "speed_mean/std"},
            {"offset": 1, "name": "fwd_gap", "scale": "gap_mean/std"},
        ]
    return {
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "threshold_raw": 100.0,
        "threshold_scaled": (100.0 - gap_mean) / (gap_std or 1.0),
        "channel_layout": layout,
    }


def apply_scaler_to_vendor_block(X: np.ndarray, scaler: dict) -> np.ndarray:
    """Z-score the (B, seq_len, n_ch) vendor block. Schema detected from n_ch."""
    out = X.astype(np.float32).copy()
    s_mean = scaler["speed_mean"]; s_std = scaler["speed_std"] or 1.0
    g_mean = scaler["gap_mean"]; g_std = scaler["gap_std"] or 1.0
    n_ch = out.shape[2]
    speed_cols, gap_cols = _speed_gap_offsets(n_ch)
    speed_set = set(speed_cols); gap_set = set(gap_cols)
    for j in range(n_ch):
        if j in speed_set:
            out[:, :, j] = (out[:, :, j] - s_mean) / s_std
        elif j in gap_set:
            out[:, :, j] = (out[:, :, j] - g_mean) / g_std
        # other columns (aux in v1) are passthrough
    return out


def build_feature_matrix(
    X_vendor_scaled: np.ndarray, X_extras: np.ndarray | None, feature_set: str,
) -> np.ndarray:
    """Per-tick concat then flatten — must match the live inference layout.

    Tree-based trainers want a flat (N, F) matrix; this function is the
    single source of truth for how the layout is composed. Match this exactly
    in apps/prediction/live_features.merge_for_predictor.
    """
    if feature_set == "vendor":
        return X_vendor_scaled.reshape(X_vendor_scaled.shape[0], -1).astype(np.float32)
    if feature_set == "rich":
        if X_extras is None:
            raise ValueError("rich feature_set requires extras")
        merged = np.concatenate([X_vendor_scaled, X_extras], axis=2)
        return merged.reshape(merged.shape[0], -1).astype(np.float32)
    raise ValueError(f"unknown feature_set {feature_set!r}")


__all__ = [
    "Split",
    "Dataset",
    "load_dataset",
    "compute_scaler",
    "apply_scaler_to_vendor_block",
    "build_feature_matrix",
]
