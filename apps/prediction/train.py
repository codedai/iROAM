"""Train a per-horizon LightGBM bunching predictor on a local dataset.

The output bundle (``model/`` directory) is layout-compatible with
``deployment/bunching_lightgbm/`` — i.e. it can be loaded by the existing
``BunchingPredictor`` class without code changes. That gives us a clean
A/B path: point ``BUNCHING_MODEL_DIR`` at the new bundle to flip serving over.

CLI:
    python -m apps.prediction.train \
        --dataset out/datasets/route29_v1 \
        --out deployment/bunching_local_v1/model

Reads the dataset manifest, loads train/val parquet shards, fits one LightGBM
classifier per horizon, calibrates per-horizon F2-optimal thresholds on the
val split, and writes the bundle.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from data_process.bunching.labels import (
    BUNCHING_THRESHOLD_M,
    N_CHANNELS,
    N_EXTRA,
)


# ─────────────────────── dataset loading ─────────────────────────────────────


@dataclass
class LoadedSplit:
    """Materialised arrays for one split."""

    X: np.ndarray            # (N, seq_len * n_features)
    extras: np.ndarray | None  # (N, seq_len * n_extra)
    Y: np.ndarray            # (N, pred_len) float32 — NaN where label missing
    n_examples: int


def _decode_window(blob: bytes, *, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(shape)


def load_split(
    manifest: dict, dataset_dir: Path, split_dates: Sequence[str],
) -> LoadedSplit:
    seq_len = manifest["seq_len"]
    pred_len = manifest["pred_len"]
    n_extra = manifest.get("n_extra", N_EXTRA)

    shards = [
        s for s in manifest["shards"] if s["service_date"] in set(split_dates)
    ]
    frames: list[pd.DataFrame] = []
    for shard in shards:
        frames.append(pd.read_parquet(dataset_dir / shard["path"]))
    if not frames:
        return LoadedSplit(
            X=np.zeros((0, seq_len * N_CHANNELS), dtype=np.float32),
            extras=np.zeros((0, seq_len * n_extra), dtype=np.float32),
            Y=np.zeros((0, pred_len), dtype=np.float32),
            n_examples=0,
        )
    df = pd.concat(frames, ignore_index=True)

    X = np.stack(
        [_decode_window(b, shape=(seq_len, N_CHANNELS)).reshape(-1) for b in df["window"]],
        axis=0,
    ).astype(np.float32)
    extras = np.stack(
        [_decode_window(b, shape=(seq_len, n_extra)).reshape(-1) for b in df["extras"]],
        axis=0,
    ).astype(np.float32)
    Y = np.stack(
        [np.frombuffer(b, dtype=np.float32) for b in df["labels"]],
        axis=0,
    ).astype(np.float32)
    return LoadedSplit(X=X, extras=extras, Y=Y, n_examples=len(df))


# ───────────────────────── per-horizon trainer ───────────────────────────────


@dataclass
class HorizonResult:
    horizon: int
    booster: lgb.Booster | None
    constant: float | None
    best_iter: int
    threshold: float
    f2_val: float
    pr_auc_val: float
    brier_val: float
    positive_rate_train: float


def _f2(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    beta2 = 4.0  # F2 weights recall 2x precision
    return (1 + beta2) * p * r / (beta2 * p + r)


def _f_beta(p: float, r: float, beta: float) -> float:
    if p + r <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _best_f2_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Legacy F2-optimal threshold finder (kept for back-compat)."""
    return _pick_threshold(probs, y, strategy="f2")


def _pick_threshold(
    probs: np.ndarray, y: np.ndarray, *, strategy: str,
) -> tuple[float, float]:
    """Per-horizon decision threshold + the metric value at that threshold.

    Strategies:
      * ``"f2"``, ``"f1"``, ``"f0.5"`` — F-beta maximisation (recall-, balanced-,
        precision-weighted). Returns (threshold, F-beta at that threshold).
      * ``"precision@<X>"`` where X ∈ (0, 1) — pick the threshold that
        maximises **recall** subject to ``precision ≥ X`` on the val split.
        Returns (threshold, recall at that threshold). If no threshold meets
        the precision constraint, returns (0.99, 0.0) — effectively "never
        alert" — which keeps the bundle loadable but signals to the eval
        harness that this horizon is infeasible at the requested precision.

    Why this matters in production: at 5–8% positive base rates, F2 picks
    thresholds around 0.1 that have ~10% precision — about nine of every ten
    alerts are false positives. ``precision@0.30`` or ``precision@0.50`` gives
    a dispatcher-honest alert volume in exchange for some recall.
    """
    thrs = np.arange(0.01, 0.99, 0.01)
    best_thr = 0.5
    best_metric = -1.0

    # Pre-extract for speed.
    y_pos = (y == 1)
    y_neg = (y == 0)
    n_pos_total = int(y_pos.sum())

    if strategy.startswith("precision@"):
        try:
            target = float(strategy.split("@", 1)[1])
        except ValueError as e:
            raise ValueError(f"bad strategy {strategy!r}") from e
        if not 0.0 < target < 1.0:
            raise ValueError(f"precision target must be in (0, 1), got {target}")
        for t in thrs:
            pred = probs >= t
            tp = int(np.sum(pred & y_pos))
            fp = int(np.sum(pred & y_neg))
            if tp + fp == 0:
                continue
            prec = tp / (tp + fp)
            if prec < target:
                continue
            rec = tp / max(1, n_pos_total)
            if rec > best_metric:
                best_metric = rec
                best_thr = float(t)
        if best_metric < 0:
            # No threshold met the precision floor. Refuse to alert at all
            # rather than ship a low-precision threshold that pretends to.
            return 0.99, 0.0
        return best_thr, max(best_metric, 0.0)

    # F-beta family.
    beta_map = {"f0.5": 0.5, "f1": 1.0, "f2": 2.0}
    if strategy not in beta_map:
        raise ValueError(f"unknown threshold strategy: {strategy!r}")
    beta = beta_map[strategy]
    for t in thrs:
        pred = probs >= t
        tp = int(np.sum(pred & y_pos))
        fp = int(np.sum(pred & y_neg))
        fn = int(np.sum((~pred) & y_pos))
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / max(1, tp + fn)
        fb = _f_beta(prec, rec, beta)
        if fb > best_metric:
            best_metric = fb
            best_thr = float(t)
    return best_thr, max(best_metric, 0.0)


def train_one_horizon(
    horizon: int,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    *, params: dict, n_estimators: int, early_stopping_rounds: int,
    threshold_strategy: str = "f2",
) -> HorizonResult:
    """Train one booster for one prediction horizon. Skip if labels are single-class."""
    finite_tr = np.isfinite(ytr)
    finite_va = np.isfinite(yva)
    ytr_use = ytr[finite_tr]
    yva_use = yva[finite_va]
    Xtr_use = Xtr[finite_tr]
    Xva_use = Xva[finite_va]

    n_pos_tr = int((ytr_use == 1).sum())
    n_neg_tr = int((ytr_use == 0).sum())
    pos_rate_tr = n_pos_tr / max(1, n_pos_tr + n_neg_tr)

    # If a horizon has no positives in train, return a constant predictor of
    # the empirical rate. Matches the vendor bundle's CONSTANT sentinel.
    if n_pos_tr == 0 or n_neg_tr == 0:
        return HorizonResult(
            horizon=horizon,
            booster=None,
            constant=pos_rate_tr,
            best_iter=0,
            threshold=0.5,
            f2_val=0.0,
            pr_auc_val=0.0,
            brier_val=float("nan"),
            positive_rate_train=pos_rate_tr,
        )

    spw = n_neg_tr / max(1, n_pos_tr)
    train_set = lgb.Dataset(Xtr_use, label=ytr_use)
    val_set = lgb.Dataset(Xva_use, label=yva_use, reference=train_set)
    booster = lgb.train(
        params={**params, "scale_pos_weight": spw, "verbose": -1},
        train_set=train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iter = int(booster.best_iteration or n_estimators)
    probs_va = booster.predict(Xva_use, num_iteration=best_iter)
    thr, _metric = _pick_threshold(probs_va, yva_use, strategy=threshold_strategy)
    # Always report F2 alongside the chosen metric so cross-strategy bundles
    # remain comparable on the same yardstick the legacy reports use.
    _, f2 = _pick_threshold(probs_va, yva_use, strategy="f2")
    pr_auc = float(average_precision_score(yva_use, probs_va)) if len(np.unique(yva_use)) > 1 else 0.0
    brier = float(brier_score_loss(yva_use, probs_va)) if len(np.unique(yva_use)) > 1 else float("nan")
    return HorizonResult(
        horizon=horizon,
        booster=booster,
        constant=None,
        best_iter=best_iter,
        threshold=thr,
        f2_val=f2,
        pr_auc_val=pr_auc,
        brier_val=brier,
        positive_rate_train=pos_rate_tr,
    )


# ──────────────────────────── bundle writer ──────────────────────────────────


def write_bundle(
    out_dir: Path,
    *,
    manifest: dict,
    train_split: dict,
    results: list[HorizonResult],
    feature_set: str,
    n_features_per_tick: int,
    extra_features: list[str] | None,
    speed_mean: float, speed_std: float,
    gap_mean: float, gap_std: float,
    threshold_strategy: str = "f2",
    n_extra: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # boosters
    for r in results:
        path = out_dir / f"booster_h{r.horizon:02d}.txt"
        if r.booster is not None:
            r.booster.save_model(str(path), num_iteration=r.best_iter)
        else:
            with open(path, "w") as f:
                f.write(f"CONSTANT\t{r.constant or 0.0:.8f}\n")

    # scaler — keep the vendor key names so BunchingPredictor can load it.
    scaler = {
        "speed_mean": float(speed_mean),
        "speed_std": float(speed_std),
        "gap_mean": float(gap_mean),
        "gap_std": float(gap_std),
        "threshold_raw": BUNCHING_THRESHOLD_M,
        "threshold_scaled": float((BUNCHING_THRESHOLD_M - gap_mean) / gap_std) if gap_std else 0.0,
        "channel_layout": [
            {"offset": 0, "name": "speed", "scale": "speed_mean/std"},
            {"offset": 1, "name": "gap", "scale": "gap_mean/std"},
            {"offset": 2, "name": "aux", "scale": "passthrough"},
        ],
    }
    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    thresholds = {
        str(r.horizon): {
            "threshold": float(r.threshold),
            "f2_val": float(r.f2_val),
            "pr_auc_val": float(r.pr_auc_val),
            "brier_val": float(r.brier_val) if not np.isnan(r.brier_val) else None,
            "best_iter": int(r.best_iter),
            "positive_rate_train": float(r.positive_rate_train),
        }
        for r in results
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    meta = {
        "model_type": "per_horizon_lightgbm",
        "framework": "lightgbm",
        "lightgbm_version": lgb.__version__,
        "numpy_version": np.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seq_len": manifest["seq_len"],
        "pred_len": manifest["pred_len"],
        "step": manifest.get("step", 2),
        "n_channels": n_features_per_tick,
        "n_features": int(manifest["seq_len"] * n_features_per_tick),
        "variant": "matched",  # legacy key, kept for predictor compatibility
        "data_root": str(manifest.get("dataset_root", "")),
        "feature_set": feature_set,
        "extra_features": extra_features or [],
        # n_extra is what the live feature builder needs to allocate its
        # extras array. We persist it explicitly so future schema bumps
        # (v1=7, v2=10) are self-describing.
        "n_extra": n_extra if n_extra is not None else len(extra_features or []),
        # vendor_schema_v: 1 = legacy 9-channel (target/u1/u2 × speed/gap/aux),
        # 2 = current 6-channel (target/d1/d2 × speed/fwd_gap). The live
        # builder branches on this to produce matching windows.
        "vendor_schema_v": int(manifest.get("vendor_schema_v", 1)),
        # extras_schema_v: 1 = legacy 7 (with leader_speed),
        # 2 = 10 (v1 + terminus), 3 = trimmed 7 (no leader_speed,
        # only dist_to_terminus_norm from the terminus group).
        "extras_schema_v": int(
            manifest.get("extras_schema_v")
            or (2 if int(manifest.get("n_extra", 0)) == 10 else 1)
        ),
        "step_seconds": manifest.get("step_seconds"),
        "route_id": manifest.get("route_id"),
        # Records how the per-horizon thresholds were chosen. Ops-grade
        # bundles use ``precision@<X>`` — the eval harness branches on this.
        "threshold_strategy": threshold_strategy,
        "split": {
            "train_dates": train_split.get("train_dates"),
            "val_dates": train_split.get("val_dates"),
            "test_dates": train_split.get("test_dates"),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ──────────────────────────── orchestration ──────────────────────────────────


def fit_and_write(
    dataset_dir: Path,
    out_dir: Path,
    *,
    feature_set: str = "vendor",
    n_estimators: int = 300,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_child_samples: int = 50,
    early_stopping_rounds: int = 20,
    threshold_strategy: str = "f2",
) -> None:
    """Train all per-horizon boosters and write the bundle to ``out_dir``."""
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split = manifest["split"]

    print(f"Loading dataset from {dataset_dir} (feature_set={feature_set})", flush=True)
    train = load_split(manifest, dataset_dir, split["train_dates"])
    val = load_split(manifest, dataset_dir, split["val_dates"])
    print(f"  train={train.n_examples}  val={val.n_examples}", flush=True)
    if train.n_examples == 0 or val.n_examples == 0:
        raise RuntimeError("dataset is empty for the chosen split")

    # Scaler stats on the train split's raw windows. LightGBM is monotonic-
    # invariant *at training time*, but the BunchingPredictor scales raw inputs
    # at inference using the bundled scaler — so if we train on raw and serve
    # on scaled, every tree split fires the wrong branch and predictions
    # collapse. Compute z-score stats here, scale both train+val, and write
    # the same stats into scaler.json so inference matches.
    seq_len = manifest["seq_len"]
    # Use the per-schema (speed, gap) offsets within one tick from data.py
    # so this code stays correct for both legacy 9-channel (v1, with aux)
    # and current 6-channel (v2, downstream chain) vendor layouts.
    from apps.prediction.data import _speed_gap_offsets
    n_ch_vendor = int(manifest.get("n_channels", N_CHANNELS))
    intra_speed, intra_gap = _speed_gap_offsets(n_ch_vendor)
    speed_cols: list[int] = []
    gap_cols: list[int] = []
    for k in range(seq_len):
        for off in intra_speed:
            speed_cols.append(k * n_ch_vendor + off)
        for off in intra_gap:
            gap_cols.append(k * n_ch_vendor + off)
    speed_vals = train.X[:, speed_cols].reshape(-1)
    gap_vals = train.X[:, gap_cols].reshape(-1)
    speed_mean = float(np.mean(speed_vals))
    speed_std = float(np.std(speed_vals) or 1.0)
    gap_mean = float(np.mean(gap_vals))
    gap_std = float(np.std(gap_vals) or 1.0)

    def _zscore_window_block(X: np.ndarray) -> np.ndarray:
        out = X.copy()
        out[:, speed_cols] = (X[:, speed_cols] - speed_mean) / (speed_std or 1.0)
        out[:, gap_cols] = (X[:, gap_cols] - gap_mean) / (gap_std or 1.0)
        return out

    Xtr_vendor = _zscore_window_block(train.X)
    Xva_vendor = _zscore_window_block(val.X)

    if feature_set == "vendor":
        Xtr, Xva = Xtr_vendor, Xva_vendor
        n_features_per_tick = N_CHANNELS
        extras_list = None
    elif feature_set == "rich":
        # Extras are passthrough at both train and inference time. CRITICAL:
        # interleave per-tick (concat on channel axis, then re-flatten) so the
        # column order matches what BunchingPredictor sees at inference, where
        # the live builder produces a single (seq_len, n_chans) window and the
        # predictor flattens it tick-major.
        n_extra = manifest.get("n_extra", N_EXTRA)
        v_tr3 = Xtr_vendor.reshape(-1, seq_len, N_CHANNELS)
        v_va3 = Xva_vendor.reshape(-1, seq_len, N_CHANNELS)
        e_tr3 = train.extras.reshape(-1, seq_len, n_extra)
        e_va3 = val.extras.reshape(-1, seq_len, n_extra)
        Xtr = np.concatenate([v_tr3, e_tr3], axis=2).reshape(v_tr3.shape[0], -1)
        Xva = np.concatenate([v_va3, e_va3], axis=2).reshape(v_va3.shape[0], -1)
        n_features_per_tick = N_CHANNELS + n_extra
        extras_list = list(manifest.get("extra_features", []))
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "feature_pre_filter": False,
    }

    pred_len = manifest["pred_len"]
    results: list[HorizonResult] = []
    t0 = time.time()
    for h in range(pred_len):
        ytr = train.Y[:, h]
        yva = val.Y[:, h]
        r = train_one_horizon(
            h, Xtr, ytr, Xva, yva,
            params=params, n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            threshold_strategy=threshold_strategy,
        )
        results.append(r)
        msg = (
            f"  h{h:02d}: thr={r.threshold:.2f} F2={r.f2_val:.3f} "
            f"PR-AUC={r.pr_auc_val:.3f} pos_rate={r.positive_rate_train:.3f} "
            f"best_iter={r.best_iter}"
        )
        if r.booster is None:
            msg += "  (CONSTANT)"
        print(msg, flush=True)
    print(f"Training done in {time.time() - t0:.1f}s", flush=True)

    write_bundle(
        out_dir,
        manifest=manifest,
        train_split=split,
        results=results,
        feature_set=feature_set,
        n_features_per_tick=n_features_per_tick,
        extra_features=extras_list,
        speed_mean=speed_mean, speed_std=speed_std,
        gap_mean=gap_mean, gap_std=gap_std,
        threshold_strategy=threshold_strategy,
        n_extra=int(manifest.get("n_extra", N_EXTRA)) if feature_set == "rich" else 0,
    )
    print(f"Wrote bundle to {out_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--feature-set", choices=("vendor", "rich"), default="vendor")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument(
        "--threshold-strategy", default="f2",
        help="Per-horizon threshold rule: f2|f1|f0.5|precision@<X>  (e.g. precision@0.30)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fit_and_write(
        dataset_dir=args.dataset,
        out_dir=args.out,
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        early_stopping_rounds=args.early_stopping_rounds,
        threshold_strategy=args.threshold_strategy,
    )


if __name__ == "__main__":
    main()
