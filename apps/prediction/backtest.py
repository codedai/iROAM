"""Chronological backtest comparing predictors on a held-out test split.

Reads the same parquet dataset the trainer used, materialises the test split,
runs N predictors over it, and reports per-horizon precision / recall / F2 /
PR-AUC / Brier. Optionally writes a JSON report and a markdown summary table.

Compared predictors (registered by name):
  * ``vendor``  — the shipped deployment/bunching_lightgbm/model bundle.
                  Geometry mismatch (5-min horizon at 10 s) is handled by only
                  scoring the horizons that overlap.
  * ``local``   — any bundle produced by ``apps.prediction.train``. Pass via
                  ``--local <path>``; can be repeated for multiple bundles.
  * ``physics`` — the deterministic gap-closure baseline.

Usage:
    python -m apps.prediction.backtest \
        --dataset out/datasets/route29_v1 \
        --local deployment/bunching_local_v1/model \
        --local deployment/bunching_local_rich_v1/model \
        --out out/eval/route29_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from data_process.bunching.labels import N_CHANNELS, N_EXTRA

# Local imports kept lazy so ``--predictors physics`` works without lightgbm.


# ────────────────────────── data loading ─────────────────────────────────────


@dataclass
class TestSplit:
    seq_len: int
    pred_len: int
    n_extra: int
    step_seconds: int
    # (N, seq_len, N_CHANNELS) raw windows
    X_raw: np.ndarray
    # (N, seq_len, n_extra) richer features
    extras_raw: np.ndarray
    # (N, pred_len) float32 (NaN if outside data window)
    Y: np.ndarray


def load_test_split(dataset_dir: Path) -> TestSplit:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    seq_len = manifest["seq_len"]
    pred_len = manifest["pred_len"]
    n_extra = manifest.get("n_extra", N_EXTRA)
    test_dates = set(manifest["split"]["test_dates"])
    shards = [s for s in manifest["shards"] if s["service_date"] in test_dates]
    frames = [pd.read_parquet(dataset_dir / s["path"]) for s in shards]
    if not frames:
        raise RuntimeError("test split is empty")
    df = pd.concat(frames, ignore_index=True)
    X = np.stack(
        [np.frombuffer(b, dtype=np.float32).reshape(seq_len, N_CHANNELS) for b in df["window"]]
    )
    extras = np.stack(
        [np.frombuffer(b, dtype=np.float32).reshape(seq_len, n_extra) for b in df["extras"]]
    )
    Y = np.stack([np.frombuffer(b, dtype=np.float32) for b in df["labels"]])
    return TestSplit(
        seq_len=seq_len,
        pred_len=pred_len,
        n_extra=n_extra,
        step_seconds=manifest.get("step_seconds", 60),
        X_raw=X,
        extras_raw=extras,
        Y=Y,
    )


# ────────────────────────── metrics helpers ──────────────────────────────────


def _f2(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    beta2 = 4.0
    return (1 + beta2) * p * r / (beta2 * p + r)


@dataclass
class HorizonMetrics:
    horizon: int
    n: int
    pos_rate: float
    threshold: float
    precision: float
    recall: float
    f2: float
    pr_auc: float
    brier: float


def score(
    probs: np.ndarray, Y: np.ndarray, thresholds: np.ndarray,
) -> list[HorizonMetrics]:
    """probs, Y: (N, pred_len). thresholds: (pred_len,)."""
    out: list[HorizonMetrics] = []
    pred_len = Y.shape[1]
    for h in range(pred_len):
        y = Y[:, h]
        p = probs[:, h]
        m = np.isfinite(y)
        y = y[m]
        p = p[m]
        if y.size == 0:
            out.append(HorizonMetrics(h, 0, 0.0, float(thresholds[h]), 0, 0, 0, 0, float("nan")))
            continue
        thr = float(thresholds[h])
        pred = p >= thr
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0
        brier = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan")
        out.append(
            HorizonMetrics(
                horizon=h,
                n=int(y.size),
                pos_rate=float((y == 1).mean()),
                threshold=thr,
                precision=float(prec),
                recall=float(rec),
                f2=float(_f2(prec, rec)),
                pr_auc=pr_auc,
                brier=brier,
            )
        )
    return out


# ──────────────────────────── predictors ─────────────────────────────────────


class _LocalLgbWrapper:
    """Thin wrapper around BunchingPredictor that handles both feature sets."""

    def __init__(self, bundle_dir: Path) -> None:
        from deployment.bunching_lightgbm import BunchingPredictor
        self._impl = BunchingPredictor(bundle_dir)
        self.meta = self._impl.metadata
        self.thresholds = np.array(
            [self._impl.thresholds[h]["threshold"] for h in range(self._impl.pred_len)],
            dtype=np.float32,
        )
        self.feature_set = self.meta.get("feature_set", "vendor")
        self.pred_len = self._impl.pred_len
        self.seq_len = self._impl.seq_len
        self.n_channels = self._impl.n_channels

    def predict_proba(self, X_raw: np.ndarray, extras_raw: np.ndarray) -> np.ndarray:
        if self.feature_set == "vendor":
            return self._impl.predict_proba(X_raw, is_scaled=False)
        # rich = concat per-tick. But the predictor's scaler only knows vendor
        # channels. We bypass scaling here: predictor sees a (seq_len, n_chans)
        # window where the first 9 channels are vendor (already in raw units)
        # and the rest are passthrough extras. We pre-scale the vendor block
        # using the bundle's scaler and pass is_scaled=True to skip rescale.
        from deployment.bunching_lightgbm.src.preprocess import scale_window
        scaler = self._impl.scaler
        batch = X_raw.shape[0]
        scaled = np.empty_like(X_raw)
        for i in range(batch):
            scaled[i] = scale_window(X_raw[i], scaler)
        merged = np.concatenate([scaled, extras_raw], axis=2).astype(np.float32)
        return self._impl.predict_proba(merged, is_scaled=True)


class _VendorWrapper:
    """Shipped 60×9 / 5-min model. Only the first few horizons line up with a
    1-min-step backtest, so we expand by repetition: vendor horizon h corresponds
    to ~10s × (h+1), and our backtest horizon H corresponds to step_seconds × (H+1).
    We map vendor predictions to backtest horizons by carrying the last vendor
    horizon forward — a deliberately weak baseline that documents what serving
    this model *is*. (We could also re-run vendor against a finer grid but the
    point is to show the upper bound of using a 5-min model for a 30-min task.)"""

    def __init__(self, bundle_dir: Path, target_pred_len: int, target_step_seconds: int) -> None:
        from deployment.bunching_lightgbm import BunchingPredictor
        self._impl = BunchingPredictor(bundle_dir)
        self.target_pred_len = target_pred_len
        self.target_step_seconds = target_step_seconds
        self.pred_len = self._impl.pred_len
        self.thresholds_vendor = np.array(
            [self._impl.thresholds[h]["threshold"] for h in range(self._impl.pred_len)],
            dtype=np.float32,
        )

    def predict_proba(self, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs (N, target_pred_len), thresholds (target_pred_len,))."""
        # Vendor expects the same (seq_len=60, 9) shape it shipped with. If our
        # test windows are shorter, we left-pad with the first frame.
        seq_len_vendor = self._impl.seq_len
        N, seq_len, _ = X_raw.shape
        if seq_len < seq_len_vendor:
            pad = np.repeat(X_raw[:, :1], seq_len_vendor - seq_len, axis=1)
            X_in = np.concatenate([pad, X_raw], axis=1)
        elif seq_len > seq_len_vendor:
            X_in = X_raw[:, -seq_len_vendor:]
        else:
            X_in = X_raw
        probs = self._impl.predict_proba(X_in, is_scaled=False)
        # Map every backtest horizon H to a vendor horizon h_v.
        out = np.zeros((N, self.target_pred_len), dtype=np.float32)
        thr = np.zeros(self.target_pred_len, dtype=np.float32)
        for H in range(self.target_pred_len):
            t_sec = (H + 1) * self.target_step_seconds
            # vendor horizon h: covers t = (h+1) * 10s
            h_v = min(self.pred_len - 1, max(0, t_sec // 10 - 1))
            out[:, H] = probs[:, h_v]
            thr[H] = self.thresholds_vendor[h_v]
        return out, thr


# ────────────────────────────── main ─────────────────────────────────────────


def run_backtest(
    dataset_dir: Path,
    *,
    local_bundles: list[Path],
    include_vendor: bool,
    include_physics: bool,
    out_dir: Path | None,
) -> dict:
    split = load_test_split(dataset_dir)
    report: dict[str, list[dict]] = {}

    # ── Physics baseline
    if include_physics:
        from apps.prediction.physics_baseline import (
            PhysicsBaseline,
            PhysicsBaselineConfig,
        )

        phys = PhysicsBaseline(
            pred_len=split.pred_len,
            config=PhysicsBaselineConfig(step_seconds=split.step_seconds),
        )
        probs = phys.predict_proba(split.X_raw, is_scaled=False)
        thr = np.full(split.pred_len, 0.5, dtype=np.float32)
        metrics = score(probs, split.Y, thr)
        report["physics"] = [asdict(m) for m in metrics]
        print(f"physics:    F2@h5={metrics[min(4, split.pred_len-1)].f2:.3f} "
              f"F2@hLast={metrics[-1].f2:.3f}", flush=True)

    # ── Vendor model
    if include_vendor:
        vendor_dir = Path(__file__).resolve().parents[2] / "deployment" / "bunching_lightgbm" / "model"
        if vendor_dir.is_dir():
            vendor = _VendorWrapper(vendor_dir, split.pred_len, split.step_seconds)
            probs, thr = vendor.predict_proba(split.X_raw)
            metrics = score(probs, split.Y, thr)
            report["vendor"] = [asdict(m) for m in metrics]
            print(f"vendor:     F2@h5={metrics[min(4, split.pred_len-1)].f2:.3f} "
                  f"F2@hLast={metrics[-1].f2:.3f}", flush=True)

    # ── Local LightGBM bundles
    for bundle in local_bundles:
        # Use the bundle dir parent for naming so "deployment/foo/model" and
        # "deployment/bar/model" don't collide as just "model".
        label = bundle.parent.name if bundle.name == "model" else bundle.name
        name = f"local:{label}"
        lcl = _LocalLgbWrapper(bundle)
        probs = lcl.predict_proba(split.X_raw, split.extras_raw)
        metrics = score(probs, split.Y, lcl.thresholds)
        report[name] = [asdict(m) for m in metrics]
        print(f"{name}: F2@h5={metrics[min(4, split.pred_len-1)].f2:.3f} "
              f"F2@hLast={metrics[-1].f2:.3f}", flush=True)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        _write_markdown_summary(out_dir / "report.md", report, split.pred_len)
        print(f"Wrote report to {out_dir}", flush=True)
    return report


def _write_markdown_summary(path: Path, report: dict, pred_len: int) -> None:
    sample_horizons = sorted({
        0,
        min(4, pred_len - 1),
        min(9, pred_len - 1),
        min(14, pred_len - 1),
        min(19, pred_len - 1),
        min(24, pred_len - 1),
        pred_len - 1,
    })
    lines: list[str] = [
        "# Bunching forecast backtest",
        "",
        "Per-horizon precision/recall/F2 on the test split.",
        "",
    ]
    for name, metrics in report.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| h | min | n | pos% | thr | P | R | F2 | PR-AUC | Brier |")
        lines.append("|---|----:|--:|-----:|----:|--:|--:|---:|------:|------:|")
        for h in sample_horizons:
            if h >= len(metrics):
                continue
            m = metrics[h]
            min_lbl = f"+{h+1}"
            lines.append(
                f"| {h} | {min_lbl} | {m['n']} | {m['pos_rate']*100:.1f} | {m['threshold']:.2f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f2']:.3f} | "
                f"{m['pr_auc']:.3f} | {m['brier']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--local", action="append", default=[], type=Path)
    p.add_argument("--no-vendor", action="store_true")
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_backtest(
        dataset_dir=args.dataset,
        local_bundles=list(args.local),
        include_vendor=not args.no_vendor,
        include_physics=not args.no_physics,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
