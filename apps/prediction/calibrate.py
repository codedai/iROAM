"""Fit per-horizon isotonic calibrators on the val split for a bag bundle.

Loads the bag's averaged predictions on val, fits one
``sklearn.isotonic.IsotonicRegression`` per horizon, persists the calibrators
as JSON (``iso_h00.json ... iso_h29.json``) inside the bundle, and writes a
``thresholds_calibrated.json`` with F2-optimal thresholds re-tuned on the
calibrated probabilities.

The JSON format is intentionally tiny and self-describing:
    { "x": [...sorted increasing...], "y": [...non-decreasing in [0,1]...] }
At inference, ``np.interp(prob, x, y)`` gives the calibrated probability.

CLI:
    python -m apps.prediction.calibrate \
        --bundle deployment/bunching_local_rich_bag8 \
        --dataset out/datasets/route29_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from apps.prediction.bagged_predictor import BaggedPredictor
from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    build_feature_matrix,
    compute_scaler,
    load_dataset,
)
from apps.prediction.train import _pick_threshold


def fit_and_save(
    bundle_dir: Path, dataset_dir: Path, *, threshold_strategy: str | None = None,
) -> None:
    """Fit isotonic calibrators on the bag's val predictions.

    ``threshold_strategy`` overrides the strategy persisted in the bag's
    manifest. If neither is set, falls back to ``f2`` for back-compat.
    """
    pred = BaggedPredictor(bundle_dir)
    ds = load_dataset(str(dataset_dir))
    feature_set = pred.metadata.get("feature_set", "rich")
    # Bag manifest persists the strategy used at training; honour it here so
    # calibrated thresholds match the operating-point story.
    strategy = (
        threshold_strategy
        or pred._manifest.get("threshold_strategy")
        or "f2"
    )

    scaler = compute_scaler(ds.train.X_vendor)
    Xva_v = apply_scaler_to_vendor_block(ds.val.X_vendor, scaler)
    extras = ds.val.X_extras if feature_set == "rich" else None
    Xva_flat = build_feature_matrix(Xva_v, extras, feature_set)
    Xva3 = Xva_flat.reshape(-1, pred.seq_len, pred.n_channels)
    print(f"Predicting on val ({Xva3.shape[0]} rows) using bag (n_bags={pred.metadata['n_bags']})…",
          flush=True)
    probs_val = pred.predict_proba(Xva3, is_scaled=True)
    Yva = ds.val.Y

    iso_dir = bundle_dir / "calibration"
    iso_dir.mkdir(parents=True, exist_ok=True)
    thr_out: dict[str, dict] = {}
    for h in range(pred.pred_len):
        y = Yva[:, h]; m = np.isfinite(y)
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            # Degenerate horizon — write an identity calibrator so the loader
            # still finds a file and predict_proba doesn't have to branch.
            iso_payload = {"x": [0.0, 1.0], "y": [0.0, 1.0], "degenerate": True}
            (iso_dir / f"iso_h{h:02d}.json").write_text(json.dumps(iso_payload))
            thr_out[str(h)] = {
                "threshold": 0.5, "f2_val": 0.0, "pr_auc_val": 0.0,
                "brier_val": None, "best_iter": 0, "positive_rate_train": 0.0,
                "method": "bag_avg_cal_degenerate",
            }
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs_val[m, h], y[m])
        # Save the piecewise-constant breakpoints in a portable form.
        x_break = np.asarray(iso.X_thresholds_, dtype=np.float64).tolist()
        y_break = np.asarray(iso.y_thresholds_, dtype=np.float64).tolist()
        iso_payload = {"x": x_break, "y": y_break, "degenerate": False}
        (iso_dir / f"iso_h{h:02d}.json").write_text(json.dumps(iso_payload))

        # Tune thresholds on calibrated val probs using the bundle's chosen
        # strategy. We also always report F2 alongside so cross-strategy
        # bundles remain comparable on the legacy yardstick.
        cal_val = np.interp(probs_val[:, h], x_break, y_break)
        thr, _ = _pick_threshold(cal_val[m], y[m], strategy=strategy)
        _, f2 = _pick_threshold(cal_val[m], y[m], strategy="f2")
        pr = float(average_precision_score(y[m], cal_val[m]))
        br = float(brier_score_loss(y[m], cal_val[m]))
        thr_out[str(h)] = {
            "threshold": float(thr),
            "f2_val": float(f2),
            "pr_auc_val": pr,
            "brier_val": br,
            "best_iter": 0,
            "positive_rate_train": 0.0,
            "method": f"bag_avg_cal/{strategy}",
        }
    (bundle_dir / "thresholds_calibrated.json").write_text(json.dumps(thr_out, indent=2))
    print(f"Wrote {pred.pred_len} isotonic calibrators to {iso_dir}", flush=True)
    print(f"Wrote calibrated thresholds to {bundle_dir / 'thresholds_calibrated.json'}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fit_and_save(args.bundle, args.dataset)


if __name__ == "__main__":
    main()
