"""Per-horizon XGBoost trainer — for ensemble diversity vs LightGBM.

Produces a bundle in a parallel layout to ``deployment/bunching_local_*``,
loadable via ``XGBPredictor``.

CLI:
    python -m apps.prediction.train_xgb \
        --dataset out/datasets/route29_v1 \
        --out deployment/bunching_xgb_rich_v1 --feature-set rich
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    build_feature_matrix,
    compute_scaler,
    load_dataset,
)
from apps.prediction.train import _best_f2_threshold
from data_process.bunching.labels import N_CHANNELS, N_EXTRA


@dataclass
class XgbHorizonResult:
    horizon: int
    booster_path: Path | None
    constant: float | None
    best_iter: int
    threshold: float
    f2_val: float
    pr_auc_val: float
    brier_val: float | None
    positive_rate_train: float


def train_one_horizon_xgb(
    h: int,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    *,
    out_dir: Path,
    n_estimators: int, max_depth: int, learning_rate: float, min_child_weight: int,
    early_stopping_rounds: int, tree_method: str,
) -> XgbHorizonResult:
    finite_tr = np.isfinite(ytr)
    finite_va = np.isfinite(yva)
    ytr_use = ytr[finite_tr].astype(np.int32)
    yva_use = yva[finite_va].astype(np.int32)
    Xtr_use = Xtr[finite_tr]
    Xva_use = Xva[finite_va]
    n_pos = int((ytr_use == 1).sum())
    n_neg = int((ytr_use == 0).sum())
    pos_rate = n_pos / max(1, n_pos + n_neg)

    if n_pos == 0 or n_neg == 0:
        return XgbHorizonResult(
            horizon=h, booster_path=None, constant=pos_rate, best_iter=0,
            threshold=0.5, f2_val=0.0, pr_auc_val=0.0, brier_val=None,
            positive_rate_train=pos_rate,
        )

    dtr = xgb.DMatrix(Xtr_use, label=ytr_use)
    dva = xgb.DMatrix(Xva_use, label=yva_use)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": max_depth,
        "eta": learning_rate,
        "min_child_weight": min_child_weight,
        "tree_method": tree_method,
        "scale_pos_weight": n_neg / max(1, n_pos),
        "verbosity": 0,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=n_estimators,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_iter = int(booster.best_iteration)
    probs_va = booster.predict(dva, iteration_range=(0, best_iter + 1))
    thr, f2 = _best_f2_threshold(probs_va, yva_use)
    pr = float(average_precision_score(yva_use, probs_va)) if len(np.unique(yva_use)) > 1 else 0.0
    br = float(brier_score_loss(yva_use, probs_va)) if len(np.unique(yva_use)) > 1 else None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"xgb_h{h:02d}.json"
    booster.save_model(str(path))
    return XgbHorizonResult(
        horizon=h, booster_path=path, constant=None, best_iter=best_iter,
        threshold=thr, f2_val=f2, pr_auc_val=pr, brier_val=br,
        positive_rate_train=pos_rate,
    )


def fit_and_write_xgb(
    dataset_dir: Path, out_dir: Path, *,
    feature_set: str,
    n_estimators: int = 400, max_depth: int = 7, learning_rate: float = 0.05,
    min_child_weight: int = 5, early_stopping_rounds: int = 20,
    tree_method: str = "hist",
) -> None:
    ds = load_dataset(str(dataset_dir))
    print(f"Loaded train={ds.train.n} val={ds.val.n} test={ds.test.n}", flush=True)

    scaler = compute_scaler(ds.train.X_vendor)
    Xtr_v = apply_scaler_to_vendor_block(ds.train.X_vendor, scaler)
    Xva_v = apply_scaler_to_vendor_block(ds.val.X_vendor, scaler)
    Xtr = build_feature_matrix(Xtr_v, ds.train.X_extras, feature_set)
    Xva = build_feature_matrix(Xva_v, ds.val.X_extras, feature_set)

    out_model = out_dir / "model"
    out_model.mkdir(parents=True, exist_ok=True)

    results: list[XgbHorizonResult] = []
    t0 = time.time()
    for h in range(ds.pred_len):
        r = train_one_horizon_xgb(
            h, Xtr, ds.train.Y[:, h], Xva, ds.val.Y[:, h],
            out_dir=out_model,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, min_child_weight=min_child_weight,
            early_stopping_rounds=early_stopping_rounds,
            tree_method=tree_method,
        )
        results.append(r)
        print(f"  h{h:02d}: thr={r.threshold:.2f} F2={r.f2_val:.3f} "
              f"PR-AUC={r.pr_auc_val:.3f} best_iter={r.best_iter}", flush=True)
    print(f"XGB training done in {time.time()-t0:.1f}s", flush=True)

    n_chans_per_tick = N_CHANNELS + (ds.n_extra if feature_set == "rich" else 0)

    (out_model / "scaler.json").write_text(json.dumps(scaler, indent=2))
    thresholds = {
        str(r.horizon): {
            "threshold": float(r.threshold),
            "f2_val": float(r.f2_val),
            "pr_auc_val": float(r.pr_auc_val),
            "brier_val": (float(r.brier_val) if r.brier_val is not None else None),
            "best_iter": int(r.best_iter),
            "positive_rate_train": float(r.positive_rate_train),
        }
        for r in results
    }
    (out_model / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    meta = {
        "model_type": "per_horizon_xgboost",
        "framework": "xgboost",
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seq_len": ds.seq_len,
        "pred_len": ds.pred_len,
        "n_channels": n_chans_per_tick,
        "n_features": int(ds.seq_len * n_chans_per_tick),
        "feature_set": feature_set,
        "extra_features": list(ds.manifest.get("extra_features", [])),
        "step_seconds": ds.manifest.get("step_seconds"),
        "route_id": ds.manifest.get("route_id"),
        "hyperparameters": {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "tree_method": tree_method,
        },
        "split": ds.manifest["split"],
    }
    (out_model / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote bundle to {out_model}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--feature-set", choices=("vendor", "rich"), default="rich")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-weight", type=int, default=5)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--tree-method", default="hist")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fit_and_write_xgb(
        dataset_dir=args.dataset,
        out_dir=args.out,
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method=args.tree_method,
    )


if __name__ == "__main__":
    main()
