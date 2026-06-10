"""Unified backtest for every model + ensemble variant on the held-out test split.

Each predictor is wrapped to expose ``predict_proba_test`` and
``predict_proba_val`` returning ``(N, pred_len)`` float32 in [0, 1]. We
materialise val + test once via ``apps.prediction.data``, run every model,
build derived ensembles (mean / median / stacked / calibrated), and emit a
report at ``out/eval/sota_v1/{report.md, report.json}``.

The point is to make the comparison apples-to-apples: identical scaler, same
val/test rows, same metrics, same threshold-tuning protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    build_feature_matrix,
    compute_scaler,
    load_dataset,
)
from apps.prediction.train import _best_f2_threshold


# ─────────────────────────── metrics ────────────────────────────────────────


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
    brier: float | None


def score(probs: np.ndarray, Y: np.ndarray, thresholds: np.ndarray) -> list[HorizonMetrics]:
    out: list[HorizonMetrics] = []
    pred_len = Y.shape[1]
    for h in range(pred_len):
        y = Y[:, h]; p = probs[:, h]
        m = np.isfinite(y); y = y[m]; p = p[m]
        if y.size == 0:
            out.append(HorizonMetrics(h, 0, 0.0, float(thresholds[h]), 0, 0, 0, 0, None))
            continue
        thr = float(thresholds[h])
        pred = p >= thr
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0
        br = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else None
        out.append(HorizonMetrics(h, int(y.size), float((y == 1).mean()), thr,
                                  float(prec), float(rec), float(_f2(prec, rec)),
                                  pr_auc, br))
    return out


def tune_thresholds(probs_val: np.ndarray, Y_val: np.ndarray) -> np.ndarray:
    """Per-horizon F2-optimal thresholds on val."""
    H = Y_val.shape[1]
    out = np.full(H, 0.5, dtype=np.float32)
    for h in range(H):
        y = Y_val[:, h]; m = np.isfinite(y)
        if m.sum() == 0:
            continue
        thr, _ = _best_f2_threshold(probs_val[m, h], y[m])
        out[h] = float(thr)
    return out


def horizon_mean_f2_5_to_30(metrics: list[HorizonMetrics], step_seconds: int) -> float:
    """Mean F2 across horizons whose lead time is in [5, 30] minutes."""
    out = []
    for m in metrics:
        minutes = (m.horizon + 1) * (step_seconds / 60.0)
        if 5.0 <= minutes <= 30.0:
            out.append(m.f2)
    return float(np.mean(out)) if out else 0.0


# ────────────────────────── per-model loaders ───────────────────────────────


def load_lgbm_bundle(bundle_dir: Path) -> Callable:
    """Return a function ``(X_flat_scaled) -> (N, H)`` for a single LightGBM bundle."""
    from deployment.bunching_lightgbm import BunchingPredictor
    pred = BunchingPredictor(bundle_dir)
    def fn(X_flat: np.ndarray) -> np.ndarray:
        # X_flat is already scaled; reshape back to (N, L, C) so predictor's
        # validate_window passes, then is_scaled=True bypasses scaling.
        seq_len = pred.seq_len; n_ch = pred.n_channels
        X3 = X_flat.reshape(-1, seq_len, n_ch)
        return pred.predict_proba(X3, is_scaled=True)
    return fn


def load_bagged_bundle(bundle_dir: Path) -> Callable:
    from apps.prediction.bagged_predictor import BaggedPredictor
    pred = BaggedPredictor(bundle_dir)
    def fn(X_flat: np.ndarray) -> np.ndarray:
        seq_len = pred.seq_len; n_ch = pred.n_channels
        X3 = X_flat.reshape(-1, seq_len, n_ch)
        return pred.predict_proba(X3, is_scaled=True)
    return fn


def load_xgb_bundle(bundle_dir: Path) -> Callable:
    import xgboost as xgb
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    H = int(meta["pred_len"])
    boosters: list[xgb.Booster | None] = []
    best_iters: list[int] = []
    thr_raw = json.loads((bundle_dir / "thresholds.json").read_text())
    for h in range(H):
        p = bundle_dir / f"xgb_h{h:02d}.json"
        if not p.exists():
            boosters.append(None); best_iters.append(0); continue
        b = xgb.Booster(); b.load_model(str(p))
        boosters.append(b); best_iters.append(int(thr_raw[str(h)]["best_iter"]))
    def fn(X_flat: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X_flat)
        out = np.zeros((X_flat.shape[0], H), dtype=np.float32)
        for h, b in enumerate(boosters):
            if b is None:
                out[:, h] = 0.0
            else:
                bi = best_iters[h]
                out[:, h] = b.predict(d, iteration_range=(0, bi + 1)).astype(np.float32)
        return out
    return fn


def load_torch_bundle(bundle_dir: Path) -> Callable:
    import torch
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    kind = meta["model_type"]
    n_chans = int(meta["n_channels"]); L = int(meta["seq_len"]); H = int(meta["pred_len"])
    arch = meta["arch"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "tcn_multitask":
        from apps.prediction.train_tcn import TCN
        model = TCN(n_channels=n_chans, n_horizons=H,
                    hidden=int(arch["hidden"]), dropout=float(arch["dropout"]))
    elif kind == "tiny_transformer_multitask":
        from apps.prediction.train_tx import TinyTransformer
        model = TinyTransformer(
            n_channels=n_chans, seq_len=L, n_horizons=H,
            d_model=int(arch["d_model"]), n_heads=int(arch["n_heads"]),
            n_layers=int(arch["n_layers"]), ff_dim=int(arch["ff_dim"]),
            dropout=float(arch["dropout"]),
        )
    else:
        raise ValueError(f"unsupported torch bundle kind {kind!r}")
    state = torch.load(bundle_dir / "weights.pt", map_location="cpu")["state_dict"]
    model.load_state_dict(state); model.to(dev); model.eval()

    def fn(X3_scaled: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(X3_scaled).float().to(dev)
            if kind == "tcn_multitask":
                t = t.transpose(1, 2)  # (B, L, C) → (B, C, L)
            p = torch.sigmoid(model(t)).cpu().numpy().astype(np.float32)
        return p
    return fn


# ───────────────────────────── pipeline ─────────────────────────────────────


def _maybe_load(bundle_dir: Path, fmt: str) -> Callable | None:
    if not bundle_dir.exists():
        return None
    try:
        if fmt == "lgbm":   return load_lgbm_bundle(bundle_dir / "model")
        if fmt == "bagged": return load_bagged_bundle(bundle_dir)
        if fmt == "xgb":    return load_xgb_bundle(bundle_dir / "model")
        if fmt == "torch":  return load_torch_bundle(bundle_dir / "model")
    except Exception as e:
        print(f"  ! failed to load {bundle_dir} ({fmt}): {e}", flush=True)
        return None
    return None


def _bundle_feature_set(bundle_dir: Path, fmt: str) -> str:
    """Peek at the bundle's metadata to find its feature_set."""
    if fmt == "bagged":
        meta_path = bundle_dir / "bag_manifest.json"
    else:
        meta_path = bundle_dir / "model" / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
        return str(meta.get("feature_set", "vendor"))
    except Exception:
        return "vendor"


def run(dataset_dir: Path, out_dir: Path, *, models_spec: list[tuple[str, str, Path]]) -> None:
    ds = load_dataset(str(dataset_dir))
    print(f"Dataset {dataset_dir}: train={ds.train.n} val={ds.val.n} test={ds.test.n}", flush=True)

    # Pre-scale once. Build BOTH the vendor-only and the rich flat matrices
    # because vendor-schema bundles expect the 9-channel layout while rich
    # bundles expect 16. Same scaler stats for both so the comparison is fair.
    scaler = compute_scaler(ds.train.X_vendor)
    Xva_v = apply_scaler_to_vendor_block(ds.val.X_vendor, scaler)
    Xte_v = apply_scaler_to_vendor_block(ds.test.X_vendor, scaler)
    Xva_flat_v = build_feature_matrix(Xva_v, None, "vendor")
    Xte_flat_v = build_feature_matrix(Xte_v, None, "vendor")
    Xva_flat_r = build_feature_matrix(Xva_v, ds.val.X_extras, "rich")
    Xte_flat_r = build_feature_matrix(Xte_v, ds.test.X_extras, "rich")
    # 3D rich used by torch models.
    Xva3 = np.concatenate([Xva_v, ds.val.X_extras], axis=2).astype(np.float32)
    Xte3 = np.concatenate([Xte_v, ds.test.X_extras], axis=2).astype(np.float32)

    # Load every available model and record its feature_set so we dispatch
    # the right flat matrix at score time.
    models: dict[str, tuple[str, str, Callable]] = {}  # name -> (fmt, feature_set, fn)
    for name, fmt, path in models_spec:
        fn = _maybe_load(path, fmt)
        if fn is None:
            continue
        fs = _bundle_feature_set(path, fmt)
        models[name] = (fmt, fs, fn)
        print(f"  loaded {name} from {path} ({fmt}, feature_set={fs})", flush=True)
    if not models:
        raise RuntimeError("no models could be loaded")

    # Score every model on val + test using the matching feature matrix.
    probs_val: dict[str, np.ndarray] = {}
    probs_test: dict[str, np.ndarray] = {}
    for name, (fmt, fs, fn) in models.items():
        if fmt in ("lgbm", "bagged", "xgb"):
            Xva_use = Xva_flat_v if fs == "vendor" else Xva_flat_r
            Xte_use = Xte_flat_v if fs == "vendor" else Xte_flat_r
            probs_val[name] = fn(Xva_use)
            probs_test[name] = fn(Xte_use)
        elif fmt == "torch":
            probs_val[name] = fn(Xva3)
            probs_test[name] = fn(Xte3)
        print(f"  scored {name}", flush=True)

    # Derived ensembles.
    base_names = list(models.keys())

    # Simple mean over all base models.
    if len(base_names) >= 2:
        probs_val["ens_mean"] = np.mean([probs_val[n] for n in base_names], axis=0).astype(np.float32)
        probs_test["ens_mean"] = np.mean([probs_test[n] for n in base_names], axis=0).astype(np.float32)
        # Median is a small robustness check.
        probs_val["ens_median"] = np.median([probs_val[n] for n in base_names], axis=0).astype(np.float32)
        probs_test["ens_median"] = np.median([probs_test[n] for n in base_names], axis=0).astype(np.float32)

    # Stacking: per-horizon logistic regression over base-model val outputs.
    if len(base_names) >= 2:
        stacked_val = np.zeros_like(probs_val[base_names[0]])
        stacked_test = np.zeros_like(probs_test[base_names[0]])
        for h in range(ds.pred_len):
            y = ds.val.Y[:, h]; m = np.isfinite(y)
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                # No usable supervision — fall back to mean.
                stacked_val[:, h] = np.mean([probs_val[n][:, h] for n in base_names], axis=0)
                stacked_test[:, h] = np.mean([probs_test[n][:, h] for n in base_names], axis=0)
                continue
            Xs_val = np.stack([probs_val[n][m, h] for n in base_names], axis=1)
            ys_val = y[m]
            lr = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
            lr.fit(Xs_val, ys_val)
            # Fill val (using all val rows; the LR was fit on the same rows so
            # this is in-sample — but we only USE val to tune the stacker, and
            # the test-set evaluation below is the honest one).
            Xs_val_full = np.stack([probs_val[n][:, h] for n in base_names], axis=1)
            stacked_val[:, h] = lr.predict_proba(Xs_val_full)[:, 1]
            Xs_test = np.stack([probs_test[n][:, h] for n in base_names], axis=1)
            stacked_test[:, h] = lr.predict_proba(Xs_test)[:, 1]
        probs_val["ens_stacked"] = stacked_val.astype(np.float32)
        probs_test["ens_stacked"] = stacked_test.astype(np.float32)

    # Isotonic calibration of each base model + ens_stacked + ens_mean.
    to_calibrate = [n for n in probs_test.keys()]
    cal_val: dict[str, np.ndarray] = {}
    cal_test: dict[str, np.ndarray] = {}
    for name in to_calibrate:
        cv = np.zeros_like(probs_val[name])
        ct = np.zeros_like(probs_test[name])
        for h in range(ds.pred_len):
            y = ds.val.Y[:, h]; m = np.isfinite(y)
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                cv[:, h] = probs_val[name][:, h]; ct[:, h] = probs_test[name][:, h]; continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs_val[name][m, h], y[m])
            cv[:, h] = iso.transform(probs_val[name][:, h])
            ct[:, h] = iso.transform(probs_test[name][:, h])
        cal_val[name + "_cal"] = cv.astype(np.float32)
        cal_test[name + "_cal"] = ct.astype(np.float32)
    probs_val.update(cal_val); probs_test.update(cal_test)

    # For each model, tune thresholds on val then score on test.
    report: dict[str, list[dict]] = {}
    headline: dict[str, dict] = {}
    step_seconds = int(ds.manifest.get("step_seconds", 60))
    for name, p_test in probs_test.items():
        thrs = tune_thresholds(probs_val[name], ds.val.Y)
        ms = score(p_test, ds.test.Y, thrs)
        report[name] = [asdict(m) for m in ms]
        headline[name] = {
            "mean_F2_5_30min": horizon_mean_f2_5_to_30(ms, step_seconds),
            "F2@5min": next((m.f2 for m in ms if m.horizon == max(0, 5 - 1)), 0.0),
            "F2@15min": next((m.f2 for m in ms if m.horizon == max(0, 15 - 1)), 0.0),
            "F2@30min": next((m.f2 for m in ms if m.horizon == max(0, 30 - 1)), 0.0),
            "mean_PR_AUC": float(np.mean([m.pr_auc for m in ms])),
            "mean_Brier": float(np.nanmean([m.brier if m.brier is not None else np.nan for m in ms])),
        }
        print(f"  {name:32s}  meanF2[5-30]={headline[name]['mean_F2_5_30min']:.3f}  "
              f"F2@5={headline[name]['F2@5min']:.3f}  F2@30={headline[name]['F2@30min']:.3f}  "
              f"PR-AUC={headline[name]['mean_PR_AUC']:.3f}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps({
        "headline": headline, "per_horizon": report,
    }, indent=2))
    _write_md(out_dir / "report.md", headline, report, step_seconds, ds.pred_len)
    print(f"Wrote {out_dir}/report.md", flush=True)


def _write_md(path: Path, headline: dict, per_horizon: dict, step_seconds: int, pred_len: int) -> None:
    lines: list[str] = [
        "# SOTA backtest — bunching prediction",
        "",
        f"Test split: chronological. Metric of record: **mean F2 across 5–30 min horizons**.",
        "",
        "## Headline",
        "",
        "| model | meanF2[5-30] | F2@+5min | F2@+15min | F2@+30min | mean PR-AUC | mean Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Sort by mean_F2_5_30min descending.
    rows = sorted(headline.items(), key=lambda kv: -kv[1]["mean_F2_5_30min"])
    for name, h in rows:
        lines.append(
            f"| {name} | **{h['mean_F2_5_30min']:.3f}** | {h['F2@5min']:.3f} | {h['F2@15min']:.3f} | "
            f"{h['F2@30min']:.3f} | {h['mean_PR_AUC']:.3f} | {h['mean_Brier']:.4f} |"
        )
    lines += ["", "## Per-horizon breakdown (selected horizons)", ""]
    sample = sorted({0, 4, 9, 14, 19, 24, pred_len - 1})
    for name, _ in rows:
        ms = per_horizon[name]
        lines += [f"### {name}", "",
                  "| h | min | n | pos% | thr | P | R | F2 | PR-AUC | Brier |",
                  "|---|----:|--:|-----:|----:|--:|--:|---:|------:|------:|"]
        for h in sample:
            if h >= len(ms): continue
            m = ms[h]
            min_lbl = f"+{int((h+1)*step_seconds/60)}"
            br_str = f"{m['brier']:.4f}" if m["brier"] is not None else "—"
            lines.append(
                f"| {h} | {min_lbl} | {m['n']} | {m['pos_rate']*100:.1f} | {m['threshold']:.2f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f2']:.3f} | "
                f"{m['pr_auc']:.3f} | {br_str} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Default registry — the eval picks up whichever bundles are present.
    repo = Path(__file__).resolve().parents[2]
    models = [
        ("lgbm_rich",         "lgbm",   repo / "deployment" / "bunching_local_rich_v1"),
        ("lgbm_vendor",       "lgbm",   repo / "deployment" / "bunching_local_v1"),
        ("lgbm_rich_bag8",    "bagged", repo / "deployment" / "bunching_local_rich_bag8"),
        ("xgb_rich",          "xgb",    repo / "deployment" / "bunching_xgb_rich_v1"),
        ("tcn_rich",          "torch",  repo / "deployment" / "bunching_tcn_v1"),
        ("tx_rich",           "torch",  repo / "deployment" / "bunching_tx_v1"),
    ]
    run(args.dataset, args.out, models_spec=models)


if __name__ == "__main__":
    main()
