"""Operational evaluation harness — answers the questions dispatchers ask.

The training-time `eval_sota.py` reports F2 / PR-AUC / Brier — academically
useful, but doesn't tell ops what they actually need to know:

  1. **If we deploy this, how many alerts will dispatchers see per hour?**
     ("Alerts per bus-hour" — proxy for cognitive load on a supervisor.)
  2. **Of the alerts we fire, how many are real?** (Precision at the chosen
     threshold, per horizon.)
  3. **At a precision floor of X%, what recall can we get?**
     (Operating-point curves — dispatchers pick a precision floor first,
     then ask what recall is achievable.)
  4. **Are the probabilities themselves trustworthy?** (Reliability /
     calibration curve — when the model says 30%, do 30% of those bunch?)
  5. **Does performance degrade in peak hours?** (Per-period breakdown.)

The harness consumes a bundle directory + a dataset; everything else
(thresholds, predictions) is derived. Output:

  out/eval/ops/<bundle>/
    report.md          — human-readable summary
    report.json        — every metric, every horizon, every period
    reliability.csv    — calibration bins for plotting (no matplotlib dep)
    pr_curves.csv      — precision/recall sweep for plotting

Bundles supported: single LightGBM, bagged LightGBM, XGBoost.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    build_feature_matrix,
    compute_scaler,
    load_dataset,
)
from apps.prediction.eval_sota import (
    _bundle_feature_set,
    load_bagged_bundle,
    load_lgbm_bundle,
    load_xgb_bundle,
)


# Operational target curves — what precision floors do dispatchers care about?
PRECISION_TARGETS = (0.20, 0.30, 0.40, 0.50, 0.60)

# Reliability diagram: bin width (≈10 bins).
RELIABILITY_BINS = np.linspace(0.0, 1.0, 11)

# Local-TZ periods for the per-period breakdown.
PERIODS = {
    "am_peak":  (7 * 60,  9 * 60 + 30),
    "midday":   (9 * 60 + 30, 15 * 60),
    "pm_peak":  (15 * 60, 18 * 60 + 30),
    "evening":  (18 * 60 + 30, 23 * 60),
}


# ─────────────────────────────── metrics ────────────────────────────────────


@dataclass
class HorizonOps:
    horizon: int
    horizon_min: float
    n: int
    pos_rate: float
    # Operating point at the bundle's chosen threshold.
    threshold: float
    precision_at_thr: float
    recall_at_thr: float
    alerts_per_100_examples: float
    # Precision-recall curve summary (PR-AUC + recall at each precision target).
    pr_auc: float
    recall_at_precision: dict[float, float | None]
    # Probability calibration.
    brier: float | None


def _alerts_per_bus_hour(probs: np.ndarray, thresholds: np.ndarray,
                          step_seconds: int) -> float:
    """Estimate alert volume per bus-hour at the bundle's chosen thresholds.

    Each row is one (bus, t_ref) example. If we fire when ANY horizon exceeds
    its threshold, the alert rate per example is the fraction of rows with
    ``(probs >= thresholds).any(axis=1)``. We then convert to per-bus-hour
    assuming ``t_ref`` is sampled every ``step_seconds``.
    """
    any_fire = (probs >= thresholds).any(axis=1)
    fire_rate_per_example = float(any_fire.mean()) if any_fire.size else 0.0
    examples_per_bus_hour = 3600.0 / max(1, step_seconds)
    return fire_rate_per_example * examples_per_bus_hour


def _recall_at_precision(probs: np.ndarray, y: np.ndarray,
                          target_precision: float) -> float | None:
    """Maximum recall achievable subject to ``precision >= target`` on this split."""
    m = np.isfinite(y)
    yy = y[m].astype(np.int32)
    pp = probs[m]
    if yy.size == 0 or len(np.unique(yy)) < 2:
        return None
    prec_arr, rec_arr, _ = precision_recall_curve(yy, pp)
    feasible = prec_arr[:-1] >= target_precision  # last point is the trivial (P=1, R=0)
    if not feasible.any():
        return None
    return float(rec_arr[:-1][feasible].max())


def _reliability_bins(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    """Per-bin observed-vs-predicted rate for a calibration diagram."""
    out = []
    m = np.isfinite(y)
    yy = y[m].astype(np.int32); pp = probs[m]
    if yy.size == 0:
        return out
    for lo, hi in zip(RELIABILITY_BINS[:-1], RELIABILITY_BINS[1:]):
        sel = (pp >= lo) & (pp < hi if hi < 1.0 else pp <= hi)
        n = int(sel.sum())
        if n == 0:
            out.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                        "mean_pred": None, "obs_rate": None})
            continue
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
            "mean_pred": float(pp[sel].mean()),
            "obs_rate": float(yy[sel].mean()),
        })
    return out


def _score_horizon(probs: np.ndarray, y: np.ndarray, threshold: float,
                    horizon: int, step_seconds: int) -> HorizonOps:
    m = np.isfinite(y)
    yy = y[m].astype(np.int32); pp = probs[m]
    if yy.size == 0:
        return HorizonOps(
            horizon=horizon, horizon_min=(horizon + 1) * step_seconds / 60.0,
            n=0, pos_rate=0.0, threshold=threshold,
            precision_at_thr=0.0, recall_at_thr=0.0, alerts_per_100_examples=0.0,
            pr_auc=0.0, recall_at_precision={t: None for t in PRECISION_TARGETS},
            brier=None,
        )
    pred = pp >= threshold
    tp = int(((pred) & (yy == 1)).sum())
    fp = int(((pred) & (yy == 0)).sum())
    fn = int(((~pred) & (yy == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    pr_auc = float(average_precision_score(yy, pp)) if len(np.unique(yy)) > 1 else 0.0
    br = float(brier_score_loss(yy, pp)) if len(np.unique(yy)) > 1 else None
    return HorizonOps(
        horizon=horizon, horizon_min=(horizon + 1) * step_seconds / 60.0,
        n=int(yy.size), pos_rate=float((yy == 1).mean()),
        threshold=float(threshold),
        precision_at_thr=float(prec), recall_at_thr=float(rec),
        alerts_per_100_examples=100.0 * float(pred.mean()),
        pr_auc=pr_auc,
        recall_at_precision={
            t: _recall_at_precision(pp, yy, t) for t in PRECISION_TARGETS
        },
        brier=br,
    )


def _t_ref_minutes_per_example(dates: list[str], n_per_date: dict[str, int],
                                 step_seconds: int) -> np.ndarray:
    """Approximate t_ref minute-of-day per example, used to bucket by period.

    The dataset stores ``service_date`` per example but not t_ref itself
    (it's implicit in the per-date row order, since labels.py walks the grid
    chronologically). We reconstruct an approximate t_ref by spreading each
    date's examples evenly across the operating window (06:00–22:00). This is
    enough granularity for am-peak vs midday vs pm-peak slicing — better
    would be persisting t_ref_min as a column at dataset-build time.
    """
    out = np.zeros(len(dates), dtype=np.float64)
    counters: dict[str, int] = {d: 0 for d in n_per_date}
    op_start = 6 * 60.0; op_end = 22 * 60.0
    for i, d in enumerate(dates):
        n = n_per_date[d]
        c = counters[d]
        out[i] = op_start + (op_end - op_start) * (c / max(1, n - 1))
        counters[d] += 1
    return out


def _period_for(mod: float) -> str:
    for name, (lo, hi) in PERIODS.items():
        if lo <= mod < hi:
            return name
    return "other"


# ──────────────────────────── bundle loaders ────────────────────────────────


def _load_predictor(bundle_dir: Path) -> tuple[str, Callable, dict]:
    """Detect bundle type and return (label, predictor_fn, meta)."""
    bag_manifest = bundle_dir / "bag_manifest.json"
    if bag_manifest.exists():
        return "bagged", load_bagged_bundle(bundle_dir), json.loads(bag_manifest.read_text())
    model_meta = bundle_dir / "model" / "metadata.json"
    if model_meta.exists():
        meta = json.loads(model_meta.read_text())
        if meta.get("framework") == "xgboost":
            return "xgb", load_xgb_bundle(bundle_dir / "model"), meta
        return "lgbm", load_lgbm_bundle(bundle_dir / "model"), meta
    raise FileNotFoundError(f"no recognisable bundle at {bundle_dir}")


# ───────────────────────────────── main ─────────────────────────────────────


def run(bundle_dir: Path, dataset_dir: Path, out_dir: Path) -> dict:
    fmt, predict_fn, bundle_meta = _load_predictor(bundle_dir)
    feature_set = bundle_meta.get("feature_set", "vendor")
    step_seconds = int(bundle_meta.get("step_seconds", 60))
    pred_len = int(bundle_meta.get("pred_len", 30))

    # Thresholds: each bundle persists per-horizon thresholds in either
    # thresholds.json (top-level for bag) or model/thresholds.json (flat).
    if (bundle_dir / "thresholds_calibrated.json").exists():
        thr_path = bundle_dir / "thresholds_calibrated.json"
    elif (bundle_dir / "thresholds.json").exists():
        thr_path = bundle_dir / "thresholds.json"
    else:
        thr_path = bundle_dir / "model" / "thresholds.json"
    thr_dict = json.loads(thr_path.read_text())
    thresholds = np.array([float(thr_dict[str(h)]["threshold"])
                            for h in range(pred_len)], dtype=np.float32)

    ds = load_dataset(str(dataset_dir))
    print(f"Bundle {bundle_dir} (fmt={fmt}, feature_set={feature_set})", flush=True)
    print(f"  train={ds.train.n} val={ds.val.n} test={ds.test.n}", flush=True)

    # Scale + build flat feature matrix for the test split.
    scaler = compute_scaler(ds.train.X_vendor)
    Xte_v = apply_scaler_to_vendor_block(ds.test.X_vendor, scaler)
    Xte_flat = build_feature_matrix(
        Xte_v,
        ds.test.X_extras if feature_set == "rich" else None,
        feature_set,
    )
    probs_te = predict_fn(Xte_flat)
    Y_te = ds.test.Y

    # Overall per-horizon ops table.
    per_horizon: list[HorizonOps] = []
    for h in range(pred_len):
        per_horizon.append(_score_horizon(
            probs_te[:, h], Y_te[:, h], float(thresholds[h]),
            horizon=h, step_seconds=step_seconds,
        ))

    # Headline rollup: cover ALL horizons the bundle serves, intersected with
    # the dispatcher-relevant ceiling (30 min). A 10-min bundle gets a 1-10
    # min headline; a 30-min bundle gets the 1-30 min full picture. The
    # previous hard-coded 5-30 min window hid v3's strong short-horizon
    # behaviour entirely.
    max_band_min = min(30.0, pred_len * step_seconds / 60.0)
    in_band = [m for m in per_horizon if 1.0 <= m.horizon_min <= max_band_min]
    headline = {
        "mean_precision_at_thr": float(np.mean([m.precision_at_thr for m in in_band])),
        "mean_recall_at_thr":    float(np.mean([m.recall_at_thr    for m in in_band])),
        "mean_pr_auc":           float(np.mean([m.pr_auc           for m in in_band])),
        "mean_brier":            float(np.nanmean([m.brier or np.nan for m in in_band])),
        "alerts_per_bus_hour":   _alerts_per_bus_hour(probs_te, thresholds, step_seconds),
        "mean_recall_at_precision": {
            f"{t:.2f}": float(np.nanmean([
                (m.recall_at_precision[t] if m.recall_at_precision[t] is not None else np.nan)
                for m in in_band
            ])) for t in PRECISION_TARGETS
        },
    }

    # Per-period breakdown — recompute metrics over each time-of-day bucket.
    n_per_date: dict[str, int] = {}
    for d in ds.test.dates:
        n_per_date[d] = n_per_date.get(d, 0) + 1
    t_ref_est = _t_ref_minutes_per_example(ds.test.dates, n_per_date, step_seconds)
    period_idx = np.array([_period_for(t) for t in t_ref_est])
    per_period: dict[str, dict] = {}
    for name in (*PERIODS.keys(), "other"):
        sel = period_idx == name
        if not sel.any():
            continue
        prs, recs = [], []
        for h in range(pred_len):
            hm = (h + 1) * step_seconds / 60.0
            if not (1.0 <= hm <= max_band_min):
                continue
            y = Y_te[sel, h]; p = probs_te[sel, h]
            m = np.isfinite(y); y = y[m].astype(np.int32); p = p[m]
            if y.size == 0:
                continue
            pred = p >= thresholds[h]
            tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
            fn = int(((~pred) & (y == 1)).sum())
            prs.append(tp / max(1, tp + fp)); recs.append(tp / max(1, tp + fn))
        per_period[name] = {
            "n_examples": int(sel.sum()),
            "mean_precision_in_band": float(np.mean(prs)) if prs else 0.0,
            "mean_recall_in_band":    float(np.mean(recs)) if recs else 0.0,
            "alerts_per_bus_hour": _alerts_per_bus_hour(probs_te[sel], thresholds, step_seconds),
        }

    # Reliability bins: pool all in-band horizons (one diagram per bundle is
    # readable; per-horizon × 30 is too much). Use raw probabilities.
    pooled_probs = []
    pooled_y = []
    for h in range(pred_len):
        hm = (h + 1) * step_seconds / 60.0
        if not (1.0 <= hm <= max_band_min):
            continue
        y = Y_te[:, h]; m = np.isfinite(y)
        pooled_probs.append(probs_te[m, h]); pooled_y.append(y[m])
    rel_bins = _reliability_bins(
        np.concatenate(pooled_probs) if pooled_probs else np.array([]),
        np.concatenate(pooled_y)     if pooled_y     else np.array([]),
    )

    # Write outputs.
    bundle_label = bundle_dir.name
    out_sub = out_dir / bundle_label
    out_sub.mkdir(parents=True, exist_ok=True)

    report = {
        "bundle": str(bundle_dir),
        "bundle_meta": {k: bundle_meta.get(k) for k in
                         ("feature_set", "step_seconds", "pred_len", "n_extra",
                          "trained_at", "threshold_strategy", "calibrated")},
        "test_n": ds.test.n,
        "headline_band_min": float(max_band_min),
        "headline": headline,
        "per_horizon": [asdict(m) for m in per_horizon],
        "per_period": per_period,
        "reliability_bins": rel_bins,
        "precision_targets": list(PRECISION_TARGETS),
    }
    (out_sub / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))

    # Reliability CSV for plotting.
    with (out_sub / "reliability.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"])
        for b in rel_bins:
            w.writerow([b["bin_lo"], b["bin_hi"], b["n"], b["mean_pred"], b["obs_rate"]])

    _write_md(out_sub / "report.md", report, step_seconds, pred_len)
    print(f"Wrote {out_sub}/report.md", flush=True)
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not json-serialisable: {type(o)}")


def _write_md(path: Path, report: dict, step_seconds: int, pred_len: int) -> None:
    h = report["headline"]
    band = report["headline_band_min"]
    lines = [
        f"# Ops evaluation — {report['bundle']}",
        "",
        f"Test split: n = {report['test_n']} examples.",
        f"Bundle meta: feature_set=`{report['bundle_meta'].get('feature_set')}`, "
        f"threshold_strategy=`{report['bundle_meta'].get('threshold_strategy')}`, "
        f"calibrated=`{report['bundle_meta'].get('calibrated')}`",
        "",
        f"## Headline (mean across +1–{band:.0f} min horizons, at the bundle's chosen thresholds)",
        "",
        f"- **precision = {h['mean_precision_at_thr']:.3f}**",
        f"- **recall    = {h['mean_recall_at_thr']:.3f}**",
        f"- **alerts per bus-hour = {h['alerts_per_bus_hour']:.2f}**",
        f"- mean PR-AUC = {h['mean_pr_auc']:.3f}",
        f"- mean Brier  = {h['mean_brier']:.4f}",
        "",
        "## Recall achievable at precision floor (operating curve)",
        "",
        f"| precision floor | mean recall in [1, {band:.0f}] min |",
        "|---|---:|",
    ]
    for t, r in h["mean_recall_at_precision"].items():
        lines.append(f"| ≥ {float(t):.2f} | {r:.3f} |")
    lines += ["", f"## Per-period breakdown (1–{band:.0f} min mean P / R)", "",
              "| period | n | precision | recall | alerts/bus-hr |",
              "|---|---:|---:|---:|---:|"]
    for name, m in report["per_period"].items():
        lines.append(
            f"| {name} | {m['n_examples']} | {m['mean_precision_in_band']:.3f} | "
            f"{m['mean_recall_in_band']:.3f} | {m['alerts_per_bus_hour']:.2f} |"
        )
    lines += ["", "## Per-horizon detail (selected horizons)", "",
              "| h | min | n | pos% | thr | P | R | alerts/100 | PR-AUC | Brier | recall@P=0.30 | recall@P=0.50 |",
              "|---|----:|--:|-----:|----:|--:|--:|--:|---:|---:|---:|---:|"]
    sample = sorted({0, 4, 9, 14, 19, 24, pred_len - 1})
    for h in sample:
        if h >= len(report["per_horizon"]): continue
        m = report["per_horizon"][h]
        br = f"{m['brier']:.4f}" if m["brier"] is not None else "—"
        # ``recall_at_precision`` keys may be floats (in-memory) or strings
        # (after JSON round-trip via report.json). Handle both.
        rec = m["recall_at_precision"]
        r30 = rec.get(0.3, rec.get("0.3"))
        r50 = rec.get(0.5, rec.get("0.5"))
        r30s = f"{r30:.2f}" if r30 is not None else "—"
        r50s = f"{r50:.2f}" if r50 is not None else "—"
        lines.append(
            f"| {h} | +{m['horizon_min']:.0f} | {m['n']} | {m['pos_rate']*100:.1f} | "
            f"{m['threshold']:.2f} | {m['precision_at_thr']:.2f} | {m['recall_at_thr']:.2f} | "
            f"{m['alerts_per_100_examples']:.1f} | {m['pr_auc']:.2f} | {br} | {r30s} | {r50s} |"
        )
    lines += ["", f"## Reliability (pooled +1–{band:.0f} min)", "",
              "| bin | n | mean_pred | obs_rate |",
              "|---|--:|---:|---:|"]
    for b in report["reliability_bins"]:
        mp = f"{b['mean_pred']:.3f}" if b["mean_pred"] is not None else "—"
        orate = f"{b['obs_rate']:.3f}" if b["obs_rate"] is not None else "—"
        lines.append(f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | {b['n']} | {mp} | {orate} |")
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path,
                   help="Path to bundle root (single or bagged)")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.bundle, args.dataset, args.out)


if __name__ == "__main__":
    main()
