"""Feature diagnostics for a bagged-LightGBM bunching bundle.

Three views over the input feature set:

  1. **Gain importance** — sum of LightGBM split-gain across every booster,
     every horizon, every bag member, then aggregate to (a) per-channel
     totals and (b) per-tick totals. Per-channel tells you which *kinds*
     of features matter; per-tick tells you whether all 20 history ticks
     are pulling weight or only the most recent few.

  2. **Permutation importance** — for each channel, shuffle its values
     across all examples in the val split, re-predict, and measure the
     drop in mean PR-AUC across horizons. More honest than gain (which
     can be biased toward high-cardinality features); a 0 here means the
     model genuinely doesn't use that channel.

  3. **Pairwise correlation** — for each pair of channels, average the
     Pearson correlation across the 20 history ticks on a train sample.
     |ρ| > 0.95 is a near-duplicate; the model likely splits on one and
     ignores the other.

Together these answer "which features should I add / drop / keep?".

Usage:
    python -m apps.prediction.diag_features \
        --bundle deployment/bunching_local_rich_bag8_v5_10min_p30 \
        --dataset out/datasets/route29_v5 \
        --out out/diag/v5_features.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    build_feature_matrix,
    compute_scaler,
    load_dataset,
)


def _channel_names(bundle_meta: dict, n_extra: int) -> list[str]:
    """Reconstruct per-tick channel names from bundle metadata."""
    vsv = int(bundle_meta.get("vendor_schema_v", 1))
    if vsv == 2:
        vendor = ["target_speed", "target_fwd_gap",
                  "d1_speed", "d1_fwd_gap",
                  "d2_speed", "d2_fwd_gap"]
    else:
        vendor = ["target_speed", "target_gap", "target_aux",
                  "u1_speed", "u1_gap", "u1_aux",
                  "u2_speed", "u2_gap", "u2_aux"]
    extras = list(bundle_meta.get("extra_features") or [])
    if len(extras) != n_extra:
        # Fallback: synthesise generic names if metadata is missing/wrong.
        extras = [f"ex_{i:02d}" for i in range(n_extra)]
    return vendor + extras


def gain_importance(bundle_dir: Path) -> tuple[np.ndarray, dict]:
    """Returns (gain[bag, h, feature_flat_index], meta_for_naming).

    Sums to (n_features,) array of total gain across all bags + horizons,
    and per-(channel, tick) breakdowns are derived from the flat layout.
    """
    import lightgbm as lgb

    manifest = json.loads((bundle_dir / "bag_manifest.json").read_text())
    n_bags = int(manifest["n_bags"]); pred_len = int(manifest["pred_len"])
    seq_len = int(manifest["seq_len"]); n_channels = int(manifest["n_channels"])
    n_features = seq_len * n_channels

    out = np.zeros((n_bags, pred_len, n_features), dtype=np.float64)
    for k in range(n_bags):
        bag_model_dir = bundle_dir / f"bag_{k:02d}" / "model"
        for h in range(pred_len):
            p = bag_model_dir / f"booster_h{h:02d}.txt"
            head = p.open().readline()
            if head.startswith("CONSTANT"):
                continue
            b = lgb.Booster(model_file=str(p))
            g = b.feature_importance(importance_type="gain")
            # LightGBM returns importance over the features it actually saw
            # (= n_features). Length should match; if not, pad/truncate.
            if g.shape[0] == n_features:
                out[k, h, :] = g.astype(np.float64)
            else:
                m = min(n_features, g.shape[0])
                out[k, h, :m] = g[:m]
    return out, manifest


def permutation_importance(
    bundle_dir: Path, dataset_dir: Path, *, sample_size: int = 5000, seed: int = 2026,
) -> tuple[np.ndarray, list[str]]:
    """Per-channel PR-AUC drop when its values are shuffled across rows.

    Mean PR-AUC over horizons that have both classes; we measure baseline
    PR-AUC once, then shuffle each channel (across ALL its tick positions
    simultaneously) and re-score. ``importance[ch] = baseline - shuffled``.
    Positive = the model uses it.
    """
    from apps.prediction.bagged_predictor import BaggedPredictor

    rng = np.random.default_rng(seed)
    pred = BaggedPredictor(bundle_dir)
    feature_set = pred.metadata.get("feature_set", "rich")
    ds = load_dataset(str(dataset_dir))

    # Sample val to keep the permutation pass cheap (n_channels × predict).
    n = ds.val.n
    take = min(sample_size, n)
    idx = rng.choice(n, size=take, replace=False)
    Xv = ds.val.X_vendor[idx]; Xe = ds.val.X_extras[idx]; Yv = ds.val.Y[idx]

    scaler = compute_scaler(ds.train.X_vendor)
    Xv_scaled = apply_scaler_to_vendor_block(Xv, scaler)

    extras_for_merge = Xe if feature_set == "rich" else None
    base_flat = build_feature_matrix(Xv_scaled, extras_for_merge, feature_set)

    # Baseline PR-AUC (mean over horizons).
    n_features = base_flat.shape[1]
    seq_len = ds.seq_len
    n_channels = n_features // seq_len
    base_probs = _predict_via_bagged(pred, base_flat, seq_len, n_channels)
    base_pr = _mean_pr_auc(base_probs, Yv)

    importances = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        col_idx = [t * n_channels + ch for t in range(seq_len)]
        # Shuffle the channel ACROSS rows (preserving its own per-tick
        # joint distribution within a row). This is the standard
        # permutation-importance protocol.
        flat = base_flat.copy()
        perm = rng.permutation(take)
        flat[:, col_idx] = base_flat[perm][:, col_idx]
        probs = _predict_via_bagged(pred, flat, seq_len, n_channels)
        importances[ch] = base_pr - _mean_pr_auc(probs, Yv)

    n_extra = pred.metadata.get("n_extra") or (n_channels - 9)
    names = _channel_names(pred.metadata, int(n_extra))[:n_channels]
    return importances, names


def _predict_via_bagged(pred, flat: np.ndarray, seq_len: int, n_channels: int) -> np.ndarray:
    X3 = flat.reshape(-1, seq_len, n_channels)
    return pred.predict_proba(X3, is_scaled=True)


def _mean_pr_auc(probs: np.ndarray, Y: np.ndarray) -> float:
    """Mean PR-AUC over the horizons the BUNDLE serves (Y may carry more —
    e.g. a v2 dataset with 30 labels evaluated on a pred_len=10 bundle —
    so clamp to ``probs.shape[1]``)."""
    H = min(probs.shape[1], Y.shape[1])
    out = []
    for h in range(H):
        y = Y[:, h]; m = np.isfinite(y)
        if m.sum() and len(np.unique(y[m])) > 1:
            out.append(average_precision_score(y[m], probs[m, h]))
    return float(np.mean(out)) if out else 0.0


def channel_correlations(dataset_dir: Path, *, sample_size: int = 10000, seed: int = 2026,
                          ) -> tuple[np.ndarray, list[str]]:
    """Channel × channel correlations (averaged across the 20 history ticks)
    on a sample of train rows."""
    ds = load_dataset(str(dataset_dir))
    seq_len = ds.seq_len; n_extra = ds.n_extra
    n = ds.train.n
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    # Concatenate vendor + extras to a (sample, seq_len, n_channels_total) tensor.
    Xv = ds.train.X_vendor[idx]; Xe = ds.train.X_extras[idx]
    X3 = np.concatenate([Xv, Xe], axis=2)
    n_channels = X3.shape[2]
    # For each pair, compute corr per tick then mean.
    corr = np.eye(n_channels)
    for k in range(seq_len):
        c = np.corrcoef(X3[:, k, :].T)
        c = np.where(np.isnan(c), 0.0, c)
        corr += c
    corr = (corr - np.eye(n_channels)) / seq_len  # subtract the implicit
    np.fill_diagonal(corr, 1.0)

    manifest = ds.manifest
    names = _channel_names({
        "vendor_schema_v": manifest.get("vendor_schema_v", 1),
        "extra_features": manifest.get("extra_features"),
    }, int(n_extra))[:n_channels]
    return corr, names


# ──────────────────────────────── report ────────────────────────────────────


def render_report(
    bundle_dir: Path,
    dataset_dir: Path,
    *,
    sample_size: int,
    out_path: Path,
) -> dict:
    print(f"Bundle: {bundle_dir}", flush=True)
    print(f"Dataset: {dataset_dir}", flush=True)

    print("• Computing gain importance...", flush=True)
    gain, bag_man = gain_importance(bundle_dir)
    seq_len = int(bag_man["seq_len"]); n_channels = int(bag_man["n_channels"])
    n_extra = int(bag_man.get("n_extra", max(0, n_channels - 9)))
    names = _channel_names(bag_man, n_extra)[:n_channels]
    # Aggregate over bags + horizons.
    total_gain = gain.sum(axis=(0, 1))   # (n_features,)
    # Per-channel sum (over all 20 ticks of that channel).
    per_channel = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        cols = [t * n_channels + ch for t in range(seq_len)]
        per_channel[ch] = total_gain[cols].sum()
    # Per-tick sum (over all channels at that tick).
    per_tick = np.zeros(seq_len, dtype=np.float64)
    for k in range(seq_len):
        cols = [k * n_channels + ch for ch in range(n_channels)]
        per_tick[k] = total_gain[cols].sum()
    gain_pct_per_channel = 100.0 * per_channel / max(per_channel.sum(), 1.0)
    gain_pct_per_tick = 100.0 * per_tick / max(per_tick.sum(), 1.0)

    print("• Computing permutation importance...", flush=True)
    perm_imp, _ = permutation_importance(bundle_dir, dataset_dir, sample_size=sample_size)

    print("• Computing pairwise correlations...", flush=True)
    corr, _ = channel_correlations(dataset_dir, sample_size=sample_size * 2)

    # Find correlated pairs.
    pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if abs(corr[i, j]) >= 0.80:
                pairs.append((names[i], names[j], float(corr[i, j])))
    pairs.sort(key=lambda r: -abs(r[2]))

    # Persist + render.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Feature diagnostics — {bundle_dir.name}", "",
        f"Dataset: `{dataset_dir.name}`  ·  sample_size={sample_size}", "",
        "## Per-channel importance", "",
        "Gain = LightGBM split-gain summed over all bags × all horizons × all 20 history ticks of that channel.  "
        "PermImp = mean PR-AUC drop when the channel is shuffled on the val split (higher = more important).",
        "",
        "| # | channel | gain (%) | PermImp |",
        "|---|---|---:|---:|",
    ]
    # Sort by gain%.
    order = list(np.argsort(-per_channel))
    for ch in order:
        lines.append(
            f"| {ch} | `{names[ch]}` | {gain_pct_per_channel[ch]:.2f} | {perm_imp[ch]:+.4f} |"
        )
    lines += ["", "## Per-tick importance (gain summed across channels)", "",
              "Recent ticks (closer to t_ref) should dominate. If trailing ticks "
              "contribute a lot, history length may be longer than necessary.", "",
              "| tick (rel to t_ref) | gain (%) |", "|---:|---:|"]
    for k in range(seq_len):
        rel = -(seq_len - 1 - k)
        lines.append(f"| {rel:+d} | {gain_pct_per_tick[k]:.2f} |")

    lines += ["", "## Strongly-correlated channel pairs (|ρ| ≥ 0.80)", ""]
    if not pairs:
        lines.append("_None._")
    else:
        lines += ["| a | b | ρ |", "|---|---|---:|"]
        for a, b, r in pairs:
            lines.append(f"| `{a}` | `{b}` | {r:+.3f} |")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}", flush=True)

    return {
        "channels": names,
        "gain_pct_per_channel": gain_pct_per_channel.tolist(),
        "perm_importance": perm_imp.tolist(),
        "gain_pct_per_tick": gain_pct_per_tick.tolist(),
        "high_corr_pairs": pairs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sample-size", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    render_report(args.bundle, args.dataset, sample_size=args.sample_size, out_path=args.out)


if __name__ == "__main__":
    main()
