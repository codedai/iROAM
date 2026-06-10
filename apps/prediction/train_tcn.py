"""Small dilated TCN trainer — multi-task 30-horizon head.

Design rationale (this is a CPU/GPU-cheap experiment, not a SOTA-paper bake-off):

* Input: (B, C=16, L=20) — channels-first, raw units (vendor block z-scored
  before feeding so it shares scale with the gradient-boosted baselines).
* Backbone: 3 dilated 1D conv blocks (dilation 1, 2, 4). Each block is
  Conv1d → GroupNorm → GELU → Dropout. Receptive field after 3 blocks at
  kernel=3 is 1 + 2*(1+2+4) = 15 ticks — enough to see most of the 20-tick
  window without padding artefacts on the edges.
* Head: global average pool over time → Linear → 30 logits (one per horizon).
* Loss: BCEWithLogitsLoss with per-horizon ``pos_weight`` for class imbalance
  + per-horizon NaN masking (a bus that doesn't last 30 min only supervises
  the horizons it has labels for).
* Optimizer: AdamW + cosine LR + early stop on val PR-AUC mean across horizons.

Output bundle (``<out>/model/``):
  weights.pt          # state_dict
  scaler.json         # speed/gap z-score stats (shared with the GBM bundles)
  thresholds.json     # per-horizon F2-optimal thresholds, tuned on val
  metadata.json       # arch + geometry + provenance
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss
from torch.utils.data import DataLoader, TensorDataset

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    compute_scaler,
    load_dataset,
)
from apps.prediction.train import _best_f2_threshold
from data_process.bunching.labels import N_CHANNELS


def _conv_block(in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> nn.Sequential:
    """Dilated causal-padded conv block. Padding = (k-1)*d so output length == input."""
    pad = (kernel - 1) * dilation // 2  # symmetric padding, not strictly causal — fine for 30 min horizon
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class TCN(nn.Module):
    def __init__(self, *, n_channels: int, n_horizons: int, hidden: int = 96,
                 dropout: float = 0.20) -> None:
        super().__init__()
        self.block1 = _conv_block(n_channels, hidden, kernel=3, dilation=1, dropout=dropout)
        self.block2 = _conv_block(hidden,     hidden, kernel=3, dilation=2, dropout=dropout)
        self.block3 = _conv_block(hidden,     hidden, kernel=3, dilation=4, dropout=dropout)
        # Residual shortcut from raw input to last block; helps gradient flow
        # in a tiny network and gives the model a cheap path to a near-linear
        # baseline.
        self.proj_in = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z) + self.proj_in(x)  # residual at the trunk's tail
        z = z.mean(dim=-1)                    # global average pool
        return self.head(z)                   # (B, n_horizons)


def _masked_bce_with_logits(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor,
) -> torch.Tensor:
    """BCE with per-horizon pos_weight; NaN labels masked out, mean over valid."""
    mask = ~torch.isnan(y)
    y_safe = torch.where(mask, y, torch.zeros_like(y))
    # Manual BCE so we can apply per-horizon pos_weight + sample mask.
    log_sig_pos = F.logsigmoid(logits)
    log_sig_neg = F.logsigmoid(-logits)
    loss_per = -(pos_weight * y_safe * log_sig_pos + (1 - y_safe) * log_sig_neg)
    loss = (loss_per * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def _pos_weight_per_horizon(Y: np.ndarray) -> np.ndarray:
    """Per-horizon (#neg / #pos), NaNs ignored. Clamp [1, 50] to avoid extreme."""
    out = np.ones(Y.shape[1], dtype=np.float32)
    for h in range(Y.shape[1]):
        y = Y[:, h]
        m = np.isfinite(y)
        if m.sum() == 0:
            continue
        pos = float((y[m] == 1).sum())
        neg = float((y[m] == 0).sum())
        if pos > 0:
            out[h] = max(1.0, min(50.0, neg / pos))
    return out


@dataclass
class HorizonThr:
    horizon: int
    threshold: float
    f2_val: float
    pr_auc_val: float
    brier_val: float | None


def _tune_thresholds(probs_val: np.ndarray, Y_val: np.ndarray) -> list[HorizonThr]:
    out: list[HorizonThr] = []
    for h in range(Y_val.shape[1]):
        y = Y_val[:, h]
        m = np.isfinite(y)
        if m.sum() == 0:
            out.append(HorizonThr(h, 0.5, 0.0, 0.0, None)); continue
        p = probs_val[m, h]; yy = y[m]
        thr, f2 = _best_f2_threshold(p, yy)
        pr = float(average_precision_score(yy, p)) if len(np.unique(yy)) > 1 else 0.0
        br = float(brier_score_loss(yy, p)) if len(np.unique(yy)) > 1 else None
        out.append(HorizonThr(h, float(thr), float(f2), pr, br))
    return out


def fit_and_write_tcn(
    dataset_dir: Path, out_dir: Path, *,
    epochs: int = 40, batch_size: int = 512, lr: float = 2e-3,
    weight_decay: float = 1e-4, hidden: int = 96, dropout: float = 0.20,
    patience: int = 6, device: str | None = None, seed: int = 2026,
) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}", flush=True)

    ds = load_dataset(str(dataset_dir))
    scaler = compute_scaler(ds.train.X_vendor)
    # Build 3D tensors: vendor scaled + extras passthrough, then channel-concat.
    Xtr_v = apply_scaler_to_vendor_block(ds.train.X_vendor, scaler)
    Xva_v = apply_scaler_to_vendor_block(ds.val.X_vendor, scaler)
    Xtr = np.concatenate([Xtr_v, ds.train.X_extras], axis=2)  # (N, L, C)
    Xva = np.concatenate([Xva_v, ds.val.X_extras], axis=2)
    Ytr, Yva = ds.train.Y, ds.val.Y
    n_chans = Xtr.shape[-1]
    L = Xtr.shape[1]
    H = Ytr.shape[1]

    pos_w = torch.tensor(_pos_weight_per_horizon(Ytr), dtype=torch.float32, device=dev)
    print(f"n_chans={n_chans} L={L} H={H} pos_weight range=[{pos_w.min():.1f}, {pos_w.max():.1f}]", flush=True)

    # Channels-first for Conv1d.
    Xtr_t = torch.from_numpy(Xtr.transpose(0, 2, 1)).float()
    Xva_t = torch.from_numpy(Xva.transpose(0, 2, 1)).float()
    Ytr_t = torch.from_numpy(Ytr).float()
    Yva_t = torch.from_numpy(Yva).float()
    dl_tr = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=(dev.type == "cuda"))
    dl_va = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size * 2, shuffle=False,
                       num_workers=2, pin_memory=(dev.type == "cuda"))

    model = TCN(n_channels=n_chans, n_horizons=H, hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TCN params={n_params:,}", flush=True)

    best_pr = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0; n_steps = 0
        for xb, yb in dl_tr:
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = _masked_bce_with_logits(logits, yb, pos_w)
            loss.backward()
            opt.step()
            tr_loss += loss.item(); n_steps += 1
        sched.step()

        # Val
        model.eval()
        probs_list, y_list = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(dev, non_blocking=True)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                probs_list.append(p); y_list.append(yb.numpy())
        probs_va = np.concatenate(probs_list, axis=0)
        y_va = np.concatenate(y_list, axis=0)

        # Mean PR-AUC across horizons that have both classes.
        prs = []
        for h in range(H):
            y = y_va[:, h]; m = np.isfinite(y)
            if m.sum() and len(np.unique(y[m])) > 1:
                prs.append(average_precision_score(y[m], probs_va[m, h]))
        mean_pr = float(np.mean(prs)) if prs else 0.0
        print(f"  epoch {ep:02d}: train_loss={tr_loss/max(1,n_steps):.4f}  "
              f"val_meanPR={mean_pr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if mean_pr > best_pr + 1e-4:
            best_pr = mean_pr; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = ep; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop after {bad_epochs} bad epochs", flush=True); break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    # Final val probs at best weights, then tune thresholds.
    probs_list = []
    with torch.no_grad():
        for xb, _ in dl_va:
            xb = xb.to(dev, non_blocking=True)
            probs_list.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs_va = np.concatenate(probs_list, axis=0)
    thresholds = _tune_thresholds(probs_va, Yva)
    print(f"Best val meanPR={best_pr:.4f} @ epoch {best_epoch}", flush=True)

    out_model = out_dir / "model"
    out_model.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_model / "weights.pt")
    (out_model / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (out_model / "thresholds.json").write_text(json.dumps({
        str(t.horizon): {
            "threshold": t.threshold,
            "f2_val": t.f2_val,
            "pr_auc_val": t.pr_auc_val,
            "brier_val": t.brier_val,
            "best_iter": best_epoch,
            "positive_rate_train": 0.0,
            "method": "tcn",
        } for t in thresholds
    }, indent=2))
    (out_model / "metadata.json").write_text(json.dumps({
        "model_type": "tcn_multitask",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seq_len": L,
        "pred_len": H,
        "n_channels": n_chans,
        "n_features": int(L * n_chans),
        "feature_set": "rich",
        "extra_features": list(ds.manifest.get("extra_features", [])),
        "step_seconds": ds.manifest.get("step_seconds"),
        "route_id": ds.manifest.get("route_id"),
        "arch": {
            "hidden": hidden, "dropout": dropout,
            "blocks": [
                {"kernel": 3, "dilation": 1},
                {"kernel": 3, "dilation": 2},
                {"kernel": 3, "dilation": 4},
            ],
            "params": n_params,
        },
        "training": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "best_epoch": best_epoch,
            "best_meanPR_val": best_pr,
        },
        "split": ds.manifest["split"],
    }, indent=2))
    print(f"Wrote bundle to {out_model}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fit_and_write_tcn(
        dataset_dir=args.dataset, out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, hidden=args.hidden, dropout=args.dropout,
        patience=args.patience, device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
