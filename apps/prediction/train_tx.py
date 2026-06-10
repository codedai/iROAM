"""Tiny Transformer encoder trainer — multi-task 30-horizon head.

A 20-tick × 16-channel window doesn't need a large model. We use:

* Per-tick linear projection to d_model=64.
* Learned positional embedding of length 20.
* 2-layer Transformer encoder, 4 heads, ff_dim=128, GELU, dropout=0.1.
* Mean pool over time → Linear → 30 logits.

Same NaN-masked BCE-with-logits loss + per-horizon pos_weight as the TCN.

Output bundle layout matches the TCN bundle (weights.pt + metadata + scaler +
thresholds), so the same DLPredictor wrapper serves both.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from apps.prediction.data import (
    apply_scaler_to_vendor_block,
    compute_scaler,
    load_dataset,
)
from apps.prediction.train_tcn import (
    _masked_bce_with_logits,
    _pos_weight_per_horizon,
    _tune_thresholds,
)


class TinyTransformer(nn.Module):
    def __init__(self, *, n_channels: int, seq_len: int, n_horizons: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        # Learned positional embedding — sequences are short and fixed-length,
        # so the simplest choice does well.
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C)
        z = self.input_proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)               # (B, L, d_model)
        z = z.mean(dim=1)                 # global average pool over time
        return self.head(z)               # (B, n_horizons)


def fit_and_write_tx(
    dataset_dir: Path, out_dir: Path, *,
    epochs: int = 40, batch_size: int = 512, lr: float = 1e-3,
    weight_decay: float = 1e-4, d_model: int = 64, n_layers: int = 2,
    n_heads: int = 4, ff_dim: int = 128, dropout: float = 0.1,
    patience: int = 6, device: str | None = None, seed: int = 2026,
) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}", flush=True)

    ds = load_dataset(str(dataset_dir))
    scaler = compute_scaler(ds.train.X_vendor)
    Xtr_v = apply_scaler_to_vendor_block(ds.train.X_vendor, scaler)
    Xva_v = apply_scaler_to_vendor_block(ds.val.X_vendor, scaler)
    Xtr = np.concatenate([Xtr_v, ds.train.X_extras], axis=2)   # (N, L, C)
    Xva = np.concatenate([Xva_v, ds.val.X_extras], axis=2)
    Ytr, Yva = ds.train.Y, ds.val.Y
    n_chans = Xtr.shape[-1]; L = Xtr.shape[1]; H = Ytr.shape[1]
    pos_w = torch.tensor(_pos_weight_per_horizon(Ytr), dtype=torch.float32, device=dev)
    print(f"n_chans={n_chans} L={L} H={H} pos_weight range=[{pos_w.min():.1f}, {pos_w.max():.1f}]",
          flush=True)

    Xtr_t = torch.from_numpy(Xtr).float()
    Xva_t = torch.from_numpy(Xva).float()
    Ytr_t = torch.from_numpy(Ytr).float()
    Yva_t = torch.from_numpy(Yva).float()
    dl_tr = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=(dev.type == "cuda"))
    dl_va = DataLoader(TensorDataset(Xva_t, Yva_t), batch_size=batch_size * 2, shuffle=False,
                       num_workers=2, pin_memory=(dev.type == "cuda"))

    model = TinyTransformer(
        n_channels=n_chans, seq_len=L, n_horizons=H,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ff_dim=ff_dim, dropout=dropout,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TX params={n_params:,}", flush=True)

    best_pr = -1.0; best_state = None; best_epoch = -1; bad_epochs = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tr_loss = 0.0; n_steps = 0
        for xb, yb in dl_tr:
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = _masked_bce_with_logits(model(xb), yb, pos_w)
            loss.backward()
            opt.step(); tr_loss += loss.item(); n_steps += 1
        sched.step()

        model.eval(); probs_list, y_list = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                p = torch.sigmoid(model(xb.to(dev, non_blocking=True))).cpu().numpy()
                probs_list.append(p); y_list.append(yb.numpy())
        probs_va = np.concatenate(probs_list, axis=0); y_va = np.concatenate(y_list, axis=0)
        prs = []
        for h in range(H):
            y = y_va[:, h]; m = np.isfinite(y)
            if m.sum() and len(np.unique(y[m])) > 1:
                prs.append(average_precision_score(y[m], probs_va[m, h]))
        mean_pr = float(np.mean(prs)) if prs else 0.0
        print(f"  epoch {ep:02d}: train_loss={tr_loss/max(1,n_steps):.4f}  "
              f"val_meanPR={mean_pr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if mean_pr > best_pr + 1e-4:
            best_pr = mean_pr
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = ep; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop after {bad_epochs} bad epochs", flush=True); break

    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    probs_list = []
    with torch.no_grad():
        for xb, _ in dl_va:
            probs_list.append(torch.sigmoid(model(xb.to(dev, non_blocking=True))).cpu().numpy())
    probs_va = np.concatenate(probs_list, axis=0)
    thresholds = _tune_thresholds(probs_va, Yva)
    print(f"Best val meanPR={best_pr:.4f} @ epoch {best_epoch}", flush=True)

    out_model = out_dir / "model"; out_model.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_model / "weights.pt")
    (out_model / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (out_model / "thresholds.json").write_text(json.dumps({
        str(t.horizon): {
            "threshold": t.threshold, "f2_val": t.f2_val,
            "pr_auc_val": t.pr_auc_val, "brier_val": t.brier_val,
            "best_iter": best_epoch, "positive_rate_train": 0.0, "method": "transformer",
        } for t in thresholds
    }, indent=2))
    (out_model / "metadata.json").write_text(json.dumps({
        "model_type": "tiny_transformer_multitask",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seq_len": L, "pred_len": H, "n_channels": n_chans,
        "n_features": int(L * n_chans),
        "feature_set": "rich",
        "extra_features": list(ds.manifest.get("extra_features", [])),
        "step_seconds": ds.manifest.get("step_seconds"),
        "route_id": ds.manifest.get("route_id"),
        "arch": {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "ff_dim": ff_dim, "dropout": dropout, "params": n_params,
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fit_and_write_tx(
        dataset_dir=args.dataset, out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, ff_dim=args.ff_dim,
        dropout=args.dropout, patience=args.patience, device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
