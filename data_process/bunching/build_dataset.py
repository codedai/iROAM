"""Bulk-build bunching-prediction datasets for one route across many days.

CLI:
    python -m data_process.bunching.build_dataset \
        --route 29 --dir 0 --dir 1 \
        --since 2026-04-22 --until 2026-05-26 \
        --out out/datasets/route29

Output: one parquet shard per (date, direction) under ``--out``, plus a
``manifest.json`` that records the schema, the date range, and a
chronological train/val/test split (used by the trainer).

We persist the 60×9 vendor-schema window AND the 60×N_EXTRA richer-feature
window flattened into the parquet so the trainer can pick either.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from db.session import SessionLocal

from .labels import (
    DEFAULT_PRED_LEN,
    DEFAULT_SEQ_LEN,
    DEFAULT_STEP_SECONDS,
    EXTRA_FEATURES,
    EXTRAS_SCHEMA_V,
    N_CHANNELS,
    N_EXTRA,
    VENDOR_SCHEMA_V,
    LabelledExample,
    extract_for_date,
)


def _row_from_example(ex: LabelledExample) -> dict:
    return {
        "service_date": ex.service_date,
        "route_id": ex.route_id,
        "direction_id": ex.direction_id,
        "trip_id": ex.trip_id,
        "start_date": ex.start_date,
        "vehicle_id": ex.vehicle_id,
        "bus_index": ex.bus_index,
        "t_ref_min": ex.t_ref_min,
        "stop_idx_at_ref": ex.stop_idx_at_ref,
        "forward_gap_at_ref": ex.forward_gap_at_ref,
        # Flatten arrays into single bytes blobs — much faster than wide
        # columnar storage for these dense small tensors.
        "window": ex.window.astype(np.float32).tobytes(),
        "extras": ex.extras.astype(np.float32).tobytes(),
        "labels": ex.labels.astype(np.float32).tobytes(),
        "label_gaps": ex.label_gaps.astype(np.float32).tobytes(),
    }


def _date_range(since: date, until: date) -> Iterable[date]:
    d = since
    while d <= until:
        yield d
        d += timedelta(days=1)


def build(
    route_id: str,
    directions: list[int],
    since: date,
    until: date,
    out_dir: Path,
    *,
    step_seconds: int = DEFAULT_STEP_SECONDS,
    seq_len: int = DEFAULT_SEQ_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dates = list(_date_range(since, until))

    with SessionLocal() as session:
        shard_index: list[dict] = []
        for d in all_dates:
            for direction in directions:
                examples = extract_for_date(
                    session, route_id=route_id, direction_id=direction, service_date=d,
                    step_seconds=step_seconds, seq_len=seq_len, pred_len=pred_len,
                )
                if not examples:
                    continue
                rows = [_row_from_example(ex) for ex in examples]
                df = pd.DataFrame(rows)
                shard = out_dir / f"route{route_id}_d{direction}_{d.isoformat()}.parquet"
                df.to_parquet(shard, index=False)
                shard_index.append(
                    {
                        "path": shard.name,
                        "service_date": d.isoformat(),
                        "direction_id": direction,
                        "n_examples": len(rows),
                    }
                )
                print(f"  {d} dir={direction} → {len(rows)} examples → {shard.name}", flush=True)

    # Chronological split on unique service dates: oldest train_frac → train,
    # next val_frac → val, remainder → test. This matches the design used by
    # the deployed vendor bundle and avoids same-day leakage.
    dates_used = sorted({s["service_date"] for s in shard_index})
    n = len(dates_used)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n - n_train > 1 else 0
    train_dates = dates_used[:n_train]
    val_dates = dates_used[n_train : n_train + n_val]
    test_dates = dates_used[n_train + n_val :]

    manifest = {
        "route_id": route_id,
        "directions": directions,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "step_seconds": step_seconds,
        "seq_len": seq_len,
        "n_channels": N_CHANNELS,
        "n_extra": N_EXTRA,
        "extra_features": list(EXTRA_FEATURES),
        "vendor_schema_v": VENDOR_SCHEMA_V,
        "extras_schema_v": EXTRAS_SCHEMA_V,
        "pred_len": pred_len,
        "shards": shard_index,
        "split": {
            "train_dates": train_dates,
            "val_dates": val_dates,
            "test_dates": test_dates,
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(s["n_examples"] for s in shard_index)
    print(
        f"Done: {total} examples across {len(shard_index)} shards. "
        f"train={len(train_dates)}d val={len(val_dates)}d test={len(test_dates)}d → {out_dir}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", required=True, help="GTFS route_id")
    p.add_argument("--dir", action="append", type=int, default=[], dest="directions",
                   help="Direction id (repeat for both; default = 0,1)")
    p.add_argument("--since", required=True, type=date.fromisoformat)
    p.add_argument("--until", required=True, type=date.fromisoformat)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--pred-len", type=int, default=DEFAULT_PRED_LEN)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    directions = args.directions or [0, 1]
    build(
        route_id=args.route,
        directions=directions,
        since=args.since,
        until=args.until,
        out_dir=args.out,
        step_seconds=args.step_seconds,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
