"""Train and persist a per-horizon LightGBM bunching classifier.

This is the only script in this bundle that touches the training pickles.
At inference time the `model/` folder is fully self-contained: 30 LightGBM
boosters (.txt native format), a scaler, per-horizon thresholds, and a
metadata manifest.

Input data layout (produced by data_process/build_chronological.py)
--------------------------------------------------------------------
A folder like ``filtered/training_data/chrono/`` containing:

    scaler.pkl                  # 4-tuple (speed_mean, speed_std, gap_mean, gap_std)
    matched/scaled_chrono_train.pkl
    matched/scaled_chrono_val.pkl
    matched/scaled_chrono_test.pkl

Each pickle is a list of ``(X, Y, CAT)`` tuples where
    X: (seq_len, 3 + step*3) scaled float32         # e.g. (60, 9) for step=2
    Y: (pred_len, 3) scaled float32                 # Y[:,0] is scaled forward gap
    CAT: auxiliary categorical metadata (unused here)

Channel layout inside X (repeats per upstream bus, starting with the target bus):
    offset 0: speed     (scaled with speed_mean/std)
    offset 1: gap       (scaled with gap_mean/std)
    offset 2: auxiliary (passed through unchanged)

Bunching label at horizon h: ``Y[h, 0] < (threshold_raw - gap_mean) / gap_std``.
Default ``threshold_raw = 100`` (metres).

Usage
-----
    python train.py \
        --data_root /home/jiahao/Documents/iroam_qt/filtered/training_data/chrono \
        --variant matched \
        --step 2 --seq_len 60 --pred_len 30 \
        --out_dir ./model \
        --seed 2021

Retraining on your own data: produce pickles in the same schema and pass
``--data_root`` pointing at that folder.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def _load_split(root: Path, variant: str, flag: str):
    pkl = root / variant / f'scaled_chrono_{flag}.pkl'
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    X = np.stack([d[0] for d in data]).astype(np.float32)
    Y = np.stack([d[1] for d in data]).astype(np.float32)
    return X, Y


def _load_scaler(root: Path) -> tuple[float, float, float, float]:
    with open(root / 'scaler.pkl', 'rb') as f:
        speed_mean, speed_std, gap_mean, gap_std = pickle.load(f)
    return float(speed_mean), float(speed_std), float(gap_mean), float(gap_std)


def _tune_f2_threshold(prob: np.ndarray, y: np.ndarray,
                       beta: float = 2.0) -> tuple[float, float]:
    """Return (best_threshold, best_fbeta). Grid over 1% increments."""
    grid = np.concatenate([[0.0, 1.0], np.linspace(0.01, 0.99, 99)])
    best_thr, best_f = 0.5, -1.0
    b2 = beta * beta
    y = y.astype(np.int32)
    for t in grid:
        pred = (prob >= t).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        denom = b2 * prec + rec
        if denom == 0:
            continue
        f = (1 + b2) * prec * rec / denom
        if f > best_f:
            best_f, best_thr = f, float(t)
    return best_thr, float(best_f)


def main():
    p = argparse.ArgumentParser(
        description='Train per-horizon LightGBM bunching classifiers.'
    )
    p.add_argument('--data_root', required=True,
                   help='Folder containing scaler.pkl and {variant}/scaled_chrono_*.pkl')
    p.add_argument('--variant', default='matched', choices=['matched', 'full'])
    p.add_argument('--step', type=int, default=2,
                   help='Number of upstream buses to include (enc_in = 3 + step*3).')
    p.add_argument('--seq_len', type=int, default=60)
    p.add_argument('--pred_len', type=int, default=30)
    p.add_argument('--threshold_raw', type=float, default=100.0,
                   help='Gap (metres) below which a bus is considered bunched.')
    p.add_argument('--out_dir', default='./model',
                   help='Where to write booster_h*.txt, scaler.json, etc.')
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--n_estimators', type=int, default=300)
    p.add_argument('--num_leaves', type=int, default=63)
    p.add_argument('--max_depth', type=int, default=-1)
    p.add_argument('--learning_rate', type=float, default=0.05)
    p.add_argument('--min_child_samples', type=int, default=50)
    p.add_argument('--early_stopping_rounds', type=int, default=20)
    args = p.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print('ERROR: lightgbm not installed. Run: pip install lightgbm',
              file=sys.stderr)
        return 1

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[train] loading data from {data_root} (variant={args.variant})')
    X_tr, Y_tr = _load_split(data_root, args.variant, 'train')
    X_va, Y_va = _load_split(data_root, args.variant, 'val')
    X_te, Y_te = _load_split(data_root, args.variant, 'test')
    speed_mean, speed_std, gap_mean, gap_std = _load_scaler(data_root)

    n_ch = 3 + args.step * 3
    X_tr = X_tr[:, :, :n_ch].reshape(len(X_tr), -1)
    X_va = X_va[:, :, :n_ch].reshape(len(X_va), -1)
    X_te = X_te[:, :, :n_ch].reshape(len(X_te), -1)

    thr_scaled = (args.threshold_raw - gap_mean) / gap_std
    Y_tr_bin = (Y_tr[:, :, 0] < thr_scaled).astype(np.int32)
    Y_va_bin = (Y_va[:, :, 0] < thr_scaled).astype(np.int32)
    Y_te_bin = (Y_te[:, :, 0] < thr_scaled).astype(np.int32)

    print(f'[train] n_train={len(X_tr)}  n_val={len(X_va)}  n_test={len(X_te)}')
    print(f'[train] features per sample: {X_tr.shape[1]} '
          f'(= seq_len {args.seq_len} × n_channels {n_ch})')
    print(f'[train] positive rate (train, last step): '
          f'{Y_tr_bin[:, -1].mean():.3f}')

    thresholds: dict[int, dict] = {}
    test_probs = np.zeros_like(Y_te_bin, dtype=np.float32)

    t_start = time.time()
    for h in range(args.pred_len):
        y_h = Y_tr_bin[:, h]
        y_va_h = Y_va_bin[:, h]
        y_te_h = Y_te_bin[:, h]

        booster_path = out_dir / f'booster_h{h:02d}.txt'

        if len(np.unique(y_h)) < 2:
            print(f'  horizon {h:02d}: only one class in train; '
                  f'storing constant prob={y_h.mean():.3f}')
            test_probs[:, h] = y_h.mean()
            # Write a 1-line sentinel file so the predictor can detect it.
            with open(booster_path, 'w') as f:
                f.write(f'CONSTANT\t{float(y_h.mean()):.6f}\n')
            thresholds[h] = {
                'threshold': 0.5,
                'f2_val': 0.0,
                'constant': float(y_h.mean()),
            }
            continue

        pos_rate = float(y_h.mean())
        scale_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)

        clf = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            learning_rate=args.learning_rate,
            min_child_samples=args.min_child_samples,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(
            X_tr, y_h,
            eval_set=[(X_va, y_va_h)],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds,
                                          verbose=False)],
        )
        clf.booster_.save_model(str(booster_path))

        prob_va = clf.predict_proba(X_va)[:, 1]
        prob_te = clf.predict_proba(X_te)[:, 1]
        test_probs[:, h] = prob_te

        thr, f2 = _tune_f2_threshold(prob_va, y_va_h, beta=2.0)
        thresholds[h] = {
            'threshold': float(thr),
            'f2_val': float(f2),
            'best_iter': int(clf.best_iteration_ or clf.n_estimators),
            'positive_rate_train': pos_rate,
        }

        if h % 5 == 0 or h == args.pred_len - 1:
            print(f'  horizon {h:02d}: iter={clf.best_iteration_}  '
                  f'val_f2={f2:.3f}  thr={thr:.3f}  '
                  f'test_prob_mean={prob_te.mean():.3f}')

    elapsed = time.time() - t_start
    print(f'[train] finished {args.pred_len} boosters in {elapsed:.1f}s')

    # scaler.json
    with open(out_dir / 'scaler.json', 'w') as f:
        json.dump({
            'speed_mean': speed_mean,
            'speed_std': speed_std,
            'gap_mean': gap_mean,
            'gap_std': gap_std,
            'threshold_raw': args.threshold_raw,
            'threshold_scaled': float(thr_scaled),
            'channel_layout': [
                {'offset': 0, 'name': 'speed', 'scale': 'speed_mean/std'},
                {'offset': 1, 'name': 'gap',   'scale': 'gap_mean/std'},
                {'offset': 2, 'name': 'aux',   'scale': 'passthrough'},
            ],
        }, f, indent=2)

    # thresholds.json
    with open(out_dir / 'thresholds.json', 'w') as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)

    # metadata.json
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({
            'model_type': 'per_horizon_lightgbm',
            'framework': 'lightgbm',
            'lightgbm_version': lgb.__version__,
            'numpy_version': np.__version__,
            'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'step': args.step,
            'n_channels': n_ch,
            'n_features': int(X_tr.shape[1]),
            'variant': args.variant,
            'data_root': str(data_root),
            'seed': args.seed,
            'hyperparameters': {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'num_leaves': args.num_leaves,
                'learning_rate': args.learning_rate,
                'min_child_samples': args.min_child_samples,
                'early_stopping_rounds': args.early_stopping_rounds,
            },
        }, f, indent=2)

    # Test-set metrics as a sanity check row
    overall_prob = test_probs.ravel()
    overall_y = Y_te_bin.ravel()
    mean_prob = float(overall_prob.mean())
    pos_rate_test = float(overall_y.mean())
    print(f'[train] test overall: mean_prob={mean_prob:.3f}  '
          f'positive_rate={pos_rate_test:.3f}')

    # Also write one scaled test sample to examples/ for demonstration.
    examples_dir = Path(__file__).resolve().parent / 'examples'
    examples_dir.mkdir(exist_ok=True)
    demo_window = X_te[0].reshape(args.seq_len, n_ch).astype(np.float32)
    demo_label_any = int(Y_te_bin[0].max())
    np.save(examples_dir / 'example_input.npy', demo_window)
    with open(examples_dir / 'example_input.json', 'w') as f:
        json.dump({
            'note': 'Scaled 60×9 window from the chrono test split (index 0). '
                    'Channels 0-2 target bus, 3-5 upstream#1, 6-8 upstream#2.',
            'shape': list(demo_window.shape),
            'ground_truth_any_bunching_in_next_30_steps': demo_label_any,
            'ground_truth_per_horizon': Y_te_bin[0].tolist(),
        }, f, indent=2)

    print(f'[train] wrote {out_dir}/ + examples/example_input.{{npy,json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
