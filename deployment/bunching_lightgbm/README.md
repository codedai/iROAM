# Bus Bunching Predictor — LightGBM deployment bundle

A self-contained per-horizon bunching-probability predictor for live transit
monitoring. Trained on TTC route 29 AVL data (2024); the interface is
route-agnostic — retrain on any feed that produces the same feature schema.

## Why LightGBM

This bundle uses gradient-boosted trees, not the deep time-series models from
the research side of the repo. The motivation:

- **Matches or beats SOTA on this task.** On the chronological split,
  LightGBM's per-horizon F2 is competitive with PatchTST/iTransformer while
  being simpler and cheaper.
- **Deploys without a GPU.** Inference is CPU-only, ~sub-millisecond per bus.
- **Well-calibrated by default.** Isotonic-calibrated probabilities integrate
  cleanly with threshold-based alerting.
- **Tiny, portable artefacts.** 30 `*.txt` boosters + four JSON files. No
  PyTorch, no ONNX, no custom runtime.

If you need the deep-model comparison context, see the top-level `CLAUDE.md`
and `results/comparison.md` in the parent repo.

---

## Folder layout

```
bunching_lightgbm/
├── README.md                 # this file
├── requirements.txt          # just lightgbm + numpy
├── train.py                  # retrain from your own pickles
├── model/                    # produced by train.py — the deployable artefact
│   ├── booster_h00.txt       # LightGBM booster, horizon 0 (next step)
│   ├── booster_h01.txt       # ...
│   ├── booster_h29.txt       # horizon 29 (30 steps ahead)
│   ├── scaler.json           # speed/gap mean-std + bunching threshold
│   ├── thresholds.json       # F2-optimal decision threshold per horizon
│   └── metadata.json         # training config + data provenance
├── src/
│   ├── __init__.py           # public API re-exports
│   ├── predict.py            # BunchingPredictor class
│   ├── preprocess.py         # raw↔scaled conversion
│   └── schema.py             # input shape validation
├── examples/
│   ├── example_input.npy     # one scaled 60×9 window (chrono test sample 0)
│   ├── example_input.json    # ground-truth label for the above
│   ├── run_example.py        # end-to-end: load model, predict, print
│   └── integration_snippet.py # template for live monitoring loop
└── tests/
    └── test_end_to_end.py    # sanity check runnable from this folder alone
```

---

## Install

```bash
pip install -r requirements.txt
```

That is the complete dependency list — `lightgbm >= 4` and `numpy >= 1.24`.
Python 3.9+ is assumed.

---

## Quick start (inference)

```python
from pathlib import Path
import numpy as np
from src import BunchingPredictor

predictor = BunchingPredictor(Path('model'))

# raw-unit window: (seq_len=60, n_channels=9) in m/s and metres
raw_window = np.load('examples/example_input.npy')   # (actually scaled, see below)

# is_scaled=True if the input is already z-score scaled (training units).
# is_scaled=False if the input is in raw units — the bundled scaler is applied.
probs = predictor.predict_proba(raw_window, is_scaled=True)  # (1, 30)

alert = predictor.alert(raw_window, is_scaled=True)[0]
# alert = {
#   'any_alert':        True,
#   'first_alert_step': 4,           # earliest horizon above F2 threshold
#   'max_prob':         0.72,
#   'max_prob_step':    7,
#   'per_horizon':      [0.11, 0.15, ..., 0.68],
# }
```

Smoke test:

```bash
python examples/run_example.py
python tests/test_end_to_end.py
```

---

## Input schema

The model consumes a rolling **`(seq_len, n_channels)` window** of features
for one bus and its upstream neighbours. Defaults: `seq_len=60` (60 one-second
or one-step ticks of history), `n_channels = 3 + step*3` where `step` is the
number of upstream buses included. The shipped model uses `step=2`, so
`n_channels=9` and the total feature count is `540`.

Channel order, repeated once per bus (target first, then upstream#1, upstream#2):

| offset | name  | raw unit | scaling             |
|--------|-------|----------|---------------------|
| 0      | speed | m/s      | `(x − speed_mean) / speed_std` |
| 1      | gap   | m        | `(x − gap_mean) / gap_std`     |
| 2      | aux   | —        | passed through unchanged       |

`aux` is a route-position / categorical feature preserved from the training
pipeline. If you do not have it in your live feed, fill it with the
training-time mean (0.0 in scaled space is usually safe).

At inference time the predictor accepts either:
- **already-scaled** windows (`is_scaled=True`, default), or
- **raw-unit** windows (`is_scaled=False`) — the bundled scaler in
  `model/scaler.json` is applied automatically.

The raw bunching threshold is 100 metres (gap below this = bunched). The
scaled threshold is `(100 − gap_mean) / gap_std` and is stored in
`scaler.json`; it is used at training time to produce the binary labels.

### Where do the upstream-bus features come from?

You must maintain a rolling (seq_len) buffer per active bus. On every AVL
tick for bus `B`:

1. Compute `B.speed` and `B.forward_gap` (metres along the shape to the
   nearest leader bus ahead in the same direction).
2. Identify the 1st and 2nd upstream buses — the two buses most recently
   behind `B` on the same shape — and grab their speed/gap as well.
3. Append the 9-d row to `B`'s buffer.

The exact logic for upstream-bus identification is **not** included here
because it is GTFS-shape-specific. See `iroam_simulator/` in the parent repo
for the reference implementation.

---

## Output

`predict_proba(x)` → `(batch, pred_len)` probabilities in `[0, 1]`. Horizon
`h` corresponds to "bunching (gap < 100 m) at time-step `t + h + 1`" for the
target bus in each window.

`alert(x)` → per-sample dict using the per-horizon F2-optimal thresholds
stored in `thresholds.json` (tuned on the val split at training time).

`predict_scalar(x, mode='max')` → `(batch,)` a single "will bunch soon?"
number per sample. `'max'` is the operational default; `'last'` aligns with
the academic last-step metric; `'mean'` is a smooth aggregate.

---

## Retraining on your own data

Produce a folder with the same layout as `filtered/training_data/chrono/`:

```
your_data/
├── scaler.pkl                 # pickle of tuple (speed_mean, speed_std, gap_mean, gap_std)
└── matched/
    ├── scaled_chrono_train.pkl   # list of (X, Y, CAT) tuples, X=(60,9), Y=(30,3)
    ├── scaled_chrono_val.pkl
    └── scaled_chrono_test.pkl
```

Then:

```bash
python train.py \
    --data_root /path/to/your_data \
    --variant matched \
    --step 2 --seq_len 60 --pred_len 30 \
    --out_dir ./model \
    --seed 2021
```

Training takes a few minutes on a modern desktop CPU. Output overwrites
`model/*.txt` and the three JSONs. The script also regenerates
`examples/example_input.{npy,json}` from the new test split.

For the pipeline that builds these pickles from raw AVL + GTFS, see
`data_process/` in the parent repo.

---

## Integration pattern for a monitoring system

See `examples/integration_snippet.py` for a complete template. The gist:

```python
predictor = BunchingPredictor('model')       # once at service start
buffer = {}                                  # bus_id -> deque(maxlen=60) of rows

def on_avl_tick(bus_id, raw_row):            # raw_row shape (9,)
    buf = buffer.setdefault(bus_id, deque(maxlen=predictor.seq_len))
    buf.append(raw_row)
    if len(buf) < predictor.seq_len:
        return                               # warming up
    window = np.stack(buf, axis=0)
    result = predictor.alert(window, is_scaled=False)[0]
    if result['any_alert']:
        raise_alert(bus_id, result)
```

Cost per bus per tick on a single CPU core: ≈1 ms for all 30 boosters. The
predictor is thread-safe for concurrent reads (no hidden state is mutated
during `predict_*`).

---

## Model performance reference

From the parent repo's chronological evaluation (seeds 2021–2023,
`v2_chrono`, test split = 2024-10-27 → 2024-12-26):

| horizon (steps) | precision | recall | F2  | PR-AUC |
|-----------------|-----------|--------|-----|--------|
| 0               | 0.75      | 0.93   | 0.88| 0.89   |
| 5               | 0.26      | 0.89   | 0.60| 0.43   |
| 15              | 0.20      | 0.84   | 0.50| 0.28   |
| 29              | 0.16      | 0.79   | 0.43| 0.20   |

Recall stays above 80% even 29 steps out; precision degrades as the horizon
extends (expected — bunching becomes less deterministic further ahead).

---

## Config quick reference

Hyperparameters used for the shipped model:

| key                    | value |
|------------------------|-------|
| seq_len                | 60    |
| pred_len               | 30    |
| step (upstream buses)  | 2     |
| n_channels per tick    | 9     |
| n_estimators           | 300   |
| num_leaves             | 63    |
| learning_rate          | 0.05  |
| min_child_samples      | 50    |
| early_stopping_rounds  | 20    |
| bunching threshold     | gap < 100 m |
| scale_pos_weight       | auto, per horizon = (1 − pos_rate) / pos_rate |

All of these are overridable via `train.py --help`.

---

## Troubleshooting

- **`Input must be 2D or 3D`** — your window is the wrong shape. Expected
  `(60, 9)` or `(batch, 60, 9)`.
- **`Input contains N non-finite values`** — an AVL row had NaN/Inf. Impute
  before calling the predictor (forward-fill is usually fine).
- **All probabilities ≈ 0 or the positive rate** — the training data for a
  horizon was single-class; the corresponding `booster_h*.txt` is a
  `CONSTANT\t<rate>` sentinel. Retrain on a larger split.
- **Predictions look shifted** — make sure `is_scaled` matches your input
  units. Passing already-scaled data as `is_scaled=False` double-scales it
  and produces nonsense.
