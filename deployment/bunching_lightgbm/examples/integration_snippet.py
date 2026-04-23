"""Sketch of how to embed the predictor in a live transit-monitoring loop.

This is a *template* — the AVL-ingestion / upstream-bus bookkeeping bits are
project-specific and left as stubs.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import BunchingPredictor  # noqa: E402

# --- one-time setup on service start ----------------------------------------
PREDICTOR = BunchingPredictor(ROOT / 'model')
SEQ_LEN = PREDICTOR.seq_len          # 60 steps by default
N_CHANNELS = PREDICTOR.n_channels    # 9 for step=2

# Ring buffer per bus — each bus id maps to a deque of the last SEQ_LEN
# 9-d feature rows. You are responsible for populating these from your AVL
# feed; the row convention is:
#   [ target_speed, target_gap, target_aux,
#     upstream1_speed, upstream1_gap, upstream1_aux,
#     upstream2_speed, upstream2_gap, upstream2_aux ]
# in *raw* units (m/s for speed, metres for gap).
_buffers: dict[str, deque] = {}


def record_tick(bus_id: str, row: np.ndarray) -> None:
    if bus_id not in _buffers:
        _buffers[bus_id] = deque(maxlen=SEQ_LEN)
    _buffers[bus_id].append(np.asarray(row, dtype=np.float32))


def predict_for(bus_id: str) -> dict | None:
    buf = _buffers.get(bus_id)
    if buf is None or len(buf) < SEQ_LEN:
        return None  # not enough history yet
    window = np.stack(buf, axis=0)   # (SEQ_LEN, N_CHANNELS) in raw units
    # is_scaled=False → the predictor will apply the bundled scaler.
    return PREDICTOR.alert(window, is_scaled=False)[0]


# --- example live loop ------------------------------------------------------
def live_loop_example():
    while True:
        snapshot = fetch_avl_snapshot()  # you supply: list[(bus_id, row)]
        for bus_id, row in snapshot:
            record_tick(bus_id, row)
            result = predict_for(bus_id)
            if result and result['any_alert']:
                publish_alert({
                    'bus_id': bus_id,
                    'lead_steps': result['first_alert_step'],
                    'confidence': result['max_prob'],
                })
        time.sleep(60)  # 1-minute AVL polling cadence


# --- stubs ------------------------------------------------------------------
def fetch_avl_snapshot():
    raise NotImplementedError('Wire this to your AVL feed.')


def publish_alert(payload: dict):
    raise NotImplementedError('Wire this to your alerting channel.')


if __name__ == '__main__':
    print('This file is a template; run run_example.py for a working demo.')
