"""End-to-end smoke test: load the predictor and run it on the bundled sample.

Usage
-----
    cd deployment/bunching_lightgbm
    python examples/run_example.py

Expected output: probability vector of length 30, plus an alert dict.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import BunchingPredictor  # noqa: E402


def main() -> int:
    model_dir = ROOT / 'model'
    predictor = BunchingPredictor(model_dir)
    print(f'loaded predictor from {model_dir}')
    print(f'  seq_len={predictor.seq_len}  pred_len={predictor.pred_len}  '
          f'n_channels={predictor.n_channels}  n_features={predictor.n_features}')

    sample_path = HERE / 'example_input.npy'
    window = np.load(sample_path)
    print(f'loaded {sample_path.name} with shape {window.shape}')

    # 1. Probabilities over the 30-step horizon (input is already scaled).
    probs = predictor.predict_proba(window, is_scaled=True)[0]
    print('\nper-horizon bunching probability:')
    for h in range(0, predictor.pred_len, 5):
        print(f'  step {h:2d}: {probs[h]:.3f}')
    print(f'  step {predictor.pred_len - 1:2d}: {probs[-1]:.3f}')

    # 2. Operational alert using tuned F2 thresholds.
    alert = predictor.alert(window, is_scaled=True)[0]
    print('\nalert summary:')
    print(f'  any_alert        : {alert["any_alert"]}')
    print(f'  first_alert_step : {alert["first_alert_step"]}')
    print(f'  max_prob         : {alert["max_prob"]:.3f} '
          f'(at step {alert["max_prob_step"]})')

    # 3. Ground-truth label from the training pickle (for sanity).
    gt_path = HERE / 'example_input.json'
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        print('\nground truth:')
        print(f'  any bunching in next 30 steps: '
              f'{bool(gt.get("ground_truth_any_bunching_in_next_30_steps", 0))}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
