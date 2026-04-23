"""Minimal end-to-end test runnable without the parent repo.

    cd deployment/bunching_lightgbm
    python -m tests.test_end_to_end
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import BunchingPredictor  # noqa: E402


def main() -> int:
    predictor = BunchingPredictor(ROOT / 'model')
    window = np.load(ROOT / 'examples' / 'example_input.npy')

    assert window.shape == (predictor.seq_len, predictor.n_channels), window.shape

    probs = predictor.predict_proba(window, is_scaled=True)
    assert probs.shape == (1, predictor.pred_len), probs.shape
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()

    scalar = predictor.predict_scalar(window, mode='max', is_scaled=True)
    assert scalar.shape == (1,)
    assert 0.0 <= float(scalar[0]) <= 1.0

    alert = predictor.alert(window, is_scaled=True)[0]
    assert set(alert) >= {'any_alert', 'first_alert_step', 'max_prob',
                           'max_prob_step', 'per_horizon'}

    batch = np.stack([window, window], axis=0)
    probs_batch = predictor.predict_proba(batch, is_scaled=True)
    assert probs_batch.shape == (2, predictor.pred_len)
    assert np.allclose(probs_batch[0], probs_batch[1])

    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
