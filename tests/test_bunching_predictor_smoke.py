"""Prove the bundled BunchingPredictor works inside this project's import graph.

The bundle ships its own ``tests/test_end_to_end.py`` that runs standalone from
its folder; this test covers the integration path the API takes — importing via
``deployment.bunching_lightgbm`` and using the singleton loader in
``apps.api.services.bunching_predictor`` — so a future refactor of either side
fails loudly here rather than at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = REPO_ROOT / "deployment" / "bunching_lightgbm"


pytestmark = pytest.mark.skipif(
    not (BUNDLE_ROOT / "model" / "metadata.json").exists(),
    reason="bunching_lightgbm model bundle not present",
)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


skip_if_no_lightgbm = pytest.mark.skipif(
    not _lightgbm_available(), reason="lightgbm not installed"
)


@skip_if_no_lightgbm
def test_predictor_loads_and_scores_bundled_example():
    from apps.api.services.bunching_predictor import get_predictor, reset_cache

    reset_cache()
    predictor = get_predictor()
    assert predictor.seq_len == 60
    assert predictor.pred_len == 30
    assert predictor.n_channels == 9

    window = np.load(BUNDLE_ROOT / "examples" / "example_input.npy")
    assert window.shape == (60, 9)

    probs = predictor.predict_proba(window, is_scaled=True)
    assert probs.shape == (1, 30)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()

    alert = predictor.alert(window, is_scaled=True)[0]
    assert {"any_alert", "first_alert_step", "max_prob", "max_prob_step", "per_horizon"} <= alert.keys()


@skip_if_no_lightgbm
def test_predictor_singleton_returns_same_instance():
    from apps.api.services.bunching_predictor import get_predictor, reset_cache

    reset_cache()
    a = get_predictor()
    b = get_predictor()
    assert a is b
