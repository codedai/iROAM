"""Lazy singleton around the bundled ``BunchingPredictor``.

Loading the predictor means reading 30 LightGBM booster files + 3 JSON blobs and
initialising a ``lightgbm`` runtime. ~100 ms on first touch, which is why we
cache it per process. LightGBM inference itself is thread-safe for reads, so
the cached instance is safe to share across FastAPI's thread pool.

Set ``BUNCHING_MODEL_DIR`` to point at an alternate bundle (e.g. for A/B tests
or retrained variants); otherwise the bundled ``deployment/bunching_lightgbm/model``
is used.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Resolve the bundled model dir once at module load. Walks up from this file:
#   apps/api/services/bunching_predictor.py  →  <repo_root>/deployment/bunching_lightgbm/model
_BUNDLE_ROOT = Path(__file__).resolve().parents[3] / "deployment" / "bunching_lightgbm"
_DEFAULT_MODEL_DIR = _BUNDLE_ROOT / "model"


class PredictorUnavailable(RuntimeError):
    """Raised when the model bundle is present but cannot be loaded (bad files,
    missing ``lightgbm`` runtime, etc.). Kept distinct from the stdlib errors so
    the API layer can translate it to a 503."""


@lru_cache(maxsize=2)
def _load(model_dir: str):  # noqa: ANN202 — returns BunchingPredictor
    # Import lazily so server startup doesn't pull in lightgbm when forecasting
    # is never requested, and so test environments without lightgbm can still
    # exercise the rest of the API.
    try:
        from deployment.bunching_lightgbm import BunchingPredictor
    except Exception as exc:
        raise PredictorUnavailable(
            f"cannot import BunchingPredictor from deployment bundle: {exc!r}"
        ) from exc

    path = Path(model_dir)
    if not path.is_dir():
        raise PredictorUnavailable(f"model directory not found: {path}")

    try:
        return BunchingPredictor(path)
    except Exception as exc:
        raise PredictorUnavailable(
            f"failed to initialise BunchingPredictor from {path}: {exc!r}"
        ) from exc


def get_predictor():  # noqa: ANN201 — returns BunchingPredictor
    """Return the process-local ``BunchingPredictor`` instance."""
    return _load(os.environ.get("BUNCHING_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))


def reset_cache() -> None:
    """Test hook: drop the cached predictor so ``get_predictor`` rebuilds it."""
    _load.cache_clear()
