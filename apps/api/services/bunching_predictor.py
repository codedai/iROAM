"""Lazy singleton around the deployed bunching predictor.

The bundle path can be either:
  * a flat single-model bundle (``<dir>/booster_h*.txt`` + scaler/thresholds/meta)
    served by ``deployment.bunching_lightgbm.BunchingPredictor``, or
  * a bagged bundle (``<dir>/bag_manifest.json`` + per-bag subdirs) served by
    ``apps.prediction.bagged_predictor.BaggedPredictor`` — averages K members
    and optionally applies isotonic calibration.

Pick the bundle via ``BUNCHING_MODEL_DIR``. Defaults walk a small priority list
so ops can drop in a new bundle without touching code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_DEPLOY_ROOT = Path(__file__).resolve().parents[3] / "deployment"

# Resolution order, first that exists wins. Top of the list = current SOTA.
_DEFAULT_CANDIDATES = (
    # v6 — extras schema v4: dropped 3 redundant location/dwell channels
    # (all ρ ≈ ±0.998 with stop_index_norm or PermImp ≈ 0) and added two
    # physics-derived ones — rel_speed_to_d1 (target − d1 speed) and
    # target_accel_3tick. +22% relative precision at the calibrated
    # threshold vs v5 at the same alert volume; same PR-AUC and Brier.
    # See out/diag/v5_features.md (motivation) and out/diag/v6_features.md
    # (the new features are actively used by the model).
    _DEPLOY_ROOT / "bunching_local_rich_bag8_v6_10min_p30",
    # v5 — extras schema v3 (drop leader_speed/dist_to_terminus_m/
    # time_to_terminus_min). Kept for instant rollback.
    _DEPLOY_ROOT / "bunching_local_rich_bag8_v5_10min_p30",
    # v4 — vendor schema v2 + extras schema v2 (10 extras with
    # leader_speed + 3 terminus channels). Kept for instant rollback.
    _DEPLOY_ROOT / "bunching_local_rich_bag8_v4_10min_p30",
    # v3 — vendor schema v1 (upstream+aux), kept for instant rollback if
    # v4 misbehaves. Same shape (pred_len=10).
    _DEPLOY_ROOT / "bunching_local_rich_bag8_v3_10min_p30",
    # v2 — 30-min horizon, kept as deeper fallback. Pair with
    # ``FORECAST_HORIZON_CAP_MIN=10`` to enforce the same 10-min product
    # at serving time.
    _DEPLOY_ROOT / "bunching_local_rich_bag8_v2_p30",
    _DEPLOY_ROOT / "bunching_local_rich_bag8",
    _DEPLOY_ROOT / "bunching_local_rich_v1" / "model",
    _DEPLOY_ROOT / "bunching_local_v1" / "model",
    _DEPLOY_ROOT / "bunching_lightgbm" / "model",        # legacy 2024 vendor bundle (last resort)
)


class PredictorUnavailable(RuntimeError):
    """Bundle is present but failed to load (bad files, missing runtime, …)."""


def _is_bag_bundle(path: Path) -> bool:
    return path.is_dir() and (path / "bag_manifest.json").is_file()


@lru_cache(maxsize=4)
def _load(model_dir: str):  # noqa: ANN202
    path = Path(model_dir)
    if not path.exists():
        raise PredictorUnavailable(f"model path not found: {path}")

    # Bagged?
    if _is_bag_bundle(path):
        try:
            from apps.prediction.bagged_predictor import BaggedPredictor
            return BaggedPredictor(path)
        except Exception as exc:
            raise PredictorUnavailable(
                f"failed to initialise BaggedPredictor from {path}: {exc!r}"
            ) from exc

    # Single-bundle path. The caller may have pointed at the bundle root
    # rather than at ``<bundle>/model/`` — accept either.
    if not (path / "metadata.json").is_file() and (path / "model" / "metadata.json").is_file():
        path = path / "model"

    try:
        from deployment.bunching_lightgbm import BunchingPredictor
    except Exception as exc:
        raise PredictorUnavailable(
            f"cannot import BunchingPredictor from deployment bundle: {exc!r}"
        ) from exc
    try:
        return BunchingPredictor(path)
    except Exception as exc:
        raise PredictorUnavailable(
            f"failed to initialise BunchingPredictor from {path}: {exc!r}"
        ) from exc


def _resolve_default() -> str:
    for cand in _DEFAULT_CANDIDATES:
        if cand.exists():
            return str(cand)
    # Final fallback so the loader still raises a useful error.
    return str(_DEFAULT_CANDIDATES[-1])


def get_predictor():  # noqa: ANN201
    """Return the process-local predictor instance.

    Resolution:
      1. ``BUNCHING_MODEL_DIR`` env var (explicit override).
      2. First existing entry in ``_DEFAULT_CANDIDATES``.
    """
    override = os.environ.get("BUNCHING_MODEL_DIR")
    return _load(override or _resolve_default())


def reset_cache() -> None:
    """Test hook: drop the cached predictor so ``get_predictor`` rebuilds it."""
    _load.cache_clear()
