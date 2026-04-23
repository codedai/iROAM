"""Deployment bundle for the LightGBM bunching predictor.

Thin re-export so the parent app can do::

    from deployment.bunching_lightgbm import BunchingPredictor

without the submodule gymnastics the bundle's own ``src`` package assumes (it
is otherwise designed to be used standalone from inside its own folder).
"""

from .src import BunchingPredictor, scale_window, unscale_gap, validate_window

__all__ = [
    "BunchingPredictor",
    "scale_window",
    "unscale_gap",
    "validate_window",
]
