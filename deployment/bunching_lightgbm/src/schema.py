"""Input-shape validation for the bunching predictor."""

from __future__ import annotations

import numpy as np


class SchemaError(ValueError):
    """Raised when an input window does not match the expected shape."""


def validate_window(x: np.ndarray, seq_len: int, n_channels: int,
                    *, allow_batch: bool = True) -> np.ndarray:
    """Return ``x`` as a float32 ndarray with shape ``(batch, seq_len, n_channels)``.

    Accepts either a single window ``(seq_len, n_channels)`` or a batch
    ``(batch, seq_len, n_channels)``. Any other shape raises ``SchemaError``.

    Does NOT check whether the values are already scaled — use
    ``preprocess.scale_window`` if your input is still in raw units.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        if not allow_batch:
            raise SchemaError(
                f'Expected batched (batch, {seq_len}, {n_channels}), got {arr.shape}'
            )
        if arr.shape != (seq_len, n_channels):
            raise SchemaError(
                f'Expected a single window of shape ({seq_len}, {n_channels}), '
                f'got {arr.shape}'
            )
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[1:] != (seq_len, n_channels):
            raise SchemaError(
                f'Expected (batch, {seq_len}, {n_channels}), got {arr.shape}'
            )
    else:
        raise SchemaError(
            f'Input must be 2D or 3D, got {arr.ndim}D tensor of shape {arr.shape}'
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise SchemaError(f'Input contains {n_bad} non-finite values.')
    return arr
