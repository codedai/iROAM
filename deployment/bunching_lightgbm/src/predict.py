"""Bunching predictor: loads per-horizon LightGBM boosters and serves predictions.

Designed to be embedded inside a transit-monitoring service. Load once at
startup; call ``predict_proba`` / ``alert`` on every incoming AVL snapshot.

Everything needed to run this class lives under ``<model_dir>/``:
    booster_h00.txt ... booster_h{pred_len-1}.txt    # per-horizon boosters
    scaler.json                                       # normalization stats
    thresholds.json                                   # F2-optimal decision thresholds
    metadata.json                                     # training config
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .preprocess import scale_window
from .schema import validate_window


class BunchingPredictor:
    """Per-horizon bunching-probability predictor backed by LightGBM boosters.

    Parameters
    ----------
    model_dir : path
        Folder produced by ``train.py`` containing booster_h*.txt, scaler.json,
        thresholds.json, metadata.json.
    """

    def __init__(self, model_dir: str | Path):
        try:
            import lightgbm as lgb
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'lightgbm is required at inference time. pip install lightgbm'
            ) from e
        self._lgb = lgb

        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f'Model dir not found: {self.model_dir}')

        with open(self.model_dir / 'metadata.json') as f:
            self.metadata: dict = json.load(f)
        with open(self.model_dir / 'scaler.json') as f:
            self.scaler: dict = json.load(f)
        with open(self.model_dir / 'thresholds.json') as f:
            self.thresholds: dict[int, dict] = {
                int(k): v for k, v in json.load(f).items()
            }

        self.seq_len: int = self.metadata['seq_len']
        self.pred_len: int = self.metadata['pred_len']
        self.n_channels: int = self.metadata['n_channels']
        self.n_features: int = self.metadata['n_features']

        self._boosters: list = []
        self._constants: dict[int, float] = {}
        for h in range(self.pred_len):
            path = self.model_dir / f'booster_h{h:02d}.txt'
            if not path.exists():
                raise FileNotFoundError(f'Missing booster for horizon {h}: {path}')
            with open(path) as f:
                head = f.readline()
            if head.startswith('CONSTANT\t'):
                self._constants[h] = float(head.split('\t', 1)[1])
                self._boosters.append(None)
            else:
                self._boosters.append(self._lgb.Booster(model_file=str(path)))

    # ------------------------------------------------------------------ core
    def _flatten(self, x: np.ndarray) -> np.ndarray:
        arr = validate_window(x, self.seq_len, self.n_channels)
        return arr.reshape(arr.shape[0], -1)

    def predict_proba(self, x: np.ndarray, *, is_scaled: bool = True) -> np.ndarray:
        """Return per-horizon bunching probabilities.

        Parameters
        ----------
        x : ndarray
            A window ``(seq_len, n_channels)`` or a batch
            ``(batch, seq_len, n_channels)``.
        is_scaled : bool
            ``True`` (default) if ``x`` is already in the scaled units the
            model was trained on. ``False`` means raw units (m/s, metres)
            and will be scaled with the bundled scaler before inference.

        Returns
        -------
        ndarray of shape ``(batch, pred_len)`` containing probabilities in [0, 1].
        """
        if not is_scaled:
            x = np.asarray(x, dtype=np.float32)
            squeeze = x.ndim == 2
            if squeeze:
                x = x[None, ...]
            x = np.stack([scale_window(w, self.scaler) for w in x])
            if squeeze:
                x = x[0]
        flat = self._flatten(x)
        out = np.zeros((flat.shape[0], self.pred_len), dtype=np.float32)
        for h in range(self.pred_len):
            if h in self._constants:
                out[:, h] = self._constants[h]
            else:
                out[:, h] = self._boosters[h].predict(flat).astype(np.float32)
        return out

    def predict_scalar(self, x: np.ndarray, *, mode: str = 'max',
                        is_scaled: bool = True) -> np.ndarray:
        """Collapse per-horizon probabilities into a single scalar per sample.

        mode='max' (default): highest probability across horizons — the
        operationally useful "will this bus bunch *at any point* in the
        next ``pred_len`` steps?"
        mode='last': probability at the final horizon.
        mode='mean': mean probability across horizons.
        """
        p = self.predict_proba(x, is_scaled=is_scaled)
        if mode == 'max':
            return p.max(axis=1)
        if mode == 'last':
            return p[:, -1]
        if mode == 'mean':
            return p.mean(axis=1)
        raise ValueError(f'unknown mode {mode!r}')

    def alert(self, x: np.ndarray, *, is_scaled: bool = True) -> list[dict]:
        """For each sample, return a dict describing the alert status.

        Each dict has:
            any_alert        : bool, whether any horizon exceeds its F2 threshold
            first_alert_step : earliest horizon step above threshold, or None
            max_prob         : highest probability across horizons
            max_prob_step    : horizon where max_prob occurs
            per_horizon      : full probability vector
        """
        probs = self.predict_proba(x, is_scaled=is_scaled)
        thrs = np.array([self.thresholds[h]['threshold']
                         for h in range(self.pred_len)], dtype=np.float32)
        exceed = probs >= thrs  # (batch, pred_len)
        results = []
        for i in range(probs.shape[0]):
            any_hit = bool(exceed[i].any())
            first = int(np.argmax(exceed[i])) if any_hit else None
            max_idx = int(np.argmax(probs[i]))
            results.append({
                'any_alert': any_hit,
                'first_alert_step': first,
                'max_prob': float(probs[i, max_idx]),
                'max_prob_step': max_idx,
                'per_horizon': probs[i].tolist(),
            })
        return results
