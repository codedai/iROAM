"""Top-level forecast orchestration.

Given a materialised slice of the iROAM dataset (same object list the ``/buses``
endpoint assembles), pick the eligible "running" buses at ``t_ref``, build their
feature windows, batch them through the shipped LightGBM predictor, and emit
per-bus + aggregate output.

The payload shape is tuned for what the dashboard's ForecastPanel needs: each
per-bus entry carries enough to show a bar in the at-risk list, and the aggregate
``horizon_summary`` drives the 5-minute probability curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apps.analytics.anomalies import BusTrajectory

from .bunching_predictor import PredictorUnavailable, get_predictor
from .forecast_features import (
    SEQ_LEN,
    STEP_SECONDS,
    BusWindowResult,
    build_bus_window,
)


@dataclass(frozen=True)
class ForecastResult:
    t_ref_min: float
    horizon_steps: int
    step_seconds: int
    thresholds: list[float]
    per_bus: list[dict[str, Any]]
    horizon_summary: dict[str, list[float]]
    num_buses_total: int
    num_running: int
    num_eligible: int


def run_forecast(
    buses: list[BusTrajectory],
    *,
    num_stops: int,
    t_ref_min: float,
    freshness_s: float = 90.0,
    edge_exclude: int = 2,
    predictor: Any | None = None,
) -> ForecastResult:
    """Run the bunching predictor for every "running" bus in ``buses``.

    ``predictor`` override exists for tests; production path calls ``get_predictor``.
    """
    pred = predictor if predictor is not None else get_predictor()

    # Build windows / eligibility verdicts for every bus.
    results: list[BusWindowResult] = []
    for bus in buses:
        results.append(
            build_bus_window(
                bus,
                buses,
                t_ref_min=t_ref_min,
                num_stops=num_stops,
                freshness_s=freshness_s,
                edge_exclude=edge_exclude,
            )
        )

    eligible = [r for r in results if r.window is not None]

    # "Running" = fresh + on-route (edge-exclude passed OR failed because of edge
    # only). We count edge-excluded buses as running for the telemetry pane so a
    # user can see "14/18 running are being forecast, 4 near termini excluded".
    def _is_running(r: BusWindowResult) -> bool:
        if r.window is not None:
            return True
        reason = r.reason or ""
        return reason.startswith("stop_idx=") or reason.startswith(
            "no upstream bus on any tick"
        )

    num_running = sum(1 for r in results if _is_running(r))

    # Batch inference over eligible windows.
    per_bus_out: dict[int, dict[str, Any]] = {}
    if eligible:
        batch = np.stack([r.window for r in eligible], axis=0).astype(np.float32)
        probs = pred.predict_proba(batch, is_scaled=False)
        alerts = pred.alert(batch, is_scaled=False)
        for r, alert in zip(eligible, alerts, strict=True):
            per_bus_out[r.bus_index] = {
                "eligible": True,
                "ineligible_reason": None,
                "stop_idx": _round_or_none(r.stop_idx_at_ref),
                "any_alert": bool(alert["any_alert"]),
                "first_alert_step": alert["first_alert_step"],
                "max_prob": float(alert["max_prob"]),
                "max_prob_step": int(alert["max_prob_step"]),
                "per_horizon": [float(x) for x in alert["per_horizon"]],
            }
        thresholds = [
            float(pred.thresholds[h]["threshold"]) for h in range(pred.pred_len)
        ]
        any_alert_rate = (probs >= np.array(thresholds, dtype=np.float32)).mean(axis=0)
        mean_prob = probs.mean(axis=0)
    else:
        probs = np.zeros((0, SEQ_LEN), dtype=np.float32)  # placeholder
        thresholds = [
            float(pred.thresholds[h]["threshold"]) for h in range(pred.pred_len)
        ]
        any_alert_rate = np.zeros(pred.pred_len, dtype=np.float32)
        mean_prob = np.zeros(pred.pred_len, dtype=np.float32)

    # Per-bus payload, preserving input order, including ineligible buses.
    out_rows: list[dict[str, Any]] = []
    for bus, r in zip(buses, results, strict=True):
        row: dict[str, Any] = {
            "bus_id": bus.bus_index,
            "trip_id": bus.trip_id,
            "vehicle_id": bus.vehicle_id,
        }
        row.update(
            per_bus_out.get(
                r.bus_index,
                {
                    "eligible": False,
                    "ineligible_reason": r.reason,
                    "stop_idx": _round_or_none(r.stop_idx_at_ref),
                    "any_alert": None,
                    "first_alert_step": None,
                    "max_prob": None,
                    "max_prob_step": None,
                    "per_horizon": None,
                },
            )
        )
        out_rows.append(row)

    return ForecastResult(
        t_ref_min=float(t_ref_min),
        horizon_steps=int(pred.pred_len),
        step_seconds=STEP_SECONDS,
        thresholds=thresholds,
        per_bus=out_rows,
        horizon_summary={
            "any_alert_rate": [float(x) for x in any_alert_rate.tolist()],
            "mean_prob": [float(x) for x in mean_prob.tolist()],
        },
        num_buses_total=len(buses),
        num_running=int(num_running),
        num_eligible=len(eligible),
    )


def _round_or_none(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


__all__ = ["run_forecast", "ForecastResult", "PredictorUnavailable"]
