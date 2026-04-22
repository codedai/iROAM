"""Tests for ``apps.analytics.upsample``.

Guard the boundary-insertion algorithm (carried verbatim from the legacy
pipeline) so a refactor doesn't silently shift any boundary by a second.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from apps.analytics.upsample import (
    compute_moving_speed,
    last_step_clean_up,
    upsample_df,
)


def _ts(hh: int, mm: int, ss: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(2026, 4, 20, hh, mm, ss, tzinfo=timezone.utc))


def test_compute_moving_speed_fills_leading_nan() -> None:
    df = pd.DataFrame(
        {
            "datetime": [_ts(8, 0, 0), _ts(8, 0, 10), _ts(8, 0, 30)],
            "travel_distance_m": [0.0, 100.0, 300.0],
        }
    )
    out = compute_moving_speed(df)
    # leading NaN filled with 0.0; then 10 m/s, then 10 m/s
    assert out["moving_speed_m_s"].tolist() == pytest.approx([0.0, 10.0, 10.0])


def test_compute_moving_speed_on_empty_df() -> None:
    df = pd.DataFrame(columns=["datetime", "travel_distance_m"])
    out = compute_moving_speed(df)
    assert out.empty
    assert "moving_speed_m_s" in out.columns


def test_upsample_inserts_boundary_rows_tagged_unobserved() -> None:
    df = pd.DataFrame(
        {
            "datetime": [_ts(8, 0, 0), _ts(8, 0, 30)],
            "travel_distance_m": [0.0, 300.0],
            "moving_speed_m_s": [0.0, 10.0],
            "observed": [True, True],
            "trip_id": ["T1", "T1"],
        }
    )
    out = upsample_df(df, resolution_seconds=10)
    # Boundaries at t=0, 10, 20 (next boundary 30 == next row, excluded).
    assert len(out) == 3
    assert [t.second for t in out["datetime"]] == [0, 10, 20]
    # Distance = current_travel + partial_delta * next_row.speed = 0 + (0,10,20) * 10
    assert out["travel_distance_m"].tolist() == pytest.approx([0.0, 100.0, 200.0])
    # All synthetic -> observed=False
    assert out["observed"].tolist() == [False, False, False]
    # trip_id inherited from current/next per nearer-midpoint logic.
    assert out["trip_id"].tolist() == ["T1", "T1", "T1"]


def test_upsample_nearer_midpoint_picks_from_current_or_next() -> None:
    # Distances 0 -> 300. Midpoint = 150. Boundaries at t=0,10,20 -> dist = 0,100,200.
    # dist_candidate < 150 pulls 'current'; >= 150 pulls 'next'.
    df = pd.DataFrame(
        {
            "datetime": [_ts(8, 0, 0), _ts(8, 0, 30)],
            "travel_distance_m": [0.0, 300.0],
            "moving_speed_m_s": [0.0, 10.0],
            "observed": [True, True],
            "vehicle_id": ["V_CURRENT", "V_NEXT"],
        }
    )
    out = upsample_df(df, resolution_seconds=10)
    assert out["vehicle_id"].tolist() == ["V_CURRENT", "V_CURRENT", "V_NEXT"]


def test_upsample_returns_empty_when_fewer_than_two_rows() -> None:
    df = pd.DataFrame(
        {
            "datetime": [_ts(8, 0, 0)],
            "travel_distance_m": [0.0],
            "moving_speed_m_s": [0.0],
            "observed": [True],
        }
    )
    out = upsample_df(df, resolution_seconds=10)
    assert out.empty


def test_upsample_skips_non_positive_delta() -> None:
    # Two rows at identical timestamps — pipeline shouldn't divide by zero.
    df = pd.DataFrame(
        {
            "datetime": [_ts(8, 0, 0), _ts(8, 0, 0)],
            "travel_distance_m": [0.0, 50.0],
            "moving_speed_m_s": [0.0, 0.0],
            "observed": [True, True],
        }
    )
    out = upsample_df(df, resolution_seconds=10)
    assert out.empty


def test_last_step_clean_up_rounds_and_reorders() -> None:
    df = pd.DataFrame(
        {
            "moving_speed_m_s": [1.23456],
            "observed": [True],
            "datetime": [_ts(8, 0, 0)],
            "travel_distance_m": [123.456],
            "trip_id": ["T1"],
            "start_date": ["20260420"],
            "route_id": ["29"],
            "extra_unused_col": ["should_be_dropped"],
        }
    )
    out = last_step_clean_up(df)
    # Reordered: trip_id, start_date appear before datetime; extra col dropped.
    assert list(out.columns)[:2] == ["trip_id", "start_date"]
    assert "extra_unused_col" not in out.columns
    assert out["travel_distance_m"].iloc[0] == 123.46
    assert out["moving_speed_m_s"].iloc[0] == 1.23
