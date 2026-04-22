"""Upsample the trip trajectory to a fixed temporal resolution.

``upsample_df`` is carried over verbatim from the legacy
``data_process/clean_and_combine.py`` with one addition — each emitted row is
tagged ``observed=False`` (it is a synthetic boundary point), while source
rows that happen to be carried through the nearer-midpoint logic inherit
``observed=True`` from the input. The algorithm: for each consecutive pair of
real rows, insert one synthetic row at every ``resolution_seconds`` boundary
between them; distance is interpolated via the next row's speed; other columns
are copied from whichever of (current, next) is closer in distance.
"""

from __future__ import annotations

import pandas as pd


_FINAL_COLUMN_ORDER = [
    "trip_id", "start_date", "service_date", "route_id", "direction_id", "shape_id",
    "vehicle_id",
    "datetime", "time_offset_seconds",
    "travel_distance_m", "moving_speed_m_s",
    "observed",
    "occupancy_status", "source_vehicle_position_id",
]


def compute_moving_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``moving_speed_m_s`` from distance/time diffs.

    The leading NaN is filled with 0 so upsample can use it on the first gap.
    """
    if df.empty:
        return df.assign(moving_speed_m_s=pd.Series(dtype=float))
    out = df.copy().sort_values("datetime").reset_index(drop=True)
    dist_diff = out["travel_distance_m"].diff()
    time_diff = out["datetime"].diff().dt.total_seconds()
    speed = dist_diff / time_diff
    speed = speed.replace([float("inf"), float("-inf")], pd.NA)
    # Align: speed[i] describes movement from row i-1 -> row i. upsample_df
    # wants the "next row's speed" when bridging row i -> row i+1, which is
    # exactly speed[i+1]. So no shift here — the old pipeline used the same
    # convention (speed stored on the arriving row).
    out["moving_speed_m_s"] = speed.fillna(0.0)
    return out


def upsample_df(df: pd.DataFrame, resolution_seconds: int) -> pd.DataFrame:
    """Insert rows at fixed time boundaries between every consecutive pair.

    Each output row has ``observed`` set to False (it is synthesized). Source
    rows are NOT appended here — the boundary-point logic preserves one row's
    identity at each midpoint, matching the legacy pipeline's behavior.

    Requires columns: ``datetime``, ``travel_distance_m``, ``moving_speed_m_s``.
    """
    if len(df) < 2:
        return pd.DataFrame(columns=df.columns)

    rows = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i].copy()
        next_row = df.iloc[i + 1].copy()

        t_current = current_row["datetime"]
        t_next = next_row["datetime"]

        t_current_travel = current_row["travel_distance_m"]
        t_next_travel = next_row["travel_distance_m"]
        middle_travel = (t_current_travel + t_next_travel) / 2
        t_next_speed = next_row["moving_speed_m_s"]

        total_delta = (t_next - t_current).total_seconds()
        if total_delta <= 0:
            continue

        epoch_current = int(t_current.timestamp())
        first_boundary = (epoch_current // resolution_seconds) * resolution_seconds
        if first_boundary < epoch_current:
            first_boundary += resolution_seconds
        while first_boundary < epoch_current:
            first_boundary += resolution_seconds

        candidate = first_boundary
        epoch_next = int(t_next.timestamp())
        while candidate < epoch_next:
            t_candidate = pd.to_datetime(candidate, unit="s", utc=True)
            partial_delta = (t_candidate - t_current).total_seconds()
            dist_candidate = t_current_travel + (partial_delta * t_next_speed)

            if dist_candidate < middle_travel:
                new_row = current_row.copy()
            else:
                new_row = next_row.copy()
            new_row["moving_speed_m_s"] = t_next_speed
            new_row["datetime"] = t_candidate
            new_row["travel_distance_m"] = dist_candidate
            new_row["observed"] = False
            rows.append(new_row)
            candidate += resolution_seconds

    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def last_step_clean_up(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric fields and reorder columns to the canonical output schema."""
    if df.empty:
        return df
    out = df.copy()
    if "travel_distance_m" in out.columns:
        out["travel_distance_m"] = out["travel_distance_m"].round(2)
    if "moving_speed_m_s" in out.columns:
        out["moving_speed_m_s"] = out["moving_speed_m_s"].round(2)
    existing = [c for c in _FINAL_COLUMN_ORDER if c in out.columns]
    return out[existing]
