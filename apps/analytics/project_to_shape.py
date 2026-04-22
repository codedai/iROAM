"""Project GPS points onto a route shape to get ``travel_distance_m``.

Pure: input DataFrame + LineString -> DataFrame with two added columns
(``travel_distance_m``, ``orthogonal_distance_m``) and outliers dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from apps.analytics.shapes import transform_lonlat_to_3857


def project_trajectory(
    df: pd.DataFrame,
    shape_line: LineString,
    *,
    max_orthogonal_distance_m: float = 200.0,
) -> pd.DataFrame:
    """Add ``travel_distance_m`` + ``orthogonal_distance_m``; drop far-off points.

    Expects columns ``latitude`` and ``longitude`` (WGS84).
    """
    if df.empty:
        return df.assign(travel_distance_m=pd.Series(dtype=float),
                         orthogonal_distance_m=pd.Series(dtype=float))

    travel = np.empty(len(df), dtype=float)
    orth = np.empty(len(df), dtype=float)

    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    for i in range(len(df)):
        x, y = transform_lonlat_to_3857(float(lons[i]), float(lats[i]))
        pt = Point(x, y)
        d = shape_line.project(pt)
        nearest = shape_line.interpolate(d)
        travel[i] = d
        orth[i] = pt.distance(nearest)

    out = df.copy()
    out["travel_distance_m"] = travel
    out["orthogonal_distance_m"] = orth
    out = out[out["orthogonal_distance_m"] <= max_orthogonal_distance_m].reset_index(drop=True)
    return out
