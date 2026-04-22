"""Unit tests for ``apps.analytics.project_to_shape.project_trajectory``.

Uses synthetic shapely LineStrings in EPSG:3857 — no GTFS, no DB. The
transformer in ``shapes.py`` converts lon/lat to EPSG:3857 meters, so we
build a LineString in EPSG:3857 directly and feed matching lat/lon that
transform onto known points on that line.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from apps.analytics.project_to_shape import project_trajectory

_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon, lat = _TO_4326.transform(x, y)
    return float(lon), float(lat)


@pytest.fixture
def shape_line() -> LineString:
    # L-shape in EPSG:3857 meters: (0,0) -> (100,0) -> (100,100). Total length 200m.
    return LineString([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])


def _df_from_xy(points: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for x, y in points:
        lon, lat = _to_lonlat(x, y)
        rows.append({"latitude": lat, "longitude": lon})
    return pd.DataFrame(rows)


def test_projection_distances_on_straight_segment(shape_line: LineString) -> None:
    df = _df_from_xy([(0.0, 0.0), (25.0, 0.0), (50.0, 0.0), (100.0, 0.0)])
    out = project_trajectory(df, shape_line, max_orthogonal_distance_m=1.0)
    assert len(out) == 4
    assert out["travel_distance_m"].tolist() == pytest.approx([0.0, 25.0, 50.0, 100.0], abs=0.01)
    assert (out["orthogonal_distance_m"] < 0.01).all()


def test_projection_around_corner(shape_line: LineString) -> None:
    # After the corner, distance-along-shape is 100m (corner) + y coord.
    df = _df_from_xy([(100.0, 10.0), (100.0, 50.0), (100.0, 100.0)])
    out = project_trajectory(df, shape_line, max_orthogonal_distance_m=1.0)
    assert out["travel_distance_m"].tolist() == pytest.approx([110.0, 150.0, 200.0], abs=0.01)


def test_outliers_beyond_max_orthogonal_are_dropped(shape_line: LineString) -> None:
    # Two near-the-line points and one far outlier 500m off.
    df = _df_from_xy([(50.0, 0.0), (50.0, 500.0), (75.0, 0.0)])
    out = project_trajectory(df, shape_line, max_orthogonal_distance_m=200.0)
    assert len(out) == 2
    assert out["travel_distance_m"].tolist() == pytest.approx([50.0, 75.0], abs=0.01)


def test_empty_input_returns_empty_with_columns(shape_line: LineString) -> None:
    df = pd.DataFrame(columns=["latitude", "longitude"])
    out = project_trajectory(df, shape_line)
    assert out.empty
    assert "travel_distance_m" in out.columns
    assert "orthogonal_distance_m" in out.columns


def test_preserves_other_columns(shape_line: LineString) -> None:
    rows = []
    for x, y, tag in [(0.0, 0.0, "a"), (50.0, 0.0, "b"), (100.0, 0.0, "c")]:
        lon, lat = _to_lonlat(x, y)
        rows.append({"latitude": lat, "longitude": lon, "tag": tag})
    df = pd.DataFrame(rows)
    out = project_trajectory(df, shape_line, max_orthogonal_distance_m=1.0)
    assert out["tag"].tolist() == ["a", "b", "c"]
