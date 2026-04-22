"""Build ``dict[shape_id, shapely.LineString]`` (EPSG:3857) from ``shapes.txt``.

Stored in EPSG:3857 because ``LineString.project`` returns CRS-native units;
we want meters. Transformation is done once at build time per shape.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def build_linestrings(shapes_df: pd.DataFrame) -> dict[str, LineString]:
    """One ``LineString`` per ``shape_id`` in EPSG:3857 (meter units).

    Vertices are ordered by ``shape_pt_sequence``.
    """
    if shapes_df.empty:
        return {}
    df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
    result: dict[str, LineString] = {}
    for shape_id, group in df.groupby("shape_id", sort=False):
        lons = group["shape_pt_lon"].to_numpy()
        lats = group["shape_pt_lat"].to_numpy()
        xs, ys = _TO_3857.transform(lons, lats)
        if len(xs) >= 2:
            result[str(shape_id)] = LineString(zip(xs, ys))
    return result


@lru_cache(maxsize=1)
def _cached_shapes_key(shapes_id: int) -> dict[str, LineString]:  # pragma: no cover
    # Placeholder for future caching by DataFrame identity; not used yet.
    raise NotImplementedError


def transform_lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    """Transform a single (lon, lat) point to EPSG:3857 meters."""
    x, y = _TO_3857.transform(lon, lat)
    return float(x), float(y)
