"""Optional CSV export of trajectory frames, grouped by (route_id, service_date, direction_id)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_day_csvs(out_dir: Path, frames_by_key: dict[tuple[str, str, int], list[pd.DataFrame]]) -> list[Path]:
    """Write one CSV per ``(route_id, service_date, direction_id)`` bucket.

    Filenames follow the legacy pipeline's shape: ``{route}_{service_date}_dir{N}.csv``.
    Returns the list of written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (route_id, service_date, direction_id), frames in frames_by_key.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        dir_token = f"dir{direction_id}" if direction_id is not None else "dirNA"
        path = out_dir / f"{route_id}_{service_date}_{dir_token}.csv"
        combined.to_csv(path, index=False)
        written.append(path)
    return written
