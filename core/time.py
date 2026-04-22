"""Time helpers. All datetimes in the system are timezone-aware UTC.

GTFS-Realtime timestamps are Unix epoch seconds (UTC). Conversion to
``America/Toronto`` happens only in presentation layers (dashboard).
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TORONTO_TZ = ZoneInfo("America/Toronto")


def utc_now() -> datetime:
    """Return the current UTC time as a tz-aware datetime."""
    return datetime.now(tz=timezone.utc)


def epoch_to_utc(epoch_seconds: int | float | None) -> datetime | None:
    """Convert Unix epoch seconds to a UTC datetime, or None if the input is falsy.

    GTFS-RT encodes "unset" as 0 (or the field missing); callers should pass
    None for "missing". We additionally treat 0 as missing because it's never
    a valid TTC timestamp.
    """
    if epoch_seconds in (None, 0):
        return None
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)


def to_toronto(dt: datetime | None) -> datetime | None:
    """Convert a UTC datetime to America/Toronto for display."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TORONTO_TZ)
