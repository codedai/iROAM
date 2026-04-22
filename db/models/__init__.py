"""ORM model package.

Importing this package registers every table on ``Base.metadata``.
"""

from db.models.feed_fetch_log import FeedFetchLog
from db.models.raw_snapshot import RawGtfsrtSnapshot
from db.models.trip_trajectory import AnalyticsRun, TripTrajectory
from db.models.vehicle_position import VehiclePosition

__all__ = [
    "AnalyticsRun",
    "FeedFetchLog",
    "RawGtfsrtSnapshot",
    "TripTrajectory",
    "VehiclePosition",
]
