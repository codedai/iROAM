"""Normalizer unit tests — ensure the pure FeedMessage→ORM mapping is correct."""

from __future__ import annotations

from datetime import datetime, timezone

from google.transit import gtfs_realtime_pb2

from apps.collector.normalizer import build_trip_updates, extract_header
from tests._factories import make_feed_message


def _now() -> datetime:
    return datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)


def test_extract_header_basic() -> None:
    msg = make_feed_message(feed_timestamp=1_700_000_000)
    h = extract_header(msg)
    assert h.gtfs_realtime_version == "2.0"
    assert h.incrementality == "FULL_DATASET"
    assert h.feed_header_timestamp is not None
    assert int(h.feed_header_timestamp.timestamp()) == 1_700_000_000
    assert h.entity_count == 0


def test_build_trip_updates_fan_out_and_fields() -> None:
    msg = make_feed_message(
        entities=[
            {
                "id": "e1",
                "trip_id": "T1",
                "route_id": "501",
                "direction_id": 0,
                "start_date": "20260421",
                "start_time": "08:15:00",
                "vehicle_id": "V1",
                "vehicle_label": "Streetcar 4501",
                "timestamp": 1_700_000_000,
                "delay": -30,
                "stop_time_updates": [
                    {
                        "stop_sequence": 1,
                        "stop_id": "S1",
                        "arrival": {"time": 1_700_000_100, "delay": 10, "uncertainty": 5},
                    },
                    {
                        "stop_sequence": 2,
                        "stop_id": "S2",
                        "departure": {"time": 1_700_000_200, "delay": -5},
                        "schedule_relationship": gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SCHEDULED,
                    },
                ],
            },
            {
                "id": "e2",
                "trip_id": "T2",
                "route_id": "504",
            },
        ],
    )

    now = _now()
    rows = build_trip_updates(msg, fetched_at=now, feed_header_timestamp=now)
    assert len(rows) == 2

    tu1, tu2 = rows
    assert tu1.entity_id == "e1"
    assert tu1.trip_id == "T1"
    assert tu1.route_id == "501"
    assert tu1.direction_id == 0
    assert tu1.start_date == "20260421"
    assert tu1.start_time == "08:15:00"
    assert tu1.vehicle_id == "V1"
    assert tu1.vehicle_label == "Streetcar 4501"
    assert tu1.delay_seconds == -30
    assert tu1.trip_update_timestamp is not None
    assert tu1.fetched_at == now
    assert len(tu1.stop_times) == 2

    st1, st2 = tu1.stop_times
    assert st1.stop_sequence == 1
    assert st1.stop_id == "S1"
    assert st1.arrival_delay == 10
    assert st1.arrival_uncertainty == 5
    assert st1.departure_time is None

    assert st2.stop_id == "S2"
    assert st2.departure_delay == -5
    assert st2.schedule_relationship == "SCHEDULED"

    # Entity without a vehicle / timestamp / delay → all optional fields None.
    assert tu2.vehicle_id is None
    assert tu2.trip_update_timestamp is None
    assert tu2.delay_seconds is None
    assert tu2.stop_times == []


def test_build_trip_updates_skips_non_trip_update_entities() -> None:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    msg.header.timestamp = 1_700_000_000
    e = msg.entity.add()
    e.id = "vp-only"
    # Populate vehicle only — no trip_update.
    e.vehicle.trip.trip_id = "irrelevant"

    rows = build_trip_updates(msg, fetched_at=_now(), feed_header_timestamp=None)
    assert rows == []
