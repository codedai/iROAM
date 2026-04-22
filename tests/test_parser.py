"""Parser unit tests — no DB, no network."""

from __future__ import annotations

import pytest

from apps.collector.parser import ParseError, parse_feed_message
from tests._factories import make_feed_message


def test_parse_roundtrip_preserves_fields() -> None:
    msg = make_feed_message(
        feed_timestamp=1_700_000_111,
        entities=[
            {
                "id": "tu-1",
                "trip_id": "12345",
                "route_id": "501",
                "direction_id": 0,
                "start_date": "20260421",
                "start_time": "12:34:56",
                "vehicle_id": "4321",
                "timestamp": 1_700_000_100,
                "delay": 42,
                "stop_time_updates": [
                    {"stop_sequence": 1, "stop_id": "A", "arrival": {"time": 1_700_000_200, "delay": 10}},
                    {"stop_sequence": 2, "stop_id": "B", "departure": {"time": 1_700_000_300, "delay": -5}},
                ],
            }
        ],
    )
    payload = msg.SerializeToString()

    parsed = parse_feed_message(payload)
    assert parsed.header.gtfs_realtime_version == "2.0"
    assert parsed.header.timestamp == 1_700_000_111
    assert len(parsed.entity) == 1
    entity = parsed.entity[0]
    assert entity.id == "tu-1"
    assert entity.trip_update.trip.trip_id == "12345"
    assert entity.trip_update.delay == 42
    assert len(entity.trip_update.stop_time_update) == 2


def test_parse_empty_payload_raises() -> None:
    with pytest.raises(ParseError):
        parse_feed_message(b"")


def test_parse_malformed_payload_raises() -> None:
    with pytest.raises(ParseError):
        parse_feed_message(b"\x00\x01\x02 this is not a feed message")


def test_parse_falls_back_to_text_format() -> None:
    """TTC's endpoint returns protobuf text format by default; we accept it."""
    from google.protobuf import text_format

    msg = make_feed_message(
        feed_timestamp=1_700_000_999,
        entities=[{"id": "tx", "trip_id": "TX", "route_id": "R"}],
    )
    text_bytes = text_format.MessageToString(msg).encode("utf-8")
    parsed = parse_feed_message(text_bytes)
    assert parsed.header.timestamp == 1_700_000_999
    assert len(parsed.entity) == 1
    assert parsed.entity[0].trip_update.trip.trip_id == "TX"
