"""Download a fresh TTC GTFS-RT VehiclePositions payload and save it to tests/fixtures/.

Run manually::

    python -m scripts.capture_sample
"""

from __future__ import annotations

import sys
from pathlib import Path

from apps.collector.fetcher import fetch_bytes
from apps.collector.parser import parse_feed_message
from core.config import get_settings
from core.logging import configure_logging, get_logger

_logger = get_logger(__name__)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "sample_vehicle_positions.pb"
)


def main() -> int:
    configure_logging()
    settings = get_settings()
    url = settings.gtfs_rt_vehicle_positions_url
    _logger.info("capturing_sample", extra={"url": url})

    result = fetch_bytes(url)
    message = parse_feed_message(result.content)
    entity_count = len(message.entity)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(result.content)
    _logger.info(
        "sample_saved",
        extra={
            "path": str(FIXTURE_PATH),
            "bytes": len(result.content),
            "entities": entity_count,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
