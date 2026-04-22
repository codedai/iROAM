"""Collector CLI.

Usage::

    python -m apps.collector.main --once
    python -m apps.collector.main --loop
    python -m apps.collector.main --loop --interval 15
    python -m apps.collector.main --once --feed vehicle-positions   # default
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType

from apps.collector.feed_specs import get_spec
from apps.collector.runner import run_once
from core.config import get_settings
from core.constants import CANONICAL_FEED
from core.logging import configure_logging, get_logger
from db.session import SessionLocal

_logger = get_logger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTC GTFS-RT VehiclePositions collector")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run one cycle and exit")
    mode.add_argument("--loop", action="store_true", help="poll continuously")
    p.add_argument(
        "--interval",
        type=int,
        default=None,
        help="seconds between polls (overrides COLLECTOR_INTERVAL_SECONDS)",
    )
    p.add_argument(
        "--feed",
        type=str,
        default=CANONICAL_FEED,
        help=f"feed name (default: {CANONICAL_FEED})",
    )
    return p.parse_args(argv)


def _cycle(feed_name: str) -> bool:
    """Run one cycle with a fresh session. Returns True on success."""
    spec = get_spec(feed_name)
    with SessionLocal() as session:
        outcome = run_once(session, spec)
        return outcome.success


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = get_settings()
    interval = args.interval if args.interval is not None else settings.collector_interval_seconds

    if args.once:
        ok = _cycle(args.feed)
        return 0 if ok else 1

    stopping = False

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        _logger.info("signal_received", extra={"signal": signum})
        stopping = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _logger.info(
        "collector_start",
        extra={"feed": args.feed, "interval_seconds": interval},
    )
    while not stopping:
        start = time.monotonic()
        try:
            _cycle(args.feed)
        except Exception:  # noqa: BLE001 — loop must not die on one cycle
            _logger.exception("cycle_unhandled_error")
        elapsed = time.monotonic() - start
        remaining = max(0.0, interval - elapsed)
        for _ in range(int(remaining * 10)):
            if stopping:
                break
            time.sleep(0.1)

    _logger.info("collector_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
