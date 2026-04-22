"""Apply Alembic migrations programmatically.

Useful for container entrypoints where invoking the ``alembic`` CLI would
require shipping the ini. This calls the same code paths using a programmatic
``Config`` that points at ``alembic.ini``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from core.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def upgrade_head() -> None:
    configure_logging()
    ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    cfg = Config(str(ini_path))
    _logger.info("alembic_upgrade_head_start")
    command.upgrade(cfg, "head")
    _logger.info("alembic_upgrade_head_done")


if __name__ == "__main__":
    upgrade_head()
