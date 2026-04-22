"""Declarative base for all ORM models.

Importing a model module (e.g. ``db.models.trip_update``) registers it with
``Base.metadata`` and makes it visible to Alembic's autogenerate.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base."""

    pass


# Importing the models package has the side effect of registering every table
# on ``Base.metadata``. Alembic's env.py relies on this.
def register_all_models() -> None:
    """Force-import every model module so Base.metadata is fully populated.

    Called by Alembic's env.py and anywhere we need an exhaustive metadata view
    (e.g. integration tests that use ``Base.metadata.create_all``).
    """
    # Local import to avoid circulars at module load time.
    from db import models  # noqa: F401
