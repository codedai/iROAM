"""Engine + session factory.

Engine is lazily created the first time ``get_engine()`` or ``SessionLocal()``
is called, so importing this module is cheap and safe during tests.
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy Engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide SessionLocal factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal


def session_scope() -> Iterator[Session]:
    """Context-manager-style session yielder for scripts and the API.

    The caller is responsible for commit/rollback. Used as a FastAPI dependency
    via ``Depends(get_db)``; for ad-hoc usage, prefer ``with SessionLocal() as s``.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


# Public alias so callers can simply ``from db.session import SessionLocal``.
def SessionLocal() -> Session:  # noqa: N802 — keep the familiar factory name
    return get_session_factory()()
