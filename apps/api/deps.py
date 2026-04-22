"""FastAPI dependencies.

``get_db`` yields a Session that is closed on request completion.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from db.session import SessionLocal


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
