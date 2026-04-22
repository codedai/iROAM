"""Shared pytest fixtures.

Database tests require a reachable Postgres. We default to the compose instance
at ``localhost:5433`` but honor ``TEST_DATABASE_URL`` if set. When unreachable,
DB-dependent tests are skipped rather than failed, so pure unit tests still run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from db.base import Base, register_all_models


def _default_test_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://ttc:ttc@localhost:5433/ttc_gtfsrt_test",
    )


@pytest.fixture(scope="session")
def test_db_url() -> str:
    return _default_test_url()


@pytest.fixture(scope="session")
def db_engine(test_db_url: str) -> Iterator[Engine]:
    """Session-scoped engine; creates schema once, drops at end."""
    register_all_models()
    try:
        engine = create_engine(test_db_url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres at {test_db_url} not reachable: {exc}")

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Per-test session using a SAVEPOINT that rolls back at teardown.

    We wrap every test in a transaction so inserts from one test don't leak
    into another; each ``commit()`` inside the test is redirected to a nested
    SAVEPOINT, honoring application code that calls ``session.commit()``.
    """
    connection = db_engine.connect()
    outer_txn = connection.begin()

    SessionFactory = sessionmaker(bind=connection, autoflush=False, expire_on_commit=False, future=True)
    session = SessionFactory()

    # Start a SAVEPOINT; restart it whenever the app code commits.
    nested = connection.begin_nested()

    from sqlalchemy import event

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess: Session, trans) -> None:  # type: ignore[no-untyped-def]
        nonlocal nested
        if trans.nested and not trans._parent.nested:  # type: ignore[attr-defined]
            nested = connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        outer_txn.rollback()
        connection.close()
