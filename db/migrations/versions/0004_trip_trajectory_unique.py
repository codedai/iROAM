"""unique index on trip_trajectories(trip_id, start_date, datetime)

Replaces the non-unique ``ix_tt_trip_start_dt`` with a unique
``ux_trip_trajectories_instance_dt``. Enforces the analytics runner's
delete-then-insert contract at the schema level so any future code path
that tries to simple-append is rejected by the database rather than
silently doubling the dataset.

The migration **refuses to run** if existing rows already violate the
constraint — prints the duplicate count and instructs the operator to
run ``python -m scripts.db_reset --yes-i-am-sure`` (or otherwise
deduplicate) before retrying. This is intentional: silently dropping
old rows during a schema migration would be too destructive to do
implicitly.

Revision ID: 0004_tt_unique
Revises: 0003_trajectories
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0004_tt_unique"
down_revision = "0003_trajectories"
branch_labels = None
depends_on = None


_DUP_CHECK_SQL = """
SELECT COUNT(*)
FROM (
    SELECT trip_id, start_date, datetime
    FROM trip_trajectories
    GROUP BY trip_id, start_date, datetime
    HAVING COUNT(*) > 1
) dup
"""


def upgrade() -> None:
    conn = op.get_bind()
    dup_count = conn.execute(text(_DUP_CHECK_SQL)).scalar_one()
    if dup_count and dup_count > 0:
        raise RuntimeError(
            f"Cannot add unique index: trip_trajectories already contains "
            f"{dup_count} duplicate (trip_id, start_date, datetime) groups. "
            "Clean up before migrating — e.g. "
            "`python -m scripts.db_reset --yes-i-am-sure` then re-run the "
            "analytics pipeline, or delete offending rows manually."
        )

    op.drop_index("ix_tt_trip_start_dt", table_name="trip_trajectories")
    op.create_index(
        "ux_trip_trajectories_instance_dt",
        "trip_trajectories",
        ["trip_id", "start_date", "datetime"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_trip_trajectories_instance_dt", table_name="trip_trajectories")
    op.create_index(
        "ix_tt_trip_start_dt",
        "trip_trajectories",
        ["trip_id", "start_date", "datetime"],
    )
