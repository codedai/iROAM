"""add analytics_runs and trip_trajectories

Stores the per-run metadata and the per-point upsampled trajectory output of
``apps/analytics``. Both append-only, indexed on the canonical lookup paths.

Revision ID: 0003_trajectories
Revises: 0002_pivot_vp
Create Date: 2026-04-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_trajectories"
down_revision = "0002_pivot_vp"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analytics_runs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("service_date", sa.Date, nullable=False),
        sa.Column("route_id", sa.String(32), nullable=True),
        sa.Column("config_json", postgresql.JSONB, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("rows_written", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index(
        "ix_analytics_runs_service_date_started",
        "analytics_runs",
        ["service_date", sa.text("started_at DESC")],
    )
    op.create_index("ix_analytics_runs_status", "analytics_runs", ["status"])

    op.create_table(
        "trip_trajectories",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "run_id",
            sa.BigInteger,
            sa.ForeignKey("analytics_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trip_id", sa.String(64), nullable=False),
        sa.Column("start_date", sa.String(8), nullable=False),
        sa.Column("service_date", sa.Date, nullable=False),
        sa.Column("route_id", sa.String(32), nullable=True),
        sa.Column("direction_id", sa.SmallInteger, nullable=True),
        sa.Column("shape_id", sa.String(32), nullable=True),
        sa.Column("vehicle_id", sa.String(64), nullable=True),
        sa.Column("datetime", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("time_offset_seconds", sa.Integer, nullable=True),
        sa.Column("travel_distance_m", postgresql.DOUBLE_PRECISION, nullable=False),
        sa.Column("moving_speed_m_s", sa.Float, nullable=True),
        sa.Column("observed", sa.Boolean, nullable=False),
        sa.Column("occupancy_status", sa.String(32), nullable=True),
        sa.Column(
            "source_vehicle_position_id",
            sa.BigInteger,
            sa.ForeignKey("vehicle_positions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_tt_trip_start_dt",
        "trip_trajectories",
        ["trip_id", "start_date", "datetime"],
    )
    op.create_index(
        "ix_tt_route_service_dt",
        "trip_trajectories",
        ["route_id", "service_date", "datetime"],
    )
    op.create_index("ix_tt_run", "trip_trajectories", ["run_id"])


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS trip_trajectories CASCADE")
    op.execute("DROP TABLE IF EXISTS analytics_runs CASCADE")
