"""Application configuration, loaded from environment / .env via pydantic-settings.

All apps (api, collector, dashboard) import ``get_settings()`` to read config.
Values are cached; mutate environment + restart the process to re-read.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the environment. See ``.env.example`` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://ttc:ttc@localhost:5433/ttc_gtfsrt",
        description="SQLAlchemy DSN. Must use the psycopg (v3) driver.",
    )

    # Realtime feeds. VehiclePositions is the canonical MVP feed.
    # Pin the binary wire format — the TTC endpoint otherwise serves
    # protobuf text which is ~7x larger and only intended for inspection.
    gtfs_rt_vehicle_positions_url: str = (
        "https://gtfsrt.ttc.ca/vehicles/position?format=binary"
    )
    gtfs_rt_trip_updates_url: str = "https://gtfsrt.ttc.ca/trips/update?format=binary"
    gtfs_rt_alerts_url: str = "https://gtfsrt.ttc.ca/alerts?format=binary"

    # Collector
    collector_interval_seconds: int = 20
    collector_http_timeout_seconds: float = 10.0
    collector_http_retries: int = 2
    collector_user_agent: str = "ttc-gtfsrt-platform/0.2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "*"

    # Dashboard
    dashboard_api_base_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    # Tuning
    active_vehicle_window_minutes: int = 5
    max_page_size: int = 5000

    # Analytics (apps/analytics)
    gtfs_static_dir: Path = Field(
        default=Path("Complete GTFS"),
        description="Directory containing GTFS static .txt files (trips, shapes, stops, ...).",
    )
    analytics_upsample_resolution_s: int = 10
    analytics_max_orthogonal_distance_m: float = 200.0

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.api_cors_origins.strip()
        if raw == "*" or not raw:
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
