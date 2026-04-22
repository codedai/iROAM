"""Streamlit dashboard home — platform summary for VehiclePositions."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from apps.dashboard.api_client import (
    APIError,
    feed_status_vehicle_positions,
    health,
    vehicles_latest,
)

st.set_page_config(page_title="TTC GTFS-RT", page_icon=":bus:", layout="wide")

st.title("TTC GTFS-Realtime — Platform Dashboard")
st.caption(
    "Append-only ingestion of the TTC VehiclePositions feed. "
    "Use the sidebar to drill into the live map, routes, a single vehicle, or replay."
)

try:
    h = health()
except APIError as exc:
    st.error(f"API error: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach API: {exc}")
    st.stop()

status_ok = h.get("db_ok") and h.get("status") == "ok"
if status_ok:
    st.success(f"API reachable — DB OK — canonical feed: `{h.get('feed_name')}`")
else:
    st.warning("API degraded")

try:
    status = feed_status_vehicle_positions()
except APIError as exc:
    st.error(f"Could not load feed status: {exc}")
    st.stop()

try:
    latest = vehicles_latest(minutes=5, limit=5000)
except APIError as exc:
    st.error(f"Could not load latest vehicles: {exc}")
    latest = []

active_vehicle_count = len(latest)
active_route_count = len({r.get("route_id") for r in latest if r.get("route_id")})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active vehicles (5m)", active_vehicle_count)
c2.metric("Active routes (5m)", active_route_count)
success_rate = status.get("success_rate_last_hour")
c3.metric(
    "Success rate (1h)",
    f"{success_rate * 100:.1f}%" if success_rate is not None else "n/a",
)
lag = status.get("lag_seconds")
c4.metric(
    "Feed lag",
    f"{lag:.0f}s" if lag is not None else "n/a",
)

c5, c6, c7 = st.columns(3)
c5.metric("Fetches last hour", status.get("fetches_last_hour", 0))
c6.metric("Successes (1h)", status.get("successes_last_hour", 0))
c7.metric("Failures (1h)", status.get("failures_last_hour", 0))

st.divider()

left, right = st.columns([2, 3])

with left:
    st.subheader("Last fetch")
    st.json(
        {
            "feed_name": status.get("feed_name"),
            "last_fetched_at": status.get("last_fetched_at"),
            "last_success_at": status.get("last_success_at"),
            "last_http_status": status.get("last_http_status"),
            "last_entity_count": status.get("last_entity_count"),
            "last_feed_header_timestamp": status.get("last_feed_header_timestamp"),
            "last_error_type": status.get("last_error_type"),
            "last_error_message": status.get("last_error_message"),
        }
    )

with right:
    st.subheader("Top routes (latest 5-minute window)")
    if latest:
        counts = Counter(
            r.get("route_id") for r in latest if r.get("route_id")
        ).most_common(15)
        df = pd.DataFrame(counts, columns=["route_id", "active_vehicles"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No vehicle observations in the last 5 minutes yet.")

st.info(
    "Use the sidebar → **Live Fleet Map** for a real-time scatter of every active "
    "vehicle, **Route Explorer** for route-scoped fleet health, "
    "**Vehicle Detail** for a single vehicle's trail, or **Replay** to play back a "
    "time window."
)
