"""Route Explorer — fleet-health aggregates + vehicle map for one route."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st

from apps.dashboard.api_client import (
    APIError,
    route_metrics,
    route_vehicles_latest,
    routes_index,
)

st.set_page_config(page_title="Route Explorer", layout="wide")
st.title("Route Explorer")

with st.sidebar:
    st.header("Filters")
    window_minutes = st.slider("Metrics window (minutes)", 5, 180, 15)
    latest_minutes = st.slider("Map staleness (minutes)", 1, 60, 5)

try:
    routes = routes_index(minutes=window_minutes)
except APIError as exc:
    st.error(f"Could not load route list: {exc}")
    st.stop()

if not routes:
    st.info(f"No routes observed in the last {window_minutes} minutes.")
    st.stop()

route_id = st.selectbox("Route", options=sorted(routes))

try:
    metrics = route_metrics(route_id, window_minutes=window_minutes)
except APIError as exc:
    st.error(f"Could not load route metrics: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active vehicles", metrics.get("active_vehicle_count", 0))
avg_speed = metrics.get("avg_speed_mps")
c2.metric(
    "Avg speed",
    f"{avg_speed * 3.6:.1f} km/h" if avg_speed is not None else "n/a",
)
max_speed = metrics.get("max_speed_mps")
c3.metric(
    "Max speed",
    f"{max_speed * 3.6:.1f} km/h" if max_speed is not None else "n/a",
)
latest_fetched = metrics.get("latest_fetched_at")
if latest_fetched:
    ts = pd.to_datetime(latest_fetched, utc=True)
    age = (datetime.now(tz=timezone.utc) - ts.to_pydatetime()).total_seconds()
    c4.metric("Latest obs age", f"{int(age)}s")
else:
    c4.metric("Latest obs age", "n/a")

left, right = st.columns(2)
with left:
    st.subheader("Status breakdown")
    sb = metrics.get("status_breakdown") or {}
    if sb:
        st.dataframe(
            pd.DataFrame(sorted(sb.items()), columns=["status", "count"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No status data.")
with right:
    st.subheader("Occupancy breakdown")
    ob = metrics.get("occupancy_breakdown") or {}
    if ob:
        st.dataframe(
            pd.DataFrame(sorted(ob.items()), columns=["occupancy", "count"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No occupancy data.")

st.divider()
st.subheader("Vehicles on route (latest)")

try:
    rows = route_vehicles_latest(route_id, minutes=latest_minutes, limit=5000)
except APIError as exc:
    st.error(f"Could not load vehicles: {exc}")
    st.stop()

if not rows:
    st.info("No vehicles on this route in the staleness window.")
    st.stop()

df = pd.DataFrame(rows)
df = df.dropna(subset=["latitude", "longitude"]).copy()
if df.empty:
    st.warning("Vehicles returned but none had coordinates.")
    st.stop()

df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
now = datetime.now(tz=timezone.utc)
df["age_seconds"] = (now - df["fetched_at"]).dt.total_seconds().astype(int)
df["speed_kph"] = (df["speed_mps"].fillna(0) * 3.6).round(1)
df["color"] = [[52, 152, 219]] * len(df)

bbox = metrics.get("bbox")
if bbox:
    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
else:
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

view_state = pdk.ViewState(
    latitude=center_lat, longitude=center_lon, zoom=12, pitch=0
)
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[longitude, latitude]",
    get_fill_color="color",
    get_radius=60,
    radius_min_pixels=4,
    radius_max_pixels=14,
    pickable=True,
    opacity=0.9,
)
tooltip = {
    "html": (
        "<b>Vehicle {vehicle_id}</b><br/>"
        "Trip: {trip_id}<br/>"
        "Speed: {speed_kph} km/h<br/>"
        "Status: {current_status}<br/>"
        "Stop seq: {current_stop_sequence}<br/>"
        "Age: {age_seconds}s"
    ),
    "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"},
}
st.pydeck_chart(
    pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )
)

with st.expander("Raw rows"):
    display_cols = [
        "vehicle_id",
        "trip_id",
        "latitude",
        "longitude",
        "bearing",
        "speed_kph",
        "current_status",
        "current_stop_sequence",
        "stop_id",
        "occupancy_status",
        "age_seconds",
        "fetched_at",
    ]
    present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[present], use_container_width=True, hide_index=True)
