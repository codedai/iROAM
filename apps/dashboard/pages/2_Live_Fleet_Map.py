"""Live Fleet Map — every active vehicle plotted from /vehicles/latest."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st

from apps.dashboard.api_client import APIError, routes_index, vehicles_latest

st.set_page_config(page_title="Live Fleet Map", layout="wide")
st.title("Live Fleet Map")
st.caption("Latest observation per vehicle within the staleness window.")

with st.sidebar:
    st.header("Filters")
    minutes = st.slider("Staleness window (minutes)", 1, 60, 5)
    try:
        routes = routes_index(minutes=max(minutes, 15))
    except APIError as exc:
        st.error(f"Could not load route list: {exc}")
        routes = []
    route_choice = st.selectbox(
        "Route filter", options=["(all routes)", *sorted(routes)]
    )
    color_by = st.radio("Color by", ["route", "speed", "status"], horizontal=True)
    show_bearing = st.checkbox("Show bearing arrows", value=False)
    auto_refresh = st.checkbox("Auto-refresh (20s)", value=False)

route_id = None if route_choice == "(all routes)" else route_choice

try:
    rows = vehicles_latest(route_id=route_id, minutes=minutes, limit=5000)
except APIError as exc:
    st.error(f"Could not load vehicles: {exc}")
    st.stop()

if not rows:
    st.info("No vehicles in this window.")
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


def _color_for_route(route: str | None) -> list[int]:
    """Stable pseudo-random color from route_id hash."""
    if not route:
        return [150, 150, 150]
    h = hash(route) & 0xFFFFFF
    return [(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF]


def _color_for_speed(mps: float | None) -> list[int]:
    if mps is None or pd.isna(mps):
        return [150, 150, 150]
    # 0 m/s = red, 15 m/s (~54 kph) = green
    ratio = max(0.0, min(1.0, float(mps) / 15.0))
    r = int(230 * (1 - ratio))
    g = int(200 * ratio + 40)
    return [r, g, 60]


_STATUS_COLOR = {
    "INCOMING_AT": [241, 196, 15],
    "STOPPED_AT": [231, 76, 60],
    "IN_TRANSIT_TO": [46, 204, 113],
}


def _color_for_status(status: str | None) -> list[int]:
    return _STATUS_COLOR.get(status or "", [120, 120, 180])


if color_by == "route":
    df["color"] = df["route_id"].apply(_color_for_route)
elif color_by == "speed":
    df["color"] = df["speed_mps"].apply(_color_for_speed)
else:
    df["color"] = df["current_status"].apply(_color_for_status)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Plotted vehicles", len(df))
c2.metric("Distinct routes", df["route_id"].nunique(dropna=True))
avg_age = int(df["age_seconds"].mean())
c3.metric("Avg position age", f"{avg_age}s")
mean_speed = df["speed_mps"].mean(skipna=True)
c4.metric(
    "Avg speed",
    f"{mean_speed * 3.6:.1f} km/h" if pd.notna(mean_speed) else "n/a",
)

view_state = pdk.ViewState(
    latitude=float(df["latitude"].mean()),
    longitude=float(df["longitude"].mean()),
    zoom=11,
    pitch=0,
)

scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[longitude, latitude]",
    get_fill_color="color",
    get_radius=40,
    radius_min_pixels=3,
    radius_max_pixels=10,
    pickable=True,
    opacity=0.85,
)

layers = [scatter]

if show_bearing:
    bearing_df = df.dropna(subset=["bearing"]).copy()
    if not bearing_df.empty:
        bearing_df["text"] = bearing_df["bearing"].apply(lambda b: "↑")
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=bearing_df,
                get_position="[longitude, latitude]",
                get_text="text",
                get_angle="-bearing",
                get_size=18,
                get_color=[30, 30, 30],
                billboard=False,
            )
        )

tooltip = {
    "html": (
        "<b>Vehicle {vehicle_id}</b><br/>"
        "Route: {route_id}<br/>"
        "Trip: {trip_id}<br/>"
        "Speed: {speed_kph} km/h<br/>"
        "Status: {current_status}<br/>"
        "Occupancy: {occupancy_status}<br/>"
        "Age: {age_seconds}s"
    ),
    "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"},
}

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )
)

with st.expander("Raw rows"):
    display_cols = [
        "vehicle_id",
        "route_id",
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

if auto_refresh:
    import time

    time.sleep(20)
    st.rerun()
