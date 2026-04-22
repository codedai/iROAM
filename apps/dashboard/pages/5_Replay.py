"""Replay — play back a time window on a map using /replay/vehicles."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from apps.dashboard.api_client import APIError, replay_vehicles, routes_index

st.set_page_config(page_title="Replay", layout="wide")
st.title("Replay")
st.caption("Scrub a time window of append-only observations on the map.")

default_end = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
default_start = default_end - timedelta(minutes=15)

with st.sidebar:
    st.header("Window")
    start_date = st.date_input("Start date (UTC)", value=default_start.date())
    start_time = st.time_input("Start time (UTC)", value=default_start.time())
    end_date = st.date_input("End date (UTC)", value=default_end.date())
    end_time = st.time_input("End time (UTC)", value=default_end.time())

    try:
        routes = routes_index(minutes=180)
    except APIError as exc:
        st.error(f"Could not load route list: {exc}")
        routes = []

    route_choice = st.selectbox(
        "Route filter", options=["(all routes)", *sorted(routes)]
    )
    limit = st.number_input(
        "Max rows", min_value=500, max_value=20000, value=5000, step=500
    )
    bucket_seconds = st.select_slider(
        "Snapshot bucket (seconds)", options=[10, 15, 20, 30, 60], value=30
    )

start = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
end = datetime.combine(end_date, end_time, tzinfo=timezone.utc)

if end <= start:
    st.error("End must be after start.")
    st.stop()

route_id = None if route_choice == "(all routes)" else route_choice

try:
    rows = replay_vehicles(
        start=start, end=end, route_id=route_id, limit=int(limit)
    )
except APIError as exc:
    st.error(f"Could not load replay: {exc}")
    st.stop()

if not rows:
    st.info("No observations in this window.")
    st.stop()

df = pd.DataFrame(rows)
df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
df = df.dropna(subset=["latitude", "longitude"]).copy()
if df.empty:
    st.warning("Rows returned but none had coordinates.")
    st.stop()

df["speed_kph"] = (df["speed_mps"].fillna(0) * 3.6).round(1)

capped = len(df) >= int(limit)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Distinct vehicles", df["vehicle_id"].nunique(dropna=True))
c3.metric("Distinct routes", df["route_id"].nunique(dropna=True))
span_min = (df["fetched_at"].max() - df["fetched_at"].min()).total_seconds() / 60
c4.metric("Covered span", f"{span_min:.1f} min")

if capped:
    st.warning(
        f"Row cap of {int(limit)} reached — narrow the window or a route filter for "
        "full coverage."
    )

st.subheader("Rows per minute")
per_min = (
    df.assign(minute=df["fetched_at"].dt.floor("min"))
    .groupby("minute")
    .size()
    .reset_index(name="rows")
)
chart = (
    alt.Chart(per_min)
    .mark_bar()
    .encode(
        x=alt.X("minute:T", title="Minute (UTC)"),
        y=alt.Y("rows:Q", title="Observations"),
        tooltip=["minute:T", "rows:Q"],
    )
    .properties(height=180)
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Snapshot")

t_min = df["fetched_at"].min().to_pydatetime()
t_max = df["fetched_at"].max().to_pydatetime()

snapshot_time = st.slider(
    "Snapshot time (UTC)",
    min_value=t_min,
    max_value=t_max,
    value=t_max,
    step=timedelta(seconds=int(bucket_seconds)),
    format="YYYY-MM-DD HH:mm:ss",
)

window_start = snapshot_time - timedelta(seconds=int(bucket_seconds))
snap = df[(df["fetched_at"] > window_start) & (df["fetched_at"] <= snapshot_time)]

if snap.empty:
    st.info("No observations in this snapshot bucket.")
else:
    # latest row per vehicle inside the bucket
    snap = snap.sort_values("fetched_at").drop_duplicates(
        subset=["vehicle_id"], keep="last"
    )
    snap = snap.copy()
    snap["color"] = [[52, 152, 219]] * len(snap)
    snap["fetched_at_str"] = snap["fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    view_state = pdk.ViewState(
        latitude=float(snap["latitude"].mean()),
        longitude=float(snap["longitude"].mean()),
        zoom=11,
        pitch=0,
    )
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=snap,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=50,
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.85,
    )
    tooltip = {
        "html": (
            "<b>Vehicle {vehicle_id}</b><br/>"
            "Route: {route_id}<br/>"
            "Speed: {speed_kph} km/h<br/>"
            "Status: {current_status}<br/>"
            "At: {fetched_at_str}"
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
    st.caption(
        f"Showing {len(snap)} vehicles in bucket "
        f"({window_start:%H:%M:%S} → {snapshot_time:%H:%M:%S} UTC)"
    )

with st.expander("All returned rows"):
    display_cols = [
        "fetched_at",
        "vehicle_id",
        "route_id",
        "trip_id",
        "latitude",
        "longitude",
        "speed_kph",
        "current_status",
        "occupancy_status",
    ]
    present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[present], use_container_width=True, hide_index=True)
