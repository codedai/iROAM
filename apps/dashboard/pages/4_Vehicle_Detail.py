"""Vehicle Detail — latest state, recent trail, speed chart for one vehicle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from apps.dashboard.api_client import APIError, vehicle_history, vehicle_latest

st.set_page_config(page_title="Vehicle Detail", layout="wide")
st.title("Vehicle Detail")

with st.sidebar:
    st.header("Inputs")
    vehicle_id = st.text_input("Vehicle ID", value="")
    history_minutes = st.slider("History window (minutes)", 5, 240, 60)
    history_limit = st.number_input(
        "Max rows", min_value=50, max_value=5000, value=1000, step=50
    )

if not vehicle_id:
    st.info("Enter a vehicle_id in the sidebar to inspect.")
    st.stop()

try:
    latest = vehicle_latest(vehicle_id)
except APIError as exc:
    if exc.status == 404:
        st.warning(f"No observations found for vehicle_id={vehicle_id!r}.")
    else:
        st.error(str(exc))
    st.stop()

st.subheader(f"Latest — vehicle {latest.get('vehicle_id')}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Route", latest.get("route_id") or "—")
c2.metric("Trip", latest.get("trip_id") or "—")
speed_mps = latest.get("speed_mps")
c3.metric(
    "Speed",
    f"{speed_mps * 3.6:.1f} km/h" if speed_mps is not None else "n/a",
)
c4.metric("Status", latest.get("current_status") or "—")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Stop seq", latest.get("current_stop_sequence") or "—")
c6.metric("Stop ID", latest.get("stop_id") or "—")
c7.metric("Occupancy", latest.get("occupancy_status") or "—")
bearing = latest.get("bearing")
c8.metric("Bearing", f"{bearing:.0f}°" if bearing is not None else "n/a")

st.caption(
    f"Last fetched at {latest.get('fetched_at')} "
    f"(vehicle timestamp: {latest.get('vehicle_timestamp')})"
)

st.divider()
st.subheader(f"Trail — last {history_minutes} minutes")

end = datetime.now(tz=timezone.utc)
start = end - timedelta(minutes=history_minutes)

try:
    history = vehicle_history(
        vehicle_id, start=start, end=end, limit=int(history_limit)
    )
except APIError as exc:
    st.error(f"Could not load history: {exc}")
    st.stop()

if not history:
    st.info("No history rows in this window.")
    st.stop()

df = pd.DataFrame(history)
df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
df = df.sort_values("fetched_at")
df = df.dropna(subset=["latitude", "longitude"]).copy()

if df.empty:
    st.warning("History rows exist but none had coordinates.")
    st.stop()

df["speed_kph"] = (df["speed_mps"].fillna(0) * 3.6).round(1)

points = df[["longitude", "latitude"]].values.tolist()
path_data = [{"path": points, "color": [231, 76, 60]}]

scatter_df = df[
    ["longitude", "latitude", "speed_kph", "fetched_at", "current_status"]
].copy()
scatter_df["fetched_at"] = scatter_df["fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

view_state = pdk.ViewState(
    latitude=float(df["latitude"].iloc[-1]),
    longitude=float(df["longitude"].iloc[-1]),
    zoom=13,
    pitch=0,
)
path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_color="color",
    get_width=4,
    width_min_pixels=2,
)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=scatter_df,
    get_position="[longitude, latitude]",
    get_fill_color=[52, 152, 219],
    get_radius=25,
    radius_min_pixels=2,
    radius_max_pixels=6,
    pickable=True,
    opacity=0.75,
)
tooltip = {
    "html": (
        "<b>{fetched_at}</b><br/>"
        "Speed: {speed_kph} km/h<br/>"
        "Status: {current_status}"
    ),
    "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"},
}
st.pydeck_chart(
    pdk.Deck(
        layers=[path_layer, point_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )
)

st.subheader("Speed over time")
speed_chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("fetched_at:T", title="Time (UTC)"),
        y=alt.Y("speed_kph:Q", title="Speed (km/h)"),
        tooltip=["fetched_at:T", "speed_kph:Q", "current_status:N"],
    )
    .properties(height=240)
)
st.altair_chart(speed_chart, use_container_width=True)

with st.expander("History rows"):
    display_cols = [
        "fetched_at",
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
    ]
    present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[present], use_container_width=True, hide_index=True)
