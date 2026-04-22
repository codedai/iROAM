"""Feed Health — recent fetches + per-minute success/failure chart."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from apps.dashboard.api_client import APIError, feed_status_vehicle_positions

st.set_page_config(page_title="Feed Health", layout="wide")
st.title("Feed Health")

try:
    status = feed_status_vehicle_positions()
except APIError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Feed: `{status.get('feed_name', 'vehicle-positions')}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fetches last hour", status["fetches_last_hour"])
c2.metric("Successes", status["successes_last_hour"])
c3.metric("Failures", status["failures_last_hour"])
success_rate = status.get("success_rate_last_hour")
c4.metric(
    "Success rate",
    f"{success_rate * 100:.1f}%" if success_rate is not None else "n/a",
)

c5, c6, c7 = st.columns(3)
c5.metric(
    "Last entity count",
    status.get("last_entity_count") if status.get("last_entity_count") is not None else "n/a",
)
lag = status.get("lag_seconds")
c6.metric("Feed lag", f"{lag:.0f}s" if lag is not None else "n/a")
c7.metric(
    "Last HTTP status",
    status.get("last_http_status") if status.get("last_http_status") is not None else "n/a",
)

recent = status.get("recent", [])
if not recent:
    st.info("No fetches recorded yet.")
    st.stop()

df = pd.DataFrame(recent)
df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)

st.subheader("Recent fetches")
display_cols = [
    "fetched_at",
    "success",
    "http_status",
    "duration_ms",
    "entity_count",
    "response_bytes",
    "error_type",
    "error_message",
]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

st.subheader("Per-minute counts")
buckets = (
    df.assign(minute=df["fetched_at"].dt.floor("min"))
    .groupby(["minute", "success"])
    .size()
    .reset_index(name="count")
)
buckets["outcome"] = buckets["success"].map({True: "success", False: "failure"})
chart = (
    alt.Chart(buckets)
    .mark_bar()
    .encode(
        x=alt.X("minute:T", title="Minute (UTC)"),
        y=alt.Y("count:Q", title="Fetches"),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(domain=["success", "failure"], range=["#2ecc71", "#e74c3c"]),
        ),
        tooltip=["minute:T", "outcome:N", "count:Q"],
    )
    .properties(height=260)
)
st.altair_chart(chart, use_container_width=True)
