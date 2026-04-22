# `apps/analytics/` — trip-trajectory pipeline

Turns raw `vehicle_positions` GPS polls + the GTFS static schedule into clean,
upsampled per-trip trajectories (distance-along-route + speed, at fixed
time boundaries). One CLI invocation = one service date.

```
python -m apps.analytics.main --date YYYY-MM-DD [--route N] [--export-csv DIR]
# or via Make:
make analytics-run DATE=YYYY-MM-DD [ROUTE=N]
```

Output lands in two tables:

| Table               | What it holds                                                                  |
|---------------------|--------------------------------------------------------------------------------|
| `analytics_runs`    | One row per CLI invocation (`service_date`, `route_id`, config, status).       |
| `trip_trajectories` | One row per upsampled point, FK → `analytics_runs.id`.                         |

Both are **append-only**. Re-running the same `(date, route)` inserts a new
run; filter on `run_id = (SELECT max(id) FROM analytics_runs WHERE ...)` to
get the latest.

---

## Module layout

```
apps/analytics/
  main.py                # argparse CLI — opens SessionLocal, calls runner
  runner.py              # transaction owner; manages analytics_runs lifecycle
  pipeline.py            # per-trip-instance orchestration (no DB writes)
  gtfs_static.py         # cached loader for Complete GTFS/*.txt
  shapes.py              # shape_id → shapely LineString in EPSG:3857
  trajectory_extract.py  # VehiclePosition rows → ordered DataFrame
  project_to_shape.py    # project (lat,lon) onto a shape → travel_distance_m
  upsample.py            # compute_moving_speed + upsample_df + last_step_clean_up
  csv_export.py          # optional {route}_{date}_dir{N}.csv writer
```

The separation follows the same *pure-functions + one transaction-owning
runner* pattern as `apps/collector/`: every transform is a function on a
DataFrame, and only `runner.py` touches `session.commit()`.

---

## End-to-end data flow

```
 GTFS static                 vehicle_positions (Postgres)
      │                                 │
      ▼                                 ▼
  gtfs_static.load_all()      pipeline.list_trip_instances(session, date)
      │                                 │      returns [(trip_id, start_date), ...]
      ▼                                 │
  shapes.build_linestrings()            ▼
      │                       for each (trip_id, start_date):
      │                            │
      │                            ▼
      │                       fetch_by_trip_instance(session, trip_id, start_date)
      │                            │     → list[VehiclePosition]
      │                            ▼
      │                       trajectory_extract.build_trip_trajectory(rows, static.trips)
      │                            │     → DataFrame: datetime, lat/lon, vehicle_id,
      │                            │       trip_id, start_date, route_id, direction_id,
      │                            │       shape_id (joined), trip_start_datetime,
      │                            │       time_offset_seconds
      │                            ▼
      └──────────────────►    project_to_shape.project_trajectory(df, shape_line)
                                   │     adds travel_distance_m + orthogonal_distance_m,
                                   │     drops rows > max_orthogonal_distance_m
                                   ▼
                             upsample.compute_moving_speed(df)
                                   │     adds moving_speed_m_s (= Δdist / Δt)
                                   ▼
                             upsample.upsample_df(df, resolution_s=10)
                                   │     inserts boundary rows at every 10s tick;
                                   │     each marked observed=False
                                   ▼
                             upsample.last_step_clean_up(df)
                                   │     rounds, reorders columns
                                   ▼
                             runner._df_to_orm(df, run_id)
                                   │
                                   ▼
                             session.add_all(...); session.commit()
                                   │     (per trip instance — a crash mid-day
                                   │      keeps all earlier work)
                                   ▼
                             finalize analytics_runs.status = 'ok'
```

---

## Stage-by-stage, with the "why"

### 1. `gtfs_static.load_all()` — schedule lookup tables

Reads `Complete GTFS/*.txt` once and caches by directory mtime. The pipeline
uses:
- `trips.txt` — to resolve `trip_id → (shape_id, route_id, direction_id)`.
- `shapes.txt` — the actual route polyline, fed to `shapes.build_linestrings`.
- `stops.txt` / `stop_times.txt` — loaded but not yet consumed by analytics
  (reserved for future stop-arrival matching).

`GTFS_STATIC_DIR` env var (default `Complete GTFS`) controls the path. In
docker-compose, the directory is mounted at `/gtfs`.

### 2. `shapes.build_linestrings(shapes_df)` — projection-ready geometry

For each `shape_id`, groups vertices by `shape_pt_sequence`, builds a
`shapely.LineString` in **EPSG:3857 meters** (via a module-level
`pyproj.Transformer`). Why 3857 and not WGS84? `LineString.project(point)`
returns distance in the CRS's native units — we want meters, not degrees.
Transforming *once* at shape-build time is much cheaper than projecting each
incoming GPS point's coordinates on the fly.

Returns `dict[shape_id, LineString]` — a few hundred shapes, cheap to hold
resident.

### 3. `pipeline.list_trip_instances(session, service_date)` — find the work

A trip instance is keyed on `(trip_id, start_date)`: a `trip_id` can repeat
across service days, so we disambiguate with the date.

The TTC GTFS-RT feed **does not populate** `TripDescriptor.start_date`, so
the query falls back to a synthesized one:

```sql
COALESCE(
  start_date,
  to_char(timezone('America/Toronto', vehicle_timestamp), 'YYYYMMDD'),
  to_char(timezone('America/Toronto', fetched_at),        'YYYYMMDD')
)
```

Feeds that *do* set `start_date` (including GTFS's 24:00+ overnight
convention) are honored verbatim.

### 4. `db.queries.vehicles.fetch_by_trip_instance(trip_id, start_date)`

All `vehicle_positions` rows for that trip instance, **ascending** by
`COALESCE(vehicle_timestamp, fetched_at)`. The same effective-`start_date`
COALESCE above is used for filtering.

Chronological sort is a pipeline requirement: `compute_moving_speed` does
`df.diff()`, `project_trajectory` leaves order untouched, and `upsample_df`
iterates consecutive pairs.

### 5. `trajectory_extract.build_trip_trajectory(rows, trips_df)`

Shapes the DataFrame the rest of the pipeline consumes:

- Drops rows with null `latitude`/`longitude` (can't project them).
- Sorts by `(datetime, source_vehicle_position_id)`; dedupes exact-timestamp
  collisions keeping the latest-ingested row.
- Joins `trips_df` on `trip_id` to attach `shape_id` and fill missing
  `direction_id`.
- Parses `trip_start_datetime` from `(start_date, start_time)`, handling
  GTFS's `27:15:00`-style overnight times via `timedelta` math. Toronto is
  treated as a fixed UTC-5 offset — this is safe because the resulting
  `time_offset_seconds` is a *duration*, which is TZ-invariant.

### 6. `pipeline.process_trip_instance(...)` continues: attach `service_date`, short-circuit

After `build_trip_trajectory` runs we:
- Overwrite `df["start_date"]` with the effective one from the caller (keeps
  the DB column non-null even when the feed left it NULL).
- Attach `service_date` parsed from the YYYYMMDD string.
- Resolve `shape_id` and return empty if it's missing or unknown — we can't
  project without a shape.

### 7. `project_to_shape.project_trajectory(df, shape_line, max_orthogonal_distance_m=200)`

For each row: transform `(lat, lon) → EPSG:3857 Point`, then:
- `travel_distance_m = shape_line.project(point)` — distance along the shape
  from its start to the closest point on the shape.
- `orthogonal_distance_m = point.distance(shape_line.interpolate(travel_distance_m))`
  — how far the GPS point was from the shape itself.

Rows beyond `max_orthogonal_distance_m` (default 200 m) are dropped: they're
almost always deadheading buses, off-route detours, or bad GPS. Keeping them
would corrupt speed calculations downstream.

No numpy vectorization in v1 — shapely's per-row `project` is O(vertices),
good enough for a single service day.

### 8. `upsample.compute_moving_speed(df)`

```
moving_speed_m_s[i] = (travel_distance_m[i] - travel_distance_m[i-1])
                    / (datetime[i] - datetime[i-1]).total_seconds()
```

The leading NaN is filled with `0.0` so the upsampler has a usable value for
the first gap. Infinities (zero-time jumps) become NaN and then 0.

Speed is stored on the *arriving* row (i.e. `speed[i]` describes the motion
from `i-1 → i`). This matches what `upsample_df` expects when bridging two
real rows: it uses the *next* row's speed to interpolate distance at
boundary points.

### 9. `upsample.upsample_df(df, resolution_seconds=10)` — the critical piece

This function is carried over *verbatim* from the legacy pipeline (the only
piece that survived unchanged), because its boundary-insertion logic has
been validated by the previous project. One tiny addition: every inserted
row is tagged `observed=False`.

For each consecutive real pair `(current_row, next_row)`:

1. Let `total_delta = t_next - t_current`. Skip if ≤ 0.
2. Find the first `resolution_seconds` boundary `≥ t_current` (epoch-aligned,
   not t_current-relative — so ticks land on round seconds).
3. For each boundary `t_candidate < t_next`:
   - `partial_delta = t_candidate - t_current`
   - `dist_candidate = t_current.travel + partial_delta * t_next.speed`
   - **Nearer-midpoint rule**: if `dist_candidate < (current.travel + next.travel)/2`,
     copy identity columns (`vehicle_id`, `trip_id`, `occupancy_status`, ...)
     from `current_row`; otherwise from `next_row`. This picks whichever
     real observation the synthetic point is closer to.
   - Stamp `datetime = t_candidate`, `travel_distance_m = dist_candidate`,
     `moving_speed_m_s = t_next.speed`, `observed = False`.

Note: the source observations are **not re-appended** to the output. The
nearer-midpoint rule is what preserves their identity at each boundary —
matching the legacy pipeline's behavior.

### 10. `upsample.last_step_clean_up(df)`

Rounds `travel_distance_m` and `moving_speed_m_s` to 2 decimals; reorders
columns to the canonical output schema matching `trip_trajectories`.

### 11. `runner.run_for_date(session, service_date, ...)`

Transaction owner. Lifecycle:

```python
run = AnalyticsRun(service_date=..., status='running', config_json={...})
session.add(run); session.commit()           # 1. checkpoint start
try:
    for trip_id, start_date in list_trip_instances(...):
        df = pipeline.process_trip_instance(...)
        if df.empty: continue
        session.add_all(_df_to_orm(df, run.id))
        session.commit()                      # 2. per-trip commit
    run.status = 'ok'; run.finished_at = now(); run.rows_written = N
    session.commit()                          # 3. finalize
except Exception as exc:
    session.rollback()
    run.status = 'failed'; run.error_message = str(exc)[:8000]
    session.commit()                          # 4. record failure
    raise
```

Per-trip commit is deliberate: a crash halfway through a service day keeps
every completed trip's rows in Postgres, and the final `analytics_runs.status`
tells you whether the run is trustworthy or partial.

### 12. `csv_export.write_day_csvs(out_dir, frames_by_key)` (optional)

Only runs when `--export-csv DIR` is passed. Groups the in-memory frames by
`(route_id, service_date, direction_id)` and writes
`{route}_{date}_dir{N}.csv` — same filename shape the legacy pandas pipeline
produced, so existing downstream notebooks keep working.

---

## Output schema: `trip_trajectories`

| Column                       | Type                     | Notes                                                   |
|------------------------------|--------------------------|---------------------------------------------------------|
| `id`                         | BIGINT PK                |                                                         |
| `run_id`                     | BIGINT FK CASCADE        | → `analytics_runs.id`.                                  |
| `trip_id`                    | VARCHAR(64)              | GTFS `trip_id`.                                         |
| `start_date`                 | VARCHAR(8)               | YYYYMMDD — effective (may be synthesized).              |
| `service_date`               | DATE                     | Parsed from `start_date`.                               |
| `route_id`, `direction_id`   | VARCHAR(32), SMALLINT    | From VP, filled from `trips.txt` if missing.            |
| `shape_id`                   | VARCHAR(32)              | From `trips.txt`.                                       |
| `vehicle_id`                 | VARCHAR(64)              | Inherited per nearer-midpoint.                          |
| `datetime`                   | TIMESTAMPTZ              | Upsampled boundary time.                                |
| `time_offset_seconds`        | INTEGER                  | `datetime - trip_start_datetime`, if knowable.          |
| `travel_distance_m`          | DOUBLE PRECISION         | Meters along the shape. **Not null.**                   |
| `moving_speed_m_s`           | REAL                     | Interpolated from the bridging pair.                    |
| `observed`                   | BOOLEAN                  | Always `False` for rows this pipeline emits in v1.      |
| `occupancy_status`           | VARCHAR(32)              | Inherited per nearer-midpoint.                          |
| `source_vehicle_position_id` | BIGINT FK SET NULL       | → `vehicle_positions.id` (the "carried" row).           |

Indexes: `(trip_id, start_date, datetime)`, `(route_id, service_date, datetime)`, `(run_id)`.

---

## Configuration

From `core.config.Settings` (env-overridable):

| Setting                                | Default           | Controls                                      |
|----------------------------------------|-------------------|-----------------------------------------------|
| `gtfs_static_dir`                      | `Complete GTFS`   | Where the `.txt` files live.                  |
| `analytics_upsample_resolution_s`      | `10`              | Upsample tick in seconds.                     |
| `analytics_max_orthogonal_distance_m`  | `200.0`           | Drop GPS points farther from the shape.       |

CLI flags `--upsample-seconds`, `--max-orthogonal-distance-m` override these
per invocation.

---

## Running it

```bash
# Inside docker (matches how make analytics-run works)
docker compose run --rm \
  -v "$(pwd)/out:/out" \
  api python -m apps.analytics.main \
  --date 2026-04-22 --route 29 --export-csv /out/2026-04-22

# On the host — override DATABASE_URL to the exposed port
DATABASE_URL="postgresql+psycopg://ttc:ttc@localhost:5433/ttc_gtfsrt" \
GTFS_STATIC_DIR="./Complete GTFS" \
python -m apps.analytics.main --date 2026-04-22 --route 29
```

---

## Known limitations

- **All emitted rows are `observed=False`** in v1 — the upsample step does
  not also re-append the source rows. Querying `WHERE observed = true`
  returns nothing.
- **Per-row projection is unvectorized.** One full day × all routes takes
  a few minutes; acceptable for a nightly batch, not for per-vehicle
  real-time.
- **Fixed UTC-5 for Toronto.** DST transitions (March/November) will shift
  the effective-service-day boundary by an hour. Fine in practice because
  TTC service crosses local midnight but not the DST-switch hour.
- **No stop-time matching yet.** `stop_times.txt` is loaded but the pipeline
  doesn't align trajectories to scheduled stop arrivals — that's the next
  logical extension.
