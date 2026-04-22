import os
import logging
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
from tqdm import tqdm
import numpy as np

############################################################
# 0. Logging Setup
############################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

############################################################
# 1. Configuration
############################################################
CONFIG = {
    # File/Folder Paths
    'SOURCE_FOLDER': '/home/jiahao/Documents/iroam_qt/transsee_spider/data/ttc',
    'DEST_FOLDER': '/home/jiahao/Documents/iroam_qt/filtered',
    'SHAPE_AND_BUFFER_FILE': '/home/jiahao/Documents/iroam_qt/filtered/shapes_with_buffer.csv',

    # File patterns
    'TRIP_FILE_SUFFIX': '_trippath.csv',

    # Column names in your CSV
    'TIME_COL': 'time_val',    # the column that may have bracketed times
    'DIST_COL': 'distance',    # the distance column (e.g. '52m')
    'SPEED_COL': 'speed',      # the speed column (e.g. '6.2km/h')

    # Chunk reading (None to read full CSV at once)
    'CHUNK_SIZE': None
}

# Your CSV columns in order:
#  route_date, trip_id, fleet, route, direction, destination, time_val, lat, lon, distance,
#  speed, movement, station_name, time_offset, relative_position, scheduled_time, load, store_path

############################################################
# 2. Load Shape and Buffer
############################################################
logging.info("Loading shape and buffer data...")

shape_df = pd.read_csv(CONFIG['SHAPE_AND_BUFFER_FILE'])
shape_df['geometry'] = shape_df['geometry'].apply(loads)
shape_df['buffer'] = shape_df['buffer'].apply(loads)

shapes_gdf = gpd.GeoDataFrame(shape_df, geometry='geometry', crs="EPSG:3857")
buffers_gdf = gpd.GeoDataFrame(shape_df, geometry='buffer', crs="EPSG:3857")

logging.info("Shape and buffer data loaded successfully.")

############################################################
# 3. Time Column Reformat Function
############################################################
def reformat_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Some rows may have time_val like: ['5:23:15AM','5:23:45AM'].
    - This function keeps only the last time in that list.
    - Then sets speed/distance to 0 for that row.
    """
    time_col = CONFIG['TIME_COL']
    dist_col = CONFIG['DIST_COL']
    speed_col = CONFIG['SPEED_COL']

    # Only run if columns exist
    for col_name in [time_col, dist_col, speed_col]:
        if col_name not in df.columns:
            logging.warning(f"Column {col_name} not found in DataFrame. Skipping reformat.")
            return df

    for i in range(len(df)):
        val = df.at[i, time_col]
        if isinstance(val, str) and '[' in val and ']' in val:
            # Example: ['5:23:15AM','5:23:45AM']
            times_str = val.strip('[]').replace("'", "")  # remove brackets & single quotes
            splitted = [t.strip() for t in times_str.split(',')]
            if splitted:
                last_time = splitted[-1]
                df.at[i, time_col] = last_time
                df.at[i, speed_col] = '0km/h'
                df.at[i, dist_col] = '0m'

    return df

############################################################
# 4. Reading & Writing Functions
############################################################
def read_trip_csv(path: str) -> gpd.GeoDataFrame:
    """
    Read a trip CSV into a GeoDataFrame.
    If CHUNK_SIZE is set, read in chunks; otherwise read entire CSV at once.
    """
    chunk_size = CONFIG['CHUNK_SIZE']

    def process_df(df_chunk: pd.DataFrame) -> gpd.GeoDataFrame:
        # Reformat time column
        df_chunk = reformat_time_column(df_chunk)

        # Create geometry from lat/lon
        df_chunk['geometry'] = [
            Point(lon, lat) for lat, lon in zip(df_chunk['lat'], df_chunk['lon'])
        ]
        gdf_chunk = gpd.GeoDataFrame(
            df_chunk, geometry='geometry', crs="EPSG:4326"
        ).to_crs(buffers_gdf.crs)
        return gdf_chunk

    if chunk_size is None:
        # Single-pass read
        df = pd.read_csv(path)
        return process_df(df)
    else:
        # Multi-chunk read
        chunks = []
        for df_chunk in pd.read_csv(path, chunksize=chunk_size):
            chunks.append(process_df(df_chunk))
        return pd.concat(chunks, ignore_index=True)

def write_results(gdf: gpd.GeoDataFrame, dest_path: str, drop_cols: Optional[list] = None) -> None:
    """Write the final GeoDataFrame to CSV, dropping geometry columns."""
    if drop_cols is None:
        drop_cols = []
    valid_cols = [c for c in gdf.columns if c not in drop_cols]
    gdf[valid_cols].to_csv(dest_path, index=False)

############################################################
# 5. Geometry Helpers
############################################################
def nearest_point_on_shape(point: Point, shape_line) -> (Point, float):
    """
    Return the nearest point on the given shape line and the distance between them.
    """
    nearest_pt = shape_line.interpolate(shape_line.project(point))
    dist = point.distance(nearest_pt)
    return nearest_pt, dist

def distance_along_shape(point: Point, line) -> float:
    """
    Return the distance along the shape (line) from its start to the nearest projection of 'point'.
    """
    return line.project(point)

############################################################
# 6. Main Processing Function
############################################################
def process_folder(folder: str) -> pd.DataFrame:
    """
    Traverse the source folder (route/date/trip structure).
    Perform buffer filtering, nearest shape points, and travel distance.
    Returns a combined DataFrame of all processed results.
    """
    logging.info("Starting processing pipeline...")
    all_results = []  # keep track of final data from each trip if you want a single aggregated CSV

    if not os.path.isdir(folder):
        logging.error(f"Source folder '{folder}' not found or not a directory.")
        return pd.DataFrame()

    routes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for route_folder in tqdm(routes, desc="Routes"):
        if route_folder.endswith('backup'):
            continue
        route_path = os.path.join(folder, route_folder)

        # For each date subfolder
        dates = [d for d in os.listdir(route_path) if os.path.isdir(os.path.join(route_path, d))]
        for date_folder in tqdm(dates, desc="Dates"):
            date_path = os.path.join(route_path, date_folder)

            trip_dirs = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
            for trip_folder in trip_dirs:
                trip_path_folder = os.path.join(date_path, trip_folder)

                # Look for CSV with suffix (e.g., '_trippath.csv')
                trip_csvs = [
                    f for f in os.listdir(trip_path_folder)
                    if f.endswith(CONFIG['TRIP_FILE_SUFFIX'])
                ]
                for csv_file in trip_csvs:
                    csv_path = os.path.join(trip_path_folder, csv_file)
                    # logging.info(f"Processing file: {csv_path}")

                    # 1) Load trip data
                    try:
                        trip_gdf = read_trip_csv(csv_path)
                    except Exception as e:
                        logging.error(f"Failed to read CSV {csv_path}: {e}")
                        continue

                    # 2) Filter with buffer
                    try:
                        filtered_gdf = gpd.sjoin(trip_gdf, buffers_gdf, predicate='within')
                    except Exception as e:
                        logging.error(f"Spatial join failed on {csv_path}: {e}")
                        continue

                    # 3) Find nearest shape line points
                    nearest_points = []
                    distances = []
                    for pt in filtered_gdf.geometry:
                        min_dist = np.inf
                        best_pt = None
                        for shape_geom in shapes_gdf.geometry:
                            npt, dist = nearest_point_on_shape(pt, shape_geom)
                            if dist < min_dist:
                                min_dist = dist
                                best_pt = npt
                        nearest_points.append(best_pt)
                        distances.append(min_dist)

                    filtered_gdf['nearest_shape_point'] = nearest_points
                    filtered_gdf['shortest_distance_m'] = distances

                    # 4) Travel distance along a single shape line
                    # You might pick the correct line for that route; here we just use index=0
                    shape_line = shapes_gdf.iloc[0].geometry
                    travel_dists = [
                        distance_along_shape(pt, shape_line) for pt in nearest_points
                    ]
                    filtered_gdf['travel_distance_m'] = travel_dists

                    # 5) Save final results
                    out_dir = os.path.join(
                        CONFIG['DEST_FOLDER'],
                        route_folder, date_folder, trip_folder
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    final_file = os.path.join(out_dir, f"{trip_folder}_final.csv")
                    write_results(
                        filtered_gdf,
                        final_file,
                        drop_cols=['geometry', 'geometry_right', 'nearest_shape_point']
                    )
                    # logging.info(f"Saved final CSV: {final_file}")

                    # # 6) Gather data for a combined CSV
                    # final_df = filtered_gdf.drop(
                    #     columns=['geometry','geometry_right','nearest_shape_point'],
                    #     errors='ignore'
                    # )
                    # # Optionally add route/date/trip context
                    # final_df['route'] = route_folder
                    # final_df['date'] = date_folder
                    # final_df['trip_folder'] = trip_folder
                    # all_results.append(final_df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        logging.info("No data processed or no matching CSV files found.")
        return pd.DataFrame()



############################################################
# 7. Execute
############################################################
if __name__ == "__main__":
    combined_results = process_folder(CONFIG['SOURCE_FOLDER'])
    if not combined_results.empty:
        # Optionally write all data to one big CSV
        all_csv_path = os.path.join(CONFIG['DEST_FOLDER'], 'all_trips_combined.csv')
        combined_results.to_csv(all_csv_path, index=False)
        logging.info(f"All trips combined into: {all_csv_path}")

    logging.info("Processing pipeline complete.")
