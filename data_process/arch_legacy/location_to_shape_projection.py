import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
import numpy as np


# ---------------------------
# 0. Paths
# ---------------------------
SHAPE_AND_BUFFER_FILE = '/home/jiahao/Documents/iroam_qt/filtered/shapes_with_buffer.csv'
TRIP_RECORDS_FILE = '/home/jiahao/Documents/iroam_qt/filtered/2023-01-01/765778622/765778622_filtered.csv'


# ---------------------------
# 1. Load Shape and Trip Points
# ---------------------------
# Load shape file (with WKT geometry for shape line)
shape_df = pd.read_csv(SHAPE_AND_BUFFER_FILE)
shape_df['geometry'] = shape_df['geometry'].apply(loads)
shapes_gdf = gpd.GeoDataFrame(shape_df, geometry='geometry', crs="EPSG:3857")

# Load trip records
records_df = pd.read_csv(TRIP_RECORDS_FILE)
records_gdf = gpd.GeoDataFrame(
    records_df,
    geometry=[Point(lon, lat) for lat, lon in zip(records_df['lat'], records_df['lon'])],
    crs="EPSG:4326"
)

# Reproject trip points to match shape CRS
records_gdf = records_gdf.to_crs(shapes_gdf.crs)

# ---------------------------
# 2. Calculate Shortest Distance and Nearest Point
# ---------------------------
def nearest_point_on_shape(point, shape):
    """Calculate the nearest point on the shape line and distance."""
    nearest_point = shape.interpolate(shape.project(point))
    distance = point.distance(nearest_point)
    return nearest_point, distance

# Find nearest distance and point for each record
nearest_points = []
distances = []

for point in records_gdf.geometry:
    shortest_distance = np.inf
    nearest_shape_point = None

    for shape in shapes_gdf.geometry:
        shape_point, dist = nearest_point_on_shape(point, shape)
        if dist < shortest_distance:
            shortest_distance = dist
            nearest_shape_point = shape_point

    nearest_points.append(nearest_shape_point)
    distances.append(shortest_distance)

# Add results to GeoDataFrame
records_gdf['nearest_shape_point'] = nearest_points
records_gdf['shortest_distance_m'] = distances

# ---------------------------
# 3. Save Results
# ---------------------------
# Save as CSV with latitude/longitude of nearest shape point
records_gdf['nearest_lat'] = [p.y for p in nearest_points]
records_gdf['nearest_lon'] = [p.x for p in nearest_points]

records_gdf.drop(columns='geometry').to_csv('trip_records_with_nearest_points.csv', index=False)

# Save as GeoJSON for visualization
records_gdf.to_file('trip_records_with_nearest_points.geojson', driver='GeoJSON')

print("✅ Shortest distances and nearest points saved successfully.")
