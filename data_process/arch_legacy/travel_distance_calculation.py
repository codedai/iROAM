import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point

# ---------------------------
# 0. Paths
# ---------------------------
SHAPE_AND_BUFFER_FILE = '/home/jiahao/Documents/iroam_qt/filtered/shapes_with_buffer.csv'
TRIP_RECORDS_FILE = '/home/jiahao/Documents/iroam_qt/trip_records_with_nearest_points.csv'

# ---------------------------
# 1. Load Shape Line
# ---------------------------
shape_df = pd.read_csv(SHAPE_AND_BUFFER_FILE)
shape_df['geometry'] = shape_df['geometry'].apply(loads)
shape_line = shape_df.iloc[0].geometry  # Assuming single shape line (LineString)

# ---------------------------
# 2. Load Points on Shape
# ---------------------------
points_df = pd.read_csv(TRIP_RECORDS_FILE)  # Columns: lat, lon
points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=[Point(lon, lat) for lat, lon in zip(points_df['nearest_lat'], points_df['nearest_lon'])],
    crs="EPSG:4326"
)

# # Reproject points to match shape CRS if needed
# points_gdf = points_gdf.to_crs(shape_df.crs)

# ---------------------------
# 3. Calculate Travel Distance Along the Shape
# ---------------------------
def distance_along_shape(point, line):
    """Calculate distance along the shape from the start to the point."""
    return line.project(point)

# Calculate distances
points_gdf['travel_distance_m'] = points_gdf.geometry.apply(lambda point: distance_along_shape(point, shape_line))

# ---------------------------
# 4. Save Results
# ---------------------------
points_gdf[['lat', 'lon', 'nearest_lat', 'nearest_lon', 'travel_distance_m']].to_csv('points_with_travel_distance.csv', index=False)

# Turn the nearest point back to EPSG:3857 for plotting
points_gdf = points_gdf.to_crs("EPSG:3857")
points_gdf.to_csv('points_with_travel_distance_3857.csv', index=False)

print("✅ Travel distances calculated and saved.")
