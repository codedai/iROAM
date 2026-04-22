import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os
from shapely.wkt import loads
from shapely.geometry import Point
from tqdm import tqdm

# ---------------------------
# 1. Paths
# ---------------------------
SOURCE_FOLDER_29 = '/home/jiahao/Documents/iroam_qt/transsee_spider/data/ttc/29'
SOURCE_FOLDER_929 = '/home/jiahao/Documents/iroam_qt/transsee_spider/data/ttc/929'
DEST_FOLDER = '/home/jiahao/Documents/iroam_qt/filtered'
SHAPE_AND_BUFFER_FILE = '/home/jiahao/Documents/iroam_qt/filtered/shapes_with_buffer.csv'

# ---------------------------
# 2. Load Shape and Buffer
# ---------------------------
shape_df = pd.read_csv(SHAPE_AND_BUFFER_FILE)
shape_df['geometry'] = shape_df['geometry'].apply(loads)
shape_df['buffer'] = shape_df['buffer'].apply(loads)

shapes_gdf = gpd.GeoDataFrame(shape_df, geometry='geometry', crs="EPSG:3857")
buffers_gdf = gpd.GeoDataFrame(shape_df, geometry='buffer', crs="EPSG:3857")

# ---------------------------
# 3. Helper Functions
# ---------------------------
def load_trip_csv(trip_path):
    """Load a trip CSV into a GeoDataFrame."""
    df = pd.read_csv(trip_path)
    df['geometry'] = [Point(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
    trip_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326").to_crs(buffers_gdf.crs)
    return trip_gdf

def save_filtered_trip(filtered_gdf, output_folder, trip_file):
    """Save original and filtered trips."""
    os.makedirs(output_folder, exist_ok=True)
    # trip_gdf.to_csv(os.path.join(output_folder, f"{trip_file}_original.csv"), index=False)
    filtered_gdf.drop(columns=['index_right', 'geometry', 'geometry_right'], errors='ignore').to_csv(
        os.path.join(output_folder, f"{trip_file}_filtered.csv"), index=False
    )

def plot_trip(shape_gdf, buffer_gdf, trip_gdf, filtered_gdf, output_folder, trip_file):
    """Plot before and after filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Before Filtering
    shape_gdf.plot(ax=axes[0], color='red', linewidth=1, label='Shape')
    buffer_gdf.plot(ax=axes[0], color='lightblue', alpha=0.5, label='Buffer')
    trip_gdf.plot(ax=axes[0], color='green', markersize=3, label='Trip Points')
    ctx.add_basemap(axes[0], source=ctx.providers.OpenStreetMap.Mapnik)
    axes[0].set_title(f"Before Filtering: {trip_file}")
    # axes[0].legend()

    # After Filtering
    shape_gdf.plot(ax=axes[1], color='red', linewidth=1, label='Shape')
    buffer_gdf.plot(ax=axes[1], color='lightblue', alpha=0.5, label='Buffer')
    filtered_gdf.plot(ax=axes[1], color='green', markersize=3, label='Filtered Points')
    ctx.add_basemap(axes[1], source=ctx.providers.OpenStreetMap.Mapnik)
    axes[1].set_title(f"After Filtering: {trip_file}")

    plt.savefig(os.path.join(output_folder, f"{trip_file}_comparison.png"), dpi=300)
    plt.close()

def process_folder(source_folder):
    """Process all trip files in a source folder."""
    for date_folder in tqdm(os.listdir(source_folder), desc="Dates"):
        date_path = os.path.join(source_folder, date_folder)
        print(date_path)
        if not os.path.isdir(date_path):
            continue

        for trip_folder in os.listdir(date_path):
            if trip_folder.endswith('csv'):
                continue
            for trip_file in os.listdir(os.path.join(date_path, trip_folder)):
                if trip_file.endswith('_trippath.csv'):
                    trip_path = os.path.join(os.path.join(date_path, trip_folder), trip_file)
                    trip_gdf = load_trip_csv(trip_path)

                    # Spatial join to filter points within buffers
                    filtered_gdf = gpd.sjoin(trip_gdf, buffers_gdf, predicate="within")

                    # Save results
                    output_folder = os.path.join(DEST_FOLDER, date_folder, trip_folder)
                    save_filtered_trip(filtered_gdf, output_folder, trip_folder)

                    # # Plot results
                    # plot_trip(shapes_gdf, buffers_gdf, trip_gdf, filtered_gdf, output_folder, trip_folder)

                    print(f"Processed and saved {trip_file} in {output_folder}")

# ---------------------------
# 4. Process Both Folders
# ---------------------------
print("Processing TTC 29...")
process_folder(SOURCE_FOLDER_29)

print("Processing TTC 929...")
process_folder(SOURCE_FOLDER_929)

print("All trips processed successfully.")
