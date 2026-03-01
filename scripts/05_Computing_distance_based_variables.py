import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\03 Results'

# We need the Grid GPKG to get the exact CRS (Projection)
grid_path = os.path.join(results_path, 'mexico_5km_grid_master.gpkg')
# We need the Panel CSV to add the columns to
panel_path = os.path.join(results_path, 'MEXICO_SPATIAL_PANEL_LONG_WITH_COORDS.csv')

print("Loading Data...")
gdf_grid = gpd.read_file(grid_path)
df_panel = pd.read_csv(panel_path)

# --- 2. DEFINE EXOGENOUS ANCHORS (Lat/Lon) ---
# We define these manually. 
anchors_data = {
    'name': [
        'Border_Laredo', 'Border_Juarez', 'Border_Tijuana', 
        'Market_CDMX', 
        'Port_Manzanillo', 'Port_Veracruz'
    ],
    'type': [
        'border', 'border', 'border', 
        'market', 
        'port', 'port'
    ],
    'lat': [
        27.5038, 31.7333, 32.5149,  # Borders
        19.4326,                    # CDMX
        19.0522, 19.1738            # Ports
    ],
    'lon': [
        -99.5073, -106.4825, -117.0382, 
        -99.1332, 
        -104.3159, -96.1342
    ]
}

# Create a GeoDataFrame for anchors
gdf_anchors = gpd.GeoDataFrame(
    anchors_data,
    geometry=gpd.points_from_xy(anchors_data['lon'], anchors_data['lat']),
    crs="EPSG:4326" # Standard Lat/Lon
)

# --- 3. REPROJECT TO MATCH THE GRID ---
# We convert the anchors to Meters (LCC) to match the grid.
print(f"Projecting Anchors to match Grid CRS: {gdf_grid.crs}")
gdf_anchors = gdf_anchors.to_crs(gdf_grid.crs)

# --- 4. CALCULATE DISTANCES ---
print("Calculating Exogenous Variables...")

# We calculate distance from every Grid Centroid -> Nearest Anchor of that type
# Get Grid Centroids
centroids = gdf_grid.geometry.centroid

# A. Distance to USA (Min distance to ANY border crossing)
borders = gdf_anchors[gdf_anchors['type'] == 'border'].geometry.unary_union
# Calculate distance (in meters), divide by 1000 for km
gdf_grid['dist_usa_km'] = centroids.distance(borders) / 1000

# B. Distance to CDMX (Distance to specific point)
cdmx = gdf_anchors[gdf_anchors['name'] == 'Market_CDMX'].geometry.iloc[0]
gdf_grid['dist_cdmx_km'] = centroids.distance(cdmx) / 1000

# C. Distance to Port (Min distance to ANY port)
ports = gdf_anchors[gdf_anchors['type'] == 'port'].geometry.unary_union
gdf_grid['dist_port_km'] = centroids.distance(ports) / 1000

# --- 5. MERGE BACK TO PANEL ---
dist_features = gdf_grid[['grid_id', 'dist_usa_km', 'dist_cdmx_km', 'dist_port_km']]

# Merge
print("Merging with Panel Data...")
df_final = df_panel.merge(dist_features, on='grid_id', how='left')

# Save
output_file = os.path.join(results_path, 'MEXICO_PANEL_WITH_EXOGENOUS_VARS.csv')
df_final.to_csv(output_file, index=False)

print("\n" + "="*50)
print("SUCCESS! Variable Construction Complete.")
print(f"Saved to: {output_file}")
print("="*50)
print("New Variables for Regression/Neural Network:")
print("1. dist_usa_km  (Proxy for Nearshoring/USMCA)")
print("2. dist_cdmx_km (Proxy for Domestic Market Potential)")
print("3. dist_port_km (Proxy for International Logistics)")

