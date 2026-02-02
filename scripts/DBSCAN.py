import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\'
grid_path = os.path.join(results_path, 'mexico_5km_grid_master.gpkg')

years = [2010, 2015, 2019, 2025]

# DBSCAN Parameters
EPSILON = 1500  # 1.5 km radius
MIN_SAMPLES = 10 # Minimum 10 factories to form a cluster

# --- 2. LOAD GRID ---
print("Loading Grid...")
master_grid = gpd.read_file(grid_path)
# We need a DataFrame to store results. Start with just grid_id.
cluster_panel = pd.DataFrame({'grid_id': master_grid['grid_id']})

# --- 3. RUN CLUSTERING LOOP ---
for year in years:
    print(f"--- Running DBSCAN for {year} ---")
    
    # Load the points for this year
    points_path = os.path.join(results_path, f'denue_{year}_manufacturing.gpkg')
    gdf = gpd.read_file(points_path)
    
    # Get coordinates for DBSCAN (Must be X, Y in meters)
    coords = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    
    # RUN DBSCAN
    print(f"  Clustering {len(gdf)} points...")
    # n_jobs=-1 uses all processor cores for speed
    db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric='euclidean', n_jobs=-1).fit(coords)
    
    # Assign cluster labels to points (-1 means Noise/Not in Cluster)
    gdf['cluster_lbl'] = db.labels_
    
    # Filter: Keep only points that are actually in a cluster (label != -1)
    clustered_points = gdf[gdf['cluster_lbl'] != -1]
    print(f"  Found {len(clustered_points)} clustered points out of {len(gdf)}.")

    if len(clustered_points) > 0:
        # SPATIAL JOIN: Map clustered points to Grid Cells
        joined = gpd.sjoin(clustered_points, master_grid[['grid_id', 'geometry']], how='inner', predicate='intersects')
        
        # AGGREGATE TO GRID LEVEL
        # 1. Binary: Does this cell contain ANY clustered points?
        # 2. Intensity: How many clustered points are in this cell?
        grid_stats = joined.groupby('grid_id').agg(
            clustered_count=('cluster_lbl', 'count')
        ).reset_index()
        
        # Rename columns for the year
        grid_stats.rename(columns={'clustered_count': f'cluster_n_{year}'}, inplace=True)
        grid_stats[f'is_cluster_{year}'] = 1 # Binary dummy
        
        # Merge into our main panel
        cluster_panel = cluster_panel.merge(grid_stats, on='grid_id', how='left')
    else:
        print(f"  Warning: No clusters found for {year} with current parameters.")

    # Fill NaNs with 0 (Cells not in a cluster)
    cols_to_fix = [f'cluster_n_{year}', f'is_cluster_{year}']
    for col in cols_to_fix:
        if col in cluster_panel.columns:
            cluster_panel[col] = cluster_panel[col].fillna(0).astype(int)
        else:
            cluster_panel[col] = 0

# --- 4. SAVE CLUSTER DATASET ---
out_csv = os.path.join(results_path, 'mexico_dbscan_clusters.csv')
cluster_panel.to_csv(out_csv, index=False)
print(f"SUCCESS: Clustering data saved to {out_csv}")












########### Merge industyr counts



import pandas as pd
import os

# --- 1. SETUP PATHS ---
results_path = r'C:\'

# --- 2. LOAD FILES ---
print("Loading Sectoral Panel (Y)...")
df_y = pd.read_csv(os.path.join(results_path, 'mexico_panel_sectoral.csv'))

print("Loading Cluster Panel (X)...")
df_x = pd.read_csv(os.path.join(results_path, 'mexico_dbscan_clusters.csv'))

# --- 3. MERGE ---
# We merge on 'grid_id'. using 'left' ensures we keep all cells from the main panel.
print("Merging panels...")
master_panel = df_y.merge(df_x, on='grid_id', how='left')

# Fill any missing cluster data with 0 (just in case)
cols_to_fill = [c for c in master_panel.columns if 'cluster' in c]
master_panel[cols_to_fill] = master_panel[cols_to_fill].fillna(0).astype(int)

# --- 4. CALCULATE GROWTH VARIABLES ---
master_panel['cluster_growth_10_25'] = master_panel['cluster_n_2025'] - master_panel['cluster_n_2010']

# Example: Change in Sector 33 (Machinery) Counts
if 'count_33_2010' in master_panel.columns:
    master_panel['growth_33_10_25'] = master_panel['count_33_2025'] - master_panel['count_33_2010']

# --- 5. SAVE FINAL DATABASE ---
output_file = os.path.join(results_path, 'FINAL_MEXICO_MANUFACTURING_PANEL.csv')
print(f"Saving Final Master Panel to: {output_file}")
master_panel.to_csv(output_file, index=False)


print(f"DONE! Your database has {len(master_panel)} rows and {len(master_panel.columns)} variables.")
