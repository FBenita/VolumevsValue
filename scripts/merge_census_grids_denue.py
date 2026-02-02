# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 00:48:36 2026

@author: L03565094
"""

import pandas as pd
import geopandas as gpd
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\Users\L03565094\Dropbox\Francisco\Papers2023\Tocayo\NNI\03 Results'

print("--- LOADING COMPONENTS ---")

# A. THE SPINE (Spatial Keys)
keys_path = os.path.join(results_path, 'mexico_5km_grid_joined_mun_2010_2015_2020_2025.gpkg')
# Use ignore_geometry=True for speed, we just need the IDs
gdf_keys = gpd.read_file(keys_path, ignore_geometry=True)
keys_df = gdf_keys[['grid_id', 'CVEGEO_2010', 'CVEGEO_2015', 'CVEGEO_2020', 'CVEGEO_2025']].copy()

# CLEAN KEYS: Force 5-digit strings (01001)
for col in ['CVEGEO_2010', 'CVEGEO_2015', 'CVEGEO_2020', 'CVEGEO_2025']:
    keys_df[col] = keys_df[col].astype(str).str.split('.').str[0].str.zfill(5)

# B. THE BODY (Factory Counts)
panel_path = os.path.join(results_path, 'FINAL_MEXICO_MANUFACTURING_PANEL.csv')
panel_df = pd.read_csv(panel_path)

# C. THE SOUL (Economic Census)
census_path = os.path.join(results_path, 'mexico_manufacturing_panel_analytical_panel.csv')
census_raw = pd.read_csv(census_path)

# --- PREPARE CENSUS DATA ---
census_raw['sector_group'] = census_raw['rama'].astype(str).str.slice(0, 2)

# *** THE CRITICAL FIX IS HERE ***
# Mapping: {Census Year : Target Analysis Year}
year_map = {
    2008: 2010, 
    2013: 2015, 
    2018: 2019, 
    2023: 2025
}
census_raw['grid_year'] = census_raw['year'].map(year_map)
census_raw = census_raw.dropna(subset=['grid_year'])

# Clean Census Key
census_raw['KEY_LINK'] = census_raw['cve_mun'].astype(str).str.split('.').str[0].str.zfill(5)

# Aggregate & Pivot
econ_vars = ['value_added', 'labor_total', 'wages_total', 'machinery', 'computers']
census_agg = census_raw.groupby(['grid_year', 'KEY_LINK', 'sector_group'])[econ_vars].sum().reset_index()

census_pivot = census_agg.pivot(index=['grid_year', 'KEY_LINK'], columns='sector_group', values=econ_vars)
census_pivot.columns = [f'{col[0]}_{col[1]}' for col in census_pivot.columns]
census_pivot = census_pivot.reset_index()

# --- 3. THE MASTER MERGE ---
print("\n--- STARTING MERGE ---")
final_dfs = []
sectors = ['31', '32', '33']

# Map Analysis Year to the specific Spatial Column
year_to_key_col = {
    2010: 'CVEGEO_2010',
    2015: 'CVEGEO_2015',
    2019: 'CVEGEO_2020', # 2019 Data uses 2020 Map
    2025: 'CVEGEO_2025'
}

for year, key_col in year_to_key_col.items():
    print(f"Processing Year {year} (Using Map: {key_col})...")
    
    # Base: Grid Panel
    cols_year = ['grid_id'] + [c for c in panel_df.columns if str(year) in c]
    df_year = panel_df[cols_year].copy()
    
    # Attach Spatial ID
    df_year = df_year.merge(keys_df[['grid_id', key_col]], on='grid_id', how='left')
    
    # Attach Census Data
    census_year = census_pivot[census_pivot['grid_year'] == year].copy()
    merged = df_year.merge(census_year, left_on=key_col, right_on='KEY_LINK', how='left')
    
    # Check success
    matches = merged['value_added_31'].notna().sum()
    print(f"  > Grid Cells with Economic Data attached: {matches}")
    
    # Dasymetric Distribution
    for sector in sectors:
        count_col = f'count_{sector}_{year}'
        if count_col in merged.columns:
            # 1. Muni Total
            muni_total = merged.groupby(key_col)[count_col].transform('sum')
            # 2. Weight
            weights = merged[count_col] / muni_total
            weights = weights.fillna(0)
            # 3. Distribute
            for var in econ_vars:
                source = f'{var}_{sector}'
                target = f'{var}_{sector}_{year}'
                if source in merged.columns:
                    merged[target] = merged[source] * weights
                else:
                    merged[target] = 0

    # Clean up
    cols_to_keep = ['grid_id'] + [c for c in merged.columns if str(year) in c and c not in ['KEY_LINK', key_col]]
    final_dfs.append(merged[cols_to_keep])

# --- 4. SAVE ---
print("\n--- SAVING FINAL DATABASE ---")
final_master = pd.DataFrame({'grid_id': panel_df['grid_id'].unique()})

for df_year in final_dfs:
    final_master = final_master.merge(df_year, on='grid_id', how='left')

output_file = os.path.join(results_path, 'FINAL_FULL_SPATIAL_ECONOMIC_PANEL_READY_V2.csv')
final_master.to_csv(output_file, index=False)
print(f"DONE! File saved to: {output_file}")