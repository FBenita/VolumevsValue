# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 06:55:20 2026

@author: L03565094
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import box
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\Users\L03565094\Dropbox\Francisco\Papers2023\Tocayo\NNI\03 Results'
input_file = os.path.join(results_path, 'MEXICO_PANEL_WITH_EXOGENOUS_VARS.csv')
mun_shape_path = r'C:\Users\L03565094\Dropbox\Francisco\Papers2023\Tocayo\NNI\01 Data\Shapefiles\Admin_Divisions_2025\00mun_REPROJECTED.gpkg'
maps_folder = os.path.join(results_path, '01_Maps')

# --- 2. LOAD DATA ---
muns = gpd.read_file(mun_shape_path)
df = pd.read_csv(input_file)
end_year = int(df['year'].max())
target_sector = '33'

# Filter for Juarez (CVE_ENT 08, CVE_MUN 037)
juarez_poly = muns[(muns['CVE_ENT'] == '08') & (muns['CVE_MUN'] == '037')].copy()
mun_area_km2 = juarez_poly.geometry.area.iloc[0] / 1e6

# --- 3. CONVERT POINTS TO 5000m SQUARE POLYGONS ---
grid_size = 5000 
def make_square(row):
    half = grid_size / 2
    return box(row.x_coord - half, row.y_coord - half, 
               row.x_coord + half, row.y_coord + half)

df_latest = df[df['year'] == end_year].copy()
df_latest['geometry'] = df_latest.apply(make_square, axis=1)
gdf_grid_cells = gpd.GeoDataFrame(df_latest, geometry='geometry', crs=juarez_poly.crs)
juarez_grid_poly = gpd.sjoin(gdf_grid_cells, juarez_poly, predicate='intersects')

# --- 4. PLOTTING ---
fig, axes = plt.subplots(1, 3, figsize=(26, 12), facecolor='white')
bounds = juarez_poly.total_bounds

for i, ax in enumerate(axes):
    # a. USA BACKGROUND SHADE
    ax.fill_between([bounds[0]-10000, bounds[2]+10000], 
                    juarez_poly.geometry.total_bounds[3], 
                    juarez_poly.geometry.total_bounds[3] + 20000, 
                    color='#ECEFF1', alpha=0.6, zorder=0)
    
    # b. BORDER LINE
    ax.axhline(y=juarez_poly.geometry.total_bounds[3], color='black', 
               linestyle='--', linewidth=2, zorder=5)
    
    # c. LABELS
    ax.text(bounds[0]+2000, juarez_poly.geometry.total_bounds[3]+3000, 
            "UNITED STATES", fontsize=14, fontweight='bold', color='#455A64')

    # d. PLOT JUAREZ BASE
    juarez_poly.plot(ax=ax, color='white', edgecolor='black', linewidth=1, alpha=0.2, zorder=1)

    # PANEL A: ADMINISTRATIVE CONTEXT
    if i == 0:
        ax.set_title("A. Administrative Context\n(Study Area & Resolution)", fontsize=22, fontweight='bold', pad=30)
        juarez_poly.plot(ax=ax, color='#CFD8DC', edgecolor='#455A64', linewidth=2, zorder=2)
        # Stats Label
        ax.text(0.5, 0.05, f"Municipality: Cd. Juárez\nTotal Area: {mun_area_km2:.1f} km²", 
                transform=ax.transAxes, ha='center', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        # Distance Arrow
        ax.annotate('', xy=(bounds[2]-5000, juarez_poly.geometry.total_bounds[3]), 
                    xytext=(bounds[2]-5000, juarez_poly.geometry.total_bounds[3]-10000),
                    arrowprops=dict(arrowstyle='<->', color='#1E88E5', lw=3))
        ax.text(bounds[2]-4000, juarez_poly.geometry.total_bounds[3]-5000, 
                "Direct Border\nProximity", color='#1E88E5', fontweight='bold', fontsize=12)
        
        # Scale Bar & North Arrow
        ax.add_artist(ScaleBar(1, location='lower left', font_properties={'size': 14}))
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88),
                    arrowprops=dict(facecolor='black', width=4, headwidth=12),
                    ha='center', va='center', fontsize=20, xycoords='axes fraction')

    # PANEL B: ESTABLISHMENT DENSITY (Model 1 Dependent Variable)
    if i == 1:
        ax.set_title("B. Spatial Proxy Weights\n(Establishment Density $n_{i,t}$)", fontsize=22, fontweight='bold', pad=30)
        active_cells = juarez_grid_poly[juarez_grid_poly[f'count_{target_sector}'] > 0]
        sc2 = active_cells.plot(column=f'count_{target_sector}', ax=ax, cmap='Reds', 
                                edgecolor='black', linewidth=0.4, legend=True, 
                                legend_kwds={'shrink': 0.5}, zorder=3)
        cax2 = fig.get_axes()[-2]
        cax2.set_ylabel("Establishments (Count)", fontsize=16, fontweight='bold', labelpad=15)
        cax2.tick_params(labelsize=12)

    # PANEL C: CAPITAL INTENSITY (Model 2 Dependent Variable)
    if i == 2:
        # --- DEFINING VARIABLES ---
        labor_col = f'labor_total_{target_sector}'      # L
        capital_col = f'machinery_{target_sector}'      # K
        
        # --- CALCULATE LOG CAPITAL INTENSITY (k = K/L) ---
        juarez_grid_poly['log_k_ratio'] = np.where(
            juarez_grid_poly[labor_col] > 0,
            np.log1p(juarez_grid_poly[capital_col] / juarez_grid_poly[labor_col]),
            0
        )
        
        # TITLE UPDATE: Using TeX notation to match the paper
        ax.set_title(r"C. Variable of Interest" + "\n" + r"(Log Capital Intensity $\ln(k_{i,t})$)", 
                     fontsize=22, fontweight='bold', pad=30)
        
        # Filter only active cells for clearer plotting
        plot_data = juarez_grid_poly[juarez_grid_poly['log_k_ratio'] > 0]
        
        sc3 = plot_data.plot(column='log_k_ratio', ax=ax, cmap='plasma', 
                                   edgecolor='black', linewidth=0.4, legend=True, 
                                   legend_kwds={'shrink': 0.5}, zorder=3)
        
        cax3 = fig.get_axes()[-1]
        cax3.set_ylabel(r"Log Capital per Worker ($\ln(k)$)", fontsize=16, fontweight='bold', labelpad=15)
        cax3.tick_params(labelsize=12)

    # Zoom to show some of the US side but focus on Juarez
    ax.set_xlim([bounds[0]-2000, bounds[2]+2000])
    ax.set_ylim([bounds[1]-5000, juarez_poly.geometry.total_bounds[3]+10000])
    ax.axis('off')
    
plt.subplots_adjust(top=0.85, wspace=0.15)
plt.savefig(os.path.join(maps_folder, 'Figure_1_Final_Methodology_Juarez.png'), dpi=300, bbox_inches='tight')
print(f"[-] Final High-Precision Figure saved to: {maps_folder}")
plt.show()