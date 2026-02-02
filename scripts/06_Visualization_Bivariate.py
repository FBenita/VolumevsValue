import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\'
input_file = os.path.join(results_path, 'MEXICO_PANEL_WITH_EXOGENOUS_VARS.csv')
maps_folder = os.path.join(results_path, '01_Maps')

if not os.path.exists(maps_folder):
    os.makedirs(maps_folder)

df = pd.read_csv(input_file)
target_sector = '33'
end_year = int(df['year'].max())

# --- 2. PREP DATA ---
print(f"Generating Map 4: Bivariate for year {end_year}...")
df_biv = df[df['year'] == end_year].copy()

# Notation matches Methodology & Tables:
# n = Establishment Density (Count)
# k = Capital Intensity (Machinery/Labor)
df_biv['n'] = df_biv[f'count_{target_sector}']
k_val = df_biv[f'machinery_{target_sector}']
l_val = df_biv[f'labor_total_{target_sector}'].replace(0, np.nan)
df_biv['k'] = np.log((k_val/l_val) + 1)

# Drop NaNs and zeros (must have active firms to be plotted)
df_biv = df_biv[['grid_id', 'x_coord', 'y_coord', 'n', 'k']].dropna()
df_biv = df_biv[df_biv['n'] > 0] 

if df_biv.empty:
    print("   [!] Error: No valid data points found.")
else:
    # 3x3 Binning
    df_biv['bin_n'] = pd.qcut(df_biv['n'].rank(method='first'), 3, labels=[0, 1, 2])
    df_biv['bin_k'] = pd.qcut(df_biv['k'].rank(method='first'), 3, labels=[0, 1, 2])

    # Bivariate Palette (Pink-Blue-Purple)
    bivar_colors = ["#e8e8e8", "#b0d5df", "#64acbe", 
                    "#e4acac", "#ad9ea5", "#627f8c", 
                    "#c85a5a", "#985356", "#574249"]

    df_biv['color'] = df_biv.apply(lambda row: bivar_colors[int(row['bin_n'])*3 + int(row['bin_k'])], axis=1)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Base Map (Light Gray Context)
    df_geo_all = df[df['year'] == end_year][['x_coord', 'y_coord']].drop_duplicates()
    ax.scatter(df_geo_all['x_coord'], df_geo_all['y_coord'], color='#E0E0E0', s=1, alpha=0.4)

    # Data Points
    ax.scatter(df_biv['x_coord'], df_biv['y_coord'], color=df_biv['color'], s=15, zorder=2)

    # Title with Context

    ax.axis('off')

    # --- LEGEND WITH PRECISE METHODOLOGY NOTATION ---
    ax_leg = fig.add_axes([0.15, 0.15, 0.15, 0.15]) 
    
    for d in range(3):
        for q in range(3):
            rect = plt.Rectangle((d, q), 1, 1, facecolor=bivar_colors[d * 3 + q], edgecolor='white', lw=0.5)
            ax_leg.add_patch(rect)
            
    ax_leg.set_xlim(0, 3)
    ax_leg.set_ylim(0, 3)
    
    # X-Axis Label: "Est. Density (n)" matches Methods
    ax_leg.set_xlabel(r'Est. Density ($n$) $\rightarrow$', fontsize=10, fontweight='bold')
    
    # Y-Axis Label: "Cap. Intensity (k)" matches Table 4
    ax_leg.set_ylabel(r'Cap. Intensity ($k$) $\rightarrow$', fontsize=10, fontweight='bold')
    
    # Ticks
    ax_leg.set_xticks([0.5, 1.5, 2.5])
    ax_leg.set_xticklabels(['Low', 'Mid', 'High'], fontsize=8)
    
    ax_leg.set_yticks([0.5, 1.5, 2.5])
    ax_leg.set_yticklabels(['Low', 'Mid', 'High'], fontsize=8, rotation=90, va='center')
    
    for spine in ax_leg.spines.values():
        spine.set_visible(False)
    ax_leg.tick_params(length=0)

    # Save
    save_path = os.path.join(maps_folder, 'Map_4_Bivariate_Final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[-] Final Map 4 saved to: {save_path}")

    plt.show()
