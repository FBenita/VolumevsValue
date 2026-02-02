# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:58:30 2026

@author: L03565094
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# --- 1. CONFIGURATION ---
results_path = r'C:\Users\L03565094\Dropbox\Francisco\Papers2023\Tocayo\NNI\03 Results'
file_path = os.path.join(results_path, 'MEXICO_PANEL_WITH_EXOGENOUS_VARS.csv')
output_folder = os.path.join(results_path, '00_Final_Paper_Figures')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Loading Data...")
df = pd.read_csv(file_path)

# --- 2. DATA PREP (Standardization) ---
print("Preparing Variables (Standardizing to 100km units)...")

# 1. Create Trend
df['trend'] = df['year'] - df['year'].min()

# 2. Standardize Distances (km -> 100km)
# This is crucial for coefficient consistency with Table 2
df['dist_usa_100km'] = df['dist_usa_km'] / 100
df['dist_cdmx_100km'] = df['dist_cdmx_km'] / 100
df['dist_port_100km'] = df['dist_port_km'] / 100

# 3. Generate Interactions
df['X_USA_Trend'] = df['dist_usa_100km'] * df['trend']
df['X_CDMX_Trend'] = df['dist_cdmx_100km'] * df['trend']
df['X_Port_Trend'] = df['dist_port_100km'] * df['trend']

# --- 3. MODELING (Robustness Check) ---
# We compare strictly Export-Oriented (33) vs Domestic-Oriented (31)
# Sector 32 (Chem/Textile) is excluded as it is an intermediate input for 33
sectors = {
    '33': 'Machinery (Target)',
    '31': 'Food (Placebo)'
}

results = []

print("\n" + "="*80)
print(f"{'SECTOR':<25} | {'COEFF (per 100km)':<20} | {'P-VALUE':<10} | {'INTERPRETATION'}")
print("="*80)

for sec_code, sec_name in sectors.items():
    dep_var = f'count_{sec_code}'
    cluster_var = 'is_cluster' # Generic cluster flag
    
    # Formula using Standardized Interactions
    formula = f"{dep_var} ~ {cluster_var} + X_USA_Trend + X_CDMX_Trend + X_Port_Trend + C(year)"
    
    try:
        # Run Poisson GLM with Clustered SE
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit(cov_type='cluster', cov_kwds={'groups': df['grid_id']})
        
        target = 'X_USA_Trend'
        beta = model.params[target]
        pval = model.pvalues[target]
        conf_int = model.conf_int().loc[target]
        
        # Determine Interpretation
        direction = "Moving NORTH (Nearshoring)" if beta < 0 else "Moving SOUTH (Population)"
        sig_stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else "")
        
        # Print to Console
        print(f"{sec_name:<25} | {beta:.6f}{sig_stars:<4}       | {pval:.1e}  | {direction}")
        
        results.append({
            'Sector_Code': sec_code,
            'Sector_Name': sec_name,
            'Coeff_Dist_USA': beta,
            'Standard_Error': model.bse[target],
            'P_Value': pval,
            'Lower_CI': conf_int[0],
            'Upper_CI': conf_int[1],
            'Interpretation': direction
        })
        
    except Exception as e:
        print(f"Error modeling Sector {sec_code}: {e}")

print("="*80 + "\n")

# --- 4. EXPORT RESULTS ---
# Save the coefficients to CSV for reproducibility
if results:
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, 'Robustness_Check_Coefficients.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"[-] Robustness coefficients saved to: {csv_path}")