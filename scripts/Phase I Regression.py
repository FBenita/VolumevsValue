import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from libpysal.weights import KNN
import libpysal

# --- 1. CONFIGURATION ---
results_path = r'C:\'
output_folder = os.path.join(results_path, '00_Final_Paper_Figures')
file_path = os.path.join(results_path, 'MEXICO_PANEL_WITH_EXOGENOUS_VARS.csv')

print("Loading Data...")
df = pd.read_csv(file_path)

# --- 2. DATA PREP ---
target_sector = '33'
df['dist_usa_100km'] = df['dist_usa_km'] / 100
df['dist_cdmx_100km'] = df['dist_cdmx_km'] / 100
df['dist_port_100km'] = df['dist_port_km'] / 100
df['trend'] = df['year'] - 2010

df['X_USA_Trend'] = df['dist_usa_100km'] * df['trend']
df['X_CDMX_Trend'] = df['dist_cdmx_100km'] * df['trend']
df['X_Port_Trend'] = df['dist_port_100km'] * df['trend']
df['X_Cluster'] = df.get(f'is_cluster_{target_sector}', df.get('is_cluster', 0))
df['Y_Count'] = df[f'count_{target_sector}']

# Spatial Lags
print("Building Spatial Weights...")
df_geo = df[['grid_id', 'x_coord', 'y_coord']].drop_duplicates()
coords = list(zip(df_geo['x_coord'], df_geo['y_coord']))
w = KNN.from_array(coords, k=8)
lags = []
for year in df['year'].unique():
    sub = df[df['year'] == year].set_index('grid_id').reindex(df_geo['grid_id']).fillna(0)
    lag_cluster = libpysal.weights.lag_spatial(w, sub['X_Cluster'])
    s = pd.Series(lag_cluster, index=df_geo['grid_id'], name='W_X_Cluster').reset_index()
    s['year'] = year
    lags.append(s)
df = df.merge(pd.concat(lags), on=['grid_id', 'year'], how='left')

# CLEAN DATA (Strict match)
vars_needed = ['Y_Count', 'X_USA_Trend', 'X_CDMX_Trend', 'X_Port_Trend', 'X_Cluster', 'W_X_Cluster', 'year', 'grid_id']
df_reg = df[vars_needed].dropna().copy()
print(f"Sample Size: {len(df_reg)}")

# --- 3. ESTIMATE MODELS ---
formula = "X_USA_Trend + X_CDMX_Trend + X_Port_Trend + X_Cluster + C(year)"

print("1. Estimating OLS...")
mod_ols = smf.ols(f"Y_Count ~ {formula}", data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['grid_id']})

print("2. Estimating Poisson (Robust)...")
mod_ppml = smf.poisson(f"Y_Count ~ {formula}", data=df_reg).fit(disp=0, cov_type='cluster', cov_kwds={'groups': df_reg['grid_id']})

print("3. Estimating Spatial Poisson...")
mod_sp = smf.poisson(f"Y_Count ~ {formula} + W_X_Cluster", data=df_reg).fit(disp=0, cov_type='cluster', cov_kwds={'groups': df_reg['grid_id']})

# --- 4. FORMATTING FUNCTION ---
def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

def extract_column(res, model_type):
    # 1. Coefficients & SEs
    params = res.params
    se = res.bse
    pvals = res.pvalues
    
    # Format: "0.123*** (0.04)"
    formatted_coeffs = []
    for var in params.index:
        val = f"{params[var]:.4f}{get_stars(pvals[var])}"
        err = f"({se[var]:.4f})"
        formatted_coeffs.append(val)
        formatted_coeffs.append(err) # Add SE as a separate row below
    
    # Create Index with interleaved SE rows
    idx = []
    for var in params.index:
        idx.append(var)
        idx.append(f"{var}_SE")
        
    col_series = pd.Series(formatted_coeffs, index=idx)
    
    # 2. Diagnostics
    n_obs = int(res.nobs)
    llf = f"{res.llf:.1f}"
    aic = f"{res.aic:.1f}"
    
    # R-squared Logic
    if model_type == 'OLS':
        r2 = f"{res.rsquared_adj:.3f}"
        r2_label = "Adj. R-squared"
    else:
        # Pseudo R-squared (McFadden) = 1 - (LL_model / LL_null)
        # Statsmodels usually calculates this as res.prsquared
        r2 = f"{res.prsquared:.3f}"
        r2_label = "Pseudo R-squared"

    diagnostics = pd.Series([n_obs, r2, llf, aic], 
                            index=['Observations', r2_label, 'Log Likelihood', 'AIC'])
    
    return pd.concat([col_series, diagnostics])

# --- 5. BUILD TABLE ---
col_1 = extract_column(mod_ols, 'OLS')
col_2 = extract_column(mod_ppml, 'Poisson')
col_3 = extract_column(mod_sp, 'Spatial')

# Merge into one DataFrame
# We align on index. Note: OLS might have different R2 label, so we align carefully
final_table = pd.concat([col_1, col_2, col_3], axis=1, keys=['(1) OLS', '(2) Poisson', '(3) Spatial'])

# Reorder Rows for Readability
# We want structural vars at top, Year dummies at bottom, Diagnostics at very bottom
structural_vars = ['X_Cluster', 'X_Cluster_SE', 
                   'W_X_Cluster', 'W_X_Cluster_SE',
                   'X_USA_Trend', 'X_USA_Trend_SE', 
                   'X_CDMX_Trend', 'X_CDMX_Trend_SE', 
                   'X_Port_Trend', 'X_Port_Trend_SE']

# Filter explicitly to control order
# We extract structural, then diagnostics
final_view = final_table.loc[structural_vars]
diagnostics_view = final_table.loc[['Observations', 'Adj. R-squared', 'Pseudo R-squared', 'Log Likelihood', 'AIC']]

# Combine
paper_ready_table = pd.concat([final_view, diagnostics_view])

# --- 6. EXPORT ---
csv_path = os.path.join(output_folder, 'Table_1_Regression_Results_Final.csv')
paper_ready_table.to_csv(csv_path)

print("\n" + "="*80)
print("FINAL PAPER TABLE PREVIEW")
print("="*80)
print(paper_ready_table.fillna("-"))
print("="*80)

print(f"[-] Saved to: {csv_path}")
