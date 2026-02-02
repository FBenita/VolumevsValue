import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf

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

# Interactions
df['X_USA_Trend'] = df['dist_usa_100km'] * df['trend']
df['X_CDMX_Trend'] = df['dist_cdmx_100km'] * df['trend']
df['X_Port_Trend'] = df['dist_port_100km'] * df['trend']
df['X_Cluster'] = df.get(f'is_cluster_{target_sector}', df.get('is_cluster', 0))

# --- 3. FILTER ACTIVE CELLS ---
df['Count_Raw'] = df[f'count_{target_sector}']
df_active = df[df['Count_Raw'] > 0].copy()

# Dependent Variable (Log K/L)
labor = df_active[f'labor_total_{target_sector}']
machinery = df_active[f'machinery_{target_sector}']
df_active['K_L'] = machinery / labor.replace(0, np.nan)
df_active['ln_K_L'] = np.log(df_active['K_L'] + 1)

# Clean NAs
vars_needed = ['ln_K_L', 'X_USA_Trend', 'X_CDMX_Trend', 'X_Port_Trend', 'X_Cluster', 'year', 'grid_id']
df_reg = df_active[vars_needed].dropna().copy()
print(f"Phase 2 Sample Size: {len(df_reg)}")

# --- 4. RUN REGRESSION (POOLED OLS) ---
print("Estimating Phase 2 Model...")
mod_pooled = smf.ols("ln_K_L ~ X_USA_Trend + X_CDMX_Trend + X_Port_Trend + X_Cluster + C(year)", 
                     data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['grid_id']})

# Export Table for Documentation
# Extract coefficients and diagnostics
params = mod_pooled.params
bse = mod_pooled.bse
pvals = mod_pooled.pvalues
ci = mod_pooled.conf_int(alpha=0.05)

table_df = pd.DataFrame({
    'Coeff': params,
    'SE': bse,
    'P_Value': pvals,
    'CI_Lower': ci[0],
    'CI_Upper': ci[1]
})

# Add Diagnostics rows at the bottom
diag_df = pd.DataFrame({
    'Coeff': [mod_pooled.nobs, mod_pooled.rsquared_adj, mod_pooled.aic],
    'SE': [np.nan, np.nan, np.nan] # Empty placeholders
}, index=['Observations', 'Adj. R-Squared', 'AIC'])

final_table = pd.concat([table_df, diag_df])
csv_path = os.path.join(output_folder, 'Table_2_Capital_Intensity.csv')
final_table.to_csv(csv_path)
print(f"[-] Table 2 saved to: {csv_path}")

# --- 5. ROBUST PLOTTING (SPLIT LAYERS) ---
# Prepare Plot Data
target_vars = ['X_USA_Trend', 'X_CDMX_Trend', 'X_Port_Trend', 'X_Cluster']
plot_df = table_df.loc[table_df.index.isin(target_vars)].copy()
# Reorder to match paper logic
plot_df = plot_df.reindex(['X_Cluster', 'X_USA_Trend', 'X_CDMX_Trend', 'X_Port_Trend'][::-1])

# Assign Y-positions
plot_df['y'] = range(len(plot_df))

# SPLIT INTO TWO DATAFRAMES
sig_df = plot_df[plot_df['P_Value'] < 0.05]
insig_df = plot_df[plot_df['P_Value'] >= 0.05]

fig, ax = plt.subplots(figsize=(10, 6))

# Layer 1: Significant Points (Red)
if not sig_df.empty:
    xerr = [sig_df['Coeff'] - sig_df['CI_Lower'], sig_df['CI_Upper'] - sig_df['Coeff']]
    ax.errorbar(sig_df['Coeff'], sig_df['y'], xerr=xerr, fmt='o', color='#D32F2F', 
                capsize=5, elinewidth=2.5, markeredgewidth=2, markersize=10, 
                label='Significant (p < 0.05)', zorder=10)

# Layer 2: Insignificant Points (Gray)
if not insig_df.empty:
    xerr = [insig_df['Coeff'] - insig_df['CI_Lower'], insig_df['CI_Upper'] - insig_df['Coeff']]
    ax.errorbar(insig_df['Coeff'], insig_df['y'], xerr=xerr, fmt='o', color='#9E9E9E', 
                capsize=5, elinewidth=2.5, markeredgewidth=2, markersize=10, 
                label='Insignificant', zorder=10)

# Formatting
labels_map = {
    'X_Cluster': 'Cluster Bonus (Local)',
    'X_USA_Trend': 'Nearshoring (Dist USA)',
    'X_CDMX_Trend': 'Domestic (Dist CDMX)',
    'X_Port_Trend': 'Global (Dist Port)'
}
ax.set_yticks(plot_df['y'])
ax.set_yticklabels([labels_map.get(i, i) for i in plot_df.index], fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.8)

ax.set_xlabel('Effect on Log Capital Intensity (K/L)', fontsize=12, fontweight='bold')
ax.set_title('Figure 7: Phase 2 - The Sophistication Test\n(Drivers of Automation in Active Factories)', fontsize=14, weight='bold')
ax.grid(axis='x', linestyle=':', alpha=0.5)
ax.legend(loc='lower right')

# Save
plot_path = os.path.join(output_folder, 'Figure_7_Capital_Intensity.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[-] Figure 7 saved to: {plot_path}")

plt.show()

