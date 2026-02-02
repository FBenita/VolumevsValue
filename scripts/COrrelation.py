import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
results_path = r'C:\'
file_path = os.path.join(results_path, 'mexico_manufacturing_panel.csv')
output_folder = os.path.join(results_path, '00_Final_Paper_Figures')

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Loading Census Data...")
df = pd.read_csv(file_path)

# --- 2. DATA PREPARATION ---
df['rama'] = df['rama'].astype(str)
df['cve_mun'] = df['cve_mun'].astype(str)

# Filter for Sector 33 (Machinery & Equipment)
df_33 = df[df['rama'].str.startswith('33')].copy()

# Focus on the most recent year available (Objective Snapshot)
recent_year = df_33['year'].max()
print(f"Analyzing Sector 33 Internal Structure for Year: {recent_year}")
df_33 = df_33[df_33['year'] == recent_year]

# Define the "High-Tech" Subsectors to Test
target_ramas = ['3361', '3363', '3364', '3344', '3359']
rama_names = {
    '3361': 'Auto Assembly',
    '3363': 'Auto Parts',
    '3364': 'Aerospace',
    '3344': 'Semiconductors & Components',
    '3359': 'Electrical Equipment'
}

# --- 3. CALCULATE MUNICIPALITY-LEVEL AGGREGATES ---
# We calculate the TOTAL Sector 33 value added per municipality
total_33 = df_33.groupby('cve_mun')['value_added'].sum().rename('Total_Sector_33')

# We calculate the value added for EACH specific Rama per municipality
rama_pivot = df_33.pivot_table(index='cve_mun', columns='rama', values='value_added', aggfunc='sum').fillna(0)

# Merge them into one validation dataframe
validation_df = pd.concat([total_33, rama_pivot], axis=1).fillna(0)

# --- 4. STATISTICAL VALIDATION (Correlation & Share) ---
stats = []

for rama in target_ramas:
    if rama in validation_df.columns:
        # A. CORRELATION: Does this subsector move with the whole sector?
        # (Proxy Validity Test)
        corr = validation_df['Total_Sector_33'].corr(validation_df[rama])
        
        # B. SHARE: How much of the sector does this industry represent?
        # (Dominance Test)
        total_val = validation_df['Total_Sector_33'].sum()
        rama_val = validation_df[rama].sum()
        share = (rama_val / total_val) * 100
        
        stats.append({
            'Rama_Code': rama,
            'Industry_Name': rama_names.get(rama, rama),
            'Correlation_with_Sec33': round(corr, 4),
            'National_Share_Pct': round(share, 2),
            'Verdict': 'Dominant Driver' if share > 10 else ('Niche/Emerging' if share > 1 else 'Minor')
        })

results_df = pd.DataFrame(stats).sort_values(by='National_Share_Pct', ascending=False)

# --- 5. EXPORT TABLE (The Evidence) ---
print("\n" + "="*80)
print("VALIDATION RESULTS (Exported to CSV)")
print("="*80)
print(results_df.to_string(index=False))

csv_path = os.path.join(output_folder, 'Appendix_Validation_Table.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n[-] Table saved to: {csv_path}")

# --- 6. EXPORT PLOTS (The Visual Proof) ---
# We generate a figure with 2 subplots:
# 1. The Dominant Driver (likely Auto Parts) - Shows why Sec 33 is good.
# 2. The Niche Driver (Semiconductors) - Shows why Sec 33 misses it (and why you need K/L).

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top Driver (Auto Parts 3363 usually)
top_driver = results_df.iloc[0]
code_top = top_driver['Rama_Code']
name_top = top_driver['Industry_Name']

if code_top in validation_df.columns:
    # Log-Log plot for better visualization of skewed economic data
    sns.regplot(ax=axes[0], 
                x=np.log(validation_df[code_top] + 1), 
                y=np.log(validation_df['Total_Sector_33'] + 1),
                scatter_kws={'alpha':0.4, 'color':'#1976D2'}, line_kws={'color':'black'})
    axes[0].set_title(f'Dominant Proxy: {name_top} ({code_top})\nShare: {top_driver["National_Share_Pct"]}% | Corr: {top_driver["Correlation_with_Sec33"]}', weight='bold')
    axes[0].set_xlabel(f'Log Value Added: {name_top}')
    axes[0].set_ylabel('Log Value Added: Total Sector 33')
    axes[0].grid(True, linestyle='--', alpha=0.3)

# Plot 2: Semiconductors (3344) - The "Nearshoring" Target
semi_code = '3344'
if semi_code in validation_df.columns:
    semi_stats = results_df[results_df['Rama_Code'] == semi_code]
    if not semi_stats.empty:
        share_semi = semi_stats.iloc[0]['National_Share_Pct']
        corr_semi = semi_stats.iloc[0]['Correlation_with_Sec33']
        
        sns.regplot(ax=axes[1], 
                    x=np.log(validation_df[semi_code] + 1), 
                    y=np.log(validation_df['Total_Sector_33'] + 1),
                    scatter_kws={'alpha':0.4, 'color':'#D32F2F'}, line_kws={'color':'black'})
        axes[1].set_title(f'Niche Target: Semiconductors ({semi_code})\nShare: {share_semi}% | Corr: {corr_semi}', weight='bold')
        axes[1].set_xlabel(f'Log Value Added: Semiconductors')
        axes[1].set_ylabel('Log Value Added: Total Sector 33')
        axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_folder, 'Appendix_Validation_Plots.png')
plt.savefig(plot_path, dpi=300)

print(f"[-] Plots saved to: {plot_path}")
