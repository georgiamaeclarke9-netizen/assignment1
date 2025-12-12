# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all CSV files
df1 = pd.read_csv('Results/mcgreevy_results.csv')
df2 = pd.read_csv('Results/student_ext_results.csv')

# Standardize column names (in case of different capitalization)
def standardize_columns(df):
    df.columns = [col.upper() for col in df.columns]
    return df

df1 = standardize_columns(df1)
df2 = standardize_columns(df2)

# Label each dataframe
df1['MODEL'] = 'McGreevy Results'
df2['MODEL'] = 'Student_ext'

# Combine and sort by horizon
df = pd.concat([df1, df2])
df = df.sort_values(['MODEL', 'HORIZON'])

# Prepare data for heatmaps
# Pivot the data to have models as rows and horizons as columns
pivot_diracc = df.pivot(index='MODEL', columns='HORIZON', values='DIRACC')
pivot_mae = df.pivot(index='MODEL', columns='HORIZON', values='MAE')
pivot_rmse = df.pivot(index='MODEL', columns='HORIZON', values='RMSE')

# Ensure consistent ordering
model_order = ['McGreevy Results', 'Student_ext']
pivot_diracc = pivot_diracc.reindex(model_order)
pivot_mae = pivot_mae.reindex(model_order)
pivot_rmse = pivot_rmse.reindex(model_order)

# ============================================
# HEATMAP 1: DIRECTIONAL ACCURACY
# ============================================
fig1, ax1 = plt.subplots(figsize=(10, 4))

# Determine color range for DirAcc
diracc_min = pivot_diracc.values.min()
diracc_max = pivot_diracc.values.max()

im1 = ax1.imshow(pivot_diracc.values, cmap='RdYlGn', aspect='auto', 
                 vmin=max(diracc_min - 0.02, 0.5), vmax=min(diracc_max + 0.02, 0.8))
ax1.set_title('Directional Accuracy (DirAcc) Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Model', fontsize=14)
ax1.set_xlabel('Horizon (days)', fontsize=14)
ax1.set_xticks(np.arange(len(pivot_diracc.columns)))
ax1.set_xticklabels(pivot_diracc.columns.astype(int), fontsize=12)
ax1.set_yticks(np.arange(len(pivot_diracc.index)))
ax1.set_yticklabels(pivot_diracc.index, fontsize=13)
cbar1 = fig1.colorbar(im1, ax=ax1, label='DirAcc', fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=12)

# Add text annotations for DirAcc with larger font
for i in range(len(pivot_diracc.index)):
    for j in range(len(pivot_diracc.columns)):
        value = pivot_diracc.iloc[i, j]
        # Dynamic text color for better contrast
        color = 'black' if value > (diracc_min + diracc_max) / 2 else 'white'
        ax1.text(j, i, f'{value:.3f}',
                ha='center', va='center', color=color, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('diracc_heatmap_large.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# HEATMAP 2: MEAN ABSOLUTE ERROR
# ============================================
fig2, ax2 = plt.subplots(figsize=(10, 4))

mae_max = pivot_mae.values.max()
im2 = ax2.imshow(pivot_mae.values, cmap='YlOrRd_r', aspect='auto', vmax=mae_max)
ax2.set_title('Mean Absolute Error (MAE) Comparison', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Model', fontsize=14)
ax2.set_xlabel('Horizon (days)', fontsize=14)
ax2.set_xticks(np.arange(len(pivot_mae.columns)))
ax2.set_xticklabels(pivot_mae.columns.astype(int), fontsize=12)
ax2.set_yticks(np.arange(len(pivot_mae.index)))
ax2.set_yticklabels(pivot_mae.index, fontsize=13)
cbar2 = fig2.colorbar(im2, ax=ax2, label='MAE', fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=12)

# Add text annotations for MAE with larger font
for i in range(len(pivot_mae.index)):
    for j in range(len(pivot_mae.columns)):
        value = pivot_mae.iloc[i, j]
        # Dynamic text color based on cell brightness
        normalized_value = value / mae_max
        color = 'white' if normalized_value > 0.5 else 'black'
        ax2.text(j, i, f'{value:.3f}',
                ha='center', va='center', color=color, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('mae_heatmap_large.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# HEATMAP 3: ROOT MEAN SQUARE ERROR
# ============================================
fig3, ax3 = plt.subplots(figsize=(10, 4))

rmse_max = pivot_rmse.values.max()
im3 = ax3.imshow(pivot_rmse.values, cmap='YlOrRd_r', aspect='auto', vmax=rmse_max)
ax3.set_title('Root Mean Square Error (RMSE) Comparison', fontsize=16, fontweight='bold', pad=20)
ax3.set_ylabel('Model', fontsize=14)
ax3.set_xlabel('Horizon (days)', fontsize=14)
ax3.set_xticks(np.arange(len(pivot_rmse.columns)))
ax3.set_xticklabels(pivot_rmse.columns.astype(int), fontsize=12)
ax3.set_yticks(np.arange(len(pivot_rmse.index)))
ax3.set_yticklabels(pivot_rmse.index, fontsize=13)
cbar3 = fig3.colorbar(im3, ax=ax3, label='RMSE', fraction=0.046, pad=0.04)
cbar3.ax.tick_params(labelsize=12)

# Add text annotations for RMSE with larger font
for i in range(len(pivot_rmse.index)):
    for j in range(len(pivot_rmse.columns)):
        value = pivot_rmse.iloc[i, j]
        # Dynamic text color based on cell brightness
        normalized_value = value / rmse_max
        color = 'white' if normalized_value > 0.5 else 'black'
        ax3.text(j, i, f'{value:.3f}',
                ha='center', va='center', color=color, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('rmse_heatmap_large.png', dpi=300, bbox_inches='tight')
plt.show()
