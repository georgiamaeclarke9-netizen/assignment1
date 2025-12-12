# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all CSV files
df1 = pd.read_csv('Results/student_kmeans_k2_standardscaler_init50_NOpca_results.csv')
df2 = pd.read_csv('Results/student_ext_results.csv')
df3 = pd.read_csv('Results/part1_results_student.csv')

# Label each dataframe
df1['Model'] = 'Original KMeans'
df2['Model'] = 'Improved KMeans (RobustScaler)'
df3['Model'] = 'Part 1 Baseline (No Regimes)'

# Combine and sort by horizon
df = pd.concat([df1, df2, df3])
df = df.sort_values(['Model', 'HORIZON'])

# Prepare data for heatmaps
# Pivot the data to have models as rows and horizons as columns
pivot_diracc = df.pivot(index='Model', columns='HORIZON', values='DIRACC')
pivot_mae = df.pivot(index='Model', columns='HORIZON', values='MAE')
pivot_rmse = df.pivot(index='Model', columns='HORIZON', values='RMSE')

# Create figure with 3 heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Heatmap 1: DirAcc
im1 = axes[0].imshow(pivot_diracc.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.75)
axes[0].set_title('Directional Accuracy (DirAcc)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Model', fontsize=12)
axes[0].set_xlabel('Horizon (days)', fontsize=12)
axes[0].set_xticks(np.arange(len(pivot_diracc.columns)))
axes[0].set_xticklabels(pivot_diracc.columns.astype(int))
axes[0].set_yticks(np.arange(len(pivot_diracc.index)))
axes[0].set_yticklabels(pivot_diracc.index)
plt.colorbar(im1, ax=axes[0], label='DirAcc')

# Add text annotations for DirAcc
for i in range(len(pivot_diracc.index)):
    for j in range(len(pivot_diracc.columns)):
        text = axes[0].text(j, i, f'{pivot_diracc.iloc[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=10)

# Heatmap 2: MAE
im2 = axes[1].imshow(pivot_mae.values, cmap='YlOrRd_r', aspect='auto')  # Reversed colormap (darker = worse)
axes[1].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Horizon (days)', fontsize=12)
axes[1].set_xticks(np.arange(len(pivot_mae.columns)))
axes[1].set_xticklabels(pivot_mae.columns.astype(int))
axes[1].set_yticks(np.arange(len(pivot_mae.index)))
axes[1].set_yticklabels([])  # Hide y-axis labels for middle plot
plt.colorbar(im2, ax=axes[1], label='MAE')

# Add text annotations for MAE
for i in range(len(pivot_mae.index)):
    for j in range(len(pivot_mae.columns)):
        text = axes[1].text(j, i, f'{pivot_mae.iloc[i, j]:.3f}',
                          ha='center', va='center', color='white', fontsize=10)

# Heatmap 3: RMSE
im3 = axes[2].imshow(pivot_rmse.values, cmap='YlOrRd_r', aspect='auto')  # Reversed colormap (darker = worse)
axes[2].set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Horizon (days)', fontsize=12)
axes[2].set_xticks(np.arange(len(pivot_rmse.columns)))
axes[2].set_xticklabels(pivot_rmse.columns.astype(int))
axes[2].set_yticks(np.arange(len(pivot_rmse.index)))
axes[2].set_yticklabels([])  # Hide y-axis labels for right plot
plt.colorbar(im3, ax=axes[2], label='RMSE')

# Add text annotations for RMSE
for i in range(len(pivot_rmse.index)):
    for j in range(len(pivot_rmse.columns)):
        text = axes[2].text(j, i, f'{pivot_rmse.iloc[i, j]:.3f}',
                          ha='center', va='center', color='white', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()
