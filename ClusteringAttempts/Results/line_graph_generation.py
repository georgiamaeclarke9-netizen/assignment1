# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.

import pandas as pd
import matplotlib.pyplot as plt

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

# Create 3 plots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Colors for each model
colors = {
    'Original KMeans': 'blue', 
    'Improved KMeans (RobustScaler)': 'red',
    'Part 1 Baseline (No Regimes)': 'green'
}

# Line styles for better differentiation
line_styles = {
    'Original KMeans': '-',
    'Improved KMeans (RobustScaler)': '--',
    'Part 1 Baseline (No Regimes)': ':'
}

# DirAcc plot
for model in colors.keys():
    data = df[df['Model'] == model]
    axes[0].plot(data['HORIZON'], data['DIRACC'], marker='o', linewidth=2, 
                 label=model, color=colors[model], linestyle=line_styles[model])
axes[0].set_title('Directional Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('DirAcc', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.5, 0.75)  # Adjusted for wider range

# MAE plot
for model in colors.keys():
    data = df[df['Model'] == model]
    axes[1].plot(data['HORIZON'], data['MAE'], marker='s', linewidth=2, 
                 label=model, color=colors[model], linestyle=line_styles[model])
axes[1].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# RMSE plot
for model in colors.keys():
    data = df[df['Model'] == model]
    axes[2].plot(data['HORIZON'], data['RMSE'], marker='^', linewidth=2, 
                 label=model, color=colors[model], linestyle=line_styles[model])
axes[2].set_title('Root Mean Square Error Comparison', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Horizon (days)', fontsize=12)
axes[2].set_ylabel('RMSE', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

# Set x-ticks
horizons = sorted(df['HORIZON'].unique())
axes[2].set_xticks(horizons)

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show()
