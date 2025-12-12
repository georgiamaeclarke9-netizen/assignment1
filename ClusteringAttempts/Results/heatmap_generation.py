import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# # loading csv
# df = pd.read_csv('Results/baseline_results.csv')

# # inputing  heatmap data
# heatmap_data = df.pivot(index='Horizon', columns='K', values='DirAcc')

# # creating and showing heatmpa
# sns.heatmap(heatmap_data, annot=True, fmt=".3f")
# plt.title('DirAcc by K and Horizon')
# plt.show()

# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.
import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
df_baseline = pd.read_csv('Results/baseline_results.csv')
df_regime = pd.read_csv('Results/results.csv')  # Your regime results

# Compare K=2 (best) vs baseline
k2_data = df_regime[df_regime['K'] == 2]

plt.figure(figsize=(10, 6))

# DirAcc comparison
plt.subplot(1, 3, 1)
plt.plot(df_baseline['Horizon'], df_baseline['Mean_DirAcc'], 'b-o', label='Baseline', linewidth=2)
plt.plot(k2_data['Horizon'], k2_data['Mean_DirAcc'], 'r-o', label='K=2 Regimes', linewidth=2)
plt.title('DirAcc: Baseline vs K=2')
plt.xlabel('Horizon')
plt.ylabel('DirAcc')
plt.legend()
plt.grid(True, alpha=0.3)

# MAE comparison
plt.subplot(1, 3, 2)
plt.plot(df_baseline['Horizon'], df_baseline['Mean_MAE'], 'b-o', label='Baseline', linewidth=2)
plt.plot(k2_data['Horizon'], k2_data['Mean_MAE'], 'r-o', label='K=2 Regimes', linewidth=2)
plt.title('MAE: Baseline vs K=2')
plt.xlabel('Horizon')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

# RMSE comparison
plt.subplot(1, 3, 3)
plt.plot(df_baseline['Horizon'], df_baseline['Mean_RMSE'], 'b-o', label='Baseline', linewidth=2)
plt.plot(k2_data['Horizon'], k2_data['Mean_RMSE'], 'r-o', label='K=2 Regimes', linewidth=2)
plt.title('RMSE: Baseline vs K=2')
plt.xlabel('Horizon')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_baseline_vs_k2.png', dpi=300)
plt.show()

# Calculate improvement
print("IMPROVEMENT: K=2 vs Baseline")
print("="*40)
for horizon in [1, 5, 10, 20, 40, 50, 70, 100]:
    baseline = df_baseline[df_baseline['Horizon'] == horizon]['Mean_DirAcc'].values[0]
    k2 = k2_data[k2_data['Horizon'] == horizon]['Mean_DirAcc'].values[0]
    improvement = k2 - baseline
    pct_improvement = (improvement / baseline) * 100
    print(f"Horizon {horizon:3d} days: {baseline:.3f} → {k2:.3f} (+{improvement:.3f}, +{pct_improvement:.1f}%)")