# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_zoomed_bar_charts(csv_files, labels, output_dir='plots'):
    """
    Create zoomed bar charts from multiple CSV files.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store data
    dir_acc_data = []
    mae_data = []
    rmse_data = []
    
    # Read each CSV file
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        avg_by_k = df.groupby('K')[['Mean_DirAcc', 'Mean_MAE', 'Mean_RMSE']].mean().reset_index()
        
        dir_acc_data.append(avg_by_k.set_index('K')['Mean_DirAcc'])
        mae_data.append(avg_by_k.set_index('K')['Mean_MAE'])
        rmse_data.append(avg_by_k.set_index('K')['Mean_RMSE'])
    
    # Create dataframes for easier plotting
    dir_acc_df = pd.DataFrame({label: data for label, data in zip(labels, dir_acc_data)}).T
    mae_df = pd.DataFrame({label: data for label, data in zip(labels, mae_data)}).T
    rmse_df = pd.DataFrame({label: data for label, data in zip(labels, rmse_data)}).T
    
    # Plot settings
    bar_width = 0.15
    opacity = 0.8
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    n_methods = len(labels)
    x = np.arange(len(dir_acc_df.columns))  # K values
    
    # Function to create a zoomed plot
    def create_zoomed_plot(data_df, ylabel, title, filename, decimal_places=4):
        plt.figure(figsize=(12, 6))
        
        for i, label in enumerate(labels):
            plt.bar(x + i * bar_width - (n_methods - 1) * bar_width / 2, 
                    data_df.loc[label], 
                    bar_width,
                    alpha=opacity,
                    color=colors[i % len(colors)],
                    label=label)
        
        plt.xlabel('Number of Regimes (K)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(x, data_df.columns)
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Calculate zoom range
        min_val = data_df.min().min()
        max_val = data_df.max().max()
        data_range = max_val - min_val
        
        # Set y-axis limits with padding
        padding = 0.1  # 10% padding
        y_min = min_val - padding * data_range
        y_max = max_val + padding * data_range
        plt.ylim(y_min, y_max)
        
        # Add value labels on top of bars
        for i, label in enumerate(labels):
            for j, k in enumerate(data_df.columns):
                value = data_df.loc[label, k]
                fmt = f'{{:.{decimal_places}f}}'
                plt.text(j + i * bar_width - (n_methods - 1) * bar_width / 2, 
                        value + 0.005 * data_range, 
                        fmt.format(value), 
                        ha='center', 
                        va='bottom', 
                        fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create the three zoomed plots
    create_zoomed_plot(dir_acc_df, 
                      'Average Directional Accuracy', 
                      'Directional Accuracy Comparison (Zoomed)',
                      'directional_accuracy_zoomed.png',
                      decimal_places=4)
    
    create_zoomed_plot(mae_df, 
                      'Average Mean Absolute Error (MAE)', 
                      'MAE Comparison (Zoomed)',
                      'mae_zoomed.png',
                      decimal_places=6)
    
    create_zoomed_plot(rmse_df, 
                      'Average Root Mean Square Error (RMSE)', 
                      'RMSE Comparison (Zoomed)',
                      'rmse_zoomed.png',
                      decimal_places=6)
    

    print(f"\nZoomed plots saved to '{output_dir}' directory")

# Example usage
if __name__ == "__main__":
    # List your CSV files and their labels
    csv_files = [
        'Results/init20_results.csv',
        'Results/init30_results.csv',
        'Results/pca_results.csv',
        'Results/robust_results.csv',
        'Results/student_kmeans_standardscaler_init50_NOpca_results.csv'
    ]
    
    labels = [
        'init20',
        'init30', 
        'PCA',
        'Robust',
        'Student KMeans'
    ]

    create_zoomed_bar_charts(csv_files, labels, output_dir='method_comparison')