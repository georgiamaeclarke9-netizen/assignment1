import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

def gmm_baseline():
# 1) Loading and preprocessing the data
    df = pd.read_csv('prices.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].str.upper()
    tickers = ['XLK', 'XLP', 'XLV', 'XLF', 'XLE', 'XLI']
    
    # Weekly prices (Friday close) - converting long to wide , .ffil to keep continuity. weekly to reduce noise
    wide_weekly = (
        df[df['ticker'].isin(tickers)]
          .pivot(index='date', columns='ticker', values='adj_close')
          .resample('W-FRI').last()
          .ffill()
          .dropna()
    )
    
    # 2) Features: weekly returns, 12w vol, 6w momentum, market aggregates
    #for capturing recent behaviour across sectors. Market regimes (bull/neutral/bear)
    returns = np.log(wide_weekly / wide_weekly.shift(1)).dropna()
    features = pd.DataFrame({
        **{f'{t}_return': returns[t] for t in tickers},
        **{f'{t}_vol_12w': returns[t].rolling(12).std() for t in tickers},
        **{f'{t}_mom_6w': returns[t].rolling(6).mean() for t in tickers},
        'market_return': returns.mean(axis=1),
        'market_vol': returns.std(axis=1)
    }).dropna()
    
    # 3) Train/Test split. Splitting features into train (up to and including 2019-01-04) - 
#k-means is being fit only on train, then applied to test. Doing so avoids look-ahead.
    SPLIT_DATE = pd.Timestamp("2019-01-04")
    train_mask = features.index <= SPLIT_DATE
    test_mask = features.index > SPLIT_DATE
    X_train, X_test = features.loc[train_mask], features.loc[test_mask]
    
    # Scaling - 
    # having unscaled features can dominate in areas, train-only keeps it leakeage safe.
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)
    x_full_z = scaler.transform(features)

#2. BIC MODEL SELECTION (GitHub approach)
    # --------------------------------------------------------------------
    print("\n📊 Performing BIC model selection (Hallac et al. method)...")
    
    lowest_bic = np.inf
    bic = []
    n_components_range = range(1, 8)  # 1 to 7 regimes
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    # Store results for all models
    bic_matrix = np.zeros((len(cv_types), len(n_components_range)))
    
    for i, cv_type in enumerate(cv_types):
        for j, n_components in enumerate(n_components_range):
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                n_init=10,
                random_state=42
            )
            gmm.fit(X_train_z)
            bic_value = gmm.bic(X_train_z)
            bic_matrix[i, j] = bic_value
            
            if bic_value < lowest_bic:
                lowest_bic = bic_value
                best_gmm = gmm
                best_n_components = n_components
                best_cv_type = cv_type
    
    # --------------------------------------------------------------------
    # 3. BIC VISUALIZATION (From GitHub)
    # --------------------------------------------------------------------
    print(f"\n✅ Optimal model: K={best_n_components}, Covariance='{best_cv_type}'")
    print(f"   BIC score: {lowest_bic:.1f}")
    
    # Create BIC plot (matching GitHub style)
    plt.figure(figsize=(10, 6))
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        plt.bar(xpos, bic_matrix[i, :], width=0.2, color=color, label=cv_type)
    
    plt.xticks(n_components_range)
    plt.ylim([bic_matrix.min() * 1.01 - 0.01 * bic_matrix.max(), bic_matrix.max()])
    plt.title('BIC Score per Model (Lower is Better)', fontsize=14)
    plt.xlabel('Number of Components (Regimes)', fontsize=12)
    plt.ylabel('BIC Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gmm_bic_selection_full.png', dpi=100)
    plt.show()
    
    # --------------------------------------------------------------------
    # 4. FINAL MODEL FITTING & ANALYSIS
    # --------------------------------------------------------------------
    print("\n🔍 Analyzing the fitted GMM model...")
    
    # Fit on full data for regime analysis (as in GitHub)
    final_gmm = GaussianMixture(
        n_components=best_n_components,
        covariance_type=best_cv_type,
        n_init=50,  # More initializations for stability
        random_state=42
    )
    final_gmm.fit(X_full_z)
    hidden_states = final_gmm.predict(X_full_z)
    
    # Display regime parameters (matching GitHub output format)
    print("\n" + "-"*60)
    print("MEAN AND VARIANCE OF EACH REGIME (Annualized):")
    print("-"*60)
    
    # Annualize means (52 weeks) - approximate for weekly data
    annual_factor = 52
    for i in range(best_n_components):
        print(f"\nRegime {i+1}:")
        # Create DataFrame like GitHub
        regime_stats = pd.DataFrame({
            'Mean': final_gmm.means_[i] * annual_factor,
            'Variance': np.diag(final_gmm.covariances_[i]) * annual_factor
        }, index=features.columns)
        print(regime_stats.round(4))
    
    # --------------------------------------------------------------------
    # 5. REGIME DISTRIBUTION VISUALIZATION (From GitHub)
    # --------------------------------------------------------------------
    # Add regime labels to features
    features_with_regime = features.copy()
    features_with_regime['regime'] = hidden_states
    returns_with_regime = returns.copy()
    returns_with_regime['regime'] = hidden_states
    
    # Plot distribution of returns by regime (simplified version)
    print("\n📈 Visualizing regime distributions...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, ticker in enumerate(tickers[:6]):  # First 6 tickers
        ax = axes[idx]
        for regime in range(best_n_components):
            mask = returns_with_regime['regime'] == regime
            data = returns[ticker].iloc[mask.values]
            ax.hist(data, bins=30, alpha=0.5, label=f'Regime {regime}', density=True)
        
        ax.set_title(f'{ticker} Returns by Regime')
        ax.set_xlabel('Weekly Log Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_regime_distributions.png', dpi=100)
    plt.show()
    
    # --------------------------------------------------------------------
    # 6. SAVE REGIME DATA (Your format)
    # --------------------------------------------------------------------
    print("\n💾 Saving regime data for prediction model...")
    
    regimes = pd.Series(hidden_states, index=features.index, dtype="Int64", name="regime")
    regime_df = pd.DataFrame({
        'date': regimes.index,
        'regime': regimes,
        'regime_lag1': regimes.shift(1)
    })
    
    filename = f'gmm_regimes_k{best_n_components}_{best_cv_type}.csv'
    regime_df.to_csv(filename, index=False)
    
    # Report regime distribution
    print(f"\n📊 Regime Distribution (Total: {len(regime_df)} weeks):")
    for regime in range(best_n_components):
        count = (regime_df['regime'] == regime).sum()
        pct = count / len(regime_df) * 100
        mean_return = features_with_regime[features_with_regime['regime']==regime]['market_return'].mean() * annual_factor
        mean_vol = features_with_regime[features_with_regime['regime']==regime]['market_vol'].mean() * np.sqrt(annual_factor)
        print(f"  Regime {regime}: {count} weeks ({pct:.1f}%) | "
              f"Avg Return: {mean_return:.2%} | Avg Vol: {mean_vol:.2%}")
    
    print(f"\n✅ Saved to: {filename}")
    print("="*70)
    
    return best_n_components, best_cv_type, filename, final_gmm, features_with_regime

# Execute
if __name__ == "__main__":
    n_components, cv_type, filename, model, features_regime = gmm_baseline()
    





