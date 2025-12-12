from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from pathlib import Path
from typing import List, Optional, Tuple, Type

class Student:

    def __init__(
        self,
        config=None,
        random_state: int = 42,
        *,
        # parameters same as Part 1 CW
        n_lags=5,
        mom_windows=(5, 10, 20),
        vol_window=20,
        sma_windows=(10, 20),    
        ema_windows=(8, 16),
        rsi_window=10,
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=3,
        min_train_points=200,
        # added regime file
        regime_file='detected_regimes_k2.csv',
        
        **kwargs
    ):
        # parameters same as Part 1 CW
        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.rsi_window = int(rsi_window)
        
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)
        self.min_train_points = int(min_train_points)
        self.random_state = int(random_state)
        
        # Store regime_file for walk-forward
        self.regime_file = regime_file

        # Loading regime labels
        self.regime_df = pd.read_csv(regime_file)

        # validating required columns
        if 'date' not in self.regime_df.columns:
            raise ValueError(f"'date' column not found in {regime_file}")
        if 'regime_lag1' not in self.regime_df.columns and 'regime' not in self.regime_df.columns:
            raise ValueError(
                f"Neither 'regime_lag1' nor 'regime' found in {regime_file}. "
                "Expected headers: date, regime, regime_lag1."
            )

        # parsing dates - sorted DateTimeIndex
        self.regime_df['date'] = pd.to_datetime(self.regime_df['date'])
        self.regime_df = self.regime_df.set_index('date').sort_index()

        # Create lag if missing
        if 'regime_lag1' not in self.regime_df.columns and 'regime' in self.regime_df.columns:
            self.regime_df['regime_lag1'] = self.regime_df['regime'].shift(1)

        # casting to numeric (nullable int)
        if 'regime' in self.regime_df.columns:
            self.regime_df['regime'] = pd.to_numeric(self.regime_df['regime'], errors='coerce').astype('Int64')
        self.regime_df['regime_lag1'] = pd.to_numeric(self.regime_df['regime_lag1'], errors='coerce').astype('Int64')

        # distinct regime Ids (for one-hot features)
        self.regime_values = sorted(int(v) for v in pd.unique(self.regime_df['regime_lag1'].dropna())) or [0]

        # Overrides
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Model state
        self.pipe_ = None
        self.best_alpha_ = None
        self.fitted_ = False

        
        self._train_regime_ids = None    # regime IDs seen in train only
        self._train_columns = None       # columns (and order) seen at train

    
    # mltester stuff:
    @staticmethod
    def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
        """
        y_t = log(C_{t+H} / C_t). NaNs at the tail where future isn't available.
        """
        close = pd.Series(close).astype(float)
        return np.log(close.shift(-horizon) / close)
    
    @staticmethod
    def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
        """
        Return (DirAcc, MAE, RMSE) on overlapping, finite indices.
        """
        df = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            return (float("nan"), float("nan"), float("nan"))
        diracc = float((np.sign(df["y"]) == np.sign(df["yhat"])).mean())
        mae = float(np.abs(df["y"] - df["yhat"]).mean())
        rmse = float(np.sqrt(((df["y"] - df["yhat"]) ** 2).mean()))
        return diracc, mae, rmse
    
    def walk_forward_predict(self, prices: pd.DataFrame, horizon: int = 1, step: int = 10):
        if "Close" not in prices.columns:
            raise ValueError("prices must contain a 'Close' column")

        # Build full target once
        y_full = self.forward_log_return(prices["Close"], horizon=horizon)
        y_full.name = "y_true"
        test_dates = y_full.dropna().index

        preds = []
        for i in range(0, len(test_dates), step):
            block = test_dates[i : i + step]
            first_test = block[0]

            X_train = prices.loc[: first_test - pd.Timedelta(days=1)]
            y_train = y_full.loc[: first_test - pd.Timedelta(days=1)]

            # Create fresh model for this block
            model = Student(
                n_lags=self.n_lags,
                mom_windows=self.mom_windows,
                vol_window=self.vol_window,
                sma_windows=self.sma_windows,
                ema_windows=self.ema_windows,
                rsi_window=self.rsi_window,
                alpha_grid=self.alpha_grid,
                cv_splits=self.cv_splits,
                min_train_points=self.min_train_points,
                random_state=self.random_state,
                regime_file=self.regime_file
            )

            model.fit(X_train, y_train, meta={"horizon": horizon})
            X_pred = prices.loc[: block[-1]]
            y_hat = model.predict(X_pred, meta={"horizon": horizon})

            preds.append(y_hat.reindex(block).dropna())

        y_pred = pd.concat(preds) if preds else pd.Series(dtype=float, name="y_pred")
        y_pred.name = "y_pred"
        y_true = y_full.reindex(y_pred.index)
        y_true.name = "y_true"
        return y_true, y_pred
    
    def evaluate_ticker(self, prices: pd.DataFrame, horizon: int = 1, step: int = 10):
        #evaluating single ticker and return metrics 
        y_true, y_pred = self.walk_forward_predict(prices, horizon=horizon, step=step)
        diracc, mae, rmse = self.compute_metrics(y_true, y_pred)
        return diracc, mae, rmse
   
    
    # helper methods same as P1 CW
    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1))

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        close = pd.Series(close).astype(float)
        diff = close.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 1 - (1 / (1 + rs))
        return rsi.fillna(0.5)

    @staticmethod
    def _finite_mean(y: pd.Series) -> float:
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0
    
    # regime lookup

    def _get_regime_for_date(self, date):
        try:
            ts = pd.Timestamp(date)
            # using the lagged column to ensure never reading same-bar info
            s = self.regime_df['regime_lag1'].loc[:ts]
            if s.empty:
                return 0  # Default regime at the start of sample
            # Most recent available lagged regime
            val = s.iloc[-1]
            if pd.isna(val):
                # Walk back to last non-NaN if the very first lag is NaN
                s2 = s.dropna()
                return int(s2.iloc[-1]) if not s2.empty else 0
            return int(val)
        except Exception:
            return 0  # Default on error

    #feature creation
    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:

        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)
        
        #same as part 1 CW
        feats = {}
        
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)
        
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()
        
        feats[f"vol_{self.vol_window}"] = lr.rolling(self.vol_window, min_periods=self.vol_window).std(ddof=0)
        
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"sma_dist_{w}"] = (close - sma) / sma.replace(0, np.nan)
        
        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f"ema_dist_{w}"] = (close - ema) / ema.replace(0, np.nan)
        
        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)
        
        # creating dataframe
        F = pd.DataFrame(feats, index=X.index)
        F = F.replace([np.inf, -np.inf], np.nan)
        F = F.dropna()
        
        if F.empty:
            return F
        
        #adding regime featuresa
        regime_labels = F.index.map(self._get_regime_for_date)
        F['regime'] = regime_labels  
        
        return F
    
    # fit
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        F = self._make_features(X_train)
        
        if F.empty:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])
            self.best_alpha_ = None
            self.fitted_ = True
            return self
        
        y = y_train.reindex(F.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        F, y = F.loc[mask], y.loc[mask]

        # regime features (train) - lock one-hot width to train
        regimes_train = F['regime'].astype('Int64')
        self._train_regime_ids = sorted(pd.Series(regimes_train).dropna().astype(int).unique().tolist())
        for rid in self._train_regime_ids:
            F[f'regime_{rid}'] = (regimes_train == rid).astype(float)

        # remember the exact training column order for alignment in predict
        self._train_columns = list(F.columns)

        
        if len(y) == 0:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])
            self.best_alpha_ = None
            self.fitted_ = True
            return self
        
        if len(F) < self.min_train_points:
            mean_y = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])
            self.best_alpha_ = None
            self.fitted_ = True
            return self
        
        # alpha selection
        n_splits = min(self.cv_splits, max(2, len(F) // 200))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_alpha, best_mse = None, np.inf
        
        for a in self.alpha_grid:
            mses = []
            for tr_idx, va_idx in tscv.split(F.values):
                X_tr, X_va = F.values[tr_idx], F.values[va_idx]
                y_tr, y_va = y.values[tr_idx], y.values[va_idx]
                
                pipe = Pipeline([
                    ("scaler", RobustScaler()),
                    ("model", ElasticNet(alpha=a, l1_ratio=0.5, random_state=self.random_state))
                ])
                pipe.fit(X_tr, y_tr)
                y_hat = pipe.predict(X_va)
                mses.append(mean_squared_error(y_va, y_hat))
            
            avg_mse = float(np.mean(mses)) if mses else np.inf
            if avg_mse < best_mse:
                best_mse, best_alpha = avg_mse, a
        
        if best_alpha is None:
            best_alpha = 1.0
        self.best_alpha_ = float(best_alpha)
        
        # Final fit
        self.pipe_ = Pipeline([
            ("scaler", RobustScaler()),
            ("model", ElasticNet(alpha=self.best_alpha_, l1_ratio=0.5, random_state=self.random_state))
        ])
        self.pipe_.fit(F.values, y.values)
        self.fitted_ = True
        
        return self
    
    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:

        F = self._make_features(X)
        
        if not self.fitted_ or self.pipe_ is None:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        #prediction regime features aligned to train
        regimes_pred = F['regime'].astype('Int64')
        if self._train_regime_ids is None:
            self._train_regime_ids = []
        for rid in self._train_regime_ids:
            col = f'regime_{rid}'
            if col not in F.columns:
                F[col] = (regimes_pred == rid).astype(float)

        # align columns to the training layout (create missing with zeros, reorder)
        if self._train_columns:
            for c in self._train_columns:
                if c not in F.columns:
                    F[c] = 0.0
            F = F[self._train_columns]


        y_hat = self.pipe_.predict(F.values)
        return pd.Series(y_hat, index=F.index, name="y_pred")


# main method for running eval
def main():
    
    # parameters
    tickers = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLV"]
    horizons = [1, 5, 10, 20, 40, 50, 70, 100]
    step = 10
    data_file = "prices.csv"
    
    # testing for k=2, k=3, k=4
    for k in [2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"TESTING K={k} REGIMES")
        print('='*50)
        
        # Store results for this K
        mean_diraccuracies = []
        mean_mae = []
        mean_rmse = []
        
        # Load data once
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df.columns = [c.lower() for c in df.columns]
        
        for horizon in horizons:
            print(f"\nEvaluating horizon {horizon} days...")
            
            rows = []
            
            for t in tickers:
                # Prepare ticker data
                ticker_df = df[df['ticker'] == t].copy()
                ticker_df = ticker_df[['date', 'adj_close']]
                ticker_df.columns = ['Date', 'Close']
                ticker_df = ticker_df.set_index('Date')
                
                # Create Student instance
                student = Student(regime_file=f'detected_regimes_k{k}_robustscaler.csv')
                
                # Evaluate ticker
                diracc, mae, rmse = student.evaluate_ticker(
                    ticker_df, 
                    horizon=horizon, 
                    step=step
                )
                
                rows.append({
                    'ticker': t,
                    'diracc': diracc,
                    'mae': mae,
                    'rmse': rmse
                })
                
                print(f"{t:>6}: DirAcc={diracc:0.4f}  MAE={mae:0.6f}  RMSE={rmse:0.6f}")
            
            # Create summary
            summary = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
            if not summary.empty:
                mean_row = {
                    'ticker': 'MEAN',
                    'diracc': float(summary["diracc"].mean()),
                    'mae': float(summary["mae"].mean()),
                    'rmse': float(summary["rmse"].mean())
                }
                
                mean_diraccuracies.append(mean_row['diracc'])
                mean_mae.append(mean_row['mae'])
                mean_rmse.append(mean_row['rmse'])
                
                # printing in the same format as my P1 CW
                print(f"Horizon {horizon:3d} days: DirAcc={mean_row['diracc']:.3f} | MAE={mean_row['mae']:.4f} | RMSE={mean_row['rmse']:.4f}")
        
        # Print summary for K[2, 3, 4]
        print(f"\nK={k} Summary:")
        print(f"Average DirAcc across horizons: {np.mean(mean_diraccuracies):.3f}")
        print(f"Average MAE across horizons:    {np.mean(mean_mae):.4f}")
        print(f"Average RMSE across horizons:   {np.mean(mean_rmse):.4f}")
    
#TODO add graphs if time allows - I need to focus on experimenting

if __name__ == "__main__":

    main()