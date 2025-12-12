import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

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

    
    # helper methods (same as part 1 CW)
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

    # feature creation
    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:

        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)
        
        # same as part 1 CW
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
                    ("scaler", StandardScaler()),
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
            ("scaler", StandardScaler()),
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