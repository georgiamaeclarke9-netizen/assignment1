# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# ===============================
# 1) Loading and preprocessing
# ===============================
DATA_FILE = "prices.csv"
TICKERS   = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLV"]
H1, H2    = 20, 16                 # segment length & overlap (step = H1 - H2)
K         = 2                      # simplest choice: high vs low correlation
SEED      = 42
USE_SCALER = True                  # set False to skip standardization
SPLIT_DATE = pd.Timestamp("2019-01-04")  # train/test split boundary

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df["ticker"] = df["ticker"].str.upper()

# Weekly close (Friday), wide panel
wide_weekly = (
    df[df["ticker"].isin(TICKERS)]
      .pivot(index="date", columns="ticker", values="adj_close")
      .sort_index()
      .resample("W-FRI").last()
      .ffill()
      .dropna()
)

# Weekly log-returns
returns = np.log(wide_weekly / wide_weekly.shift(1)).dropna()

# Train/Test split masks
train_mask = returns.index <= SPLIT_DATE
test_mask  = returns.index >  SPLIT_DATE

# Optional, train-only standardization (coordinate-wise per asset)
if USE_SCALER:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(returns.loc[train_mask].values)
    X_test  = scaler.transform(returns.loc[test_mask].values)
    returns_z = pd.DataFrame(
        np.vstack([X_train, X_test]),
        index=returns.index,
        columns=returns.columns
    )
else:
    returns_z = returns.copy()

# ===============================
# 2) Segmentation (lift to measures)
# ===============================
def segment(R: pd.DataFrame, h1: int, h2: int):
    """
    Create overlapping segments of length h1, advancing by (h1 - h2).
    Returns list of segment arrays (n_segments x h1 x d) and matching segment end dates.
    """
    step = h1 - h2
    segs = [R.iloc[i-h1+1:i+1].values for i in range(h1-1, len(R)-1, step)]
    dates = [R.index[i+1] for i in range(h1-1, len(R)-1, step)]
    return segs, pd.DatetimeIndex(dates)

segs, seg_dates = segment(returns_z, H1, H2)

# ===============================
# 3) W₂ distance between measures
# ===============================
def w2(X: np.ndarray, Y: np.ndarray) -> float:
    """
    2-Wasserstein distance between two empirical measures (discrete point clouds).
    - 1-D: closed form via sorted atoms (quantile mapping)
    - d-D: optimal assignment (Hungarian) on squared Euclidean costs
    """
    # Defensive: equalize lengths if needed (should be same = H1)
    if X.shape[0] != Y.shape[0]:
        m = min(X.shape[0], Y.shape[0])
        X, Y = X[:m], Y[:m]

    # 1-D shortcut
    if X.shape[1] == 1:
        return np.sqrt(np.mean((np.sort(X[:, 0]) - np.sort(Y[:, 0]))**2))

    # d-D: squared Euclidean cost + Hungarian
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    r, c = linear_sum_assignment(C)
    return np.sqrt(C[r, c].mean())

# ===============================
# 4) WK-means (medoid centres)
# ===============================
def wkmeans(D: np.ndarray, k: int, prev_centres: np.ndarray | None = None, seed: int = SEED):
    """
    K-medoid-like clustering over distance matrix D:
    - Assign to nearest centre
    - Update centre as the index with minimum intra-cluster sum (medoid)
    - Optional: label-persistence remapping vs. previous centres
    """
    rng = np.random.default_rng(seed)
    centres = rng.choice(len(D), k, replace=False)

    for _ in range(50):
        labels = D[:, centres].argmin(1)
        new_centres = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                # Re-seed empty cluster with the farthest-from-nearest (avoid duplicates)
                candidate = int(np.argmax(D[:, centres].min(1)))
                if candidate in new_centres:
                    candidate = int(np.argmax(D[:, centres].max(1)))
                new_centres.append(candidate)
            else:
                intra = D[np.ix_(idx, idx)].sum(1)  # sum of distances to others in cluster
                new_centres.append(int(idx[int(np.argmin(intra))]))
        new_centres = np.array(new_centres, dtype=int)
        if np.array_equal(new_centres, centres):
            break
        centres = new_centres

    # Remap labels for persistence (so cluster IDs don't flip across steps)
    if prev_centres is not None:
        cost = D[np.ix_(prev_centres, centres)]
        r, c = linear_sum_assignment(cost)  # map old centres -> new centres
        remap = {new: old for old, new in zip(r, c)}
        labels = np.array([remap.get(L, L) for L in labels], dtype=int)
        centres = centres[c]

    return labels, centres

# ===============================
# 5) Leak-safe walk-forward
# ===============================
def walkforward(segs: list[np.ndarray],
                dates: pd.DatetimeIndex,
                k: int,
                h1: int,
                h2: int):
    """
    Incrementally grow the distance matrix up to each t, run wkmeans on D[:t+1,:t+1],
    and label the FORWARD interval [segment_end, segment_end + step).
    """
    step = h1 - h2
    n = len(segs)
    D = np.zeros((0, 0))      # incremental distance matrix
    rows = []
    prev_centres = None

    for t in range(n):
        # Append distances from new segment t to previous ones [0..t)
        if t == 0:
            D = np.zeros((1, 1))
        else:
            new_row = np.array([w2(segs[t], segs[j]) for j in range(t)])[None, :]
            D = np.block([
                [D,               new_row.T],
                [new_row,         np.zeros((1, 1))]
            ])

        # Start clustering only after we have enough segments
        if t < k:
            continue

        labels, prev_centres = wkmeans(D[:t+1, :t+1], k, prev_centres, seed=SEED)

        segment_end   = dates[t]                   # label computed at segment end
        regime_start  = segment_end                # apply forward
        regime_end    = segment_end + pd.Timedelta(weeks=step)

        rows.append({
            "date":   segment_end,
            "start":  regime_start,
            "end":    regime_end,
            "regime": int(labels[-1]),            # label of the current (latest) segment
        })

    return pd.DataFrame(rows)

regimes_df = walkforward(segs, seg_dates, K, H1, H2)

# ===============================
# 6) Save regimes (+ lagged)
# ===============================
# If you’ll merge regimes on the same 'date' as a model feature, use the lagged value.
regimes_df["regime_lag1"] = regimes_df["regime"].shift(1)  # leak-safe feature at 'date'

out_file = "regimes_mcgreevy_weekly.csv"
regimes_df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
