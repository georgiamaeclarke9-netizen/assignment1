# THE CREATION OF THIS FILE WAS SUPPORTED BY ARTIFICIAL INTELLIGENCE ASSISTANCE.

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

# -------------------- USER SETTINGS --------------------
DATA_FILE   = "prices.csv"
TICKERS     = ["XLK", "XLP", "XLV", "XLF", "XLE", "XLI"]
FREQ        = "W-FRI"          # weekly Friday close
H1, H2      = 20, 16           # segment length & overlap
K           = 2                # # of correlation regimes
SEED        = 42
USE_SCALER  = True             # optional, train-only standardization
SPLIT_DATE  = pd.Timestamp("2019-01-04")
OUT_FILE    = "regimes_mcgreevy_weekly.csv"
# -------------------------------------------------------

# ========== I/O & returns ==========
panel = pd.read_csv(DATA_FILE, parse_dates=["date"])
panel["ticker"] = panel["ticker"].str.upper()
wide = (
    panel[panel["ticker"].isin([t.upper() for t in TICKERS])]
      .pivot(index="date", columns="ticker", values="adj_close")
      .sort_index()
      .resample(FREQ).last()
      .ffill()
      .dropna()
)
returns = np.log(wide / wide.shift(1)).dropna()

if USE_SCALER:
    train_mask = returns.index <= SPLIT_DATE
    test_mask  = returns.index >  SPLIT_DATE
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(returns.loc[train_mask].values)
    Xte = scaler.transform(returns.loc[test_mask].values)
    R = pd.DataFrame(np.vstack([Xtr, Xte]), index=returns.index, columns=returns.columns)
else:
    R = returns.copy()

# ========== segmentation helpers ==========
def seg1d(series: pd.Series, h1: int, h2: int):
    step = h1 - h2
    S = [series.iloc[i-h1+1:i+1].values.reshape(-1,1)
         for i in range(h1-1, len(series)-1, step)]
    dates = [series.index[i+1] for i in range(h1-1, len(series)-1, step)]
    return S, pd.DatetimeIndex(dates)

def segMD(df: pd.DataFrame, h1: int, h2: int):
    step = h1 - h2
    S = [df.iloc[i-h1+1:i+1].values for i in range(h1-1, len(df)-1, step)]
    dates = [df.index[i+1] for i in range(h1-1, len(df)-1, step)]
    return S, pd.DatetimeIndex(dates)

# ========== W₂ distance (discrete measures) ==========
def w2(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] != Y.shape[0]:
        m = min(X.shape[0], Y.shape[0])
        X, Y = X[:m], Y[:m]
    if X.shape[1] == 1:
        return float(np.sqrt(np.mean((np.sort(X[:,0]) - np.sort(Y[:,0]))**2)))
    C = np.sum((X[:,None,:] - Y[None,:,:])**2, axis=2)
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r,c].mean()))

def Dmat(segs: list[np.ndarray]) -> np.ndarray:
    n = len(segs)
    D = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1, n):
            d = w2(segs[i], segs[j])
            D[i,j] = D[j,i] = d
    return D

# ========== WK-means with medoid centres ==========
def wkmeans(D: np.ndarray, k: int, prev_centres: np.ndarray | None = None, seed: int = SEED):
    rng = np.random.default_rng(seed)
    centres = rng.choice(len(D), k, replace=False)

    for _ in range(50):
        labels = D[:, centres].argmin(1)
        new_centres = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                # reseed with hardest point
                cand = int(np.argmax(D[:, centres].min(1)))
                if cand in new_centres:
                    cand = int(np.argmax(D[:, centres].max(1)))
                new_centres.append(cand)
            else:
                intra = D[np.ix_(idx, idx)].sum(1)
                new_centres.append(int(idx[int(np.argmin(intra))]))
        new_centres = np.array(new_centres, int)
        if np.array_equal(new_centres, centres):
            break
        centres = new_centres

    if prev_centres is not None:
        cost = D[np.ix_(prev_centres, centres)]
        r, c = linear_sum_assignment(cost)   # map old -> new
        remap = {new: old for old, new in zip(r, c)}
        labels = np.array([remap.get(L, L) for L in labels], int)
        centres = centres[c]

    return labels, centres

# ========== Step-1: univariate regimes per asset ==========
uni_labels: dict[str, np.ndarray] = {}
uni_segs:   dict[str, list[np.ndarray]] = {}
uni_dates:  dict[str, pd.DatetimeIndex] = {}

for col in R.columns:
    S1, dates1 = seg1d(R[col], H1, H2)
    D1 = Dmat(S1)
    labels1, _ = wkmeans(D1, k=2, prev_centres=None, seed=SEED)
    uni_labels[col] = labels1
    uni_segs[col]   = S1
    uni_dates[col]  = dates1

# ========== Transform: cluster-specific ECDF → uniforms ==========
def ecdf_from_atoms(atoms: np.ndarray):
    x = np.sort(atoms)
    def F(v: np.ndarray) -> np.ndarray:
        return np.searchsorted(x, v, side="right") / len(x)
    return F

# Build ECDFs
ecdfs: dict[tuple[str,int], callable] = {}
for col in R.columns:
    labs = uni_labels[col]
    segs = uni_segs[col]
    for lab in np.unique(labs):
        idx = np.where(labs == lab)[0]
        atoms = np.vstack([segs[i] for i in idx])[:,0]
        ecdfs[(col, int(lab))] = ecdf_from_atoms(atoms)

# Common segment dates intersection
common_dates = None
for col in R.columns:
    ds = uni_dates[col]
    common_dates = ds if common_dates is None else common_dates.intersection(ds)

# Build multivariate copula segments (h1 x d)
copula_segs: list[np.ndarray] = []
copula_dates: list[pd.Timestamp] = []
for dt in common_dates:
    cols_u = []
    ok = True
    for col in R.columns:
        ds   = uni_dates[col]
        pos  = ds.get_indexer([dt])[0]
        if pos == -1: ok = False; break
        seg  = uni_segs[col][pos][:,0]
        lab  = int(uni_labels[col][pos])
        Fcdf = ecdfs[(col, lab)]
        u    = Fcdf(seg).reshape(-1,1)
        cols_u.append(u)
    if ok:
        copula_segs.append(np.hstack(cols_u))   # (H1 x d)
        copula_dates.append(dt)
copula_dates = pd.DatetimeIndex(copula_dates)

# ========== Step-2: multivariate WK-means on copula, walk-forward ==========
def walkforward(segs: list[np.ndarray], dates: pd.DatetimeIndex, k: int, h1: int, h2: int):
    step = h1 - h2
    n = len(segs)
    D = np.zeros((0,0))
    prev_centres = None
    rows = []
    for t in range(n):
        if t == 0:
            D = np.zeros((1,1))
        else:
            new_row = np.array([w2(segs[t], segs[j]) for j in range(t)])[None,:]
            D = np.block([[D, new_row.T],[new_row, np.zeros((1,1))]])
        if t < k:
            continue
        labels, prev_centres = wkmeans(D[:t+1,:t+1], k, prev_centres, seed=SEED)
        seg_end = dates[t]
        rows.append({
            "date":   seg_end,
            "start":  seg_end,                          # applies forward
            "end":    seg_end + pd.Timedelta(weeks=step),
            "regime": int(labels[-1])
        })
    df = pd.DataFrame(rows).sort_values("date")
    df["regime_lag1"] = df["regime"].shift(1)          # leak-safe
    return df

regimes_df = walkforward(copula_segs, copula_dates, K, H1, H2)

# ========== save ==========
regimes_df.to_csv(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE}")
