# %%
# Holographic Stats Toolkit — PRD-grade statistical augmentation
# ---------------------------------------------------------------
# What this cell does:
# 1) Load time series (S_RT, lambda_p99) from CSV if found under /mnt/data (or synthetic fallback)
# 2) Spearman/Pearson with 1000× moving-block bootstrap → 95% CI
# 3) Lag-of-peak correlation with bootstrap CI
# 4) Transfer Entropy (discrete estimator) with surrogate test (circular-shift) → p-values
# 5) Sensitivity sweeps (analysis-layer): TE bins, TE delay, embedding length
# 6) Save figures & a summary CSV to /mnt/data/holo_stats/
#
# Input CSV expectations (auto-detection, case-insensitive):
#   columns: 't', 'S_RT', 'lambda_p99'   (extra columns allowed; only these three are used)
#
# If no CSV present, we synthesize a causal pair with lag ~ +6 as a demonstration.
# Re-run this cell after placing your metrics CSV(s) into /mnt/data/ (e.g., '/mnt/data/metrics_ixb.csv').

import os
import re
import glob
import math
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import the display helper (graceful fallback)
try:
    from ace_tools import display_dataframe_to_user
except Exception:
    def display_dataframe_to_user(name: str, dataframe: pd.DataFrame):
        print(f"\n[{name}]")
        print(dataframe.head(20).to_string(index=False))

# -----------------------------
# Utility: ensure output folder
# -----------------------------
OUT_DIR = "/mnt/data/holo_stats"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# -----------------------------
# 0) Data loading or synthesis
# -----------------------------
def _find_candidate_csvs() -> List[str]:
    cands = glob.glob("/mnt/data/*.csv")
    # Heuristics: prefer files with 'metrics' or 'out' in name
    cands_sorted = sorted(cands, key=lambda p: (0 if re.search(r'metric|out|ixb|ixc', os.path.basename(p), re.I) else 1, p))
    return cands_sorted

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    map_needed = {}
    # Accept variations like s_rt, srt, lambda_p99, lambda99, etc.
    def pick(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None
    col_t = pick(['t','time','step'])
    col_s = pick(['s_rt','srt','rt','entropy','s'])
    col_l = pick(['lambda_p99','lambda99','lambda_out_p99','lambda','lam','lam_p99'])
    if col_t is None:
        df['t'] = np.arange(len(df))
        col_t = 't'
    if col_s is None or col_l is None:
        return pd.DataFrame()  # signal failure
    return df.rename(columns={col_t:'t', col_s:'S_RT', col_l:'lambda_p99'})[['t','S_RT','lambda_p99']]

def load_or_synthesize() -> Tuple[pd.DataFrame, str]:
    for path in _find_candidate_csvs():
        try:
            df0 = pd.read_csv(path)
            df = _standardize_columns(df0)
            if not df.empty and len(df) >= 100:
                return df.reset_index(drop=True), path
        except Exception:
            continue
    # Synthesize if nothing usable was found
    n = 300
    # latent AR(1)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8*x[t-1] + 0.5*rng.standard_normal()
    # boundary response with lag +6
    lag = 6
    y = np.zeros(n)
    y[lag:] = 0.7*(x[:-lag]) + 0.3*rng.standard_normal(size=n-lag)
    y[:lag] = 0.2*rng.standard_normal(size=lag)
    # scale to "S_RT" ~ positive
    S_RT = (y - y.min()) / (y.max() - y.min())
    lam = (x - x.min()) / (x.max() - x.min())
    df = pd.DataFrame({'t': np.arange(n), 'S_RT': S_RT, 'lambda_p99': lam})
    return df, "<synthetic demo>"

df, source_path = load_or_synthesize()

# Save the working copy
df.to_csv(os.path.join(OUT_DIR, "working_timeseries.csv"), index=False)

# -----------------------------------------
# 1) Spearman/Pearson with 1000× bootstrap
# -----------------------------------------
def moving_block_indices(n: int, block_len: int) -> np.ndarray:
    """Return resampled indices via moving-block bootstrap (circular)."""
    n_blocks = math.ceil(n / block_len)
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        block = [(s + k) % n for k in range(block_len)]
        idx.extend(block)
    return np.array(idx[:n], dtype=int)

def bootstrap_corr(x: np.ndarray,
                   y: np.ndarray,
                   method: str = 'spearman',
                   n_boot: int = 1000,
                   block_len: int = 10,
                   seed: int = 123) -> Dict[str, float]:
    rng_local = np.random.default_rng(seed)
    n = len(x)
    # Use local RNG inside
    def _corr(a,b):
        if method == 'spearman':
            from scipy.stats import spearmanr
            r, _ = spearmanr(a, b)
            return float(r)
        else:
            # Pearson
            if np.std(a)==0 or np.std(b)==0:
                return 0.0
            return float(np.corrcoef(a,b)[0,1])
    r0 = _corr(x, y)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # moving-block bootstrap
        n_blocks = math.ceil(n / block_len)
        starts = rng_local.integers(0, n, size=n_blocks)
        idx = []
        for s in starts:
            idx.extend([(s + k) % n for k in range(block_len)])
        idx = np.array(idx[:n], dtype=int)
        boot[i] = _corr(x[idx], y[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'r': r0, 'ci_lo': float(lo), 'ci_hi': float(hi)}

# -----------------------------------------
# 2) Lag-of-peak correlation + bootstrap CI
# -----------------------------------------
def xcorr_at_lag(x: np.ndarray, y: np.ndarray, lag: int, method='spearman') -> float:
    if lag > 0:
        a, b = x[:-lag], y[lag:]
    elif lag < 0:
        a, b = x[-lag:], y[:lag]
    else:
        a, b = x, y
    if method == 'spearman':
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r)
    else:
        if np.std(a)==0 or np.std(b)==0:
            return 0.0
        return float(np.corrcoef(a,b)[0,1])

def peak_lag_with_ci(x: np.ndarray,
                     y: np.ndarray,
                     lag_max: int = 24,
                     n_boot: int = 500,
                     block_len: int = 10,
                     method: str = 'spearman',
                     seed: int = 456) -> Dict[str, float]:
    # point estimate
    lags = np.arange(-lag_max, lag_max+1)
    vals = np.array([xcorr_at_lag(x,y,l,method) for l in lags])
    idx = int(np.argmax(vals))
    best_lag = int(lags[idx])
    best_r = float(vals[idx])

    # bootstrap CI for lag
    rng_local = np.random.default_rng(seed)
    n = len(x)
    lag_samples = np.empty(n_boot, dtype=int)
    r_samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # moving-block resample aligned pairs
        n_blocks = math.ceil(n / block_len)
        starts = rng_local.integers(0, n, size=n_blocks)
        idxs = []
        for s in starts:
            idxs.extend([(s + k) % n for k in range(block_len)])
        idxs = np.array(idxs[:n], dtype=int)
        xr = x[idxs]; yr = y[idxs]
        vals_b = np.array([xcorr_at_lag(xr, yr, l, method) for l in lags])
        j = int(np.argmax(vals_b))
        lag_samples[i] = int(lags[j])
        r_samples[i] = float(vals_b[j])

    lag_lo, lag_hi = np.percentile(lag_samples, [2.5, 97.5])
    r_lo, r_hi = np.percentile(r_samples, [2.5, 97.5])

    # plot cross-correlation curve
    fig1 = plt.figure(figsize=(6,4))
    plt.plot(lags, vals, marker='o')
    plt.axvline(best_lag, linestyle='--')
    plt.xlabel("Lag (x leads +)")
    plt.ylabel(f"{method.title()} correlation")
    plt.title("Cross-correlation vs lag")
    fig1.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "xcorr_curve.png")
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    return {
        'best_lag': best_lag,
        'best_r': best_r,
        'lag_ci_lo': float(lag_lo),
        'lag_ci_hi': float(lag_hi),
        'r_ci_lo': float(r_lo),
        'r_ci_hi': float(r_hi),
        'fig_xcorr': fig1_path
    }

# -----------------------------------------
# 3) Transfer Entropy (discrete) + surrogate
# -----------------------------------------
def discretize(a: np.ndarray, n_bins: int = 8) -> np.ndarray:
    # equal-frequency binning (quantile)
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(a, qs)
    # make edges unique (avoid duplicates)
    edges = np.unique(edges)
    if len(edges) <= 2:
        # fallback to min/max
        edges = np.linspace(a.min(), a.max(), n_bins+1)
    # use right=False to include left edges
    b = np.digitize(a, edges[1:-1], right=False)
    return b.astype(int), edges

def transfer_entropy_xy(x: np.ndarray,
                        y: np.ndarray,
                        lag: int = 1,
                        n_bins: int = 8,
                        pseudocount: float = 1e-6) -> float:
    """
    TE(X->Y) with discrete estimator:
      TE = sum p(y_{t+lag}, y_t, x_t) * log p(y_{t+lag}|y_t, x_t) / p(y_{t+lag}|y_t)
    """
    if lag < 1:
        lag = 1
    # align
    y_future = y[lag:]
    y_now = y[:-lag]
    x_now = x[:-lag]

    x_b, _ = discretize(x_now, n_bins)
    y_b, _ = discretize(y_now, n_bins)
    yf_b, _ = discretize(y_future, n_bins)

    # joint histogram counts
    nb = n_bins
    counts_xyz = np.zeros((nb, nb, nb), dtype=float)  # [y_future, y_now, x_now]
    counts_yz = np.zeros((nb, nb), dtype=float)       # [y_future, y_now]
    counts_zy = np.zeros((nb, nb), dtype=float)       # [y_now, x_now]
    counts_z = np.zeros(nb, dtype=float)              # [y_now]

    for a,b,c in zip(yf_b, y_b, x_b):
        counts_xyz[a,b,c] += 1.0
        counts_yz[a,b]    += 1.0
        counts_zy[b,c]    += 1.0
        counts_z[b]       += 1.0

    N = float(len(y_future))
    # probabilities with pseudocount
    P_xyz = (counts_xyz + pseudocount) / (N + pseudocount * nb**3)
    P_yz  = (counts_yz  + pseudocount) / (N + pseudocount * nb**2)
    P_zy  = (counts_zy  + pseudocount) / (N + pseudocount * nb**2)
    P_z   = (counts_z   + pseudocount) / (N + pseudocount * nb)

    # TE = sum P(yf, y, x) * log [ P(yf | y, x) / P(yf | y) ]
    # P(yf | y, x) = P(yf, y, x) / P(y, x)
    # P(yf | y) = P(yf, y) / P(y)
    te = 0.0
    for a in range(nb):
        for b in range(nb):
            for c in range(nb):
                p_xyz = P_xyz[a,b,c]
                p_yx = P_zy[b,c]
                p_yf_y = P_yz[a,b]
                p_y = P_z[b]
                # compute conditional terms (avoid log of 0 with pseudocount)
                p1 = p_xyz / p_yx
                p2 = p_yf_y / p_y
                te += p_xyz * np.log(p1 / p2)
    return float(te)  # nats

def circular_shift(a: np.ndarray, k: int) -> np.ndarray:
    k = k % len(a)
    if k == 0:
        return a.copy()
    return np.concatenate([a[-k:], a[:-k]])

def te_with_surrogates(x: np.ndarray,
                       y: np.ndarray,
                       lag: int = 1,
                       n_bins: int = 8,
                       n_sur: int = 200,
                       seed: int = 777) -> Dict[str, float]:
    te_obs = transfer_entropy_xy(x, y, lag=lag, n_bins=n_bins)
    rng_local = np.random.default_rng(seed)
    sur_vals = np.empty(n_sur, dtype=float)
    # circular-shift surrogates (destroy cross coupling, preserve marginals/autocorr)
    for i in range(n_sur):
        k = int(rng_local.integers(1, len(x)-1))
        xs = circular_shift(x, k)
        sur_vals[i] = transfer_entropy_xy(xs, y, lag=lag, n_bins=n_bins)
    # one-sided p-value: fraction >= observed
    p = (1.0 + np.sum(sur_vals >= te_obs)) / (n_sur + 1.0)

    # plot histogram
    fig2 = plt.figure(figsize=(6,4))
    plt.hist(sur_vals, bins=30)
    plt.axvline(te_obs, linestyle='--')
    plt.xlabel("TE surrogate values (nats)")
    plt.ylabel("Count")
    plt.title(f"Surrogate test for TE(X→Y), lag={lag}, bins={n_bins}")
    fig2.tight_layout()
    fig2_path = os.path.join(OUT_DIR, f"te_surrogates_lag{lag}_bins{n_bins}.png")
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    return {
        'te_obs': te_obs,
        'p_value': float(p),
        'fig_te_surrogates': fig2_path
    }

# ---------------------------------------------------
# 4) Sensitivity sweeps (analysis-layer, no reruns)
# ---------------------------------------------------
def sensitivity_te_grid(x: np.ndarray,
                        y: np.ndarray,
                        delays: List[int] = [1,2,4,6,8],
                        bins_list: List[int] = [6,8,12],
                        n_sur: int = 200) -> pd.DataFrame:
    rows = []
    for lag in delays:
        for nb in bins_list:
            te_xy = transfer_entropy_xy(x, y, lag=lag, n_bins=nb)
            te_yx = transfer_entropy_xy(y, x, lag=lag, n_bins=nb)
            sur_xy = te_with_surrogates(x, y, lag=lag, n_bins=nb, n_sur=n_sur)
            sur_yx = te_with_surrogates(y, x, lag=lag, n_bins=nb, n_sur=n_sur)
            rows.append({
                'lag': lag,
                'bins': nb,
                'TE_x_to_y': te_xy,
                'p_x_to_y': sur_xy['p_value'],
                'TE_y_to_x': te_yx,
                'p_y_to_x': sur_yx['p_value'],
            })
    return pd.DataFrame(rows)

# -----------------------------------------
# Run the full analysis on the loaded data
# -----------------------------------------
x = df['lambda_p99'].to_numpy().astype(float)
y = df['S_RT'].to_numpy().astype(float)

# 1) Spearman/Pearson bootstrap
spearman_res = bootstrap_corr(x, y, method='spearman', n_boot=1000, block_len=10, seed=101)
pearson_res  = bootstrap_corr(x, y, method='pearson',  n_boot=1000, block_len=10, seed=102)

# 2) Peak lag with CI
lag_res = peak_lag_with_ci(x, y, lag_max=24, n_boot=500, block_len=10, method='spearman', seed=202)

# 3) TE at (a) lag=1 and (b) best lag from cross-corr
te_lag1_xy = te_with_surrogates(x, y, lag=1, n_bins=8, n_sur=200, seed=301)
te_lag1_yx = te_with_surrogates(y, x, lag=1, n_bins=8, n_sur=200, seed=302)
best_lag = max(1, int(lag_res['best_lag']))
te_best_xy = te_with_surrogates(x, y, lag=best_lag, n_bins=8, n_sur=200, seed=303)
te_best_yx = te_with_surrogates(y, x, lag=best_lag, n_bins=8, n_sur=200, seed=304)

# 4) TE sensitivity grid
sens_df = sensitivity_te_grid(x, y, delays=[1,2,4,6,8], bins_list=[6,8,12], n_sur=100)
sens_path = os.path.join(OUT_DIR, "sensitivity_te_grid.csv")
sens_df.to_csv(sens_path, index=False)

# -----------------------------------------
# Assemble summary table
# -----------------------------------------
summary = pd.DataFrame([{
    'source': source_path,
    'N': len(df),
    'spearman_r': spearman_res['r'],
    'spearman_95CI_lo': spearman_res['ci_lo'],
    'spearman_95CI_hi': spearman_res['ci_hi'],
    'pearson_r': pearson_res['r'],
    'pearson_95CI_lo': pearson_res['ci_lo'],
    'pearson_95CI_hi': pearson_res['ci_hi'],
    'best_lag': lag_res['best_lag'],
    'best_lag_95CI_lo': lag_res['lag_ci_lo'],
    'best_lag_95CI_hi': lag_res['lag_ci_hi'],
    'r_at_best_lag': lag_res['best_r'],
    'r_best_lag_95CI_lo': lag_res['r_ci_lo'],
    'r_best_lag_95CI_hi': lag_res['r_ci_hi'],
    'TE_xy_lag1': te_lag1_xy['te_obs'],
    'TE_xy_lag1_p': te_lag1_xy['p_value'],
    'TE_yx_lag1': te_lag1_yx['te_obs'],
    'TE_yx_lag1_p': te_lag1_yx['p_value'],
    'TE_xy_bestlag': te_best_xy['te_obs'],
    'TE_xy_bestlag_p': te_best_xy['p_value'],
    'TE_yx_bestlag': te_best_yx['te_obs'],
    'TE_yx_bestlag_p': te_best_yx['p_value'],
    'fig_xcorr': lag_res['fig_xcorr'],
    'fig_te_sur_lag1_xy': te_lag1_xy['fig_te_surrogates'],
    'fig_te_sur_lag1_yx': te_lag1_yx['fig_te_surrogates'],
    'fig_te_sur_best_xy': te_best_xy['fig_te_surrogates'],
    'fig_te_sur_best_yx': te_best_yx['fig_te_surrogates'],
}] )

summary_path = os.path.join(OUT_DIR, "stats_summary.csv")
summary.to_csv(summary_path, index=False)

# -----------------------------------------
# Display results to the user
# -----------------------------------------
display_dataframe_to_user("Λ³ Holographic Stats — Summary", summary)
display_dataframe_to_user("Λ³ Holographic Stats — TE Sensitivity Grid", sens_df)

# Print where to find figures/files
print("\nSaved files:")
print(f"- Summary CSV: {summary_path}")
print(f"- TE sensitivity CSV: {sens_path}")
print(f"- XCorr plot: {lag_res['fig_xcorr']}")
print(f"- TE surrogate plots:")
print(f"  * {te_lag1_xy['fig_te_surrogates']}")
print(f"  * {te_lag1_yx['fig_te_surrogates']}")
print(f"  * {te_best_xy['fig_te_surrogates']}")
print(f"  * {te_best_yx['fig_te_surrogates']}")
