import numpy as np
from typing import Tuple

def crosscorr_at_lags(a: np.ndarray, b: np.ndarray, maxlag:int=16) -> Tuple[np.ndarray, np.ndarray]:
    lags = np.arange(-maxlag, maxlag+1, 1)
    corrs = []
    a0 = (a - a.mean())/(a.std()+1e-12)
    b0 = (b - b.mean())/(b.std()+1e-12)
    for L in lags:
        if L == 0:
            aa, bb = a0, b0
        elif L > 0:
            aa, bb = a0[L:], b0[:-L]
        else:
            aa, bb = a0[:L], b0[-L:]
        if len(aa) > 1:
            corrs.append(float(np.mean(aa*bb)))
        else:
            corrs.append(np.nan)
    return lags, np.array(corrs, dtype=float)

def rankdata(a: np.ndarray) -> np.ndarray:
    temp = a.argsort(kind='mergesort')
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a))
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = cum - counts
    avg = (starts + cum - 1)/2.0
    return avg[inv]

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(x)
    yr = rankdata(y)
    xr = (xr - xr.mean())/(xr.std()+1e-12)
    yr = (yr - yr.mean())/(yr.std()+1e-12)
    return float(np.mean(xr*yr))

def discretize_quantiles(x: np.ndarray, n_bins:int=3) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins+1)
    qs[0], qs[-1] = 0.0, 1.0
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        mn, mx = float(np.min(x)), float(np.max(x)+1e-12)
        edges = np.linspace(mn, mx, n_bins+1)
    d = np.digitize(x, edges[1:-1], right=False)
    return d.astype(np.int32)

def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins:int=3) -> float:
    xq = discretize_quantiles(x, n_bins)
    yq = discretize_quantiles(y, n_bins)
    yt = yq[:-1]
    xt = xq[:-1]
    y1 = yq[1:]
    n = n_bins
    p = np.zeros((n, n, n), dtype=float)  # y1, yt, xt
    for i in range(len(y1)):
        p[y1[i], yt[i], xt[i]] += 1.0
    p /= max(1.0, p.sum())
    p_yt_xt = p.sum(axis=0) + 1e-12
    p_y1_yt = p.sum(axis=2) + 1e-12
    p_yt    = p_yt_xt.sum(axis=1) + 1e-12
    te = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                pij = p[a,b,c]
                if pij <= 0: 
                    continue
                te += pij * np.log((p[a,b,c]/p_yt_xt[b,c]) / (p_y1_yt[a,b]/p_yt[b]))
    return float(te)
