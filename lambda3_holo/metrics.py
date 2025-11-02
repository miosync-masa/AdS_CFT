"""
Causality & Correlation Metrics
================================

Pure statistical tools for analyzing holographic causality.

Functions:
- crosscorr_at_lags: Pearson correlation at multiple time lags
- spearman_corr: Rank-based monotonic correlation
- transfer_entropy: Directed information flow (X → Y)

Physical Interpretation:
- Pearson: Linear correlation strength (sensitive to outliers)
- Spearman: Monotonic correlation (robust to nonlinearity)
- TE: Causal information transfer in nats (bits/ln(2))

Usage:
    These functions are model-agnostic and can be applied to any
    time series data. For PhaseShift experiments, typical usage:
    
    >>> S = df["entropy_RT_mo"].values
    >>> lam = df["lambda_p99_A_out_pre"].values
    >>> lags, corrs = crosscorr_at_lags(S, lam, maxlag=16)
    >>> best_lag = lags[np.argmax(corrs)]
"""

import numpy as np
from typing import Tuple


def crosscorr_at_lags(
    a: np.ndarray,
    b: np.ndarray,
    maxlag: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson cross-correlation at multiple time lags.
    
    Slides series 'a' relative to series 'b' and computes correlation
    at each lag from -maxlag to +maxlag.
    
    Args:
        a: Time series (response variable, e.g., S_RT)
        b: Time series (driver variable, e.g., λ_p99)
        maxlag: Maximum lag in both directions
    
    Returns:
        lags: Array of lag values [-maxlag, ..., 0, ..., +maxlag]
        corrs: Pearson correlation at each lag
    
    Interpretation:
        - Positive lag: b leads a (b[t] → a[t+lag])
        - Negative lag: a leads b (a[t] → b[t-lag])
        - Peak at lag > 0: Causal influence from b to a
    
    Example:
        >>> lags, corrs = crosscorr_at_lags(entropy, lambda_field, maxlag=16)
        >>> best_lag = lags[np.argmax(corrs)]
        >>> if best_lag > 0:
        ...     print(f"λ drives S_RT with {best_lag}-step delay")
    """
    lags = np.arange(-maxlag, maxlag + 1, 1)
    corrs = []
    
    # Standardize series (zero mean, unit variance)
    a0 = (a - a.mean()) / (a.std() + 1e-12)
    b0 = (b - b.mean()) / (b.std() + 1e-12)
    
    for L in lags:
        if L == 0:
            aa, bb = a0, b0
        elif L > 0:
            # Positive lag: b[0:T-L] vs a[L:T]
            aa, bb = a0[L:], b0[:-L]
        else:
            # Negative lag: a[0:T+L] vs b[-L:T]
            aa, bb = a0[:L], b0[-L:]
        
        if len(aa) > 1:
            corrs.append(float(np.mean(aa * bb)))
        else:
            corrs.append(np.nan)
    
    return lags, np.array(corrs, dtype=float)


def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Compute ranks with average tie-breaking (for Spearman correlation).
    
    Assigns ranks 0, 1, 2, ... to sorted values.
    Ties receive the average of their ranks.
    
    Args:
        a: 1D array
    
    Returns:
        Array of ranks (same length as input)
    
    Example:
        >>> rankdata([10, 20, 20, 30])
        array([0., 1.5, 1.5, 3.])  # Ties at 20 get average rank 1.5
    """
    temp = a.argsort(kind='mergesort')
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a))
    
    # Handle ties: assign average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = cum - counts
    avg = (starts + cum - 1) / 2.0
    
    return avg[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation coefficient.
    
    Measures monotonic (not necessarily linear) relationship.
    Robust to outliers and nonlinear transformations.
    
    Args:
        x, y: 1D arrays (same length)
    
    Returns:
        Spearman ρ ∈ [-1, 1]
        - ρ = +1: Perfect monotonic increase
        - ρ = 0: No monotonic relationship
        - ρ = -1: Perfect monotonic decrease
    
    Physical Interpretation:
        In holographic causality experiments:
        - High |ρ|: Strong monotonic λ-S_RT relationship
        - ρ > 0.4: Statistically significant causal structure
    
    PhaseShift-IXb Result:
        Spearman(S_RT, λ_pre) = 0.461 @ lag=+6
        → Strong monotonic causality despite nonlinear RT functional
    """
    xr = rankdata(x)
    yr = rankdata(y)
    
    # Standardize ranks
    xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    
    return float(np.mean(xr * yr))


def discretize_quantiles(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """
    Discretize continuous series into quantile-based bins.
    
    Uses quantiles to ensure balanced bin populations.
    Handles edge cases where quantiles collapse.
    
    Args:
        x: Continuous 1D array
        n_bins: Number of discrete states (typically 3-5)
    
    Returns:
        Integer array with values in {0, 1, ..., n_bins-1}
    
    Note:
        Used as preprocessing for Transfer Entropy calculation.
        Too few bins → loss of information
        Too many bins → sparse histogram, unreliable TE estimate
    """
    qs = np.linspace(0, 1, n_bins + 1)
    qs[0], qs[-1] = 0.0, 1.0
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    
    # Fallback to equal-width bins if quantiles collapse
    if len(edges) <= 2:
        mn, mx = float(np.min(x)), float(np.max(x) + 1e-12)
        edges = np.linspace(mn, mx, n_bins + 1)
    
    d = np.digitize(x, edges[1:-1], right=False)
    return d.astype(np.int32)


def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = 3) -> float:
    """
    Compute Transfer Entropy: TE(X → Y) in nats.
    
    Measures directed information flow from X to Y using history length k=l=1:
    
        TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) log[p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t)]
    
    Args:
        x: Source time series (driver)
        y: Target time series (response)
        n_bins: Number of discrete states (3 recommended)
    
    Returns:
        Transfer entropy in nats (natural units)
        - TE = 0: No information transfer
        - TE > 0: X provides info about future of Y beyond Y's own history
    
    Physical Interpretation:
        - TE(λ→S_RT): Bulk criticality predicts boundary entropy
        - TE(S_RT→λ): Boundary back-reacts on bulk (HR feedback)
        - Asymmetry: TE(X→Y) ≠ TE(Y→X) reveals causal direction
    
    PhaseShift-IXb Results:
        TE(λ_pre→S) = 0.0525 nats  ← Bulk drives boundary
        TE(S→λ_pre) = 0.1314 nats  ← Stronger backreaction (HR + SOC)
    
    Note:
        k=l=1 is minimal history length. Longer histories capture
        more complex temporal dependencies but require more data.
        
        To convert to bits: TE_bits = TE_nats / ln(2) ≈ TE_nats × 1.443
    """
    xq = discretize_quantiles(x, n_bins)
    yq = discretize_quantiles(y, n_bins)
    
    # Construct history triplets: (y_t, x_t) → y_{t+1}
    yt = yq[:-1]  # y at time t
    xt = xq[:-1]  # x at time t
    y1 = yq[1:]   # y at time t+1
    
    n = n_bins
    
    # Joint histogram: p(y_{t+1}, y_t, x_t)
    p = np.zeros((n, n, n), dtype=float)
    for i in range(len(y1)):
        p[y1[i], yt[i], xt[i]] += 1.0
    p /= max(1.0, p.sum())
    
    # Marginals
    p_yt_xt = p.sum(axis=0) + 1e-12  # p(y_t, x_t)
    p_y1_yt = p.sum(axis=2) + 1e-12  # p(y_{t+1}, y_t)
    p_yt = p_yt_xt.sum(axis=1) + 1e-12  # p(y_t)
    
    # Transfer entropy
    te = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                pij = p[a, b, c]
                if pij <= 0:
                    continue
                # TE = KL[ p(y_{t+1}|y_t,x_t) || p(y_{t+1}|y_t) ]
                te += pij * np.log((p[a, b, c] / p_yt_xt[b, c]) / (p_y1_yt[a, b] / p_yt[b]))
    
    return float(te)
