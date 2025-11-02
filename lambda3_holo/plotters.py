import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import crosscorr_at_lags, transfer_entropy, spearman_corr

def plot_rt_timeseries(df: pd.DataFrame, outpath: str):
    plt.figure()
    plt.plot(df["t"].values, df["entropy_RT_mo"].values)
    plt.xlabel("t"); plt.ylabel("S_RT (multi-objective)")
    plt.title("RT-like Entropy (Perimeter + Holes + Curvature)")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_crosscorr(df: pd.DataFrame, series_driver: str, series_response: str, outpath: str, maxlag:int=16):
    x = df[series_response].values
    y = df[series_driver].values
    if np.isnan(y).any():
        idx = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            y = np.interp(idx, idx[mask], y[mask])
        else:
            y = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    lags, corrs = crosscorr_at_lags(x, y, maxlag=maxlag)
    plt.figure()
    plt.plot(lags, corrs, marker="o")
    plt.xlabel("lag (steps)")
    plt.ylabel("Pearson corr")
    plt.title(f"Cross-corr: {series_response} vs {series_driver}")
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    best_idx = int(np.nanargmax(corrs))
    return int(lags[best_idx]), float(corrs[best_idx])

def plot_transfer_entropy(df: pd.DataFrame, series_x: str, series_y: str, outpath: str, n_bins:int=3):
    x = df[series_x].values
    y = df[series_y].values
    if np.isnan(x).any():
        idx = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() >= 2:
            x = np.interp(idx, idx[mask], x[mask])
        else:
            x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0)
    TE_x_to_y = transfer_entropy(x, y, n_bins=n_bins)
    TE_y_to_x = transfer_entropy(y, x, n_bins=n_bins)
    plt.figure()
    plt.bar([0,1], [TE_x_to_y, TE_y_to_x])
    plt.xticks([0,1], [f"{series_x}→{series_y}", f"{series_y}→{series_x}"])
    plt.ylabel("Transfer Entropy (nats)")
    plt.title("Directionality via Transfer Entropy")
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    return float(TE_x_to_y), float(TE_y_to_x)
