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
    """
    Plot cross-correlation between driver (cause) and response (effect).
    
    Args:
        series_driver: Column name for driver/cause (e.g., lambda_p99_A_out_pre)
        series_response: Column name for response/effect (e.g., entropy_RT_mo)
    """
    # Get the data
    response = df[series_response].values  # S_RT (effect)
    driver = df[series_driver].values      # λ_pre (cause)
    
    # Debug: Print NaN statistics before interpolation
    print(f"[DEBUG] NaNs in {series_driver}: {np.isnan(driver).sum()}/{len(driver)}")
    print(f"[DEBUG] NaNs in {series_response}: {np.isnan(response).sum()}/{len(response)}")
    
    # Interpolate NaNs in driver
    if np.isnan(driver).any():
        idx = np.arange(len(driver))
        mask = ~np.isnan(driver)
        if mask.sum() >= 2:
            driver_interp = np.interp(idx, idx[mask], driver[mask])
            print(f"[DEBUG] Interpolated {np.isnan(driver).sum()} NaN values in driver")
        else:
            driver_interp = np.nan_to_num(driver, nan=np.nanmean(driver) if np.isfinite(np.nanmean(driver)) else 0.0)
            print(f"[DEBUG] Too few non-NaN values, using mean fill")
    else:
        driver_interp = driver
    
    # Check for NaNs in response (shouldn't have any)
    if np.isnan(response).any():
        print(f"[WARNING] Response series has NaNs, interpolating...")
        idx = np.arange(len(response))
        mask = ~np.isnan(response)
        if mask.sum() >= 2:
            response = np.interp(idx, idx[mask], response[mask])
    
    # Compute cross-correlation
    # crosscorr_at_lags expects (response, driver) order for positive lag = driver leads
    lags, corrs = crosscorr_at_lags(response, driver_interp, maxlag=maxlag)
    
    # Debug: Print correlation peak info
    best_idx = int(np.nanargmax(corrs))
    best_lag = int(lags[best_idx])
    best_corr = float(corrs[best_idx])
    print(f"[DEBUG] Best correlation: {best_corr:.3f} at lag {best_lag}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(lags, corrs, marker="o", markersize=4)
    plt.axvline(best_lag, color='red', linestyle='--', alpha=0.7, label=f'Peak: lag={best_lag}')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel("Lag (steps) [positive = driver leads]")
    plt.ylabel("Pearson correlation")
    plt.title(f"Cross-corr: {series_response} vs {series_driver}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    return best_lag, best_corr

def plot_transfer_entropy(df: pd.DataFrame, series_x: str, series_y: str, outpath: str, n_bins:int=3):
    """
    Plot transfer entropy between two series.
    
    Args:
        series_x: Source/driver series
        series_y: Target/response series
    """
    x = df[series_x].values
    y = df[series_y].values
    
    # Debug info
    print(f"[DEBUG] Computing TE between {series_x} and {series_y}")
    
    # Interpolate NaNs in x
    if np.isnan(x).any():
        idx = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() >= 2:
            x = np.interp(idx, idx[mask], x[mask])
        else:
            x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0)
    
    # Check y for NaNs (shouldn't have any for S_RT)
    if np.isnan(y).any():
        print(f"[WARNING] Target series {series_y} has NaNs, interpolating...")
        idx = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            y = np.interp(idx, idx[mask], y[mask])
    
    # Compute TE in both directions
    TE_x_to_y = transfer_entropy(x, y, n_bins=n_bins)
    TE_y_to_x = transfer_entropy(y, x, n_bins=n_bins)
    
    print(f"[DEBUG] TE({series_x}→{series_y}) = {TE_x_to_y:.4f} nats")
    print(f"[DEBUG] TE({series_y}→{series_x}) = {TE_y_to_x:.4f} nats")
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar([0,1], [TE_x_to_y, TE_y_to_x], color=['blue', 'orange'])
    plt.xticks([0,1], [f"{series_x}→{series_y}", f"{series_y}→{series_x}"])
    plt.ylabel("Transfer Entropy (nats)")
    plt.title("Directionality via Transfer Entropy")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    return float(TE_x_to_y), float(TE_y_to_x)
