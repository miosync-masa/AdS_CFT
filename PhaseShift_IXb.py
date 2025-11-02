# PhaseShift‑IXb — Align driver/response (use pre‑update λ for causality)
# ---------------------------------------------------------------------
# Record λ_p99_A_out both before (driver) and after (response) updates, and
# recompute directionality vs S_RT_mo using the *pre* metric.
#
# Exports:
#  /mnt/data/metrics_phaseshift9b.csv
#  /mnt/data/crosscorr_out_pre_phaseshift9b.png
#  /mnt/data/transfer_entropy_phaseshift9b.png
#  /mnt/data/summary_report_phaseshift9b.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse Automaton and utilities from prior cell (already defined in kernel)

ua2 = Automaton(H=44, W=44, Z=24, L_ads=1.0, alpha=0.9,
                gate_delay=1, gate_strength=0.15, c_eff_max=0.18)
ua2.boundary = ua2.coop_field()

T = 280
rows=[]
prev=None
for t in range(1, T+1):
    # Manually inline step to capture pre/post λ
    ua2.step_agents()
    ua2.boundary = ua2.coop_field()
    ua2.update_bulk()
    R = ua2.region_A(t)
    Lb_pre = ua2.K_over_V(ua2.boundary)
    out_band = outer_boundary_band(R, 1)
    in_band  = inner_boundary_band(R, 1)
    lam_p99_out_pre = float(np.percentile(Lb_pre[out_band], 99)) if out_band.any() else np.nan
    lam_p99_in_pre  = float(np.percentile(Lb_pre[in_band],  99)) if in_band.any() else np.nan

    medG = float(np.median(Lb_pre))
    if out_band.any():
        norm_tail = lam_p99_out_pre/(medG+1e-12) - 1.0
    else:
        norm_tail = 0.0
    norm_tail = float(np.clip(norm_tail, -0.5, 3.0))
    c_eff = float(np.clip(ua2.c0*(1.0 + ua2.gamma*norm_tail), 0.08, ua2.c_eff_max))

    ua2.HR(c_eff)
    applied_px = ua2.apply_pending_gate()
    new_mask = ua2.compute_gate_mask(R, Lb_pre)
    if new_mask is not None:
        ua2.pending_masks[-1] = new_mask
    ua2.update_boundary_payoff()
    ua2.SOC_tune()

    # post measures
    S_mo, parts = ua2.S_RT_multiobjective(R)
    Lb_post = ua2.K_over_V(ua2.boundary)
    lam_p99_out_post = float(np.percentile(Lb_post[out_band],99)) if out_band.any() else np.nan
    lam_p99_in_post  = float(np.percentile(Lb_post[in_band], 99)) if in_band.any() else np.nan

    rec = dict(
        t=t,
        entropy_RT_mo=S_mo,
        region_A_size=float(R.sum()),
        region_A_perimeter=parts["perimeter"],
        region_A_holes=parts["holes"],
        region_A_curvature=parts["curvature"],
        lambda_p99_A_out_pre=lam_p99_out_pre,
        lambda_p99_A_in_pre=lam_p99_in_pre,
        lambda_p99_A_out_post=lam_p99_out_post,
        lambda_p99_A_in_post=lam_p99_in_post,
        c_eff=c_eff,
        gate_applied_px=int(applied_px)
    )
    if prev is None:
        rec["dS_RT_mo"]=0.0
    else:
        rec["dS_RT_mo"]=S_mo-prev
    prev=S_mo
    rows.append(rec)

dfb = pd.DataFrame(rows)
csv_path = "/mnt/data/metrics_phaseshift9b.csv"
dfb.to_csv(csv_path, index=False)

def crosscorr_at_lags(a: np.ndarray, b: np.ndarray, maxlag:int=16):
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
        if len(aa)>1:
            corrs.append(float(np.mean(aa*bb)))
        else:
            corrs.append(np.nan)
    return lags, np.array(corrs, dtype=float)

def best_lag_corr(a: np.ndarray, b: np.ndarray, maxlag=16):
    bb = b.copy()
    if np.isnan(bb).any():
        idx = np.arange(len(bb))
        m = ~np.isnan(bb)
        if m.sum()>=2: bb = np.interp(idx, idx[m], bb[m])
        else: bb = np.nan_to_num(bb, nan=np.nanmean(bb) if np.isfinite(np.nanmean(bb)) else 0.0)
    lags, corrs = crosscorr_at_lags(a, bb, maxlag=maxlag)
    best_idx = int(np.nanargmax(corrs))
    return int(lags[best_idx]), float(corrs[best_idx]), lags, corrs, bb

S = dfb["entropy_RT_mo"].values
lam_out_pre = dfb["lambda_p99_A_out_pre"].values

best_lag, maxcorr, lags, corrs, lam_out_pre_filled = best_lag_corr(S, lam_out_pre, maxlag=16)

# Spearman at best lag
def shift_for_lag(a: np.ndarray, b: np.ndarray, lag: int):
    if lag == 0:
        return a, b
    elif lag > 0:
        return a[lag:], b[:-lag]
    else:
        return a[:lag], b[-lag:]

from math import isnan
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
    xr = rankdata(x); yr = rankdata(y)
    xr = (xr - xr.mean())/(xr.std()+1e-12)
    yr = (yr - yr.mean())/(yr.std()+1e-12)
    return float(np.mean(xr*yr))

S_l, L_l = shift_for_lag(S, lam_out_pre_filled, best_lag)
rho_s = spearman_corr(S_l, L_l) if len(S_l)>3 else np.nan

# TE with pre‑λ as driver
def discretize_quantiles(x: np.ndarray, n_bins:int=3):
    qs = np.linspace(0,1,n_bins+1); qs[0],qs[-1]=0.0,1.0
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges)<=2:
        mn,mx=float(np.min(x)), float(np.max(x)+1e-12)
        edges = np.linspace(mn,mx,n_bins+1)
    d = np.digitize(x, edges[1:-1], right=False)
    return d.astype(np.int32)

def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins:int=3) -> float:
    xq = discretize_quantiles(x, n_bins)
    yq = discretize_quantiles(y, n_bins)
    yt = yq[:-1]; xt = xq[:-1]; y1 = yq[1:]
    n = n_bins
    p = np.zeros((n,n,n), float)
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
                if pij<=0: continue
                te += pij * np.log((p[a,b,c]/p_yt_xt[b,c]) / (p_y1_yt[a,b]/p_yt[b]))
    return float(te)

TE_L_to_S = transfer_entropy(lam_out_pre_filled, S, n_bins=3)
TE_S_to_L = transfer_entropy(S, lam_out_pre_filled, n_bins=3)

with open("/mnt/data/summary_report_phaseshift9b.txt","w") as f:
    f.write(f"[PRE driver] Best Pearson cross-corr (S_RT_mo vs λ_p99_A_out_pre): {maxcorr:.3f} at lag {best_lag}\n")
    f.write(f"Spearman at best lag: {rho_s:.3f}\n")
    f.write(f"Transfer Entropy (λ_pre→S): {TE_L_to_S:.4f} nats\n")
    f.write(f"Transfer Entropy (S→λ_pre): {TE_S_to_L:.4f} nats\n")

plt.figure()
plt.plot(lags, corrs, marker='o')
plt.xlabel("lag (steps)"); plt.ylabel("Pearson corr")
plt.title("Cross-correlation S_RT_mo vs λ_p99_A_out_pre — PhaseShift‑IXb")
plt.tight_layout(); plt.savefig("/mnt/data/crosscorr_out_pre_phaseshift9b.png"); plt.close()

plt.figure()
plt.bar([0,1], [TE_L_to_S, TE_S_to_L])
plt.xticks([0,1], ["λ_pre→S", "S→λ_pre"])
plt.ylabel("Transfer Entropy (nats)")
plt.title("Directionality via Transfer Entropy (pre‑λ as driver) — PhaseShift‑IXb")
plt.tight_layout(); plt.savefig("/mnt/data/transfer_entropy_phaseshift9b.png"); plt.close()

print(f"[PRE driver] Best Pearson corr = {maxcorr:.3f} at lag {best_lag}")
print(f"Spearman (at best lag) = {rho_s:.3f}")
print(f"TE λ_pre→S = {TE_L_to_S:.4f} nats,  TE S→λ_pre = {TE_S_to_L:.4f} nats")
print(csv_path)
