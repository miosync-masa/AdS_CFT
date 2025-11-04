# PhaseShift‑X: Complete Edition
# ================================================================================
# Unified Calibration + Pre-driver Measurement + Full Causality Analysis
#
# Features:
#  - Phase 1: Calibration (burn-in) to reach steady state
#  - Phase 2: Measurement with pre/post λ recording
#  - Delayed geodesic gating (gate_delay=1, strength=0.15)
#  - Zero-lag suppression (c_eff capped at 0.18)
#  - Multi-objective RT functional (perimeter + holes + curvature)
#  - Full directionality analysis: Pearson, Spearman, Transfer Entropy
#
# Exports:
#  /mnt/data/metrics_calibration_X.csv   (Phase 1 snapshots)
#  /mnt/data/metrics_measurement_X.csv   (Phase 2 full data)
#  /mnt/data/calibration_phase_X.png
#  /mnt/data/rt_timeseries_X.png
#  /mnt/data/crosscorr_X.png
#  /mnt/data/transfer_entropy_X.png
#  /mnt/data/summary_report_X.txt
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

# Random generator
rng = np.random.default_rng(314)

# -------------------------
# Utilities
# -------------------------
def laplacian2d(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) +
        np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4*a
    )

def perimeter_len_bool8(mask: np.ndarray) -> float:
    m = mask.astype(np.int8)
    dx = np.abs(m - np.roll(m, -1, 1))
    dy = np.abs(m - np.roll(m, -1, 0))
    d1 = np.abs(m - np.roll(np.roll(m, -1, 0), -1, 1))
    d2 = np.abs(m - np.roll(np.roll(m, -1, 0),  1, 1))
    perim = dx.sum() + dy.sum() + (np.sqrt(0.5))*(d1.sum() + d2.sum())
    return float(perim)

def corner_count(mask: np.ndarray) -> int:
    A = mask.astype(np.int8)
    B = np.roll(A, -1, 0)
    C = np.roll(A, -1, 1)
    D = np.roll(np.roll(A, -1, 0), -1, 1)
    s = A + B + C + D
    return int(np.logical_or(s == 1, s == 3).sum())

def bfs_components_periodic(mask: np.ndarray) -> List[List[Tuple[int,int]]]:
    H, W = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int,int]]] = []
    for i in range(H):
        for j in range(W):
            if not mask[i,j] or vis[i,j]:
                continue
            q = [(i,j)]; vis[i,j]=True; comp=[(i,j)]
            while q:
                ci,cj = q.pop()
                for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = (ci+di)%H, (cj+dj)%W
                    if mask[ni,nj] and not vis[ni,nj]:
                        vis[ni,nj]=True
                        q.append((ni,nj)); comp.append((ni,nj))
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps

def k_th_largest_mask(mask: np.ndarray, k: int) -> np.ndarray:
    comps = bfs_components_periodic(mask)
    out = np.zeros_like(mask, dtype=bool)
    if len(comps) > k and len(comps[k])>0:
        for i,j in comps[k]:
            out[i,j]=True
    elif len(comps)>0:
        for i,j in comps[0]:
            out[i,j]=True
    return out

def dilate4(mask: np.ndarray, steps:int=1) -> np.ndarray:
    m = mask.copy()
    for _ in range(steps):
        nb = (np.roll(m,1,0) | np.roll(m,-1,0) | np.roll(m,1,1) | np.roll(m,-1,1))
        m = (m | nb)
    return m

def erode4(mask: np.ndarray, steps:int=1) -> np.ndarray:
    inv = np.logical_not(mask)
    inv_d = dilate4(inv, steps)
    return np.logical_not(inv_d)

def inner_boundary_band(mask: np.ndarray, width:int=1) -> np.ndarray:
    er = erode4(mask, steps=width)
    return (mask & (~er))

def outer_boundary_band(mask: np.ndarray, width:int=1) -> np.ndarray:
    dl = dilate4(mask, steps=width)
    return (dl & (~mask))

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
    p = np.zeros((n, n, n), dtype=float)
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

# -------------------------
# Agents & Automaton
# -------------------------
@dataclass
class Cell:
    alive: bool
    energy: float
    genome: np.ndarray
    coop: float

def init_grid(H=44, W=44, G=56) -> List[List[Cell]]:
    grid = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(Cell(
                alive=True,
                energy=float(1.0 + 0.2*rng.standard_normal()),
                genome=rng.integers(0,2,size=G,dtype=np.int8),
                coop=float(np.clip(0.5 + 0.2*rng.standard_normal(), 0, 1))
            ))
        grid.append(row)
    return grid

def mutate(genome: np.ndarray, mut_rate: float) -> np.ndarray:
    g = genome.copy()
    flips = rng.random(len(g)) < mut_rate
    g[flips] = 1 - g[flips]
    return g

class Automaton:
    def __init__(self, H=44, W=44, Z=24, L_ads=1.0, alpha=0.9,
                 gate_delay:int=1, gate_strength:float=0.15, c_eff_max:float=0.18):
        self.H,self.W,self.Z = H,W,Z
        self.L_ads=float(L_ads); self.alpha=float(alpha); self.G_N=1.0
        self.grid=init_grid(H,W,56)
        self.resource=np.ones((H,W),dtype=float)
        self.boundary=np.zeros((H,W),dtype=float)
        self.bulk=np.zeros((H,W,Z),dtype=float)
        self.c0=0.08; self.gamma=0.8; self.c_eff_max=c_eff_max
        self.theta0=0.19
        self.gate_delay = gate_delay
        self.gate_strength = gate_strength
        self.pending_masks: List[Optional[np.ndarray]] = [None]*(self.gate_delay+1)

        # seed coop pattern
        for i in range(H):
            for j in range(W):
                if ((i//4 + j//4) % 2)==0:
                    self.grid[i][j].coop=float(np.clip(self.grid[i][j].coop*1.25,0,1))
                else:
                    self.grid[i][j].coop=float(np.clip(self.grid[i][j].coop*0.75,0,1))
        cy,cx = H//2, W//2
        for i in range(H):
            for j in range(W):
                r2=(i-cy)**2+(j-cx)**2
                boost=np.exp(-r2/(H/4)**2)
                self.grid[i][j].coop=float(np.clip(self.grid[i][j].coop*(1+0.4*boost),0,1))

    def coop_field(self) -> np.ndarray:
        M = np.zeros((self.H,self.W),dtype=float)
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i][j].alive:
                    M[i,j]=self.grid[i][j].coop
        return M

    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, 1) - coop
        gy = np.roll(coop, -1, 0) - coop
        K = np.sqrt(gx*gx + gy*gy) + 1e-12
        V = np.abs(coop - float(coop.mean())) + 1e-12
        return K / V

    def step_agents(self):
        H,W = self.H,self.W
        C = self.coop_field()
        for i in range(H):
            for j in range(W):
                c = self.grid[i][j]
                if not c.alive: 
                    continue
                take = min(0.05, self.resource[i,j]); self.resource[i,j]-=take; c.energy += take
                give = min(0.02*c.coop, c.energy)
                if give>0:
                    q=0.25*give
                    self.grid[(i+1)%H][j].energy += q
                    self.grid[(i-1)%H][j].energy += q
                    self.grid[i][(j+1)%W].energy += q
                    self.grid[i][(j-1)%W].energy += q
                    c.energy -= give
                neigh = 0.25*(C[(i+1)%H][j]+C[(i-1)%H][j]+C[i][(j+1)%W]+C[i][(j-1)%W])
                c.coop += 0.05*(neigh - c.coop) + 0.01*rng.standard_normal()
                c.coop = float(np.clip(c.coop, 0, 1))
                if c.energy < 0.15:
                    c.alive=False; c.coop=0.0
                elif c.energy > 2.0 and rng.random()<0.01:
                    child = Cell(True, c.energy*0.5, mutate(c.genome,0.02), c.coop)
                    c.energy *= 0.5
                    ii,jj = (i + rng.integers(-1,2))%H, (j + rng.integers(-1,2))%W
                    self.grid[ii][jj] = child
        self.resource += 0.03*laplacian2d(self.resource) + 0.004
        self.resource = np.clip(self.resource, 0, 3)

    def update_bulk(self):
        L0 = self.K_over_V(self.boundary)
        self.bulk[...,0] = L0
        dz = 1.0/self.Z; z = (np.arange(self.Z)+1)*dz
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k])**2
            prev = self.bulk[...,k-1]
            mixed = prev + 0.012*laplacian2d(prev)
            self.bulk[...,k] = np.clip(warp * mixed, 0, None)

    def HR(self, c_eff: float):
        dz = 1.0/self.Z; z = (np.arange(self.Z)+1)*dz
        w = np.exp(-self.alpha * z); w = w / (w.sum() + 1e-12)
        back = np.tensordot(self.bulk, w, axes=(2,0))
        self.boundary += c_eff * (back - self.boundary)

    def update_boundary_payoff(self):
        B = self.boundary
        neigh = 0.25*(np.roll(B,1,0)+np.roll(B,-1,0)+np.roll(B,1,1)+np.roll(B,-1,1))
        payoff = neigh*(1.4 - 0.6*B) - 0.4*B*(1 - neigh)
        self.boundary += 0.08 * payoff
        self.boundary = np.clip(self.boundary, 0, 1)

    def SOC_tune(self):
        Lb = self.K_over_V(self.coop_field())
        delta = float(Lb.mean() - 1.0)
        self.boundary = np.clip(self.boundary - 0.01*delta, 0, 1)

    def region_A(self, step:int) -> np.ndarray:
        C = self.coop_field(); alive = C>0
        if alive.sum()==0:
            R = np.zeros((self.H,self.W),bool); R[:, :self.W//2]=True; return R
        thr = float(np.median(C[alive]))
        high = C>=thr
        k = 0 if (step%2==0) else 1
        R = k_th_largest_mask(high, k)
        if R.sum()==0:
            R = k_th_largest_mask(high, 0)
        return R

    def count_holes_nonperiodic(self, mask: np.ndarray) -> int:
        H,W = mask.shape
        inv = (~mask).astype(np.uint8)
        vis = np.zeros_like(inv, dtype=bool)
        q = deque()
        for i in range(H):
            if inv[i,0]: q.append((i,0)); vis[i,0]=True
            if inv[i,W-1]: q.append((i,W-1)); vis[i,W-1]=True
        for j in range(W):
            if inv[0,j]: q.append((0,j)); vis[0,j]=True
            if inv[H-1,j]: q.append((H-1,j)); vis[H-1,j]=True
        while q:
            ci,cj = q.popleft()
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ni, nj = ci+di, cj+dj
                if 0<=ni<H and 0<=nj<W and inv[ni,nj] and not vis[ni,nj]:
                    vis[ni,nj]=True; q.append((ni,nj))
        holes = 0
        for i in range(H):
            for j in range(W):
                if inv[i,j] and not vis[i,j]:
                    holes += 1
                    q = deque([(i,j)]); vis[i,j]=True
                    while q:
                        ci,cj = q.popleft()
                        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                            ni, nj = ci+di, cj+dj
                            if 0<=ni<H and 0<=nj<W and inv[ni,nj] and not vis[ni,nj]:
                                vis[ni,nj]=True; q.append((ni,nj))
        return holes

    def S_RT_multiobjective(self, R: np.ndarray, w_len=1.0, w_hole=2.0, w_curv=0.5) -> Tuple[float, Dict[str,float]]:
        perim = perimeter_len_bool8(R)
        holes = self.count_holes_nonperiodic(R)
        curv  = float(corner_count(R))
        S = (w_len*perim + w_hole*holes + w_curv*curv) / (4.0 * self.G_N)
        return float(S), dict(perimeter=float(perim), holes=float(holes), curvature=float(curv))

    def compute_gate_mask(self, R: np.ndarray, Lb_pre: np.ndarray) -> Optional[np.ndarray]:
        out = outer_boundary_band(R, 1)
        if not out.any():
            return None
        vals = Lb_pre[out]
        if len(vals) < 10:
            return None
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1 + 1e-12
        thresh = q3 + 1.5*iqr
        lam_p99 = np.percentile(vals, 99)
        if lam_p99 > thresh:
            return out.copy()
        return None

    def apply_pending_gate(self):
        mask = self.pending_masks.pop(0)
        self.pending_masks.append(None)
        if mask is not None and mask.any():
            B = self.boundary
            B[mask] = np.clip(B[mask] + self.gate_strength*(1.0 - B[mask]), 0, 1)
            self.boundary = B
            return int(mask.sum())
        return 0

    def step(self, step:int) -> Dict:
        """Full step with pre/post λ recording"""
        self.step_agents()
        self.boundary = self.coop_field()
        self.update_bulk()

        # Region & PRE-update λ (driver)
        R = self.region_A(step)
        Lb_pre = self.K_over_V(self.boundary)
        out_band = outer_boundary_band(R, 1)
        in_band = inner_boundary_band(R, 1)

        lam_p99_out_pre = float(np.percentile(Lb_pre[out_band], 99)) if out_band.any() else np.nan
        lam_p99_in_pre = float(np.percentile(Lb_pre[in_band], 99)) if in_band.any() else np.nan

        # Dynamic c_eff
        medG = float(np.median(Lb_pre))
        if out_band.any():
            norm_tail = lam_p99_out_pre/(medG+1e-12) - 1.0
        else:
            norm_tail = 0.0
        norm_tail = float(np.clip(norm_tail, -0.5, 3.0))
        c_eff = float(np.clip(self.c0*(1.0 + self.gamma*norm_tail), 0.08, self.c_eff_max))

        # HR update (boundary changes here!)
        self.HR(c_eff)

        # Delayed gate
        applied_px = self.apply_pending_gate()
        new_mask = self.compute_gate_mask(R, Lb_pre)
        if new_mask is not None:
            self.pending_masks[-1] = new_mask

        # Boundary payoff & SOC
        self.update_boundary_payoff()
        self.SOC_tune()

        # POST-update λ (response)
        Lb_post = self.K_over_V(self.boundary)
        lam_p99_out_post = float(np.percentile(Lb_post[out_band], 99)) if out_band.any() else np.nan
        lam_p99_in_post = float(np.percentile(Lb_post[in_band], 99)) if in_band.any() else np.nan

        # RT entropy
        S_mo, parts = self.S_RT_multiobjective(R)

        return dict(
            t=step,
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

# -------------------------
# Analysis Functions
# -------------------------
def best_lag_corr(a: np.ndarray, b: np.ndarray, maxlag=16) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    bb = b.copy()
    if np.isnan(bb).any():
        idx = np.arange(len(bb))
        m = ~np.isnan(bb)
        if m.sum() >= 2:
            bb = np.interp(idx, idx[m], bb[m])
        else:
            bb = np.nan_to_num(bb, nan=np.nanmean(bb) if np.isfinite(np.nanmean(bb)) else 0.0)
    lags, corrs = crosscorr_at_lags(a, bb, maxlag=maxlag)
    best_idx = int(np.nanargmax(corrs))
    return int(lags[best_idx]), float(corrs[best_idx]), lags, corrs, bb

def shift_for_lag(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag == 0:
        return a, b
    elif lag > 0:
        return a[lag:], b[:-lag]
    else:
        return a[:lag], b[-lag:]

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PhaseShift-X: Complete Edition")
    print("Calibration + Pre-driver Measurement + Full Causality Analysis")
    print("=" * 70)
    
    # Initialize
    ua = Automaton(H=44, W=44, Z=24, L_ads=1.0, alpha=0.9,
                   gate_delay=1, gate_strength=0.15, c_eff_max=0.18)
    ua.boundary = ua.coop_field()
    
    # ----- Phase 1: Calibration (Burn-in) -----
    T_burn = 280
    print(f"\n[Phase 1] Calibration: {T_burn} steps (reaching steady state)...")
    calibration_records = []
    
    for t in range(1, T_burn+1):
        rec = ua.step(t)
        if t % 10 == 0:
            calibration_records.append({
                't': t,
                'phase': 'calibration',
                'entropy_RT_mo': rec['entropy_RT_mo'],
                'region_A_size': rec['region_A_size'],
                'c_eff': rec['c_eff']
            })
            if t % 30 == 0:
                print(f"  t={t:3d}: S_RT={rec['entropy_RT_mo']:.3f}, region_size={rec['region_A_size']:.0f}")
    
    df_calib = pd.DataFrame(calibration_records)
    df_calib.to_csv("/content/metrics_calibration_X.csv", index=False)
    
    # ----- Phase 2: Measurement (Pre-driver recording) -----
    T_meas = 280
    print(f"\n[Phase 2] Measurement: {T_meas} steps (recording pre/post λ)...")
    measurement_records = []
    prev_S = None
    
    for t in range(T_burn+1, T_burn+T_meas+1):
        rec = ua.step(t)
        
        # dS calculation
        if prev_S is None:
            rec["dS_RT_mo"] = 0.0
        else:
            rec["dS_RT_mo"] = rec["entropy_RT_mo"] - prev_S
        prev_S = rec["entropy_RT_mo"]
        
        rec['phase'] = 'measurement'
        measurement_records.append(rec)
        
        if (t - T_burn) % 50 == 0:
            print(f"  t={t:3d}: S_RT={rec['entropy_RT_mo']:.3f}, λ_pre={rec['lambda_p99_A_out_pre']:.3f}")
    
    df_meas = pd.DataFrame(measurement_records)
    df_meas.to_csv("/content/metrics_measurement_X.csv", index=False)
    
    # ----- Phase 3: Causality Analysis -----
    print(f"\n[Phase 3] Causality Analysis...")
    
    S = df_meas["entropy_RT_mo"].values
    lam_pre = df_meas["lambda_p99_A_out_pre"].values
    
    # Pearson cross-correlation
    best_lag, maxcorr, lags, corrs, lam_filled = best_lag_corr(S, lam_pre, maxlag=16)
    
    # Spearman at best lag
    S_lag, L_lag = shift_for_lag(S, lam_filled, best_lag)
    rho_s = spearman_corr(S_lag, L_lag) if len(S_lag) > 3 else np.nan
    
    # Transfer Entropy (both directions)
    TE_L_to_S = transfer_entropy(lam_filled, S, n_bins=3)
    TE_S_to_L = transfer_entropy(S, lam_filled, n_bins=3)
    
    # Summary report
    with open("/content/summary_report_X.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PhaseShift-X: Complete Causality Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Calibration phase: {T_burn} steps\n")
        f.write(f"Measurement phase: {T_meas} steps\n")
        f.write(f"Total steps: {T_burn + T_meas}\n\n")
        f.write("--- Configuration ---\n")
        f.write(f"Gate delay: 1 step\n")
        f.write(f"Gate strength: 0.15\n")
        f.write(f"c_eff cap: 0.18\n")
        f.write(f"RT functional: perimeter + holes (Euler) + curvature\n\n")
        f.write("--- Directionality Metrics ---\n")
        f.write(f"Best Pearson corr (S vs λ_pre): {maxcorr:.4f} at lag {best_lag}\n")
        f.write(f"Spearman at best lag: {rho_s:.4f}\n")
        f.write(f"Transfer Entropy (λ_pre → S): {TE_L_to_S:.4f} nats\n")
        f.write(f"Transfer Entropy (S → λ_pre): {TE_S_to_L:.4f} nats\n\n")
        f.write("--- Interpretation ---\n")
        if TE_S_to_L > TE_L_to_S:
            f.write(f"✓ Dominant causality: S → λ_pre (ΔTE = {TE_S_to_L - TE_L_to_S:.4f})\n")
            f.write("  Entropy drives boundary structure changes.\n")
        else:
            f.write(f"✓ Dominant causality: λ_pre → S (ΔTE = {TE_L_to_S - TE_S_to_L:.4f})\n")
            f.write("  Boundary structure drives entropy changes.\n")
        f.write("\n" + "=" * 70 + "\n")
    
    # ----- Visualizations -----
    print(f"\n[Phase 4] Generating visualizations...")
    
    # 1. Calibration phase
    plt.figure(figsize=(10, 4))
    plt.plot(df_calib['t'], df_calib['entropy_RT_mo'], 'o-', alpha=0.7, color='steelblue')
    plt.axvline(T_burn, color='red', linestyle='--', linewidth=2, label='Measurement start')
    plt.xlabel('Time step', fontsize=12)
    plt.ylabel('S_RT (multi-objective)', fontsize=12)
    plt.title('Phase 1: Calibration — System Equilibration', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/calibration_phase_X.png', dpi=150)
    plt.close()
    
    # 2. RT entropy time series (measurement phase)
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(df_meas['t'], df_meas['entropy_RT_mo'], linewidth=1.5, color='darkgreen')
    plt.ylabel('S_RT', fontsize=11)
    plt.title('Phase 2: Measurement — RT Entropy & Pre-driver λ', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_meas['t'], df_meas['lambda_p99_A_out_pre'], linewidth=1.5, color='darkorange')
    plt.xlabel('Time step', fontsize=12)
    plt.ylabel('λ_p99 (pre)', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/rt_timeseries_X.png', dpi=150)
    plt.close()
    
    # 3. Cross-correlation
    plt.figure(figsize=(10, 5))
    plt.plot(lags, corrs, 'o-', linewidth=2, markersize=6, color='navy')
    plt.axvline(best_lag, color='red', linestyle='--', linewidth=2, label=f'Best lag = {best_lag}')
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    plt.xlabel('Lag (steps)', fontsize=12)
    plt.ylabel('Pearson correlation', fontsize=12)
    plt.title('Cross-correlation: S_RT vs λ_p99_A_out_pre', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/crosscorr_X.png', dpi=150)
    plt.close()
    
    # 4. Transfer Entropy
    plt.figure(figsize=(8, 6))
    colors = ['#2E86AB' if TE_L_to_S > TE_S_to_L else '#A23B72', 
              '#A23B72' if TE_S_to_L > TE_L_to_S else '#2E86AB']
    bars = plt.bar([0, 1], [TE_L_to_S, TE_S_to_L], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xticks([0, 1], ['λ_pre → S', 'S → λ_pre'], fontsize=12)
    plt.ylabel('Transfer Entropy (nats)', fontsize=12)
    plt.title('Directionality via Transfer Entropy', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, [TE_L_to_S, TE_S_to_L])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/content/transfer_entropy_X.png', dpi=150)
    plt.close()
    


    # ----- Final summary -----
    print("\n" + "=" * 70)
    print("COMPLETE! Summary:")
    print("=" * 70)
    print(f"Calibration: {T_burn} steps → Measurement: {T_meas} steps")
    print(f"Best Pearson corr: {maxcorr:.4f} at lag {best_lag}")
    print(f"Spearman (at best lag): {rho_s:.4f}")
    print(f"TE λ_pre→S: {TE_L_to_S:.4f} nats")
    print(f"TE S→λ_pre: {TE_S_to_L:.4f} nats")
    
    if TE_S_to_L > TE_L_to_S:
        print(f"\n✓ Dominant causality: S → λ_pre (ΔTE = {TE_S_to_L - TE_L_to_S:.4f})")
        print("  → Entropy DRIVES boundary structure changes!")
    else:
        print(f"\n✓ Dominant causality: λ_pre → S (ΔTE = {TE_L_to_S - TE_S_to_L:.4f})")
        print("  → Boundary structure DRIVES entropy changes!")
    
    print("\n" + "=" * 70)
    print("Exported files:")
    print("  • /content/metrics_calibration_X.csv")
    print("  • /content/metrics_measurement_X.csv")
    print("  • /content/calibration_phase_X.png")
    print("  • /content/rt_timeseries_X.png")
    print("  • /content/crosscorr_X.png")
    print("  • /content/transfer_entropy_X.png")
    print("  • /content/summary_report_X.txt")
    print("=" * 70 + "\n")
