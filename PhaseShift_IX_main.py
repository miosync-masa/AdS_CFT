# PhaseShift‑IX — Final Causality Proof (delayed gating + capped coupling + TE)
# ---------------------------------------------------------------------------
# Implements the user's four requests in one run:
#  1) Delayed geodesic gating: gate_delay=1 (apply at t+1), gate strength=0.15
#  2) Zero‑lag suppression: c_eff upper bound = 0.18
#  3) Directionality metrics: Pearson (lag scan), Spearman (at best lag), Transfer Entropy
#  4) Multi‑objective RT functional: perimeter + holes (Euler) + curvature (corner count)
#
# Exports:
#  /mnt/data/metrics_phaseshift9.csv
#  /mnt/data/rt_mo_timeseries_phaseshift9.png
#  /mnt/data/crosscorr_out_phaseshift9.png
#  /mnt/data/transfer_entropy_phaseshift9.png
#  /mnt/data/summary_report_phaseshift9.txt
#
# Notes:
#  - Pure NumPy/Pandas/Matplotlib (no SciPy). BFS utilities coded inline.
#  - Periodic BC for dynamics; non‑periodic for hole counting (Euler inside/outside).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

rng = np.random.default_rng(913)

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
    H, W = mask.shape
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
    # Spearman via average ranks
    temp = a.argsort(kind='mergesort')
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a))
    # handle ties: average same values
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = cum - counts
    avg = (starts + cum - 1)/2.0
    ranks = avg[inv]
    return ranks

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(x)
    yr = rankdata(y)
    xr = (xr - xr.mean())/(xr.std()+1e-12)
    yr = (yr - yr.mean())/(yr.std()+1e-12)
    return float(np.mean(xr*yr))

# ---- Transfer Entropy (discrete, k=l=1, quantized bins) ----
def discretize_quantiles(x: np.ndarray, n_bins:int=3) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins+1)
    qs[0], qs[-1] = 0.0, 1.0
    edges = np.quantile(x, qs)
    # ensure unique edges
    edges = np.unique(edges)
    if len(edges) <= 2:
        # fall back to equal width
        mn, mx = float(np.min(x)), float(np.max(x)+1e-12)
        edges = np.linspace(mn, mx, n_bins+1)
    d = np.digitize(x, edges[1:-1], right=False)
    return d.astype(np.int32)

def transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins:int=3) -> float:
    # TE X->Y with history length 1: T = sum p(y_{t+1}, y_t, x_t) log p(y_{t+1}|y_t,x_t)/p(y_{t+1}|y_t)
    xq = discretize_quantiles(x, n_bins)
    yq = discretize_quantiles(y, n_bins)
    # align triplets
    yt = yq[:-1]
    xt = xq[:-1]
    y1 = yq[1:]
    # joint histogram counts
    n = n_bins
    p = np.zeros((n, n, n), dtype=float)  # y1, yt, xt
    for i in range(len(y1)):
        p[y1[i], yt[i], xt[i]] += 1.0
    p /= max(1.0, p.sum())
    # marginals
    p_yt_xt = p.sum(axis=0) + 1e-12
    p_y1_yt = p.sum(axis=2) + 1e-12
    p_yt    = p_yt_xt.sum(axis=1) + 1e-12
    # TE
    te = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                pij = p[a,b,c]
                if pij <= 0: 
                    continue
                te += pij * np.log((p[a,b,c]/p_yt_xt[b,c]) / (p_y1_yt[a,b]/p_yt[b]))
    return float(te)  # nats

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
        self.pending_masks: List[Optional[np.ndarray]] = [None]*(self.gate_delay+1)  # circular queue

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

    # ----- dynamics -----
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

    # ----- geometry (A region & multi‑objective RT) -----
    def region_A(self, step:int) -> np.ndarray:
        C = self.coop_field(); alive = C>0
        if alive.sum()==0:
            R = np.zeros((self.H,self.W),bool); R[:, :self.W//2]=True; return R
        thr = float(np.median(C[alive]))
        high = C>=thr
        k = 0 if (step%2==0) else 1  # alternate largest/second largest
        R = k_th_largest_mask(high, k)
        if R.sum()==0:
            R = k_th_largest_mask(high, 0)
        return R

    def count_holes_nonperiodic(self, mask: np.ndarray) -> int:
        # Flood fill on complement without wrap‑around; components not touching border are holes.
        H,W = mask.shape
        inv = (~mask).astype(np.uint8)
        vis = np.zeros_like(inv, dtype=bool)
        from collections import deque
        q = deque()
        # mark border background as outside
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
        # Remaining unvisited background components are holes
        holes = 0
        for i in range(H):
            for j in range(W):
                if inv[i,j] and not vis[i,j]:
                    holes += 1
                    # flood this hole
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

    # ----- delayed geodesic gating -----
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
        # pop next pending mask; apply to boundary_state: B <- B + s*(1-B)
        mask = self.pending_masks.pop(0)
        self.pending_masks.append(None)
        if mask is not None and mask.any():
            B = self.boundary
            B[mask] = np.clip(B[mask] + self.gate_strength*(1.0 - B[mask]), 0, 1)
            self.boundary = B
            return int(mask.sum())
        return 0

    # ----- one step -----
    def step(self, step:int) -> Dict:
        self.step_agents()
        self.boundary = self.coop_field()
        self.update_bulk()

        # region and pre‑Lambda for dynamic coupling + future gate
        R = self.region_A(step)
        Lb_pre = self.K_over_V(self.boundary)
        out_band = outer_boundary_band(R, 1)

        # dynamic c_eff (capped at 0.18), based on outer tail vs global median
        medG = float(np.median(Lb_pre))
        if out_band.any():
            lam_p99_out = float(np.percentile(Lb_pre[out_band], 99))
            norm_tail = lam_p99_out/(medG+1e-12) - 1.0
        else:
            norm_tail = 0.0
        norm_tail = float(np.clip(norm_tail, -0.5, 3.0))
        c_eff = float(np.clip(self.c0*(1.0 + self.gamma*norm_tail), 0.08, self.c_eff_max))

        # HR update
        self.HR(c_eff)

        # delayed gate: apply mask scheduled earlier
        applied_px = self.apply_pending_gate()

        # schedule a new mask for (t+1)
        new_mask = self.compute_gate_mask(R, Lb_pre)
        if new_mask is not None:
            self.pending_masks[-1] = new_mask  # will be applied next step

        # boundary payoff & SOC
        self.update_boundary_payoff()
        self.SOC_tune()

        # measures
        S_mo, parts = self.S_RT_multiobjective(R)
        Lb = self.K_over_V(self.boundary)
        inb  = inner_boundary_band(R,1); outb = outer_boundary_band(R,1)
        lam_p99_in  = float(np.percentile(Lb[inb], 99)) if inb.any() else np.nan
        lam_p99_out = float(np.percentile(Lb[outb],99)) if outb.any() else np.nan
        return dict(
            t=step,
            entropy_RT_mo=S_mo,
            region_A_size=float(R.sum()),
            region_A_perimeter=parts["perimeter"],
            region_A_holes=parts["holes"],
            region_A_curvature=parts["curvature"],
            lambda_p99_A_out=lam_p99_out,
            lambda_p99_A_in=lam_p99_in,
            c_eff=c_eff,
            gate_applied_px=int(applied_px)
        )

# -------------------------
# Run and evaluate
# -------------------------
ua = Automaton(H=44, W=44, Z=24, L_ads=1.0, alpha=0.9,
               gate_delay=1, gate_strength=0.15, c_eff_max=0.18)
ua.boundary = ua.coop_field()

T = 280
rows=[]; prev=None
for t in range(1, T+1):
    rec = ua.step(t)
    if prev is None:
        dS = 0.0
    else:
        dS = rec["entropy_RT_mo"] - prev
    prev = rec["entropy_RT_mo"]
    rec["dS_RT_mo"] = dS
    rows.append(rec)

df = pd.DataFrame(rows)
csv_path = "/mnt/data/metrics_phaseshift9.csv"
df.to_csv(csv_path, index=False)

# Cross-correlation (Pearson) S_RT_mo vs λ_p99_A_out (±16)
def best_lag_corr(x: np.ndarray, y: np.ndarray, maxlag=16) -> Tuple[int, float, np.ndarray, np.ndarray]:
    yy = y.copy()
    if np.isnan(yy).any():
        xidx = np.arange(len(yy))
        mask = ~np.isnan(yy)
        if mask.sum() >= 2:
            yy = np.interp(xidx, xidx[mask], yy[mask])
        else:
            yy = np.nan_to_num(yy, nan=np.nanmean(yy) if np.isfinite(np.nanmean(yy)) else 0.0)
    lags, corrs = crosscorr_at_lags(x, yy, maxlag=maxlag)
    best_idx = int(np.nanargmax(corrs))
    return int(lags[best_idx]), float(corrs[best_idx]), lags, corrs, yy

S = df["entropy_RT_mo"].values
lam_out = df["lambda_p99_A_out"].values

best_lag, maxcorr, lags, corrs, lam_out_filled = best_lag_corr(S, lam_out, maxlag=16)

# Spearman at best lag (shift lam_out accordingly to align causality)
def shift_for_lag(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag == 0:
        return a, b
    elif lag > 0:
        return a[lag:], b[:-lag]
    else:
        return a[:lag], b[-lag:]

S_lag, lam_lag = shift_for_lag(S, lam_out_filled, best_lag)
rho_s = spearman_corr(S_lag, lam_lag) if len(S_lag)>3 else np.nan

# Transfer Entropy (nats) — both directions
# Use same aligned windows (drop first to define y_{t+1})
S_te  = S.copy()
L_te  = lam_out_filled.copy()
TE_L_to_S = transfer_entropy(L_te, S_te, n_bins=3)  # λ→S
TE_S_to_L = transfer_entropy(S_te, L_te, n_bins=3)  # S→λ

# Save summary
with open("/mnt/data/summary_report_phaseshift9.txt","w") as f:
    f.write(f"Best Pearson cross-corr (S_RT_mo vs λ_p99_A_out): {maxcorr:.3f} at lag {best_lag}\n")
    f.write(f"Spearman at best lag: {rho_s:.3f}\n")
    f.write(f"Transfer Entropy (λ→S): {TE_L_to_S:.4f} nats\n")
    f.write(f"Transfer Entropy (S→λ): {TE_S_to_L:.4f} nats\n")

# Plots
plt.figure()
plt.plot(df["t"], df["entropy_RT_mo"])
plt.xlabel("t"); plt.ylabel("S_RT (multi-objective)")
plt.title("RT-like Entropy (Perimeter + Holes + Curvature) — PhaseShift‑IX")
plt.tight_layout(); plt.savefig("/mnt/data/rt_mo_timeseries_phaseshift9.png"); plt.close()

plt.figure()
plt.plot(lags, corrs, marker='o')
plt.xlabel("lag (steps)"); plt.ylabel("Pearson corr")
plt.title("Cross-correlation S_RT_mo vs λ_p99_A_out — PhaseShift‑IX")
plt.tight_layout(); plt.savefig("/mnt/data/crosscorr_out_phaseshift9.png"); plt.close()

plt.figure()
plt.bar([0,1], [TE_L_to_S, TE_S_to_L])
plt.xticks([0,1], ["λ→S", "S→λ"])
plt.ylabel("Transfer Entropy (nats)")
plt.title("Directionality via Transfer Entropy — PhaseShift‑IX")
plt.tight_layout(); plt.savefig("/mnt/data/transfer_entropy_phaseshift9.png"); plt.close()

print(f"Best Pearson corr = {maxcorr:.3f} at lag {best_lag}")
print(f"Spearman (at best lag) = {rho_s:.3f}")
print(f"TE λ→S = {TE_L_to_S:.4f} nats,  TE S→λ = {TE_S_to_L:.4f} nats")
print(csv_path)
