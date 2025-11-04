"""
lambda3_holo/automaton.py - PhaseShift-VIII Enhanced
Simple, clean, and effective AdS/CFT automaton
Based on proven PhaseShift-VIII with minimal enhancements
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from .geometry import (
    laplacian2d, perimeter_len_bool8, corner_count, count_holes_nonperiodic,
    inner_boundary_band, outer_boundary_band, k_th_largest_mask
)

# ========== SINGLE RNG - Keep it simple! ==========
rng = np.random.default_rng(913)

# ========== Data Classes ==========
@dataclass
class Cell:
    """Agent cell with energy, genome, and cooperation"""
    alive: bool
    energy: float
    genome: np.ndarray
    coop: float

# ========== Main Automaton Class ==========
class Automaton:
    """
    PhaseShift-VIII Enhanced: Simple and effective
    - Single RNG for reproducibility
    - Immediate geodesic gating (no delay complexity)
    - Multi-objective RT entropy
    - Clean temporal ordering
    """
    
    def __init__(
        self,
        H: int = 44,
        W: int = 44,
        Z: int = 24,
        L_ads: float = 1.0,
        alpha: float = 0.9,
        c0: float = 0.08,
        gamma: float = 0.9,
        c_eff_max: float = 0.18,
        gate_strength: float = 0.25,  # Strong immediate effect
        seed: int = 913
    ):
        """Initialize with simple, proven parameters"""
        global rng
        rng = np.random.default_rng(seed)
        
        self.H, self.W, self.Z = H, W, Z
        self.L_ads = L_ads
        self.alpha = alpha
        self.c0 = c0
        self.gamma = gamma
        self.c_eff_max = c_eff_max
        self.gate_strength = gate_strength
        self.G_N = 1.0
        
        # RT weights for multi-objective
        self.w_perim = 1.0
        self.w_holes = 2.0
        self.w_curv = 0.5
        
        # Initialize state
        self.grid = self._init_grid()
        self.resource = np.ones((H, W), dtype=float)
        self.bulk = np.zeros((H, W, Z), dtype=float)
        
        # Initialize boundary from cooperation
        self.boundary = self.coop_field()
        self._apply_spatial_pattern()
    
    def _init_grid(self) -> List[List[Cell]]:
        """Initialize agent grid"""
        grid = []
        for i in range(self.H):
            row = []
            for j in range(self.W):
                row.append(Cell(
                    alive=True,
                    energy=1.0 + 0.2 * rng.standard_normal(),
                    genome=rng.integers(0, 2, size=56, dtype=np.int8),
                    coop=np.clip(0.5 + 0.2 * rng.standard_normal(), 0, 1)
                ))
            grid.append(row)
        return grid
    
    def _apply_spatial_pattern(self):
        """Apply initial spatial pattern to break symmetry"""
        H, W = self.H, self.W
        
        # Checkerboard
        for i in range(H):
            for j in range(W):
                if ((i // 4 + j // 4) % 2) == 0:
                    self.grid[i][j].coop *= 1.25
                else:
                    self.grid[i][j].coop *= 0.75
                self.grid[i][j].coop = np.clip(self.grid[i][j].coop, 0, 1)
        
        # Central boost
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                boost = np.exp(-r2 / (H / 4)**2)
                self.grid[i][j].coop = np.clip(
                    self.grid[i][j].coop * (1 + 0.4 * boost), 0, 1
                )
    
    def coop_field(self) -> np.ndarray:
        """Extract cooperation field from agents"""
        M = np.zeros((self.H, self.W), dtype=float)
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i][j].alive:
                    M[i, j] = self.grid[i][j].coop
        return M
    
    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        """Compute Lambda = K/V field"""
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, 1) - coop
        gy = np.roll(coop, -1, 0) - coop
        K = np.sqrt(gx * gx + gy * gy) + 1e-12
        V = np.abs(coop - coop.mean()) + 1e-12
        return K / V
    
    def step_agents(self):
        """Update agent dynamics"""
        H, W = self.H, self.W
        C = self.coop_field()
        
        for i in range(H):
            for j in range(W):
                c = self.grid[i][j]
                if not c.alive:
                    continue
                
                # Harvest
                take = min(0.05, self.resource[i, j])
                self.resource[i, j] -= take
                c.energy += take
                
                # Share
                give = min(0.02 * c.coop, c.energy)
                if give > 0:
                    q = 0.25 * give
                    self.grid[(i+1)%H][j].energy += q
                    self.grid[(i-1)%H][j].energy += q
                    self.grid[i][(j+1)%W].energy += q
                    self.grid[i][(j-1)%W].energy += q
                    c.energy -= give
                
                # Update cooperation
                neigh = 0.25 * (
                    C[(i+1)%H][j] + C[(i-1)%H][j] +
                    C[i][(j+1)%W] + C[i][(j-1)%W]
                )
                c.coop += 0.05 * (neigh - c.coop) + 0.01 * rng.standard_normal()
                c.coop = np.clip(c.coop, 0, 1)
                
                # Death
                if c.energy < 0.15:
                    c.alive = False
                    c.coop = 0.0
        
        # Update resources
        self.resource += 0.03 * laplacian2d(self.resource) + 0.004
        self.resource = np.clip(self.resource, 0, 3)
    
    def update_bulk(self):
        """Update bulk from boundary"""
        L0 = self.K_over_V(self.boundary)
        self.bulk[..., 0] = L0
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k])**2
            prev = self.bulk[..., k-1]
            mixed = prev + 0.012 * laplacian2d(prev)
            self.bulk[..., k] = np.clip(warp * mixed, 0, None)
    
    def HR(self, c_eff: float):
        """Holographic renormalization"""
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        w = np.exp(-self.alpha * z)
        w = w / (w.sum() + 1e-12)
        back = np.tensordot(self.bulk, w, axes=(2, 0))
        self.boundary += c_eff * (back - self.boundary)
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def update_boundary_payoff(self):
        """Game-theoretic payoff dynamics"""
        B = self.boundary
        neigh = 0.25 * (
            np.roll(B, 1, 0) + np.roll(B, -1, 0) +
            np.roll(B, 1, 1) + np.roll(B, -1, 1)
        )
        payoff = neigh * (1.4 - 0.6 * B) - 0.4 * B * (1 - neigh)
        self.boundary += 0.08 * payoff
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def SOC_tune(self, rate: float = 0.01):
        """Self-organized criticality"""
        L = self.K_over_V(self.coop_field())
        delta = L.mean() - 1.0
        self.boundary = np.clip(self.boundary - rate * delta, 0, 1)
    
    def region_A(self, step: int) -> np.ndarray:
        """Define region A (alternating k)"""
        C = self.coop_field()
        alive = C > 0
        if alive.sum() == 0:
            R = np.zeros((self.H, self.W), bool)
            R[:, :self.W//2] = True
            return R
        
        thr = np.median(C[alive])
        high = C >= thr
        k = 0 if (step % 2 == 0) else 1
        R = k_th_largest_mask(high, k)
        if R.sum() == 0:
            R = k_th_largest_mask(high, 0)
        return R
    
    def S_RT_multiobjective(self, R: np.ndarray) -> Tuple[float, Dict]:
        """Multi-objective RT entropy"""
        perim = perimeter_len_bool8(R)
        holes = count_holes_nonperiodic(R)
        curv = corner_count(R)
        
        S = (self.w_perim * perim + 
             self.w_holes * holes + 
             self.w_curv * curv) / (4.0 * self.G_N)
        
        return S, {
            'perimeter': perim,
            'holes': holes,
            'curvature': curv
        }
    
    def geodesic_gating(self, R: np.ndarray, Lb: np.ndarray) -> int:
        """Immediate geodesic gating when λ spikes"""
        out = outer_boundary_band(R, 1)
        if not out.any():
            return 0
        
        vals = Lb[out]
        if len(vals) < 10:
            return 0
        
        # Detect outlier
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1 + 1e-12
        thresh = q3 + 1.5 * iqr
        lam_p99 = np.percentile(vals, 99)
        
        if lam_p99 > thresh:
            # Apply gate immediately!
            self.boundary[out] = np.clip(
                self.boundary[out] + self.gate_strength * (1.0 - self.boundary[out]), 
                0, 1
            )
            return int(out.sum())
        return 0
    
    def step(self, step: int) -> Dict:
        """Single simulation step with clean ordering"""
        
        # 1. Update agents
        self.step_agents()
        self.boundary = self.coop_field()
        
        # 2. Update bulk
        self.update_bulk()
        
        # 3. Define region A
        R = self.region_A(step)
        
        # 4. Measure PRE values
        Lb_pre = self.K_over_V(self.boundary)
        out = outer_boundary_band(R, 1)
        inn = inner_boundary_band(R, 1)
        
        lam_p99_out = np.percentile(Lb_pre[out], 99) if out.any() else np.nan
        lam_p99_in = np.percentile(Lb_pre[inn], 99) if inn.any() else np.nan
        
        # 5. Dynamic c_eff
        med_global = np.median(Lb_pre)
        if out.any():
            norm_tail = lam_p99_out / (med_global + 1e-12) - 1.0
        else:
            norm_tail = 0.0
        norm_tail = np.clip(norm_tail, -0.5, 3.0)
        c_eff = np.clip(self.c0 * (1.0 + self.gamma * norm_tail), 
                        self.c0, self.c_eff_max)
        
        # 6. HR
        self.HR(c_eff)
        
        # 7. Geodesic gating (immediate!)
        gate_px = self.geodesic_gating(R, Lb_pre)
        
        # 8. Boundary dynamics
        self.update_boundary_payoff()
        self.SOC_tune()
        
        # 9. Measure RT entropy
        S_RT, parts = self.S_RT_multiobjective(R)
        
        return dict(
            t=step,
            entropy_RT_mo=S_RT,
            lambda_p99_A_out_pre=lam_p99_out,
            lambda_p99_A_in_pre=lam_p99_in,
            region_A_size=R.sum(),
            region_A_perimeter=parts['perimeter'],
            region_A_holes=parts['holes'],
            region_A_curvature=parts['curvature'],
            c_eff=c_eff,
            gate_applied_px=gate_px
        )
    
    def run_with_burnin(self, burn_in: int = 200, measure_steps: int = 300) -> List[Dict]:
        """Run with burn-in period"""
        print(f"[BURN-IN] Running {burn_in} steps...")
        for t in range(burn_in):
            _ = self.step(t)
        
        print(f"[MEASURE] Recording {measure_steps} steps...")
        rows = []
        for t in range(measure_steps):
            rec = self.step(burn_in + t)
            rec['t'] = t  # Reset to measurement time
            rows.append(rec)
            
            if t % 25 == 0:
                print(f"[t={t:03d}] λ={rec['lambda_p99_A_out_pre']:.1f}  "
                      f"S_RT={rec['entropy_RT_mo']:.1f}  "
                      f"gate={rec['gate_applied_px']}  "
                      f"c_eff={rec['c_eff']:.3f}")
        
        return rows
