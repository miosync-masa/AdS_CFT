"""
lambda3_holo/automaton.py - FINAL VERSION with all critical fixes
Fully reproducible AdS/CFT automaton with proper causality chain
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from numpy.random import Generator, PCG64, SeedSequence
from collections import deque

from .geometry import (
    laplacian2d, perimeter_len_bool8, corner_count, count_holes_nonperiodic,
    inner_boundary_band, outer_boundary_band, k_th_largest_mask
)
from .config import ModelConfig, AgentDynamics, ResourceDynamics, AdSCFTParams, GeodesicGating, RTWeights

# ========== RNG Utilities ==========
def make_rng(seed: int, stream_tag: str) -> Generator:
    """Create an independent RNG stream from seed and tag"""
    ss = SeedSequence(seed, spawn_key=[hash(stream_tag) & 0xffffffff])
    return Generator(PCG64(ss))

# ========== Data Classes ==========
@dataclass
class Cell:
    """Agent cell with energy, genome, and cooperation level"""
    alive: bool
    energy: float
    genome: np.ndarray
    coop: float

# ========== Main Automaton Class ==========
class Automaton:
    """
    AdS/CFT-aware self-evolving automaton with FIXED causality chain:
    λ(t) → [HR/gate] → S_RT(t) → boundary(t+1) → bulk(t+1)
    
    Critical fixes:
    - Boundary NOT overwritten by coop_field
    - Exact gate delay (no off-by-one)
    - Proper temporal ordering
    - Agent-boundary weak coupling
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        # Grid geometry (legacy params)
        H: int = 44,
        W: int = 44,
        Z: int = 24,
        # AdS/CFT params (legacy)
        L_ads: float = 1.0,
        alpha: float = 0.9,
        c0: float = 0.06,
        gamma: float = 0.9,
        c_eff_max: float = 0.15,  # Zero-lag suppression
        # Gating params (legacy)
        gate_delay: int = 1,
        gate_strength: float = 0.15,  # Slightly stronger
        # SOC (legacy)
        soc_rate: float = 0.01,
        # Random seed
        seed: int = 913
    ):
        """Initialize automaton with proper configuration"""
        
        self.seed = seed
        
        # Parse configuration
        if config is not None:
            self.H = config.H
            self.W = config.W
            self.Z = config.Z
            self.agent = config.agent
            self.resource_config = config.resource 
            self.ads_cft = config.ads_cft
            self.gating = config.gating
            self.rt_weights = config.rt_weights
            self.seed = config.seed
        else:
            # Legacy: construct from individual params
            self.H = H
            self.W = W
            self.Z = Z
            self.agent = AgentDynamics()
            self.resource_config = ResourceDynamics() 
            self.ads_cft = AdSCFTParams(
                L_ADS=L_ads,
                ALPHA=alpha,
                C0=c0,
                GAMMA=gamma,
                C_EFF_MAX=c_eff_max,
                BULK_DIFFUSION=0.10  # Reduced for sharper spikes
            )
            self.gating = GeodesicGating(
                GATE_DELAY=gate_delay,
                GATE_STRENGTH=gate_strength
            )
            self.rt_weights = RTWeights()
        
        # Shortcuts
        self.L_ads = self.ads_cft.L_ADS
        self.alpha = self.ads_cft.ALPHA
        self.c0 = self.ads_cft.C0
        self.gamma = self.ads_cft.GAMMA
        self.c_eff_max = self.ads_cft.C_EFF_MAX
        self.gate_delay = self.gating.GATE_DELAY
        self.gate_strength = self.gating.GATE_STRENGTH
        self.soc_rate = self.ads_cft.SOC_RATE
        self.G_N = self.ads_cft.G_N
        
        # Phase-aware SOC rates
        self.soc_rate_burnin = 0.02  # Strong during burn-in
        self.soc_rate_measure = 0.0  # OFF during measurement
        self.phase = 'init'
        
        # Initialize RNG streams
        self.rng_core = make_rng(self.seed, "core")
        self.rng_gate = make_rng(self.seed, "gate")
        self.rng_noise = make_rng(self.seed, "noise")
        self.rng_init = make_rng(self.seed, "init")
        
        # Full state reset
        self.reset_state()

    def reset_state(self):
        """Complete state reset with proper initialization"""
        H, W, Z = self.H, self.W, self.Z
        
        # Clear all arrays
        self.bulk = np.zeros((H, W, Z), dtype=float)
        self.boundary = np.zeros((H, W), dtype=float)
        self.resource = np.ones((H, W), dtype=float)
        
        # FIX: Proper gate delay queue (exactly gate_delay slots)
        L = max(1, int(self.gate_delay))
        self.pending_gates = deque([None]*L, maxlen=L)
        
        # Lambda history for z-score
        self._lambda_hist = deque(maxlen=50)
        
        # Current c_eff
        self.c_eff_current = self.c0
        
        # Initialize grid
        self.grid = self._init_grid()
        self._apply_spatial_pattern()
        
        # Initialize boundary (only once!)
        self.boundary = self.coop_field()
        self._apply_spatial_seed()

    def _init_grid(self) -> List[List[Cell]]:
        """Initialize agent grid"""
        grid = []
        for i in range(self.H):
            row = []
            for j in range(self.W):
                row.append(Cell(
                    alive=True,
                    energy=float(1.0 + 0.2 * self.rng_init.standard_normal()),
                    genome=self.rng_init.integers(0, 2, size=56, dtype=np.int8),
                    coop=float(np.clip(0.5 + 0.2 * self.rng_init.standard_normal(), 0, 1))
                ))
            grid.append(row)
        return grid
    
    def _apply_spatial_pattern(self):
        """Apply checkerboard + Gaussian patterns"""
        H, W = self.H, self.W
        for i in range(H):
            for j in range(W):
                if ((i // 4 + j // 4) % 2) == 0:
                    self.grid[i][j].coop = float(np.clip(self.grid[i][j].coop * 1.25, 0, 1))
                else:
                    self.grid[i][j].coop = float(np.clip(self.grid[i][j].coop * 0.75, 0, 1))
        
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                boost = np.exp(-r2 / (H / 4)**2)
                self.grid[i][j].coop = float(np.clip(self.grid[i][j].coop * (1 + 0.4 * boost), 0, 1))
    
    def _apply_spatial_seed(self):
        """Add EXTREME spatial roughness to break λ=1 lock-in"""
        H, W = self.H, self.W
        B = self.boundary
        
        for i in range(H):
            for j in range(W):
                if ((i // 2 + j // 2) & 1) == 0:
                    B[i, j] = np.clip(B[i, j] * 1.5 + 0.1, 0, 1)
                else:
                    B[i, j] = np.clip(B[i, j] * 0.5 - 0.1, 0, 1)
        
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                bump = np.exp(-r2 / (H / 6)**2)
                B[i, j] = np.clip(B[i, j] + 0.3 * bump, 0, 1)
        
        noise = self.rng_init.uniform(-0.1, 0.1, (H, W))
        B[:] = np.clip(B + noise, 0, 1)
        self.boundary = B

    def coop_field(self) -> np.ndarray:
        """Extract cooperation field from agent grid"""
        M = np.zeros((self.H, self.W), dtype=float)
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i][j].alive:
                    M[i, j] = self.grid[i][j].coop
        return M
    
    def step_agents(self):
        """Update agents WITHOUT overwriting boundary!"""
        H, W = self.H, self.W
        C = self.coop_field()
        
        harvest_rate = self.agent.HARVEST_RATE
        share_rate = self.agent.SHARE_RATE
        coop_update_rate = self.agent.COOP_UPDATE_RATE
        thermal_noise = self.agent.THERMAL_NOISE
        death_threshold = self.agent.DEATH_THRESHOLD
        birth_threshold = self.agent.BIRTH_THRESHOLD
        birth_prob = self.agent.BIRTH_PROBABILITY
        mutation_rate = self.agent.MUTATION_RATE
        
        for i in range(H):
            for j in range(W):
                c = self.grid[i][j]
                if not c.alive:
                    continue
                
                take = min(harvest_rate, self.resource[i, j])
                self.resource[i, j] -= take
                c.energy += take
                
                give = min(share_rate * c.coop, c.energy)
                if give > 0:
                    q = 0.25 * give
                    self.grid[(i + 1) % H][j].energy += q
                    self.grid[(i - 1) % H][j].energy += q
                    self.grid[i][(j + 1) % W].energy += q
                    self.grid[i][(j - 1) % W].energy += q
                    c.energy -= give
                
                neigh = 0.25 * (
                    C[(i + 1) % H][j] + C[(i - 1) % H][j] +
                    C[i][(j + 1) % W] + C[i][(j - 1) % W]
                )
                c.coop += coop_update_rate * (neigh - c.coop) 
                c.coop += thermal_noise * self.rng_noise.standard_normal()
                c.coop = float(np.clip(c.coop, 0, 1))
                
                if c.energy < death_threshold:
                    c.alive = False
                    c.coop = 0.0
                elif c.energy > birth_threshold and self.rng_core.random() < birth_prob:
                    child = Cell(
                        True,
                        c.energy * 0.5,
                        self._mutate(c.genome, mutation_rate),
                        c.coop
                    )
                    c.energy *= 0.5
                    ii = (i + self.rng_core.integers(-1, 2)) % H
                    jj = (j + self.rng_core.integers(-1, 2)) % W
                    self.grid[ii][jj] = child
        
        # Update resources
        self.resource += self.resource_config.RESOURCE_DIFFUSION * laplacian2d(self.resource)
        self.resource += self.resource_config.RESOURCE_REPLENISH
        self.resource = np.clip(self.resource, 0, self.resource_config.RESOURCE_MAX)
        
        # CRITICAL: NO boundary overwrite here!
        # self.boundary = self.coop_field()  ← DELETED!
    
    def agents_to_boundary_coupling(self, rate: float = 0.20):
        """NEW: Weak coupling from agents to boundary"""
        C = self.coop_field()
        self.boundary += rate * (C - self.boundary)
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def _mutate(self, genome: np.ndarray, mut_rate: float) -> np.ndarray:
        """Mutate genome"""
        g = genome.copy()
        flips = self.rng_core.random(len(g)) < mut_rate
        g[flips] = 1 - g[flips]
        return g
    
    def update_bulk(self):
        """Update bulk from CURRENT boundary"""
        L0 = self.K_over_V(self.boundary)
        self.bulk[..., 0] = L0
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k])**2
            prev = self.bulk[..., k - 1]
            mixed = prev + self.ads_cft.BULK_DIFFUSION * laplacian2d(prev)
            self.bulk[..., k] = np.clip(warp * mixed, 0, None)
    
    def update_bulk_from(self, Bsrc: np.ndarray):
        """NEW: Update bulk from specified source"""
        L0 = self.K_over_V(Bsrc)
        self.bulk[..., 0] = L0
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k])**2
            prev = self.bulk[..., k - 1]
            mixed = prev + self.ads_cft.BULK_DIFFUSION * laplacian2d(prev)
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
        self.boundary += self.ads_cft.BOUNDARY_PAYOFF_RATE * payoff
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        """Compute Lambda = K/V field"""
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, 1) - coop
        gy = np.roll(coop, -1, 0) - coop
        K = np.sqrt(gx * gx + gy * gy) + 1e-10
        V = np.abs(coop - float(coop.mean())) + 1e-10
        return K / V
    
    def SOC_tune(self):
        """Phase-aware SOC"""
        rate = self.soc_rate_burnin if self.phase == 'burnin' else self.soc_rate_measure
        if rate == 0.0:
            return
        
        Lb = self.K_over_V(self.coop_field())
        delta = float(Lb.mean() - self.ads_cft.LAMBDA_CRITICAL)
        self.boundary = np.clip(self.boundary - rate * delta, 0, 1)
    
    def region_A(self, step: int) -> np.ndarray:
        """Define region A (consider using k=0 fixed for stability)"""
        C = self.coop_field()
        alive = C > 0
        if alive.sum() == 0:
            R = np.zeros((self.H, self.W), bool)
            R[:, :self.W // 2] = True
            return R
        thr = float(np.median(C[alive]))
        high = C >= thr
        k = 0  # Fixed largest cluster for stability
        R = k_th_largest_mask(high, k)
        if R.sum() == 0:
            R = k_th_largest_mask(high, 0)
        return R
    
    def S_RT_multiobjective(
        self,
        R: np.ndarray,
        w_len: Optional[float] = None,
        w_hole: Optional[float] = None,
        w_curv: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Multi-objective RT entropy"""
        if w_len is None:
            w_len = self.rt_weights.WEIGHT_PERIMETER
        if w_hole is None:
            w_hole = self.rt_weights.WEIGHT_HOLES
        if w_curv is None:
            w_curv = self.rt_weights.WEIGHT_CURVATURE
        
        perim = perimeter_len_bool8(R)
        holes = count_holes_nonperiodic(R)
        curv = float(corner_count(R))
        S = (w_len * perim + w_hole * holes + w_curv * curv) / (4.0 * self.G_N)
        
        return float(S), dict(
            perimeter=float(perim),
            holes=float(holes),
            curvature=float(curv)
        )
    
    def _apply_pending_gate_exact(self) -> int:
        """FIX: Apply gate with exact delay (no off-by-one)"""
        if self.gate_delay <= 0:
            return 0
        
        mask = self.pending_gates.popleft()  # Exactly delay steps ago
        if mask is not None and mask.any():
            self.boundary[mask] += self.gate_strength * (1.0 - self.boundary[mask])
            self.boundary = np.clip(self.boundary, 0, 1)
            applied = int(mask.sum())
        else:
            applied = 0
        
        # Advance queue
        self.pending_gates.append(None)
        return applied
    
    def _detect_outlier_enhanced(self, series_window: np.ndarray) -> bool:
        """Enhanced outlier detection"""
        q1, q3 = np.percentile(series_window, [25, 75])
        iqr = max(q3 - q1, 1e-12)
        if series_window[-1] > q3 + 1.5 * iqr:
            return True
        
        mean = float(series_window.mean())
        std = float(series_window.std() + 1e-12)
        if (series_window[-1] - mean) / std > 2.5:
            return True
        return False
    
    def _step_internal(self, step: int, record: bool = True) -> Optional[Dict]:
        """CRITICAL: Proper causal ordering"""
        
        # A) Pre-snapshot (this is "time t boundary")
        B_pre = self.boundary.copy()
        Lambda_b_pre = self.K_over_V(B_pre)
        R_pre = self.region_A(step)
        out_band_pre = outer_boundary_band(R_pre, 1)
        in_band_pre = inner_boundary_band(R_pre, 1)
        
        lam_p99_out_pre = float(np.percentile(Lambda_b_pre[out_band_pre], 99)) if out_band_pre.any() else np.nan
        lam_p99_in_pre = float(np.percentile(Lambda_b_pre[in_band_pre], 99)) if in_band_pre.any() else np.nan
        
        # B) Bulk from PRE boundary
        self.update_bulk_from(B_pre)
        
        # C) HR with delayed c_eff
        self.HR(self.c_eff_current)
        
        # D) Apply delayed gate
        gate_px = self._apply_pending_gate_exact()
        
        # E) Boundary dynamics
        self.update_boundary_payoff()
        self.SOC_tune()
        
        # F) Measure S_RT with POST geometry
        R_post = self.region_A(step)
        S_mo, parts = self.S_RT_multiobjective(R_post)
        
        # G) Detect new spike and enqueue
        self._lambda_hist.append(lam_p99_out_pre)
        new_mask = None
        if len(self._lambda_hist) == self._lambda_hist.maxlen:
            window = np.array(self._lambda_hist)
            if self._detect_outlier_enhanced(window) and out_band_pre.any():
                vals = Lambda_b_pre[out_band_pre]
                p98 = np.percentile(vals, 98)
                cand = out_band_pre & (Lambda_b_pre >= p98)
                new_mask = cand if cand.any() else None
        
        # Multi-fire suppression and enqueue
        if self.gate_delay > 0 and not any(m is not None for m in self.pending_gates):
            self.pending_gates[-1] = new_mask
        
        # H) Compute next c_eff (z-score amplification)
        z = 0.0
        if len(self._lambda_hist) >= 10:
            arr = np.array(self._lambda_hist, dtype=float)
            mu, sd = float(arr.mean()), float(arr.std() + 1e-12)
            z = (lam_p99_out_pre - mu) / sd
        
        self.c_eff_current = float(np.clip(
            self.c0 * (1.0 + self.gamma * max(0.0, z)),
            self.c0,
            self.c_eff_max
        ))
        
        # I) NOW update agents (AFTER boundary measurements)
        self.step_agents()
        
        # J) Agent→Boundary weak coupling
        self.agents_to_boundary_coupling(rate=0.20)
        
        # K) Micro noise
        self.boundary += 0.002 * self.rng_noise.standard_normal(self.boundary.shape)
        self.boundary = np.clip(self.boundary, 0, 1)
        
        if not record:
            return None
        
        # Debug logging
        if step % 25 == 0:
            print(f"[t={step:03d}] λ_pre={lam_p99_out_pre:.3f}  S_RT={S_mo:.3f}  "
                  f"gate_px={gate_px}  c_eff={self.c_eff_current:.3f}")
        
        return dict(
            t=step,
            entropy_RT_mo=S_mo,
            region_A_size=float(R_post.sum()),
            region_A_perimeter=parts["perimeter"],
            region_A_holes=parts["holes"],
            region_A_curvature=parts["curvature"],
            lambda_p99_A_out_pre=lam_p99_out_pre,
            lambda_p99_A_in_pre=lam_p99_in_pre,
            lambda_p99_A_out_post=np.nan,
            lambda_p99_A_in_post=np.nan,
            c_eff=self.c_eff_current,
            gate_applied_px=int(gate_px)
        )
    
    def run_with_burnin(self, burn_in: int = 250, measure_steps: int = 300) -> List[Dict]:
        """Run with phase-aware SOC"""
        self.phase = 'burnin'
        print(f"[BURN-IN] Running {burn_in} steps with SOC={self.soc_rate_burnin}...")
        for t in range(burn_in):
            self._step_internal(t, record=False)
        
        self.phase = 'measure'
        print(f"[MEASURE] Recording {measure_steps} steps with SOC={self.soc_rate_measure}...")
        rows = []
        for t in range(measure_steps):
            rec = self._step_internal(burn_in + t, record=True)
            rec['t'] = t
            rows.append(rec)
        
        return rows
    
    def step_once(self, step: int, weights: Optional[Tuple[float, float, float]] = None) -> Dict:
        """Legacy interface"""
        if weights is not None:
            self.rt_weights.WEIGHT_PERIMETER = weights[0]
            self.rt_weights.WEIGHT_HOLES = weights[1]
            self.rt_weights.WEIGHT_CURVATURE = weights[2]
        
        return self._step_internal(step, record=True)
