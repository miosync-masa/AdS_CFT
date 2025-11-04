"""
lambda3_holo/automaton.py - Complete version with all required patches
Fully reproducible AdS/CFT automaton with proper temporal ordering
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
    """
    Create an independent RNG stream from seed and tag.
    Ensures complete independence across subsystems.
    """
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
    AdS/CFT-aware self-evolving automaton with holographic renormalization
    and delayed geodesic gating. Fixed temporal ordering ensures positive lag.
    
    Key features:
    - Complete RNG independence between subsystems
    - Proper delay queue with deque for exact timing
    - Correct temporal ordering: λ(t) → gate(t+delay) → S_RT(t+delay)
    - Burn-in period support for stability
    - Zero-lag suppression via c_eff delay
    - Phase-aware SOC control to prevent λ=1 lock-in
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
        c0: float = 0.06,  # Slightly reduced
        gamma: float = 0.9,
        c_eff_max: float = 0.15,  # Reduced for zero-lag suppression
        # Gating params (legacy)
        gate_delay: int = 1,
        gate_strength: float = 0.15,
        # SOC (legacy)
        soc_rate: float = 0.01,
        # Random seed
        seed: int = 913
    ):
        """
        Initialize automaton with either ModelConfig or individual parameters.
        
        Args:
            config: Complete ModelConfig object (overrides all other params)
            H, W, Z: Grid dimensions
            seed: Random seed for reproducibility
        """
        
        # Store seed for reproducibility
        self.seed = seed
        
        # Parse configuration
        if config is not None:
            # Use config object
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
                C_EFF_MAX=c_eff_max
            )
            self.gating = GeodesicGating(
                GATE_DELAY=gate_delay,
                GATE_STRENGTH=gate_strength
            )
            self.rt_weights = RTWeights()
        
        # Shortcuts for frequently used values
        self.L_ads = self.ads_cft.L_ADS
        self.alpha = self.ads_cft.ALPHA
        self.c0 = self.ads_cft.C0
        self.gamma = self.ads_cft.GAMMA
        self.c_eff_max = self.ads_cft.C_EFF_MAX
        self.gate_delay = self.gating.GATE_DELAY
        self.gate_strength = self.gating.GATE_STRENGTH
        self.soc_rate = self.ads_cft.SOC_RATE
        self.G_N = self.ads_cft.G_N
        
        # PATCH 1: SOC rates for different phases
        self.soc_rate_burnin = 0.02      # Increased from 0.01
        self.soc_rate_measure = 0.0      # Complete OFF during measurement!
        self.phase = 'init'
        
        # Initialize independent RNG streams
        self.rng_core = make_rng(self.seed, "core")
        self.rng_gate = make_rng(self.seed, "gate")
        self.rng_noise = make_rng(self.seed, "noise")
        self.rng_init = make_rng(self.seed, "init")
        
        # Full state reset
        self.reset_state()

    def _init_grid(self) -> List[List[Cell]]:
        """Initialize agent grid using dedicated init RNG stream"""
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
        """Apply deterministic spatial patterns (checkerboard + Gaussian)"""
        H, W = self.H, self.W
        
        # Checkerboard pattern
        for i in range(H):
            for j in range(W):
                if ((i // 4 + j // 4) % 2) == 0:
                    self.grid[i][j].coop = float(np.clip(self.grid[i][j].coop * 1.25, 0, 1))
                else:
                    self.grid[i][j].coop = float(np.clip(self.grid[i][j].coop * 0.75, 0, 1))
        
        # Gaussian boost at center
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
        
        # Enhanced checkerboard (2x2 blocks)
        for i in range(H):
            for j in range(W):
                if ((i // 2 + j // 2) & 1) == 0:
                    B[i, j] = np.clip(B[i, j] * 1.5 + 0.1, 0, 1)
                else:
                    B[i, j] = np.clip(B[i, j] * 0.5 - 0.1, 0, 1)
        
        # Strong Gaussian bump
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                bump = np.exp(-r2 / (H / 6)**2)
                B[i, j] = np.clip(B[i, j] + 0.3 * bump, 0, 1)
        
        # Strong noise
        noise = self.rng_init.uniform(-0.1, 0.1, (H, W))
        B[:] = np.clip(B + noise, 0, 1)
        
        self.boundary = B

    def reset_state(self):
        """Complete state reset to ensure reproducibility"""
        H, W, Z = self.H, self.W, self.Z
        
        # Clear all arrays
        self.bulk = np.zeros((H, W, Z), dtype=float)
        self.boundary = np.zeros((H, W), dtype=float)
        self.resource = np.ones((H, W), dtype=float)
        
        # Use deque for exact delay (FIFO queue)
        self.pending_gates = deque(maxlen=self.gate_delay + 1)
        for _ in range(self.gate_delay + 1):
            self.pending_gates.append(None)
        
        # PATCH 3: Lambda history with shorter window for z-score
        self._lambda_hist = deque(maxlen=50)
        
        # Current c_eff
        self.c_eff_current = self.c0
        
        # Initialize grid
        self.grid = self._init_grid()
        self._apply_spatial_pattern()
        
        # Initialize boundary
        self.boundary = self.coop_field()
        self._apply_spatial_seed()

    def coop_field(self) -> np.ndarray:
        """Extract cooperation field from agent grid"""
        M = np.zeros((self.H, self.W), dtype=float)
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i][j].alive:
                    M[i, j] = self.grid[i][j].coop
        return M
    
    def step_agents(self):
        """Update agent dynamics using dedicated RNG streams"""
        H, W = self.H, self.W
        C = self.coop_field()
        
        # Get config values
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
                
                # Harvest resources
                take = min(harvest_rate, self.resource[i, j])
                self.resource[i, j] -= take
                c.energy += take
                
                # Share with neighbors
                give = min(share_rate * c.coop, c.energy)
                if give > 0:
                    q = 0.25 * give
                    self.grid[(i + 1) % H][j].energy += q
                    self.grid[(i - 1) % H][j].energy += q
                    self.grid[i][(j + 1) % W].energy += q
                    self.grid[i][(j - 1) % W].energy += q
                    c.energy -= give
                
                # Update cooperation
                neigh = 0.25 * (
                    C[(i + 1) % H][j] + C[(i - 1) % H][j] +
                    C[i][(j + 1) % W] + C[i][(j - 1) % W]
                )
                c.coop += coop_update_rate * (neigh - c.coop) 
                c.coop += thermal_noise * self.rng_noise.standard_normal()
                c.coop = float(np.clip(c.coop, 0, 1))
                
                # Death check
                if c.energy < death_threshold:
                    c.alive = False
                    c.coop = 0.0
                
                # Birth
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
        
        # Update resource field
        self.resource += self.resource_config.RESOURCE_DIFFUSION * laplacian2d(self.resource)
        self.resource += self.resource_config.RESOURCE_REPLENISH
        self.resource = np.clip(self.resource, 0, self.resource_config.RESOURCE_MAX)
    
    def _mutate(self, genome: np.ndarray, mut_rate: float) -> np.ndarray:
        """Mutate genome using core RNG"""
        g = genome.copy()
        flips = self.rng_core.random(len(g)) < mut_rate
        g[flips] = 1 - g[flips]
        return g
    
    def update_bulk(self):
        """Update bulk Lambda field with AdS warping"""
        L0 = self.K_over_V(self.boundary)
        self.bulk[..., 0] = L0
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k])**2
            prev = self.bulk[..., k - 1]
            mixed = prev + self.ads_cft.BULK_DIFFUSION * laplacian2d(prev)
            self.bulk[..., k] = np.clip(warp * mixed, 0, None)
    
    def HR(self, c_eff: float):
        """Holographic renormalization: bulk → boundary feedback"""
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        w = np.exp(-self.alpha * z)
        w = w / (w.sum() + 1e-12)
        back = np.tensordot(self.bulk, w, axes=(2, 0))
        self.boundary += c_eff * (back - self.boundary)
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def update_boundary_payoff(self):
        """Update boundary via game-theoretic payoff"""
        B = self.boundary
        neigh = 0.25 * (
            np.roll(B, 1, 0) + np.roll(B, -1, 0) +
            np.roll(B, 1, 1) + np.roll(B, -1, 1)
        )
        payoff = neigh * (1.4 - 0.6 * B) - 0.4 * B * (1 - neigh)
        self.boundary += self.ads_cft.BOUNDARY_PAYOFF_RATE * payoff
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        """Compute Lambda = K/V field with increased epsilon"""
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, 1) - coop
        gy = np.roll(coop, -1, 0) - coop
        K = np.sqrt(gx * gx + gy * gy) + 1e-10
        V = np.abs(coop - float(coop.mean())) + 1e-10
        return K / V
    
    def SOC_tune(self):
        """Self-organized criticality with phase-aware rate"""
        rate = self.soc_rate_burnin if self.phase == 'burnin' else self.soc_rate_measure
        
        # Skip completely during measurement phase
        if rate == 0.0:
            return
        
        Lb = self.K_over_V(self.coop_field())
        delta = float(Lb.mean() - self.ads_cft.LAMBDA_CRITICAL)
        self.boundary = np.clip(self.boundary - rate * delta, 0, 1)
    
    def region_A(self, step: int) -> np.ndarray:
        """Define region A for RT entropy calculation"""
        C = self.coop_field()
        alive = C > 0
        if alive.sum() == 0:
            R = np.zeros((self.H, self.W), bool)
            R[:, :self.W // 2] = True
            return R
        thr = float(np.median(C[alive]))
        high = C >= thr
        k = 0 if (step % 2 == 0) else 1
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
        """Compute multi-objective RT entropy"""
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
    
    def _compute_gate_mask_exact(self, Lambda_b: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Compute gate mask deterministically"""
        out_band = outer_boundary_band(R, 1)
        if not out_band.any():
            return None
        
        vals = Lambda_b[out_band]
        p98 = np.percentile(vals, 98)
        mask = out_band & (Lambda_b >= p98)
        return mask if mask.any() else None
    
    def _apply_pending_gate_exact(self) -> int:
        """PATCH 2: Apply gate from exactly delay steps ago (apply-once FIFO)"""
        if len(self.pending_gates) > 0:
            # Pop oldest mask and immediately append None to maintain length
            mask = self.pending_gates.popleft()
            self.pending_gates.append(None)
            if mask is not None and mask.any():
                self.boundary[mask] += self.gate_strength * (1.0 - self.boundary[mask])
                self.boundary = np.clip(self.boundary, 0, 1)
                return int(mask.sum())
        return 0
    
    def _detect_outlier_enhanced(self, series_window: np.ndarray) -> bool:
        """PATCH 4: Enhanced outlier detection with relaxed Z threshold"""
        q1, q3 = np.percentile(series_window, [25, 75])
        iqr = max(q3 - q1, 1e-12)
        if series_window[-1] > q3 + 1.5 * iqr:
            return True
        
        mean = float(series_window.mean())
        std = float(series_window.std() + 1e-12)
        # Reduced threshold: 3.0 → 2.5
        if (series_window[-1] - mean) / std > 2.5:
            return True
        return False
    
    def _step_internal(self, step: int, record: bool = True) -> Optional[Dict]:
        """Fixed temporal ordering with phase-aware SOC"""
        
        # ===== A. MEASURE PRE-LAMBDA =====
        B_pre = self.boundary.copy()
        Lambda_b_pre = self.K_over_V(B_pre)
        
        R = self.region_A(step)
        out_band = outer_boundary_band(R, 1)
        in_band = inner_boundary_band(R, 1)
        
        lam_p99_out_pre = float(np.percentile(Lambda_b_pre[out_band], 99)) if out_band.any() else np.nan
        lam_p99_in_pre = float(np.percentile(Lambda_b_pre[in_band], 99)) if in_band.any() else np.nan
        
        # ===== B. PHYSICS UPDATES =====
        self.step_agents()
        self.boundary = self.coop_field()
        
        # ===== C. BULK UPDATE =====
        self.update_bulk()
        
        # ===== D. HOLOGRAPHIC RENORMALIZATION =====
        self.HR(self.c_eff_current)
        
        # ===== E. APPLY PENDING GATE =====
        gate_px = self._apply_pending_gate_exact()
        
        # ===== F. BOUNDARY UPDATES =====
        self.update_boundary_payoff()
        self.SOC_tune()
        
        # ===== G. MEASURE POST-S_RT =====
        S_mo, parts = self.S_RT_multiobjective(R)
        
        # PATCH 5: Add micro dithering noise
        self.boundary += 0.002 * self.rng_noise.standard_normal(self.boundary.shape)
        self.boundary = np.clip(self.boundary, 0, 1)
        
        # ===== H. DETECT AND ENQUEUE NEW GATE =====
        self._lambda_hist.append(lam_p99_out_pre)
        if len(self._lambda_hist) == self._lambda_hist.maxlen:
            window = np.array(self._lambda_hist)
            if self._detect_outlier_enhanced(window):
                if out_band.any():
                    vals = Lambda_b_pre[out_band]
                    p98 = np.percentile(vals, 98)
                    new_mask = out_band & (Lambda_b_pre >= p98)
                    new_mask = new_mask if new_mask.any() else None
                else:
                    new_mask = None
            else:
                new_mask = None
        else:
            new_mask = None

        self.pending_gates[-1] = new_mask
        # Don't append to deque here - popleft/append is handled in _apply_pending_gate_exact
        
        # ===== I. COMPUTE NEXT c_eff (PATCH 3: z-score amplification) =====
        if len(self._lambda_hist) >= 10:
            arr = np.array(self._lambda_hist, dtype=float)
            mu = float(arr.mean())
            sd = float(arr.std() + 1e-12)
            z = (lam_p99_out_pre - mu) / sd
        else:
            z = 0.0
        
        self.c_eff_current = float(np.clip(
            self.c0 * (1.0 + self.gamma * max(0.0, z)),  # Only amplify for z > 0
            self.c0,
            self.c_eff_max
        ))
        
        if not record:
            return None
        
        # Debug logging
        if step % 25 == 0:
            pending_has_gate = any(m is not None for m in self.pending_gates)
            print(f"[t={step:03d}] λ_pre={lam_p99_out_pre:.3f} "
                  f"S_RT={S_mo:.3f} queue={'Y' if pending_has_gate else 'N'} "
                  f"c_eff={self.c_eff_current:.3f}")
        
        return dict(
            t=step,
            entropy_RT_mo=S_mo,
            region_A_size=float(R.sum()),
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
        """Run simulation with phase-aware SOC control"""
        # Burn-in phase with normal SOC
        self.phase = 'burnin'
        print(f"[BURN-IN] Running {burn_in} steps with SOC={self.soc_rate_burnin}...")
        for t in range(burn_in):
            self._step_internal(t, record=False)
        
        # Measurement phase with SOC completely OFF
        self.phase = 'measure'
        print(f"[MEASURE] Recording {measure_steps} steps with SOC={self.soc_rate_measure}...")
        rows = []
        for t in range(measure_steps):
            rec = self._step_internal(burn_in + t, record=True)
            rec['t'] = t
            rows.append(rec)
        
        return rows
    
    def step_once(self, step: int, weights: Optional[Tuple[float, float, float]] = None) -> Dict:
        """Legacy interface: Execute one complete timestep"""
        if weights is not None:
            self.rt_weights.WEIGHT_PERIMETER = weights[0]
            self.rt_weights.WEIGHT_HOLES = weights[1]
            self.rt_weights.WEIGHT_CURVATURE = weights[2]
        
        return self._step_internal(step, record=True)
