"""
lambda3_holo/automaton.py - Complete version with fixed causality
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
        c0: float = 0.08,
        gamma: float = 0.8,
        c_eff_max: float = 0.17,  # Reduced from 0.18 for zero-lag suppression
        # Gating params (legacy)
        gate_delay: int = 1,
        gate_strength: float = 0.12,  # Reduced from 0.15
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
        
        # SOC rates for different phases (CRITICAL!)
        self.soc_rate_burnin = 0.01      # Normal during burn-in
        self.soc_rate_measure = 0.001    # Reduced during measurement!
        self.phase = 'init'              # Track current phase
        
        # Initialize independent RNG streams
        self.rng_core = make_rng(self.seed, "core")      # Agent dynamics
        self.rng_gate = make_rng(self.seed, "gate")      # Gating decisions
        self.rng_noise = make_rng(self.seed, "noise")    # Thermal noise
        self.rng_init = make_rng(self.seed, "init")      # Initialization
        
        # Full state reset
        self.reset_state()
    
    def reset_state(self):
        """
        Complete state reset to ensure reproducibility.
        Clears all hidden states and reinitializes from scratch.
        """
        H, W, Z = self.H, self.W, self.Z
        
        # Clear all arrays
        self.bulk = np.zeros((H, W, Z), dtype=float)
        self.boundary = np.zeros((H, W), dtype=float)
        self.resource = np.ones((H, W), dtype=float)
        
        # CRITICAL: Use deque for exact delay (FIFO queue)
        self.pending_gates = deque(maxlen=self.gate_delay + 1)
        for _ in range(self.gate_delay + 1):
            self.pending_gates.append(None)
        
        # Lambda history for outlier detection
        self._lambda_hist = deque(maxlen=96)  # Use deque with larger window
        
        # Current c_eff (delayed by 1 step to avoid zero-lag)
        self.c_eff_current = self.c0
        
        # Initialize grid with deterministic RNG
        self.grid = self._init_grid()
        
        # Apply spatial patterns
        self._apply_spatial_pattern()
        
        # Initialize boundary from cooperation field
        self.boundary = self.coop_field()
        
        # CRITICAL: Apply extra spatial seed to prevent λ=1 lock-in
        self._apply_spatial_seed()
    
    def _apply_spatial_seed(self):
        """Add extra spatial roughness to prevent λ=1 lock-in"""
        H, W = self.H, self.W
        B = self.boundary
        
        # Enhanced checkerboard pattern (3x3 blocks for more variation)
        for i in range(H):
            for j in range(W):
                if ((i // 3 + j // 3) & 1) == 0:
                    B[i, j] = np.clip(B[i, j] * 1.25 + 0.02, 0, 1)
                else:
                    B[i, j] = np.clip(B[i, j] * 0.75 - 0.02, 0, 1)
        
        # Stronger Gaussian bump at center
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                bump = np.exp(-r2 / (H / 4)**2)
                B[i, j] = np.clip(B[i, j] + 0.15 * bump, 0, 1)
        
        # Add random noise for initial diversity
        noise = self.rng_init.uniform(-0.05, 0.05, (H, W))
        B[:] = np.clip(B + noise, 0, 1)
        
        self.boundary = B
    
    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        """Compute Lambda = K/V field with increased epsilon"""
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, 1) - coop
        gy = np.roll(coop, -1, 0) - coop
        K = np.sqrt(gx * gx + gy * gy) + 1e-10  # Increased epsilon
        V = np.abs(coop - float(coop.mean())) + 1e-10
        return K / V
    
    def SOC_tune(self):
        """Self-organized criticality with phase-aware rate"""
        # Use different rates for burn-in vs measurement
        rate = self.soc_rate_burnin if self.phase == 'burnin' else self.soc_rate_measure
        
        Lb = self.K_over_V(self.coop_field())
        delta = float(Lb.mean() - self.ads_cft.LAMBDA_CRITICAL)
        self.boundary = np.clip(self.boundary - rate * delta, 0, 1)
    
    def _detect_outlier_enhanced(self, series_window: np.ndarray) -> bool:
        """Enhanced outlier detection with multiple methods"""
        # Tukey's method
        q1, q3 = np.percentile(series_window, [25, 75])
        iqr = max(q3 - q1, 1e-12)
        if series_window[-1] > q3 + 1.5 * iqr:
            return True
        
        # Z-score method (backup)
        mean = np.mean(series_window)
        std = np.std(series_window) + 1e-12
        if (series_window[-1] - mean) / std > 3.0:
            return True
        
        return False
    
    def _step_internal(self, step: int, record: bool = True) -> Optional[Dict]:
        """
        Fixed temporal ordering with phase-aware SOC
        """
        
        # ===== A. MEASURE PRE-LAMBDA (before any updates) =====
        B_pre = self.boundary.copy()  # Snapshot before changes
        Lambda_b_pre = self.K_over_V(B_pre)
        
        # Get region A for measurements
        R = self.region_A(step)
        out_band = outer_boundary_band(R, 1)
        in_band = inner_boundary_band(R, 1)
        
        lam_p99_out_pre = float(np.percentile(Lambda_b_pre[out_band], 99)) if out_band.any() else np.nan
        lam_p99_in_pre = float(np.percentile(Lambda_b_pre[in_band], 99)) if in_band.any() else np.nan
        
        # ===== B. PHYSICS UPDATES (agents, resources) =====
        self.step_agents()
        self.boundary = self.coop_field()
        
        # ===== C. BULK UPDATE (from previous boundary state) =====
        self.update_bulk()
        
        # ===== D. HOLOGRAPHIC RENORMALIZATION (use previous c_eff) =====
        self.HR(self.c_eff_current)  # Use delayed c_eff!
        
        # ===== E. APPLY PENDING GATE (from delay queue) =====
        gate_px = self._apply_pending_gate_exact()
        
        # ===== F. BOUNDARY UPDATES (payoff, SOC with phase-aware rate) =====
        self.update_boundary_payoff()
        self.SOC_tune()  # Now uses phase-dependent rate!
        
        # ===== G. MEASURE POST-S_RT (after all updates) =====
        S_mo, parts = self.S_RT_multiobjective(R)
        
        # ===== H. DETECT AND ENQUEUE NEW GATE (for future) =====
        self._lambda_hist.append(lam_p99_out_pre)
        if len(self._lambda_hist) == self._lambda_hist.maxlen:
            window = np.array(self._lambda_hist)
            if self._detect_outlier_enhanced(window):  # Use enhanced detection
                # Compute mask from outer band
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
        
        # Append to deque (automatically pops oldest if full)
        self.pending_gates.append(new_mask)
        
        # ===== I. COMPUTE NEXT c_eff (for next step) =====
        medG = float(np.median(Lambda_b_pre))
        if medG > 0:
            norm_tail = (lam_p99_out_pre / medG - 1.0) if out_band.any() else 0.0
        else:
            norm_tail = 0.0
        norm_tail = float(np.clip(norm_tail, -0.5, 3.0))
        self.c_eff_current = float(np.clip(
            self.c0 * (1.0 + self.gamma * norm_tail), 
            self.c0, 
            self.c_eff_max
        ))
        
        if not record:
            return None
        
        # Debug logging every 25 steps
        if step % 25 == 0:
            queue_head = 'Y' if (len(self.pending_gates) > 0 and self.pending_gates[0] is not None) else 'N'
            print(f"[t={step:03d}] λ_pre={lam_p99_out_pre:.3f} "
                  f"S_RT={S_mo:.3f} queue_head={queue_head} c_eff={self.c_eff_current:.3f}")
        
        return dict(
            t=step,
            entropy_RT_mo=S_mo,
            region_A_size=float(R.sum()),
            region_A_perimeter=parts["perimeter"],
            region_A_holes=parts["holes"],
            region_A_curvature=parts["curvature"],
            lambda_p99_A_out_pre=lam_p99_out_pre,
            lambda_p99_A_in_pre=lam_p99_in_pre,
            lambda_p99_A_out_post=np.nan,  # Not measured to avoid confusion
            lambda_p99_A_in_post=np.nan,
            c_eff=self.c_eff_current,
            gate_applied_px=int(gate_px)
        )
    
    def run_with_burnin(self, burn_in: int = 250, measure_steps: int = 300) -> List[Dict]:
        """
        Run simulation with phase-aware SOC control.
        
        Args:
            burn_in: Number of steps with normal SOC (default 250)
            measure_steps: Number of steps with reduced SOC (default 300)
            
        Returns:
            List of measurement dictionaries
        """
        # Burn-in phase with normal SOC
        self.phase = 'burnin'
        print(f"[BURN-IN] Running {burn_in} steps with SOC={self.soc_rate_burnin}...")
        for t in range(burn_in):
            self._step_internal(t, record=False)
        
        # Measurement phase with reduced SOC
        self.phase = 'measure'
        print(f"[MEASURE] Recording {measure_steps} steps with SOC={self.soc_rate_measure}...")
        rows = []
        for t in range(measure_steps):
            rec = self._step_internal(burn_in + t, record=True)
            rec['t'] = t  # Reset to 0-based for measurements
            rows.append(rec)
        
        return rows
    
    def step_once(self, step: int, weights: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Legacy interface: Execute one complete timestep.
        For compatibility with existing code.
        """
        # Set weights if provided
        if weights is not None:
            self.rt_weights.WEIGHT_PERIMETER = weights[0]
            self.rt_weights.WEIGHT_HOLES = weights[1]
            self.rt_weights.WEIGHT_CURVATURE = weights[2]
        
        return self._step_internal(step, record=True)
