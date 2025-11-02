"""
Λ³-AdS/CFT Core Module
=====================

This module implements the core Λ³ self-evolving automaton coupled with
AdS/CFT holographic correspondence. It provides the complete machinery for
demonstrating spacetime emergence from critical quantum information.

Key components:
- Cell: Agent with genome, cooperation, energy
- Automaton: Full Λ³ × AdS/CFT dynamics
- Geodesic Gating: Discrete RT surface rewiring
- Holographic Renormalization: Bulk → Boundary feedback

"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from .geometry import (
    outer_boundary_band,
    inner_boundary_band,
    k_th_largest_mask,
    laplacian2d
)

# ============================================================================
# Constants
# ============================================================================

# Agent dynamics
HARVEST_RATE: float = 0.05
SHARE_RATE: float = 0.02
COOP_UPDATE_RATE: float = 0.05
THERMAL_NOISE: float = 0.01
DEATH_THRESHOLD: float = 0.15
BIRTH_THRESHOLD: float = 2.0
BIRTH_PROBABILITY: float = 0.01
MUTATION_RATE: float = 0.02

# Resource dynamics
RESOURCE_DIFFUSION: float = 0.03
RESOURCE_REPLENISH: float = 0.004
RESOURCE_MAX: float = 3.0

# AdS/CFT parameters
BULK_DIFFUSION: float = 0.012
BOUNDARY_PAYOFF_RATE: float = 0.08
SOC_RATE: float = 0.01
LAMBDA_CRITICAL: float = 1.0

# RT entropy weights
WEIGHT_PERIMETER: float = 1.0
WEIGHT_HOLES: float = 2.0
WEIGHT_CURVATURE: float = 0.5

# ============================================================================
# Agent Definition
# ============================================================================

@dataclass
class Cell:
    """
    Single agent in the Λ³ automaton.
    
    Attributes:
        alive: Survival state
        energy: Internal energy reserve
        genome: Binary array of length G (genetic code)
        coop: Cooperation propensity in [0, 1]
    """
    alive: bool
    energy: float
    genome: np.ndarray  # shape: (G,), dtype: int8
    coop: float


def init_grid(
    H: int, 
    W: int, 
    G: int, 
    rng: np.random.Generator
) -> List[List[Cell]]:
    """
    Initialize grid with random agents.
    
    Args:
        H: Grid height
        W: Grid width
        G: Genome length
        rng: NumPy random generator
        
    Returns:
        2D list of Cell objects
    """
    grid = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(Cell(
                alive=True,
                energy=float(1.0 + 0.2 * rng.standard_normal()),
                genome=rng.integers(0, 2, size=G, dtype=np.int8),
                coop=float(np.clip(0.5 + 0.2 * rng.standard_normal(), 0, 1))
            ))
        grid.append(row)
    return grid


def mutate(
    genome: np.ndarray, 
    rate: float, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply random bit flips to genome.
    
    Args:
        genome: Binary array
        rate: Mutation probability per bit
        rng: Random generator
        
    Returns:
        Mutated genome copy
    """
    g = genome.copy()
    flips = rng.random(len(g)) < rate
    g[flips] = 1 - g[flips]
    return g


# ============================================================================
# Λ³-AdS/CFT Automaton
# ============================================================================

class Automaton:
    """
    Complete Λ³ self-evolving automaton with AdS/CFT holography.
    
    This class implements:
    - Agent dynamics (birth, death, cooperation, resource competition)
    - Bulk AdS geometry with warp factor
    - Holographic renormalization (bulk → boundary)
    - Boundary CFT dynamics (Prisoner's Dilemma-like payoff)
    - Self-Organized Criticality (SOC) toward Λ=1
    - Geodesic Gating (delayed RT surface rewiring)
    
    The key observable is the correlation between:
    - λ_p99 (critical quantum information in bulk)
    - S_RT (holographic entanglement entropy)
    
    Parameters:
        H, W: Grid dimensions
        Z: Number of AdS bulk layers
        G: Genome length
        L_ads: AdS radius (geometry scale)
        alpha: HR weight decay (exp(-alpha * z))
        gate_delay: Steps to delay geodesic gating
        gate_strength: Magnitude of boundary nudge
        c_eff_max: Upper bound on holographic coupling
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        H: int = 44,
        W: int = 44,
        Z: int = 24,
        G: int = 56,
        L_ads: float = 1.0,
        alpha: float = 0.9,
        gate_delay: int = 1,
        gate_strength: float = 0.15,
        c_eff_max: float = 0.18,
        seed: int = 913
    ):
        """Initialize the Λ³-AdS/CFT automaton."""
        self.H = H
        self.W = W
        self.Z = Z
        self.G = G
        self.L_ads = float(L_ads)
        self.alpha = float(alpha)
        self.G_N = 1.0  # Newton's constant (normalized)
        
        # Geodesic gating parameters
        self.gate_delay = gate_delay
        self.gate_strength = gate_strength
        self.pending_masks: List[Optional[np.ndarray]] = [None] * (gate_delay + 1)
        
        # Dynamic coupling parameters
        self.c0 = 0.08
        self.gamma = 0.8
        self.c_eff_max = c_eff_max
        
        # SOC reference
        self.theta0 = 0.19
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # State arrays
        self.grid: List[List[Cell]] = init_grid(H, W, G, self.rng)
        self.resource = np.ones((H, W), dtype=float)
        self.boundary = np.zeros((H, W), dtype=float)
        self.bulk = np.zeros((H, W, Z), dtype=float)
        
        # Initialize with spatial pattern
        self._add_spatial_seed()
    
    def _add_spatial_seed(self) -> None:
        """Add checkerboard + Gaussian pattern to initial cooperation."""
        H, W = self.H, self.W
        
        # Checkerboard
        for i in range(H):
            for j in range(W):
                if ((i // 4 + j // 4) % 2) == 0:
                    self.grid[i][j].coop = float(
                        np.clip(self.grid[i][j].coop * 1.25, 0, 1)
                    )
                else:
                    self.grid[i][j].coop = float(
                        np.clip(self.grid[i][j].coop * 0.75, 0, 1)
                    )
        
        # Gaussian bump at center
        cy, cx = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                r2 = (i - cy)**2 + (j - cx)**2
                boost = np.exp(-r2 / (H / 4)**2)
                self.grid[i][j].coop = float(
                    np.clip(self.grid[i][j].coop * (1 + 0.4 * boost), 0, 1)
                )

    # ========================================================================
    # Core Dynamics
    # ========================================================================
    
    def coop_field(self) -> np.ndarray:
        """Extract cooperation field from grid."""
        M = np.zeros((self.H, self.W), dtype=float)
        for i in range(self.H):
            for j in range(self.W):
                if self.grid[i][j].alive:
                    M[i, j] = self.grid[i][j].coop
        return M
    
    def K_over_V(self, B: np.ndarray) -> np.ndarray:
        """
        Compute Λ = K/|V| field from boundary state.
        
        K (kinetic): Gradient magnitude (instability)
        V (potential): Deviation from mean (structure)
        
        Args:
            B: Boundary cooperation field
            
        Returns:
            Λ field (energy density ratio)
        """
        coop = np.clip(B, 0, 1)
        gx = np.roll(coop, -1, axis=1) - coop
        gy = np.roll(coop, -1, axis=0) - coop
        K = np.sqrt(gx * gx + gy * gy) + 1e-12
        V = np.abs(coop - float(coop.mean())) + 1e-12
        return K / V

    def step_agents(self) -> None:
            """Execute one step of agent dynamics."""
            H, W = self.H, self.W
            C = self.coop_field()
            
            for i in range(H):
                for j in range(W):
                    c = self.grid[i][j]
                    if not c.alive:
                        continue
                    
                    # Harvest resource
                    take = min(HARVEST_RATE, self.resource[i, j])
                    self.resource[i, j] -= take
                    c.energy += take
                    
                    # Share with neighbors
                    give = min(SHARE_RATE * c.coop, c.energy)
                    if give > 0:
                        q = 0.25 * give
                        self.grid[(i + 1) % H][j].energy += q
                        self.grid[(i - 1) % H][j].energy += q
                        self.grid[i][(j + 1) % W].energy += q
                        self.grid[i][(j - 1) % W].energy += q
                        c.energy -= give
                    
                    # Update cooperation
                    neigh = 0.25 * (
                        C[(i + 1) % H][j] +
                        C[(i - 1) % H][j] +
                        C[i][(j + 1) % W] +
                        C[i][(j - 1) % W]
                    )
                    c.coop += COOP_UPDATE_RATE * (neigh - c.coop)
                    c.coop += THERMAL_NOISE * self.rng.standard_normal()
                    c.coop = float(np.clip(c.coop, 0, 1))
                    
                    # Death
                    if c.energy < DEATH_THRESHOLD:
                        c.alive = False
                        c.coop = 0.0
                    
                    # Birth
                    elif c.energy > BIRTH_THRESHOLD:
                        if self.rng.random() < BIRTH_PROBABILITY:
                            child = Cell(
                                alive=True,
                                energy=c.energy * 0.5,
                                genome=mutate(c.genome, MUTATION_RATE, self.rng),
                                coop=c.coop
                            )
                            c.energy *= 0.5
                            ii = (i + self.rng.integers(-1, 2)) % H
                            jj = (j + self.rng.integers(-1, 2)) % W
                            self.grid[ii][jj] = child
            
            # Resource dynamics
            self.resource += RESOURCE_DIFFUSION * laplacian2d(self.resource)
            self.resource += RESOURCE_REPLENISH
            self.resource = np.clip(self.resource, 0, RESOURCE_MAX)

    # ========================================================================
    # AdS/CFT Holography
    # ========================================================================
    
    def update_bulk(self) -> None:
        """
        Update AdS bulk geometry with warp factor.
        
        The bulk Lambda field is computed from boundary via:
        Lambda(x, y, z) = warp(z) * [Lambda_boundary + diffusion]
        
        where warp(z) = (L_ads / z)^2 implements anti-de Sitter geometry.
        """
        Lambda0 = self.K_over_V(self.boundary)
        self.bulk[..., 0] = Lambda0
        
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz  # z in (0, 1]
        
        for k in range(1, self.Z):
            warp = (self.L_ads / z[k]) ** 2
            prev = self.bulk[..., k - 1]
            mixed = prev + BULK_DIFFUSION * laplacian2d(prev)
            self.bulk[..., k] = np.clip(warp * mixed, 0, None)
    
    def holographic_renormalization(self, c_eff: float) -> None:
        """
        Apply holographic renormalization: bulk → boundary feedback.
        
        Uses exponentially decaying weights favoring shallow z layers:
        w(z) ∝ exp(-alpha * z)
        
        Args:
            c_eff: Effective coupling strength (dynamically adjusted)
        """
        dz = 1.0 / self.Z
        z = (np.arange(self.Z) + 1) * dz
        
        # Shallow-biased weights
        w = np.exp(-self.alpha * z)
        w = w / (w.sum() + 1e-12)
        
        # Project bulk onto boundary
        back = np.tensordot(self.bulk, w, axes=(2, 0))
        
        # Update boundary
        self.boundary += c_eff * (back - self.boundary)
    
    def update_boundary_payoff(self) -> None:
        """
        Update boundary via Prisoner's Dilemma-like payoff.
        
        Cooperation benefits from cooperating neighbors but
        is exploited by defectors.
        """
        B = self.boundary
        neigh = 0.25 * (
            np.roll(B, 1, axis=0) +
            np.roll(B, -1, axis=0) +
            np.roll(B, 1, axis=1) +
            np.roll(B, -1, axis=1)
        )
        payoff = neigh * (1.4 - 0.6 * B) - 0.4 * B * (1 - neigh)
        self.boundary += BOUNDARY_PAYOFF_RATE * payoff
        self.boundary = np.clip(self.boundary, 0, 1)
    
    def SOC_tune(self) -> None:
        """
        Self-Organized Criticality: tune boundary toward Λ=1.
        """
        Lb = self.K_over_V(self.coop_field())
        delta = float(Lb.mean() - LAMBDA_CRITICAL)
        self.boundary = np.clip(
            self.boundary - SOC_RATE * delta, 
            0, 1
        )
