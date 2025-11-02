"""
Configuration & Physical Constants for Λ³-AdS/CFT Automaton
=============================================================

This module defines all physical, biological, and geometric parameters
for the holographic causality experiments.

Physical Interpretation:
- Agent dynamics: Cooperation-driven resource sharing & evolution
- Resource dynamics: Diffusive field with replenishment
- AdS/CFT: Bulk-boundary holographic correspondence
- RT entropy: Multi-objective entanglement proxy
"""

from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Agent Dynamics Constants
# ============================================================================

@dataclass
class AgentDynamics:
    """
    Parameters governing individual agent behavior.
    
    Physical Interpretation:
    - Harvesting: Conversion of environmental resources to internal energy
    - Sharing: Cooperative redistribution (altruism cost)
    - Cooperation: Local alignment via neighbor influence
    - Birth/Death: Population regulation via energy thresholds
    """
    
    # Resource harvesting
    HARVEST_RATE: float = 0.05
    """Fraction of local resource converted to internal energy per step
    
    Physical meaning: Metabolic efficiency × foraging rate
    Typical range: [0.01, 0.1]
    """
    
    # Cooperation & sharing
    SHARE_RATE: float = 0.02
    """Energy sharing rate with neighbors (altruism cost)
    
    Physical meaning: Cooperative investment strength
    Scaled by individual's cooperation level (coop ∈ [0,1])
    """
    
    COOP_UPDATE_RATE: float = 0.05
    """Learning rate for cooperation alignment with neighbors
    
    Physical meaning: Social influence / conformity pressure
    Higher values → faster convergence to local norms
    """
    
    THERMAL_NOISE: float = 0.01
    """Stochastic fluctuation in cooperation updates
    
    Physical meaning: Behavioral randomness / exploration
    Prevents deterministic lock-in
    """
    
    # Population dynamics
    DEATH_THRESHOLD: float = 0.15
    """Minimum energy required for survival
    
    Below this threshold → agent dies (alive=False)
    """
    
    BIRTH_THRESHOLD: float = 2.0
    """Energy surplus enabling reproduction
    
    Above this threshold → probabilistic offspring creation
    """
    
    BIRTH_PROBABILITY: float = 0.01
    """Reproduction chance per step when energy > BIRTH_THRESHOLD
    
    Keeps population density regulated
    """
    
    MUTATION_RATE: float = 0.02
    """Bit-flip probability per genome bit during reproduction
    
    Drives genetic diversity & evolutionary adaptation
    """


# ============================================================================
# Resource Field Dynamics
# ============================================================================

@dataclass
class ResourceDynamics:
    """
    Environmental resource field parameters.
    
    Physical Interpretation:
    - Diffusion: Spatial redistribution (e.g., nutrient spreading)
    - Replenishment: External influx (e.g., sunlight, rainfall)
    - Max capacity: Carrying capacity of environment
    """
    
    RESOURCE_DIFFUSION: float = 0.03
    """Diffusion coefficient for resource field
    
    Physical meaning: Spatial mixing rate (Laplacian operator)
    Units: [length²/time] in dimensionless grid units
    """
    
    RESOURCE_REPLENISH: float = 0.004
    """Constant replenishment rate per grid cell per step
    
    Physical meaning: External energy input (primary production)
    Maintains non-equilibrium steady state
    """
    
    RESOURCE_MAX: float = 3.0
    """Maximum resource density per cell
    
    Physical meaning: Environmental carrying capacity
    Prevents unbounded accumulation
    """


# ============================================================================
# AdS/CFT Holographic Parameters
# ============================================================================

@dataclass
class AdSCFTParams:
    """
    Anti-de Sitter / Conformal Field Theory parameters.
    
    Physical Interpretation:
    - Bulk: Higher-dimensional AdS spacetime (emergent gravity)
    - Boundary: Lower-dimensional CFT (quantum field)
    - Holographic Renormalization: Bulk → Boundary feedback
    - SOC: Self-organized criticality toward Λ=1
    """
    
    # Bulk geometry
    BULK_DIFFUSION: float = 0.012
    """Bulk field diffusion coefficient
    
    Physical meaning: Information spreading in AdS interior
    Related to bulk viscosity / entanglement scrambling
    """
    
    L_ADS: float = 1.0
    """AdS radius (geometry scale)
    
    Sets the curvature scale: ds² ∝ (L/z)² 
    Normalized to 1.0 for dimensionless units
    """
    
    ALPHA: float = 0.9
    """Holographic renormalization decay rate
    
    HR weights: w(z) = exp(-α·z)
    Higher α → stronger bias toward shallow bulk layers
    Physical meaning: UV/IR mixing control
    """
    
    # Boundary CFT
    BOUNDARY_PAYOFF_RATE: float = 0.08
    """Update rate for boundary field via game-theoretic payoff
    
    Implements Prisoner's Dilemma-like interaction
    Models spontaneous pattern formation
    """
    
    # Dynamic coupling
    C0: float = 0.08
    """Base holographic coupling strength (c_eff minimum)
    
    c_eff = C0 * (1 + γ * ΔΛ), capped at C_EFF_MAX
    """
    
    GAMMA: float = 0.8
    """Coupling amplification factor
    
    Relates bulk criticality (ΔΛ) to HR feedback strength
    Higher γ → stronger bulk-boundary resonance
    """
    
    C_EFF_MAX: float = 0.18
    """Maximum HR coupling (zero-lag suppression)
    
    Critical for causal structure:
    - Too high → instantaneous correlation dominates
    - This value → delayed causality emerges
    """
    
    # Self-organized criticality
    SOC_RATE: float = 0.01
    """Tuning rate toward critical point Λ=1
    
    Physical meaning: Adaptive control strength
    Boundary adjusts to maintain bulk near-criticality
    """
    
    LAMBDA_CRITICAL: float = 1.0
    """Target critical value for Λ = K/V
    
    Phase transition point:
    - Λ < 1: Stable (replication)
    - Λ ≈ 1: Critical (edge of chaos)
    - Λ > 1: Unstable (evolution/death)
    """
    
    G_N: float = 1.0
    """Newton's constant (normalized)
    
    Appears in RT formula: S_RT = Area / (4 G_N)
    Set to 1.0 for Planck units
    """


# ============================================================================
# Geodesic Gating (Delayed RT Surface Rewiring)
# ============================================================================

@dataclass
class GeodesicGating:
    """
    Delayed geodesic gating for causal structure control.
    
    Physical Interpretation:
    - Detects bulk criticality spikes (λ_p99 outliers)
    - Schedules boundary nudge after gate_delay steps
    - Implements discrete RT minimal surface rewiring
    """
    
    GATE_DELAY: int = 1
    """Number of steps to delay gate application
    
    Critical for causality:
    - delay=0: Instantaneous (no causal lag)
    - delay=1: One-step lag (standard)
    - delay=2: Extended lag (stronger separation)
    
    Experimentally optimal: delay=1 for Spearman 0.461 @ lag=+6
    """
    
    GATE_STRENGTH: float = 0.15
    """Boundary boost magnitude when gate activates
    
    B[mask] ← B[mask] + strength * (1 - B[mask])
    
    Physical meaning: RT surface "jump" amplitude
    Lower values (0.15 vs 0.25) → cleaner causal signal
    """
    
    IQR_MULTIPLIER: float = 1.5
    """Tukey outlier threshold for gate triggering
    
    Gate activates if: λ_p99 > Q3 + IQR_MULTIPLIER * IQR
    Standard statistical outlier detection
    """


# ============================================================================
# RT Entropy Multi-Objective Functional
# ============================================================================

@dataclass
class RTWeights:
    """
    Weights for multi-objective Ryu-Takayanagi-like entropy.
    
    S_RT = (w_len·perimeter + w_hole·holes + w_curv·curvature) / (4 G_N)
    
    Physical Interpretation:
    - Perimeter: Entanglement area (standard RT formula)
    - Holes: Topological complexity (Euler characteristic proxy)
    - Curvature: Geometric frustration (corner count)
    
    Motivation:
    Pure area law insufficient for discrete lattice → add topology & curvature
    """
    
    WEIGHT_PERIMETER: float = 1.0
    """Weight for boundary length (area law term)
    
    Standard RT contribution: minimal surface area
    """
    
    WEIGHT_HOLES: float = 2.0
    """Weight for hole count (topology term)
    
    Euler characteristic proxy: χ = 1 - holes
    Higher weight → topology dominates entropy
    
    Counted via non-periodic flood fill (edges as exterior)
    """
    
    WEIGHT_CURVATURE: float = 0.5
    """Weight for corner count (geometry term)
    
    Discrete curvature proxy: corners in 2×2 neighborhoods
    Captures geometric frustration of boundary
    """


# ============================================================================
# Preset Configurations
# ============================================================================

@dataclass
class ModelConfig:
    """
    Complete model configuration combining all parameter groups.
    """
    
    # Grid geometry
    H: int = 44
    W: int = 44
    Z: int = 24  # Bulk depth
    
    # Parameter groups
    agent: AgentDynamics = None
    resource: ResourceDynamics = None
    ads_cft: AdSCFTParams = None
    gating: GeodesicGating = None
    rt_weights: RTWeights = None
    
    # Random seed
    seed: int = 913
    
    def __post_init__(self):
        """Initialize parameter groups with defaults if not provided."""
        if self.agent is None:
            self.agent = AgentDynamics()
        if self.resource is None:
            self.resource = ResourceDynamics()
        if self.ads_cft is None:
            self.ads_cft = AdSCFTParams()
        if self.gating is None:
            self.gating = GeodesicGating()
        if self.rt_weights is None:
            self.rt_weights = RTWeights()


# ============================================================================
# Experimental Presets
# ============================================================================

def config_phaseshift_ixb() -> ModelConfig:
    """
    PhaseShift-IXb configuration (gate_delay=1, λ_pre driver).
    
    Results:
    - Pearson: 0.347 @ lag=+6
    - Spearman: 0.461 @ lag=+6 ★ (target > 0.4 achieved)
    - TE(λ→S): 0.0525, TE(S→λ): 0.1314
    """
    return ModelConfig(
        H=44, W=44, Z=24,
        gating=GeodesicGating(GATE_DELAY=1, GATE_STRENGTH=0.15),
        ads_cft=AdSCFTParams(C_EFF_MAX=0.18),
        seed=913
    )


def config_phaseshift_ixc() -> ModelConfig:
    """
    PhaseShift-IXc configuration (gate_delay=2, extended lag).
    
    Results:
    - Pearson: 0.275 @ lag=+12
    - Spearman: 0.313 @ lag=+12
    
    Note: Longer lag but weaker amplitude vs IXb
    """
    return ModelConfig(
        H=44, W=44, Z=24,
        gating=GeodesicGating(GATE_DELAY=2, GATE_STRENGTH=0.15),
        ads_cft=AdSCFTParams(C_EFF_MAX=0.18),
        seed=913
    )


# ============================================================================
# Parameter Validation
# ============================================================================

def validate_config(config: ModelConfig) -> Tuple[bool, str]:
    """
    Check configuration for physical consistency.
    
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    # Grid must be positive
    if config.H <= 0 or config.W <= 0 or config.Z <= 0:
        errors.append("Grid dimensions must be positive")
    
    # Energy thresholds
    if config.agent.DEATH_THRESHOLD >= config.agent.BIRTH_THRESHOLD:
        errors.append("DEATH_THRESHOLD must be < BIRTH_THRESHOLD")
    
    # Diffusion stability (rough check)
    if config.resource.RESOURCE_DIFFUSION > 0.25:
        errors.append("RESOURCE_DIFFUSION too high (numerical instability risk)")
    
    # HR coupling bounds
    if config.ads_cft.C0 > config.ads_cft.C_EFF_MAX:
        errors.append("C0 must be <= C_EFF_MAX")
    
    # Gate parameters
    if config.gating.GATE_DELAY < 0:
        errors.append("GATE_DELAY must be non-negative")
    if not (0 < config.gating.GATE_STRENGTH < 1):
        errors.append("GATE_STRENGTH must be in (0, 1)")
    
    # RT weights
    if any(w < 0 for w in [
        config.rt_weights.WEIGHT_PERIMETER,
        config.rt_weights.WEIGHT_HOLES,
        config.rt_weights.WEIGHT_CURVATURE
    ]):
        errors.append("RT weights must be non-negative")
    
    if errors:
        return False, "; ".join(errors)
    return True, "OK"
