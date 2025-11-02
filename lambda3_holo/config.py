from dataclasses import dataclass

@dataclass
class ModelConfig:
    H: int = 44
    W: int = 44
    Z: int = 24
    L_ads: float = 1.0
    alpha: float = 0.9
    c0: float = 0.08
    gamma: float = 0.8
    c_eff_max: float = 0.18
    gate_delay: int = 1
    gate_strength: float = 0.15
    soc_rate: float = 0.01  # Λ self-organizing rate
    seed: int = 913

@dataclass
class RTWeights:
    w_len: float = 1.0
    w_hole: float = 2.0
    w_curv: float = 0.5
