"""
Λ³ MPO-TEBD: Large-Scale Photo-Induced Superconductivity
=========================================================

MPO (Matrix Product Operator) implementation for measuring:
- V (vorticity)
- V² (vortex pair energy)  
- Sz_total (magnetization)

WITHOUT converting MPS to dense vector!

Memory scaling:
- Dense: O(2^N) → explodes at N≈20
- MPO:   O(N × χ² × D²) → scales to N=100+

Key insight from Λ³:
- MPS = Λ³ hierarchical structure (Axiom 1)
- MPO = operator in Λ-space
- Contraction = constraint satisfaction (CSP)

Author: Masamichi & Tamaki (環)
Date: 2025-11-29
"""

import numpy as np
from scipy.linalg import expm
import time
import gc

# =============================================================================
# Pauli Matrices (local)
# =============================================================================

# Physical dimension
D_PHYS = 2

# Pauli matrices
PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# =============================================================================
# MPO Class
# =============================================================================

class MPO:
    """
    Matrix Product Operator for 1D chain
    
    Structure:
        W[i] has shape (D_L, D_R, d, d')
        where:
            D_L = left bond dimension
            D_R = right bond dimension
            d = physical input dimension (ket)
            d' = physical output dimension (bra)
    
    Convention:
        - First site: D_L = 1
        - Last site: D_R = 1
        - W[i]_{αβ}^{ss'} acts as |s'⟩⟨s|
    """
    
    def __init__(self, N):
        self.N = N
        self.W = [None] * N  # Will hold tensors
    
    @classmethod
    def identity(cls, N):
        """Create identity MPO: I ⊗ I ⊗ ... ⊗ I"""
        mpo = cls(N)
        for i in range(N):
            # Shape: (1, 1, 2, 2) - trivial bonds
            W = np.zeros((1, 1, 2, 2), dtype=np.complex128)
            W[0, 0, :, :] = PAULI_I
            mpo.W[i] = W
        return mpo
    
    @classmethod
    def sum_local(cls, N, op):
        """
        Create MPO for Σ_i O_i (sum of local operators)
        
        Example: Sz_total = Σ_i Sz_i
        
        MPO bond dimension = 2:
            W = | I  0 |
                | O  I |
        """
        mpo = cls(N)
        
        for i in range(N):
            if i == 0:
                # First site: (1, 2, 2, 2)
                W = np.zeros((1, 2, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = op      # Start sum
                W[0, 1, :, :] = PAULI_I  # Pass through
            elif i == N - 1:
                # Last site: (2, 1, 2, 2)
                W = np.zeros((2, 1, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = PAULI_I  # Collect sum
                W[1, 0, :, :] = op       # Add last
            else:
                # Bulk site: (2, 2, 2, 2)
                W = np.zeros((2, 2, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = op       # Add to sum
                W[0, 1, :, :] = PAULI_I  # Pass through
                W[1, 1, :, :] = PAULI_I  # Keep sum
            
            mpo.W[i] = W
        
        return mpo
    
    @classmethod
    def sz_total(cls, N):
        """Σ_i Sz_i"""
        return cls.sum_local(N, PAULI_Z)
    
    @classmethod
    def vorticity_1d(cls, N):
        """
        Vorticity operator for 1D chain:
        V = Σ_i 2(Sx_i Sy_{i+1} - Sy_i Sx_{i+1})
        
        This is a nearest-neighbor two-site operator.
        
        MPO structure (bond dimension D=4):
            State 0: accumulator (collects completed terms)
            State 1: Sx placed, waiting for Sy
            State 2: Sy placed, waiting for Sx
            State 3: identity pass-through
        
        Transitions:
            3 → 1: place Sx (start term)
            3 → 2: place Sy (start term)
            1 → 0: place 2Sy (complete Sx·Sy)
            2 → 0: place -2Sx (complete -Sy·Sx)
            3 → 3: identity (bulk, pass through)
            0 → 0: identity (bulk, keep accumulator)
        """
        mpo = cls(N)
        D = 4  # Bond dimension
        
        for i in range(N):
            if i == 0:
                # First site: (1, D, 2, 2)
                # Can only START terms (no identity to output)
                W = np.zeros((1, D, 2, 2), dtype=np.complex128)
                W[0, 1, :, :] = PAULI_X   # Start Sx → state 1
                W[0, 2, :, :] = PAULI_Y   # Start Sy → state 2
                W[0, 3, :, :] = PAULI_I   # Identity → state 3 (pass-through for N>2)
                
            elif i == N - 1:
                # Last site: (D, 1, 2, 2)
                # Can only COMPLETE terms
                W = np.zeros((D, 1, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = PAULI_I          # Accumulator: keep sum
                W[1, 0, :, :] = 2.0 * PAULI_Y    # Complete: Sx → 2Sy
                W[2, 0, :, :] = -2.0 * PAULI_X   # Complete: Sy → -2Sx
                # W[3, 0] = 0: identity path ends with nothing
                
            else:
                # Bulk site: (D, D, 2, 2)
                W = np.zeros((D, D, 2, 2), dtype=np.complex128)
                
                # Accumulator keeps accumulating
                W[0, 0, :, :] = PAULI_I
                
                # Complete terms and add to accumulator
                W[1, 0, :, :] = 2.0 * PAULI_Y    # Sx·Sy term
                W[2, 0, :, :] = -2.0 * PAULI_X   # -Sy·Sx term
                
                # Identity path: start new terms
                W[3, 1, :, :] = PAULI_X   # Start new Sx
                W[3, 2, :, :] = PAULI_Y   # Start new Sy
                W[3, 3, :, :] = PAULI_I   # Pass through
            
            mpo.W[i] = W
        
        return mpo
    
    @classmethod
    def xy_hamiltonian(cls, N):
        """
        XY Hamiltonian: H = Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
        
        MPO structure (bond dimension D=4):
            State 0: accumulator
            State 1: Sx placed, waiting for Sx
            State 2: Sy placed, waiting for Sy
            State 3: identity pass-through
        """
        mpo = cls(N)
        D = 4
        
        for i in range(N):
            if i == 0:
                # First site
                W = np.zeros((1, D, 2, 2), dtype=np.complex128)
                W[0, 1, :, :] = PAULI_X   # Start Sx
                W[0, 2, :, :] = PAULI_Y   # Start Sy
                W[0, 3, :, :] = PAULI_I   # Identity for N>2
                
            elif i == N - 1:
                # Last site
                W = np.zeros((D, 1, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = PAULI_I   # Keep accumulator
                W[1, 0, :, :] = PAULI_X   # Complete Sx·Sx
                W[2, 0, :, :] = PAULI_Y   # Complete Sy·Sy
                
            else:
                # Bulk site
                W = np.zeros((D, D, 2, 2), dtype=np.complex128)
                W[0, 0, :, :] = PAULI_I   # Accumulator
                W[1, 0, :, :] = PAULI_X   # Complete Sx·Sx
                W[2, 0, :, :] = PAULI_Y   # Complete Sy·Sy
                W[3, 1, :, :] = PAULI_X   # Start new Sx
                W[3, 2, :, :] = PAULI_Y   # Start new Sy
                W[3, 3, :, :] = PAULI_I   # Pass through
            
            mpo.W[i] = W
        
        return mpo


# =============================================================================
# MPS Class (Enhanced with MPO expectation)
# =============================================================================

class MPS1D_MPO:
    """
    1D MPS with MPO-based expectation values
    
    Key feature: ⟨ψ|O|ψ⟩ computed WITHOUT to_dense()!
    
    Memory: O(N × χ² × D²) instead of O(2^N)
    """
    
    def __init__(self, N, chi_max=64, init_state="up"):
        self.N = N
        self.chi_max = chi_max
        self.d = 2
        
        # Initialize MPS tensors
        self.A = []
        for i in range(N):
            A = np.zeros((1, self.d, 1), dtype=np.complex128)
            if init_state in ("up", "0", "+z"):
                A[0, 0, 0] = 1.0
            elif init_state in ("down", "1", "-z"):
                A[0, 1, 0] = 1.0
            elif init_state == "random":
                v = np.random.randn(self.d) + 1j * np.random.randn(self.d)
                v /= np.linalg.norm(v)
                A[0, :, 0] = v
            self.A.append(A)
        
        self.bond_svals = [np.array([1.0], dtype=np.float64) for _ in range(N - 1)]
    
    def apply_1site_gate(self, i, U):
        """Apply 1-site gate U (2×2) to site i"""
        A = self.A[i]  # (χL, d, χR)
        # U: (d', d), A: (χL, d, χR)
        theta = np.tensordot(U, A, axes=(1, 1))  # (d', χL, χR)
        theta = np.transpose(theta, (1, 0, 2))   # (χL, d', χR)
        self.A[i] = theta
    
    def apply_2site_gate(self, i, U, chi_max=None):
        """Apply 2-site gate with SVD truncation"""
        if chi_max is None:
            chi_max = self.chi_max
        
        A_i = self.A[i]
        A_j = self.A[i + 1]
        
        chiL, d, chiM = A_i.shape
        chiM2, d2, chiR = A_j.shape
        
        # Contract two sites
        theta = np.tensordot(A_i, A_j, axes=(2, 0))  # (χL, d, d, χR)
        
        # Apply gate
        U4 = U.reshape(2, 2, 2, 2)
        theta = np.tensordot(U4, theta, axes=([2, 3], [1, 2]))
        theta = np.transpose(theta, (2, 0, 1, 3))  # (χL, d, d, χR)
        
        # SVD
        theta_mat = theta.reshape(chiL * 2, 2 * chiR)
        U_svd, S_svd, Vh_svd = np.linalg.svd(theta_mat, full_matrices=False)
        
        # Truncate
        chi_new = min(chi_max, len(S_svd))
        U_svd = U_svd[:, :chi_new]
        S_svd = S_svd[:chi_new]
        Vh_svd = Vh_svd[:chi_new, :]
        
        # Normalize
        S_norm = np.linalg.norm(S_svd)
        if S_norm > 0:
            S_svd = S_svd / S_norm
            U_svd *= S_norm
        
        # Reconstruct
        A_i_new = U_svd.reshape(chiL, 2, chi_new)
        A_j_new = (np.diag(S_svd) @ Vh_svd).reshape(chi_new, 2, chiR)
        
        self.A[i] = A_i_new
        self.A[i + 1] = A_j_new
        self.bond_svals[i] = S_svd.copy()
    
    def expectation_mpo(self, mpo):
        """
        Compute ⟨ψ|O|ψ⟩ using MPO contraction
        
        Algorithm: Transfer matrix method (left-to-right sweep)
        
        L has shape (χ_bra, D_mpo, χ_ket)
        
        Optimized with sequential tensordot (faster than einsum!)
        
        Complexity: O(N × χ² × D² × d²)
        """
        # Initialize left boundary: (1, 1, 1) scalar
        L = np.ones((1, 1, 1), dtype=np.complex128)
        
        for i in range(self.N):
            A = self.A[i]           # (χL, d, χR)
            A_conj = np.conj(A)     # (χL, d, χR) for bra
            W = mpo.W[i]            # (D_L, D_R, d_ket, d_bra)
            
            # Sequential contraction (much faster than einsum!)
            # 
            # Step 1: L × A → temp1
            # L: (χL_bra, D_L, χL_ket), A: (χL_ket, d, χR_ket)
            # Contract χL_ket → temp1: (χL_bra, D_L, d, χR_ket)
            temp1 = np.tensordot(L, A, axes=(2, 0))
            
            # Step 2: temp1 × W → temp2
            # temp1: (χL_bra, D_L, d_ket, χR_ket)
            # W: (D_L, D_R, d_ket, d_bra)
            # Contract D_L, d_ket → temp2: (χL_bra, χR_ket, D_R, d_bra)
            temp2 = np.tensordot(temp1, W, axes=([1, 2], [0, 2]))
            
            # Step 3: temp2 × A* → L_new
            # temp2: (χL_bra, χR_ket, D_R, d_bra)
            # A*: (χL_bra, d_bra, χR_bra)
            # Contract χL_bra, d_bra → L_new: (χR_ket, D_R, χR_bra)
            L_new = np.tensordot(temp2, A_conj, axes=([0, 3], [0, 1]))
            
            # Transpose to standard order: (χR_bra, D_R, χR_ket)
            L = np.transpose(L_new, (2, 1, 0))
        
        # Final: L should be (1, 1, 1) scalar
        return L[0, 0, 0]
    
    def bond_entropies(self):
        """Compute bond entanglement entropies from singular values"""
        ent = []
        for s in self.bond_svals:
            s = np.array(s, dtype=np.float64)
            if s.size == 0:
                ent.append(0.0)
                continue
            p = s ** 2
            Z = p.sum()
            if Z <= 0:
                ent.append(0.0)
                continue
            p = p / Z
            ent.append(float(-np.sum(p * np.log(p + 1e-12))))
        return np.array(ent, dtype=np.float64)
    
    def entanglement_contour(self):
        """
        Compute entanglement contour S(i)
        S(i) = average of neighboring bond entropies
        """
        S_bond = self.bond_entropies()
        S_site = np.zeros(self.N)
        
        S_site[0] = S_bond[0]
        S_site[-1] = S_bond[-1]
        
        if self.N > 2:
            S_site[1:-1] = 0.5 * (S_bond[:-1] + S_bond[1:])
        
        return S_site


# =============================================================================
# V² via MPO-MPO product (or direct measurement)
# =============================================================================

def mpo_squared(mpo):
    """
    Compute O² as MPO
    
    If O has bond dimension D, O² has bond dimension D²
    
    For V with D=4, V² has D=16
    """
    N = mpo.N
    mpo_sq = MPO(N)
    
    for i in range(N):
        W = mpo.W[i]  # (D_L, D_R, d, d')
        D_L, D_R, d, d_prime = W.shape
        
        # W² = W ⊗ W with physical indices contracted
        # W²[a₁a₂, b₁b₂, s, s''] = Σ_{s'} W[a₁,b₁,s,s'] × W[a₂,b₂,s',s'']
        
        W_sq = np.zeros((D_L * D_L, D_R * D_R, d, d), dtype=np.complex128)
        
        for a1 in range(D_L):
            for a2 in range(D_L):
                for b1 in range(D_R):
                    for b2 in range(D_R):
                        # Matrix product over physical space
                        W_sq[a1 * D_L + a2, b1 * D_R + b2, :, :] = \
                            W[a1, b1, :, :] @ W[a2, b2, :, :]
        
        mpo_sq.W[i] = W_sq
    
    return mpo_sq


# =============================================================================
# Main Simulation with MPO
# =============================================================================

def run_mpo_tebd(
    N_sites: int = 32,
    N_pump: int = 200,
    N_free: int = 200,
    dt: float = 0.05,
    g: float = 0.3,
    omega: float = 0.3,
    oam_l: int = 1,
    chirality: int = 1,
    chi_max: int = 64,
    measure_every: int = 5,
    verbose: bool = True,
):
    """
    Λ³ TEBD with MPO-based measurements
    
    Scales to N=100+ sites!
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("🌀 Λ³ MPO-TEBD: Large-Scale Photo-Induced Superconductivity")
        print("=" * 70)
        print(f"  N_sites = {N_sites} (Memory: O(N×χ²×D²), NOT O(2^N)!)")
        print(f"  Pump: {N_pump} steps, Free: {N_free} steps")
        print(f"  dt = {dt}, g = {g}, ω = {omega}")
        print(f"  OAM: l = {oam_l}, χ = {chirality}")
        print(f"  chi_max = {chi_max}")
        print("=" * 70)
    
    # Build MPO operators (once!)
    if verbose:
        print("\n📐 Building MPO operators...")
    
    V_mpo = MPO.vorticity_1d(N_sites)
    V2_mpo = mpo_squared(V_mpo)
    Sz_mpo = MPO.sz_total(N_sites)
    
    if verbose:
        print(f"  V_mpo bond dim: {V_mpo.W[N_sites//2].shape[0]}")
        print(f"  V²_mpo bond dim: {V2_mpo.W[N_sites//2].shape[0]}")
        print(f"  Sz_mpo bond dim: {Sz_mpo.W[N_sites//2].shape[0]}")
    
    # Local operators for TEBD
    sx_loc = PAULI_X
    sy_loc = PAULI_Y
    
    # XY bond Hamiltonian
    h_bond = np.kron(sx_loc, sx_loc) + np.kron(sy_loc, sy_loc)
    U_bond = expm(-1j * dt * h_bond)
    
    # OAM phases (1D chain mapped to circle)
    oam_phases = 2.0 * np.pi * np.arange(N_sites) / float(N_sites)
    
    # Initialize MPS
    mps = MPS1D_MPO(N_sites, chi_max=chi_max, init_state="up")
    
    # Storage
    times = []
    Sz_list = []
    V_list = []
    V2_list = []
    S_contour_hist = []
    
    N_total = N_pump + N_free
    t_start = time.time()
    
    if verbose:
        print("\n🚀 Starting time evolution...")
    
    for n in range(N_total):
        t = n * dt
        phase = "PUMP" if n < N_pump else "FREE"
        
        # OAM pump (1-site gates)
        if n < N_pump:
            omega_t = omega * t
            for j in range(N_sites):
                phase_j = omega_t + chirality * oam_l * oam_phases[j]
                h_loc = g * (np.cos(phase_j) * sx_loc + np.sin(phase_j) * sy_loc)
                U_loc = expm(-1j * dt * h_loc)
                mps.apply_1site_gate(j, U_loc)
        
        # XY TEBD (even-odd Trotter)
        for i in range(0, N_sites - 1, 2):
            mps.apply_2site_gate(i, U_bond, chi_max=chi_max)
        for i in range(1, N_sites - 1, 2):
            mps.apply_2site_gate(i, U_bond, chi_max=chi_max)
        
        # Measurements (MPO-based, no to_dense()!)
        if n % measure_every == 0:
            Sz = mps.expectation_mpo(Sz_mpo).real
            V = mps.expectation_mpo(V_mpo).real
            V2 = mps.expectation_mpo(V2_mpo).real
            S_site = mps.entanglement_contour()
            
            times.append(t)
            Sz_list.append(Sz)
            V_list.append(V)
            V2_list.append(V2)
            S_contour_hist.append(S_site)
            
            if verbose and n % (measure_every * 10) == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (n + 1) * (N_total - n - 1) if n > 0 else 0
                S_max = S_site.max()
                print(f"  n={n:4d} [{phase:4s}]: "
                      f"V={V:+.4f}, V²={V2:.4f}, <Sz>={Sz:+.3f}, "
                      f"S_max={S_max:.3f}  "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")
        
        if n % 50 == 0:
            gc.collect()
    
    total_time = time.time() - t_start
    
    if verbose:
        print(f"\n✅ Simulation complete in {total_time:.1f}s")
    
    # Convert to arrays
    times = np.array(times)
    Sz_arr = np.array(Sz_list)
    V_arr = np.array(V_list)
    V2_arr = np.array(V2_list)
    S_contour = np.array(S_contour_hist)
    
    # Analysis
    pump_off_idx = N_pump // measure_every
    
    if verbose and pump_off_idx < len(V_arr):
        print("\n" + "=" * 70)
        print("📊 RESULTS ANALYSIS")
        print("=" * 70)
        
        V_pump = V_arr[pump_off_idx]
        V2_pump = V2_arr[pump_off_idx]
        V_final = V_arr[-1]
        V2_final = V2_arr[-1]
        
        print(f"\nAt pump-off (n={N_pump}):")
        print(f"  V  = {V_pump:.6f}")
        print(f"  V² = {V2_pump:.6f}")
        
        print(f"\nFinal (n={N_total}):")
        print(f"  V  = {V_final:.6f}")
        print(f"  V² = {V2_final:.6f}")
        
        V_free = V_arr[pump_off_idx:]
        V2_free = V2_arr[pump_off_idx:]
        
        print(f"\nFree evolution statistics:")
        print(f"  <V>  = {V_free.mean():.6f} ± {V_free.std():.6f}")
        print(f"  <V²> = {V2_free.mean():.6f} ± {V2_free.std():.6f}")
        
        if V2_pump > 1e-8:
            persistence = V2_final / V2_pump
            print(f"\n  V² Persistence Ratio: {persistence:.4f}")
            
            if persistence > 0.5:
                print("\n  🏆🏆🏆 PERSISTENT VORTICITY CONFIRMED! 🏆🏆🏆")
                print("  → Photo-induced topological order survives!")
            elif persistence > 0.1:
                print("\n  ⚡ Partial vorticity persistence")
            else:
                print("\n  ⚠️ Vorticity decays after pump off")
    
    return {
        'times': times,
        'Sz': Sz_arr,
        'V': V_arr,
        'V2': V2_arr,
        'S_contour': S_contour,
        'N_sites': N_sites,
        'N_pump': N_pump,
        'N_free': N_free,
        'dt': dt,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_mpo_results(results, save_path='lambda3_mpo_tebd_results.png'):
    """Plot results from MPO-TEBD simulation"""
    import matplotlib.pyplot as plt
    
    times = results['times']
    V = results['V']
    V2 = results['V2']
    Sz = results['Sz']
    S_contour = results['S_contour']
    N_pump = results['N_pump']
    dt = results['dt']
    t_pump = N_pump * dt
    N = results['N_sites']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Vorticity
    ax = axes[0, 0]
    ax.plot(times, V, 'b-', linewidth=2)
    ax.axvline(t_pump, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Pump OFF')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('V', fontsize=12)
    ax.set_title(f'Vorticity (N={N})', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: V²
    ax = axes[0, 1]
    ax.plot(times, V2, 'g-', linewidth=2)
    ax.axvline(t_pump, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Pump OFF')
    pump_idx = N_pump // results.get('measure_every', 5)
    if pump_idx < len(V2):
        ax.axhline(V2[pump_idx], color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('V²', fontsize=12)
    ax.set_title('Vortex Pair Energy', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Magnetization
    ax = axes[1, 0]
    ax.plot(times, Sz, 'purple', linewidth=2)
    ax.axvline(t_pump, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Pump OFF')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('⟨Sz⟩', fontsize=12)
    ax.set_title('Total Magnetization', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 4: Entanglement contour
    ax = axes[1, 1]
    im = ax.imshow(
        S_contour.T,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, N-1],
        cmap='viridis'
    )
    ax.axvline(t_pump, color='w', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Site i', fontsize=12)
    ax.set_title('Entanglement Contour S(i,t)', fontsize=13)
    plt.colorbar(im, ax=ax, label='S(i,t)')
    
    plt.suptitle(
        f'Λ³ MPO-TEBD: Photo-Induced Superconductivity (N={N})\n'
        f'Memory-efficient via MPO contraction!',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ [Saved] {save_path}")
    return save_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Λ³ MPO-TEBD: Memory-Efficient Large-Scale Simulation               ║
║                                                                      ║
║  Key Innovation:                                                     ║
║  ⟨ψ|V|ψ⟩ = MPS ─── MPO ─── MPS contraction                          ║
║                                                                      ║
║  Memory:                                                             ║
║  ✗ Dense: O(2^N) → explodes at N≈20                                 ║
║  ✓ MPO:   O(N × χ² × D²) → scales to N=100+                         ║
║                                                                      ║
║  Λ³ Interpretation:                                                  ║
║  • MPS = hierarchical structure (Axiom 1)                           ║
║  • MPO = operator in Λ-space                                        ║
║  • Contraction = constraint satisfaction                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run with large system size!
    results = run_mpo_tebd(
        N_sites=32,      # Can go to 64, 100+!
        N_pump=200,
        N_free=200,
        dt=0.05,
        g=0.3,
        omega=0.3,
        oam_l=1,
        chirality=+1,
        chi_max=64,
        measure_every=5,
        verbose=True,
    )
    
    # Visualize
    plot_mpo_results(results)
    
    print("\n" + "=" * 70)
    print("🎉 MPO-TEBD COMPLETE!")
    print("=" * 70)
    print("\nNow you can simulate N=100+ sites without memory explosion!")
