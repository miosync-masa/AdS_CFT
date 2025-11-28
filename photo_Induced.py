"""
Λ³ 4×4 Photo-Induced Superconductivity
======================================

Exact diagonalization on 4×4 = 16 spins
Hilbert space: 2^16 = 65,536

Memory optimization tips:
- Uses sparse matrices throughout
- Deletes intermediate objects
- Can reduce N_pump/N_free if memory issues

Requirements:
    pip install numpy scipy matplotlib

Author: Masamichi & Tamaki
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
import time
import gc

# =============================================================================
# Build 4×4 XY Model
# =============================================================================

def build_spin_operators(N):
    """Build sparse spin-1/2 operators for N sites"""
    
    Dim = 2**N
    
    # Pauli matrices
    sx = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    sy = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
    sz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    id2 = sparse.eye(2, dtype=complex, format='csr')
    
    def single_site_op(op, site):
        """Build N-site operator with 'op' at 'site', identity elsewhere"""
        ops = [id2] * N
        ops[site] = op
        result = ops[0]
        for k in range(1, N):
            result = sparse.kron(result, ops[k], format='csr')
        return result
    
    print(f"  Building operators for {N} sites (Dim={Dim})...")
    t0 = time.time()
    
    Sx = [single_site_op(sx, i) for i in range(N)]
    Sy = [single_site_op(sy, i) for i in range(N)]
    Sz = [single_site_op(sz, i) for i in range(N)]
    
    print(f"  Done in {time.time()-t0:.1f}s")
    
    return Sx, Sy, Sz


def build_4x4_system():
    """Build complete 4×4 XY model system"""
    
    Lx, Ly = 4, 4
    N = Lx * Ly  # 16
    Dim = 2**N   # 65536
    
    print("=" * 60)
    print(f"Λ³ 4×4 XY Model Builder")
    print(f"N = {N} spins, Dim = {Dim}")
    print("=" * 60)
    
    # Coordinates
    def idx(x, y):
        return y * Lx + x
    
    coords = {idx(x, y): (x, y) for y in range(Ly) for x in range(Lx)}
    
    # NN bonds (open boundary)
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            if x + 1 < Lx:
                bonds.append((i, idx(x+1, y)))
            if y + 1 < Ly:
                bonds.append((i, idx(x, y+1)))
    
    print(f"  NN bonds: {len(bonds)}")
    
    # Plaquettes for vorticity
    plaquettes = []
    for y in range(Ly - 1):
        for x in range(Lx - 1):
            bl = idx(x, y)
            br = idx(x+1, y)
            tr = idx(x+1, y+1)
            tl = idx(x, y+1)
            plaquettes.append((bl, br, tr, tl))
    
    print(f"  Plaquettes: {len(plaquettes)}")
    
    # Build operators
    Sx, Sy, Sz = build_spin_operators(N)
    
    # XY Hamiltonian: H = Σ (Sx_i Sx_j + Sy_i Sy_j)
    print("  Building H_XY...")
    t0 = time.time()
    
    H_xy = sparse.csr_matrix((Dim, Dim), dtype=complex)
    for (i, j) in bonds:
        H_xy = H_xy + Sx[i] @ Sx[j] + Sy[i] @ Sy[j]
    
    print(f"  H_XY done in {time.time()-t0:.1f}s, nnz={H_xy.nnz}")
    
    # Vorticity operator
    print("  Building V_op...")
    t0 = time.time()
    
    V_op = sparse.csr_matrix((Dim, Dim), dtype=complex)
    for (bl, br, tr, tl) in plaquettes:
        for (i, j) in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
            V_op = V_op + 2.0 * (Sx[i] @ Sy[j] - Sy[i] @ Sx[j])
    
    print(f"  V_op done in {time.time()-t0:.1f}s")
    
    # Total Sz
    Sz_total = sum(Sz)
    
    # OAM phases
    cx, cy = (Lx - 1) / 2, (Ly - 1) / 2
    oam_phases = {}
    for site, (x, y) in coords.items():
        angle = np.arctan2(y - cy, x - cx)
        if angle < 0:
            angle += 2 * np.pi
        oam_phases[site] = angle
    
    return {
        'H_xy': H_xy,
        'V_op': V_op,
        'V2_op': V_op @ V_op,  # Pre-compute V²
        'Sz_total': Sz_total,
        'Sx': Sx,
        'Sy': Sy,
        'Sz': Sz,
        'coords': coords,
        'bonds': bonds,
        'plaquettes': plaquettes,
        'oam_phases': oam_phases,
        'N': N,
        'Dim': Dim,
        'Lx': Lx,
        'Ly': Ly,
    }


# =============================================================================
# Ground State
# =============================================================================

def find_ground_state(H, k=1):
    """Find ground state using sparse eigensolver"""
    print("  Finding ground state...")
    t0 = time.time()
    
    evals, evecs = sp_linalg.eigsh(H.real, k=k, which='SA')
    psi = evecs[:, 0].astype(complex)
    psi /= np.linalg.norm(psi)
    E0 = evals[0]
    
    print(f"  Done in {time.time()-t0:.1f}s, E0 = {E0:.6f}")
    return psi, E0


# =============================================================================
# Time Evolution
# =============================================================================

def run_oam_pump_4x4(
    N_pump: int = 50,
    N_free: int = 50,
    dt: float = 0.1,
    g: float = 0.3,
    omega: float = 0.3,
    oam_l: int = 1,
    chirality: int = 1,
    measure_every: int = 5,
    make_plots: bool = True,
    save_data: bool = True,
    verbose: bool = True,
):
    """
    Run OAM pump + free evolution on 4×4 XY model
    
    Parameters
    ----------
    N_pump : int
        Number of pump steps
    N_free : int
        Number of free evolution steps
    dt : float
        Time step
    g : float
        OAM coupling strength
    omega : float
        OAM frequency
    oam_l : int
        OAM charge (angular momentum)
    chirality : int
        +1 or -1 for pump direction
    measure_every : int
        Measure observables every N steps
    make_plots : bool
        If True, make and save plots.
    save_data : bool
        If True, save .npz with results.
    verbose : bool
        If True, print progress.
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("Λ³ 4×4 OAM Pump Simulation")
        print("=" * 60)
        print(f"  Pump: {N_pump} steps")
        print(f"  Free: {N_free} steps")
        print(f"  dt = {dt}")
        print(f"  OAM: l={oam_l}, chirality={chirality}, g={g}, ω={omega}")
        print("=" * 60)
    
    # Build system
    system = build_4x4_system()
    
    H_xy = system['H_xy']
    V_op = system['V_op']
    V2_op = system['V2_op']
    Sz_total = system['Sz_total']
    Sx = system['Sx']
    Sy = system['Sy']
    oam_phases = system['oam_phases']
    N = system['N']
    Dim = system['Dim']
    
    # Ground state as initial
    psi, E0 = find_ground_state(H_xy)
    
    # Initial measurements
    Sz0 = np.vdot(psi, Sz_total.dot(psi)).real
    V0 = np.vdot(psi, V_op.dot(psi)).real
    V20 = np.vdot(psi, V2_op.dot(psi)).real
    
    if verbose:
        print(f"\nInitial state:")
        print(f"  <Sz> = {Sz0:.6f}")
        print(f"  V    = {V0:.6f}")
        print(f"  V²   = {V20:.6f}")
    
    # Storage
    times = []
    Sz_list = []
    V_list = []
    V2_list = []
    
    N_total = N_pump + N_free
    
    if verbose:
        print(f"\n--- Starting simulation ({N_total} steps) ---")
    t_start = time.time()
    
    for n in range(N_total):
        t = n * dt
        phase = "PUMP" if n < N_pump else "FREE"
        
        if n == N_pump and n > 0 and verbose:
            print("\n--- Pump OFF → Free Evolution ---\n")
        
        # Build Hamiltonian for this step
        if n < N_pump:
            # H = H_XY + H_OAM(t)
            omega_t = omega * t
            H_drive = sparse.csr_matrix((Dim, Dim), dtype=complex)
            
            for site in range(N):
                local_phase = omega_t + chirality * oam_l * oam_phases[site]
                H_drive = H_drive + g * (
                    np.cos(local_phase) * Sx[site] +
                    np.sin(local_phase) * Sy[site]
                )
            
            H_step = H_xy + H_drive
        else:
            H_step = H_xy
        
        # Time evolution: |ψ(t+dt)⟩ = exp(-i H dt) |ψ(t)⟩
        psi = sp_linalg.expm_multiply(-1j * H_step * dt, psi)
        psi /= np.linalg.norm(psi)
        
        # Measurements
        if n % measure_every == 0:
            Sz = np.vdot(psi, Sz_total.dot(psi)).real
            V = np.vdot(psi, V_op.dot(psi)).real
            V2 = np.vdot(psi, V2_op.dot(psi)).real
            
            times.append(t)
            Sz_list.append(Sz)
            V_list.append(V)
            V2_list.append(V2)
            
            if verbose and n % (measure_every * 4) == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (n + 1) * (N_total - n - 1) if n > 0 else 0
                print(f"  n={n:4d} [{phase}]: <Sz>={Sz:+.4f}, V={V:+.6f}, V²={V2:.4f}  ({elapsed:.0f}s, ETA {eta:.0f}s)")
        
        # Garbage collection every 50 steps
        if n % 50 == 0:
            gc.collect()
    
    total_time = time.time() - t_start
    if verbose:
        print(f"\nSimulation complete in {total_time:.1f}s")
    
    # Convert to arrays
    times = np.array(times)
    Sz_arr = np.array(Sz_list)
    V_arr = np.array(V_list)
    V2_arr = np.array(V2_list)
    
    # Analysis
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS ANALYSIS")
        print("=" * 60)
    
    pump_off_idx = N_pump // measure_every
    
    if pump_off_idx < len(V_arr) and len(V_arr) > pump_off_idx:
        V_pump = V_arr[pump_off_idx]
        V_final = V_arr[-1]
        V2_pump = V2_arr[pump_off_idx]
        V2_final = V2_arr[-1]
        
        if verbose:
            print(f"\nAt pump-off (n={N_pump}):")
            print(f"  V  = {V_pump:.6f}")
            print(f"  V² = {V2_pump:.6f}")
            
            print(f"\nFinal (n={N_total}):")
            print(f"  V  = {V_final:.6f}")
            print(f"  V² = {V2_final:.6f}")
        
        # Free evolution statistics
        free_V = V_arr[pump_off_idx:]
        free_V2 = V2_arr[pump_off_idx:]
        
        if verbose:
            print(f"\nFree evolution statistics:")
            print(f"  <V>  = {free_V.mean():.6f} ± {free_V.std():.6f}")
            print(f"  <V²> = {free_V2.mean():.6f} ± {free_V2.std():.6f}")
        
        if V2_pump > 1e-6 and verbose:
            persistence = V2_final / V2_pump
            print(f"\n  V² Persistence Ratio: {persistence:.4f}")
            
            if persistence > 0.5:
                print("\n  🏆🏆🏆 PERSISTENT VORTICITY CONFIRMED! 🏆🏆🏆")
                print("  → Photo-induced topological order survives!")
            elif persistence > 0.1:
                print("\n  ⚡ Partial vorticity persistence")
            else:
                print("\n  ⚠️ Vorticity decays after pump off")
    
    # Protection score
    if verbose:
        print("\n--- Topological Protection ---")
    comm = H_xy @ V_op - V_op @ H_xy
    comm_norm = sparse.linalg.norm(comm)
    H_norm = sparse.linalg.norm(H_xy)
    V_norm = sparse.linalg.norm(V_op)
    protection_score = comm_norm / (H_norm * V_norm)
    
    if verbose:
        print(f"  ||[H_XY, V]|| = {comm_norm:.4f}")
        print(f"  Protection Score = {protection_score:.6f}")
    
    # Plot
    if make_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t_pump = N_pump * dt
        
        # Magnetization
        axes[0, 0].plot(times, Sz_arr, 'b-', linewidth=1.5)
        axes[0, 0].axvline(t_pump, color='r', linestyle='--', alpha=0.7, label='Pump OFF')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel(r'$\langle S_z \rangle$')
        axes[0, 0].set_title('Total Magnetization')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vorticity
        axes[0, 1].plot(times, V_arr, 'orange', linewidth=1.5)
        axes[0, 1].axvline(t_pump, color='r', linestyle='--', alpha=0.7, label='Pump OFF')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('V')
        axes[0, 1].set_title('Vorticity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # V²
        axes[1, 0].plot(times, V2_arr, 'g-', linewidth=1.5)
        axes[1, 0].axvline(t_pump, color='r', linestyle='--', alpha=0.7, label='Pump OFF')
        axes[1, 0].axhline(V2_arr[pump_off_idx], color='gray', linestyle=':', alpha=0.5)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel(r'$V^2$')
        axes[1, 0].set_title(r'$V^2$ (Vortex Pair Energy)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # V² zoom on free evolution
        free_times = times[pump_off_idx:]
        free_V2_plot = V2_arr[pump_off_idx:]
        
        axes[1, 1].plot(free_times, free_V2_plot, 'g-', linewidth=1.5)
        axes[1, 1].axhline(free_V2_plot.mean(), color='purple', linestyle='--', 
                            label=f'mean = {free_V2_plot.mean():.4f}')
        axes[1, 1].fill_between(free_times, 
                                 free_V2_plot.mean() - free_V2_plot.std(),
                                 free_V2_plot.mean() + free_V2_plot.std(),
                                 alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel(r'$V^2$')
        axes[1, 1].set_title(r'$V^2$ During Free Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Λ³ 4×4 XY Model: OAM-Induced Photo-Superconductivity\n'
                     f'(l={oam_l}, χ={chirality}, g={g}, ω={omega})', fontsize=14)
        plt.tight_layout()
        
        filename = f'lambda3_4x4_oam_l{oam_l}_chi{chirality}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"\n[Saved] {filename}")
    
    # Also save data
    if save_data:
        np.savez('lambda3_4x4_results.npz',
                 times=times,
                 Sz=Sz_arr,
                 V=V_arr,
                 V2=V2_arr,
                 N_pump=N_pump,
                 N_free=N_free,
                 dt=dt,
                 g=g,
                 omega=omega,
                 oam_l=oam_l,
                 chirality=chirality,
                 measure_every=measure_every)
        if verbose:
            print("[Saved] lambda3_4x4_results.npz")
    
    return {
        'times': times,
        'Sz': Sz_arr,
        'V': V_arr,
        'V2': V2_arr,
        'protection_score': protection_score,
        'N_pump': N_pump,
        'N_free': N_free,
        'dt': dt,
        'measure_every': measure_every,
    }


# =============================================================================
# Extra: Chirality mirror test (χ=+1 vs χ=-1)
# =============================================================================

def chirality_mirror_test_4x4(
    N_pump=300,
    N_free=300,
    dt=0.1,
    g=0.3,
    omega=0.3,
    oam_l=1,
    measure_every=5,
):
    """
    Run OAM pump for chirality = ±1 and check mirror symmetry:
    V_+(t)  vs  -V_-(t)
    """
    print("\n" + "="*60)
    print("CHIRALITY MIRROR TEST on 4×4 [XY]")
    print("="*60)
    
    base_kwargs = dict(
        N_pump=N_pump,
        N_free=N_free,
        dt=dt,
        g=g,
        omega=omega,
        oam_l=oam_l,
        measure_every=measure_every,
        make_plots=True,
        save_data=True,
        verbose=False,
    )
    
    res_plus  = run_oam_pump_4x4(chirality=+1, **base_kwargs)
    res_minus = run_oam_pump_4x4(chirality=-1, **base_kwargs)
    
    t  = res_plus['times']
    Vp = res_plus['V']
    Vm = res_minus['V']
    
    # 相関係数 (V_+(t) と -V_-(t) の Pearson)
    corr = np.corrcoef(Vp, -Vm)[0, 1]
    print(f"Correlation(V_chi=+1, -V_chi=-1) = {corr:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, Vp,  'r-',  label='χ = +1')
    plt.plot(t, Vm,  'b-',  alpha=0.5, label='χ = -1')
    plt.plot(t, -Vm, 'k--', alpha=0.7, label='-V(χ = -1)')
    plt.axvline(N_pump*dt, color='gray', linestyle='--', label='Pump OFF')
    plt.axhline(0.0, color='black', linestyle=':')
    plt.xlabel("Time")
    plt.ylabel("V(t)")
    plt.title(f"Chirality Reversal Test (4×4, l={oam_l})\n"
              f"corr(V_+, -V_-) = {corr:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"lambda3_4x4_chirality_mirror_l{oam_l}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {fname}")
    
    return dict(times=t, V_plus=Vp, V_minus=Vm, corr=corr)


# =============================================================================
# Extra: OAM charge scan (l = 1,2,3,...) on Max|V|
# =============================================================================

def oam_charge_scan_4x4(
    l_values=(1, 2, 3),
    N_pump=300,
    N_free=300,
    dt=0.1,
    g=0.3,
    omega=0.3,
    chirality=1,
    measure_every=5,
):
    """
    Scan OAM charge l and record Max|V| and <V>_free.
    """
    print("\n" + "="*60)
    print("OAM CHARGE SCAN on 4×4 [XY]")
    print("="*60)
    
    maxV_list = []
    avgV_list = []
    
    for l in l_values:
        print(f"\n--- l = {l} ---")
        res = run_oam_pump_4x4(
            N_pump=N_pump,
            N_free=N_free,
            dt=dt,
            g=g,
            omega=omega,
            oam_l=l,
            chirality=chirality,
            measure_every=measure_every,
            make_plots=False,
            save_data=False,
            verbose=False,
        )
        V = res['V']
        pump_off_idx = res['N_pump'] // res['measure_every']
        V_free = V[pump_off_idx:]
        
        maxV = np.max(np.abs(V))
        avgV = np.mean(V_free)
        
        maxV_list.append(maxV)
        avgV_list.append(avgV)
        
        print(f"  Max|V|    = {maxV:.6f}")
        print(f"  <V>_free = {avgV:.6f}")
    
    # Plot Max|V| vs l
    l_arr = np.array(l_values, dtype=float)
    maxV_arr = np.array(maxV_list)
    avgV_arr = np.array(avgV_list)
    
    plt.figure(figsize=(7,5))
    plt.plot(l_arr, maxV_arr, 'o-', label='Max |V|')
    plt.plot(l_arr, avgV_arr, 's--', label='<V> (free)')
    plt.xlabel("OAM charge l")
    plt.ylabel("Vorticity")
    plt.title("Scaling of Vorticity with OAM charge l (4×4 XY)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fname = "lambda3_4x4_oam_maxV_vs_l.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {fname}")
    
    return dict(l_values=l_arr, maxV=maxV_arr, avgV=avgV_arr)


# =============================================================================
# TEBD / MPS layer attached to ED (1D chain version)
# Λ³ + OAM + Vorticity + Entanglement contour
# =============================================================================

from scipy.linalg import expm  # 2x2, 4x4 の局所ゲートで使用

class MPS1D:
    """
    最小限の 1D MPS 実装（open boundary）
    - d=2 (spin-1/2)
    - A[i] has shape (chi_left, d, chi_right)
    - TEBD 用に 2-site gate と 1-site gate を実装
    - bond_svals[i]: bond i (between site i and i+1) の特異値
    """

    def __init__(self, N, chi_max=64, init_state="up"):
        self.N = N
        self.chi_max = chi_max
        self.d = 2
        # A[0]: (1,d,chi1), ..., A[N-1]: (chi_{N-1},d,1)
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
            else:
                raise ValueError(f"Unknown init_state: {init_state}")
            self.A.append(A)

        # 各ボンドの特異値（エンタングルメント用）
        self.bond_svals = [np.array([0.0], dtype=np.float64) for _ in range(N - 1)]

    # ---------------------------
    # 1-site gate: U (2x2)
    # ---------------------------
    def apply_1site_gate(self, i, U):
        """
        A[i]_{α,s,β} -> Σ_s U_{s',s} A[i]_{α,s,β}
        """
        A = self.A[i]  # (chiL, d, chiR)
        # tensordot over physical index s
        # U: (d', d), A: (chiL, d, chiR)
        theta = np.tensordot(U, A, axes=(1, 1))  # (d', chiL, chiR)
        theta = np.transpose(theta, (1, 0, 2))   # (chiL, d', chiR)
        self.A[i] = theta

    # ---------------------------
    # 2-site gate: U (4x4)
    # ---------------------------
    def apply_2site_gate(self, i, U, chi_max=None):
        """
        2-site TEBD ステップ
        - gate U: 4x4 (物理 index (s_i, s_{i+1}) 上に作用)
        - bond i(i,i+1) の特異値を保存 → エンタングルメント
        """
        if chi_max is None:
            chi_max = self.chi_max
        A_i = self.A[i]      # (chiL, d, chiM)
        A_j = self.A[i + 1]  # (chiM, d, chiR)

        chiL, d, chiM = A_i.shape
        chiM2, d2, chiR = A_j.shape
        assert d == 2 and d2 == 2, "TEBDはd=2専用に書いてある"
        assert chiM == chiM2, "bond dimension mismatch"

        # 2サイトをまとめたテンソル: (chiL, s_i, s_{i+1}, chiR)
        theta = np.tensordot(A_i, A_j, axes=(2, 0))  # (chiL, d, d, chiR)

        # U を 4-index に reshape: U_{s1',s2',s1,s2}
        U4 = U.reshape(2, 2, 2, 2)
        # U_{s1',s2',s1,s2} * theta_{chiL,s1,s2,chiR} → (s1',s2',chiL,chiR)
        theta = np.tensordot(U4, theta, axes=([2, 3], [1, 2]))  # (2,2,chiL,chiR)
        theta = np.transpose(theta, (2, 0, 1, 3))               # (chiL,2,2,chiR)

        # SVD のために行列に reshape
        theta_mat = theta.reshape(chiL * 2, 2 * chiR)  # (chiL*d, d*chiR)

        U_svd, S_svd, Vh_svd = np.linalg.svd(theta_mat, full_matrices=False)

        chi_new = min(chi_max, len(S_svd))
        U_svd = U_svd[:, :chi_new]
        S_svd = S_svd[:chi_new]
        Vh_svd = Vh_svd[:chi_new, :]

        # 数値誤差用に正規化を一度まとめて入れる
        S_norm = np.linalg.norm(S_svd)
        if S_norm > 0:
            S_svd = S_svd / S_norm
            U_svd *= S_norm

        # 戻し: A_i_new: (chiL, d, chi_new), A_j_new: (chi_new, d, chiR)
        A_i_new = U_svd.reshape(chiL, 2, chi_new)
        A_j_new = (np.diag(S_svd) @ Vh_svd).reshape(chi_new, 2, chiR)

        self.A[i] = A_i_new
        self.A[i + 1] = A_j_new

        # このボンドの特異値を保存 → entanglement S(i) 用
        self.bond_svals[i] = S_svd.copy()

    # ---------------------------
    # MPS → フル状態ベクトル ψ
    # ---------------------------
    def to_dense(self):
        """
        MPS からフルベクトル ψ (2^N) を reconstruct
        → ED の H, V, V² をそのまま流用して expectation を計算するため
        """
        # A[0]: (1,d,chi1) → (d,chi1)
        psi = self.A[0][0, :, :]  # (d, chi1)

        for i in range(1, self.N - 1):
            A = self.A[i]  # (chiL, d, chiR)
            # psi: (..., chiL), A: (chiL, d, chiR)
            psi = np.tensordot(psi, A, axes=(psi.ndim - 1, 0))
            #  → (d,...,d, chiR)

        # 最後のサイト
        A_last = self.A[-1]  # (chi_{N-1}, d, 1)
        psi = np.tensordot(psi, A_last, axes=(psi.ndim - 1, 0))  # (...,d,1)
        psi = np.squeeze(psi, axis=-1)
        psi = psi.reshape(-1)  # (2^N,)

        # 数値誤差の正規化
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        return psi

    # ---------------------------
    # bond entropies S_bond(i)
    # ---------------------------
    def bond_entropies(self):
        """
        TEBD ステップ中に保存している特異値から
        各ボンドのエンタングルメント S = -Tr(ρ log ρ) を計算
        """
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


# =============================================================================
# 1D chain version of XY + V operator (ED part reused)
# =============================================================================

def build_1d_chain_xy_and_v(N_chain):
    """
    1D チェーン (open) 上の XY Hamiltonian と V, V², Sz_total を ED で構成
    - 既存の build_spin_operators(N) を再利用
    """
    print(f"\n[build_1d_chain_xy_and_v] N = {N_chain}")
    Sx_list, Sy_list, Sz_list = build_spin_operators(N_chain)
    Dim = 2 ** N_chain

    H_xy = sparse.csr_matrix((Dim, Dim), dtype=np.complex128)
    V_op = sparse.csr_matrix((Dim, Dim), dtype=np.complex128)

    # NN bonds: (0-1, 1-2, ..., N-2 - N-1)
    for i in range(N_chain - 1):
        H_xy += Sx_list[i] @ Sx_list[i + 1] + Sy_list[i] @ Sy_list[i + 1]
        V_op += 2.0 * (Sx_list[i] @ Sy_list[i + 1] - Sy_list[i] @ Sx_list[i + 1])

    V2_op = V_op @ V_op
    Sz_total = sum(Sz_list)

    return {
        "H_xy": H_xy,
        "V_op": V_op,
        "V2_op": V2_op,
        "Sz_total": Sz_total,
        "Dim": Dim,
        "N": N_chain,
    }


# =============================================================================
# TEBD OAM pump on 1D chain, attached to ED operators
# =============================================================================

def run_tebd_oam_chain(
    N_sites=16,
    N_pump=200,
    N_free=200,
    dt=0.05,
    g=0.3,
    omega=0.3,
    oam_l=1,
    chirality=1,
    chi_max=64,
    measure_every=5,
    make_plots=True,
    verbose=True,
):
    """
    1D チェーン版 Λ³ TEBD OAM-pump
    - TEBD (MPS1D) で時間発展
    - 各測定ステップで MPS → ψ(dense) に戻して
      ED 方式の V, V², Sz_total をそのまま計算
    - 同時に bond entanglement S_bond(i,t) を記録 → entanglement contour
    """

    if verbose:
        print("\n" + "=" * 60)
        print("Λ³ 1D TEBD OAM Pump (MPS attached to ED)")
        print("=" * 60)
        print(f"  N_sites  = {N_sites}")
        print(f"  Pump     = {N_pump} steps")
        print(f"  Free     = {N_free} steps")
        print(f"  dt       = {dt}")
        print(f"  OAM: l={oam_l}, χ={chirality}, g={g}, ω={omega}")
        print(f"  chi_max  = {chi_max}")
        print("=" * 60)

    # --- ED part: H, V, V², Sz_total を構成 ---
    ed_ops = build_1d_chain_xy_and_v(N_sites)
    H_xy = ed_ops["H_xy"]
    V_op = ed_ops["V_op"]
    V2_op = ed_ops["V2_op"]
    Sz_total = ed_ops["Sz_total"]

    # --- ローカルスピン演算子 (d=2) ---
    sx_loc = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy_loc = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    # sz_loc = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # 必要なら

    # XY bond Hamiltonian (2-site)
    h_bond = np.kron(sx_loc, sx_loc) + np.kron(sy_loc, sy_loc)
    U_bond = expm(-1j * dt * h_bond)  # 固定 (時間依存しない)

    # OAM phase: 1D チェーンを円周に並べた位相
    oam_phases = 2.0 * np.pi * np.arange(N_sites) / float(N_sites)

    # --- 初期状態 MPS (全部↑) ---
    mps = MPS1D(N_sites, chi_max=chi_max, init_state="up")

    # --- 記録用 ---
    times = []
    Sz_list = []
    V_list = []
    V2_list = []
    Sbond_hist = []

    N_total = N_pump + N_free
    t_start = time.time()

    if verbose:
        print("\n--- TEBD time evolution ---")

    for n in range(N_total):
        t = n * dt
        phase = "PUMP" if n < N_pump else "FREE"

        # ----- OAM pump: n < N_pump の間だけ 1-site gate をかける -----
        if n < N_pump:
            omega_t = omega * t
            for j in range(N_sites):
                phase_j = omega_t + chirality * oam_l * oam_phases[j]
                h_loc = g * (np.cos(phase_j) * sx_loc + np.sin(phase_j) * sy_loc)
                U_loc = expm(-1j * dt * h_loc)
                mps.apply_1site_gate(j, U_loc)

        # ----- XY TEBD: even bonds, then odd bonds -----
        # even bonds (0-1, 2-3, ...)
        for i in range(0, N_sites - 1, 2):
            mps.apply_2site_gate(i, U_bond, chi_max=chi_max)
        # odd bonds (1-2, 3-4, ...)
        for i in range(1, N_sites - 1, 2):
            mps.apply_2site_gate(i, U_bond, chi_max=chi_max)

        # ----- 測定 -----
        if n % measure_every == 0:
            psi = mps.to_dense()  # (2^N,)
            Sz = np.vdot(psi, Sz_total.dot(psi)).real
            V = np.vdot(psi, V_op.dot(psi)).real
            V2 = np.vdot(psi, V2_op.dot(psi)).real

            times.append(t)
            Sz_list.append(Sz)
            V_list.append(V)
            V2_list.append(V2)
            Sbond_hist.append(mps.bond_entropies())

            if verbose and n % (measure_every * 4) == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (n + 1) * (N_total - n - 1) if n > 0 else 0
                print(f"  n={n:4d} [{phase}]: "
                      f"<Sz>={Sz:+.4f}, V={V:+.6f}, V²={V2:.4f}  "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")

        if n % 50 == 0:
            gc.collect()

    total_time = time.time() - t_start
    if verbose:
        print(f"\nTEBD simulation complete in {total_time:.1f}s")

    times = np.array(times)
    Sz_arr = np.array(Sz_list)
    V_arr = np.array(V_list)
    V2_arr = np.array(V2_list)
    Sbond_hist = np.array(Sbond_hist)  # (T_meas, N_sites-1)

    # ----- 解析: persistent V²? -----
    if verbose:
        print("\n" + "=" * 60)
        print("TEBD RESULTS ANALYSIS (1D chain)")
        print("=" * 60)

    pump_off_idx = N_pump // measure_every
    if pump_off_idx < len(V_arr):
        V_pump = V_arr[pump_off_idx]
        V2_pump = V2_arr[pump_off_idx]
        V_final = V_arr[-1]
        V2_final = V2_arr[-1]

        if verbose:
            print(f"\nAt pump-off (n={N_pump}):")
            print(f"  V  = {V_pump:.6f}")
            print(f"  V² = {V2_pump:.6f}")

            print(f"\nFinal (n={N_pump + N_free}):")
            print(f"  V  = {V_final:.6f}")
            print(f"  V² = {V2_final:.6f}")

        V_free = V_arr[pump_off_idx:]
        V2_free = V2_arr[pump_off_idx:]

        if verbose:
            print(f"\nFree evolution statistics:")
            print(f"  <V>  = {V_free.mean():.6f} ± {V_free.std():.6f}")
            print(f"  <V²> = {V2_free.mean():.6f} ± {V2_free.std():.6f}")

        if V2_pump > 1e-8 and verbose:
            persistence = V2_final / V2_pump
            print(f"\n  V² Persistence Ratio: {persistence:.4f}")
            if persistence > 0.5:
                print("\n  🏆 TEBD: PERSISTENT VORTICITY CONFIRMED (1D chain) 🏆")
            elif persistence > 0.1:
                print("\n  ⚡ Partial vorticity persistence (1D chain)")
            else:
                print("\n  ⚠️ Vorticity decays after pump off (1D chain)")

    # ----- entanglement contour / bulk geometry 用に S_site(i,t) を構成 -----
    S_bond = Sbond_hist  # shape = (T_meas, N_sites-1)
    T_meas, B = S_bond.shape
    N = B + 1
    S_site = np.zeros((T_meas, N), dtype=np.float64)
    S_site[:, 0] = S_bond[:, 0]
    S_site[:, -1] = S_bond[:, -1]
    if N > 2:
        S_site[:, 1:-1] = 0.5 * (S_bond[:, :-1] + S_bond[:, 1:])

    # ----- 可視化 -----
    if make_plots and len(times) > 1:
        # (1) global Sz, V, V²
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        t_pump = N_pump * dt

        axes[0, 0].plot(times, Sz_arr, label="<Sz>")
        axes[0, 0].axvline(t_pump, color="r", linestyle="--", label="Pump OFF")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("<Sz>")
        axes[0, 0].set_title("TEBD: Magnetization (1D)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(times, V_arr, label="V")
        axes[0, 1].axvline(t_pump, color="r", linestyle="--", label="Pump OFF")
        axes[0, 1].axhline(0.0, color="k", linestyle=":")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("V")
        axes[0, 1].set_title("TEBD: Vorticity (1D)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(times, V2_arr, label="V²")
        axes[1, 0].axvline(t_pump, color="r", linestyle="--", label="Pump OFF")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("V²")
        axes[1, 0].set_title("TEBD: V² (1D)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # (2) Entanglement contour S_site(i,t) → holographic “bulk slice”
        im = axes[1, 1].imshow(
            S_site.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], 0, N - 1],
        )
        axes[1, 1].axvline(t_pump, color="w", linestyle="--", linewidth=1.0)
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("site i")
        axes[1, 1].set_title("Entanglement contour S(i,t) (TEBD)")
        fig.colorbar(im, ax=axes[1, 1], label="S(i,t)")

        plt.suptitle(
            f"Λ³ 1D TEBD OAM Pump (N={N_sites}, l={oam_l}, χ={chirality}, g={g}, ω={omega})",
            fontsize=14,
        )
        plt.tight_layout()
        fname = f"lambda3_tebd_1d_oam_l{oam_l}_chi{chirality}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"[Saved] {fname}")

        # bulk geometry 的な図をもう1枚出してもいい：
        # S(i,t) を「bulk の高さ」として 2D カラーマップで眺めるだけでも
        # 立派な “Λ³ holography” の可視化になる

    return {
        "times": times,
        "Sz": Sz_arr,
        "V": V_arr,
        "V2": V2_arr,
        "S_bond": S_bond,
        "S_site": S_site,
    }

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Λ³ Photo-Induced Superconductivity on 4×4 XY Model              ║
║                                                                  ║
║  MPS = Λ³ computational implementation                           ║
║  - Axiom 1 (Hierarchy): Site → Bond → Plaquette                 ║
║  - Axiom 2 (Non-commutative): Gate order matters                ║
║  - Axiom 3 (Conservation): Unitary preserves norm               ║
║  - Axiom 5 (Pulsation): Truncation = ΔΛ_C                       ║
║                                                                  ║
║  Key test: Does V² persist after pump off?                       ║
║  If yes → Topological photo-induced order!                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ===== Basic single run (same asこれまで) =====
    results = run_oam_pump_4x4(
        N_pump=300,      # Pump steps
        N_free=300,      # Free evolution steps
        dt=0.1,          # Time step
        g=0.3,           # OAM coupling
        omega=0.3,       # OAM frequency
        oam_l=1,         # OAM charge
        chirality=-1,    # Pump direction
        measure_every=5  # Measurement interval
    )
    # そのあと、同じパラメータスケールで 1D TEBD 版を走らせる
    results_tebd = run_tebd_oam_chain(
        N_sites=16,      # 4×4 と同じ Hilbert 次元
        N_pump=200,
        N_free=200,
        dt=0.05,         # 少し細かくしてもよい
        g=0.3,
        omega=0.3,
        oam_l=1,
        chirality=+1,
        chi_max=64,
        measure_every=5,
        make_plots=True,
        verbose=True,
    )
    
    # ===== 追加の検証（必要ならアンコメントして使う） =====
    # 1) chirality ±1 鏡映テスト
    #mirror_res = chirality_mirror_test_4x4(
    #     N_pump=300, N_free=300, dt=0.1, g=0.3, omega=0.3,
    #     oam_l=1, measure_every=5
    #)
    #
    # 2) OAM charge l = 1,2,3 スキャン
    #scan_res = oam_charge_scan_4x4(
    #     l_values=(1,2,3),
    #     N_pump=300, N_free=300, dt=0.1, g=0.3, omega=0.3,
    #     chirality=1, measure_every=5
    #)
    
    print("\n" + "=" * 60)
    print("UNIFIED ED + TEBD Λ³ SIMULATION COMPLETE")
    print("=" * 60)
