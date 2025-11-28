"""
Λ³ Experiment Runner & Parameter Scans
=======================================

Automated parameter scans and comparison tools

Features:
1. Chirality scan (χ = ±1)
2. OAM charge scan (l = 1, 2, 3, ...)
3. Ladder size scan (L values)
4. chi_max convergence test
5. Comparison & benchmark plots

Author: Masamichi & Tamaki (環)
Date: 2025-11-29
"""

import numpy as np
import matplotlib.pyplot as plt
from lambda3_ladder_tebd import run_ladder_tebd
from analysis_tools import compute_topological_order_parameter
import time
import os


# =============================================================================
# Chirality Scan
# =============================================================================

def chirality_scan(
    L=10,
    oam_l=1,
    N_pump=200,
    N_free=200,
    dt=0.05,
    g=0.3,
    omega=0.3,
    chi_max=64,
    measure_every=5,
    verbose=True,
):
    """
    Run simulations for chirality = +1 and -1
    
    Check mirror symmetry: V_+(t) vs -V_-(t)
    
    Returns
    -------
    dict with results for both chiralities
    """
    
    if verbose:
        print("\n" + "="*70)
        print("CHIRALITY SCAN (χ = ±1)")
        print("="*70)
        print(f"  L={L}, l={oam_l}, N_pump={N_pump}, N_free={N_free}")
        print("="*70)
    
    results = {}
    
    for chirality in [+1, -1]:
        if verbose:
            print(f"\n--- Running χ = {chirality:+d} ---")
        
        res = run_ladder_tebd(
            L=L,
            N_pump=N_pump,
            N_free=N_free,
            dt=dt,
            g=g,
            omega=omega,
            oam_l=oam_l,
            chirality=chirality,
            chi_max=chi_max,
            measure_every=measure_every,
            make_plots=False,
            verbose=verbose,
        )
        
        results[f'{chirality:+d}'] = res
    
    return results


def compare_chiralities(results, save_path=None, show=True):
    """
    Compare results from chirality scan
    
    Parameters
    ----------
    results : dict
        Output from chirality_scan()
    """
    
    res_p = results['+1']
    res_m = results['-1']
    
    times = res_p['times']
    V_p = res_p['V']
    V_m = res_m['V']
    V2_p = res_p['V2']
    V2_m = res_m['V2']
    
    # Correlation: V_+ vs -V_-
    corr = np.corrcoef(V_p, -V_m)[0, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Vorticity comparison
    ax1 = axes[0, 0]
    ax1.plot(times, V_p, 'r-', linewidth=2, label='χ = +1')
    ax1.plot(times, V_m, 'b-', linewidth=2, alpha=0.7, label='χ = -1')
    ax1.plot(times, -V_m, 'k--', linewidth=1.5, alpha=0.5, label='-V(χ=-1)')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('V')
    ax1.set_title(f'Vorticity: Mirror Symmetry (corr={corr:.4f})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # (2) V² comparison
    ax2 = axes[0, 1]
    ax2.plot(times, V2_p, 'r-', linewidth=2, label='χ = +1')
    ax2.plot(times, V2_m, 'b-', linewidth=2, alpha=0.7, label='χ = -1')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('V²')
    ax2.set_title('V² (Should be identical)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # (3) Entanglement comparison
    ax3 = axes[1, 0]
    S_p = res_p['S_contour'].mean(axis=1)
    S_m = res_m['S_contour'].mean(axis=1)
    
    ax3.plot(times, S_p, 'r-', linewidth=2, label='χ = +1')
    ax3.plot(times, S_m, 'b-', linewidth=2, alpha=0.7, label='χ = -1')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mean S')
    ax3.set_title('Mean Entanglement')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # (4) Topological order comparison
    ax4 = axes[1, 1]
    top_p = compute_topological_order_parameter(res_p)
    top_m = compute_topological_order_parameter(res_m)
    
    categories = ['TOP', 'Persistence']
    values_p = [top_p['TOP'], top_p['persistence']]
    values_m = [top_m['TOP'], top_m['persistence']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x - width/2, values_p, width, label='χ = +1', color='red', alpha=0.7)
    ax4.bar(x + width/2, values_m, width, label='χ = -1', color='blue', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel('Value')
    ax4.set_title('Topological Order Parameters')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Λ³ Chirality Scan: χ = +1 vs -1', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("CHIRALITY SCAN SUMMARY")
    print("="*70)
    print(f"  V correlation (χ=+1 vs -χ=-1): {corr:.6f}")
    print(f"\n  χ = +1:")
    print(f"    TOP = {top_p['TOP']:.4f}")
    print(f"    Persistence = {top_p['persistence']:.4f}")
    print(f"\n  χ = -1:")
    print(f"    TOP = {top_m['TOP']:.4f}")
    print(f"    Persistence = {top_m['persistence']:.4f}")
    print("="*70)
    
    return fig, axes


# =============================================================================
# OAM Charge Scan
# =============================================================================

def oam_charge_scan(
    L=10,
    oam_charges=(1, 2, 3),
    chirality=1,
    N_pump=200,
    N_free=200,
    dt=0.05,
    g=0.3,
    omega=0.3,
    chi_max=64,
    measure_every=5,
    verbose=True,
):
    """
    Scan OAM charge l = 1, 2, 3, ...
    
    Returns
    -------
    dict with results for each l
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"OAM CHARGE SCAN (l = {oam_charges})")
        print("="*70)
        print(f"  L={L}, χ={chirality}, N_pump={N_pump}, N_free={N_free}")
        print("="*70)
    
    results = {}
    
    for l in oam_charges:
        if verbose:
            print(f"\n--- Running l = {l} ---")
        
        res = run_ladder_tebd(
            L=L,
            N_pump=N_pump,
            N_free=N_free,
            dt=dt,
            g=g,
            omega=omega,
            oam_l=l,
            chirality=chirality,
            chi_max=chi_max,
            measure_every=measure_every,
            make_plots=False,
            verbose=verbose,
        )
        
        results[l] = res
    
    return results


def compare_oam_charges(results, save_path=None, show=True):
    """
    Compare results from OAM charge scan
    """
    
    oam_charges = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (1) V evolution
    ax1 = axes[0, 0]
    for l in oam_charges:
        times = results[l]['times']
        V = results[l]['V']
        ax1.plot(times, V, linewidth=2, label=f'l={l}', alpha=0.7)
    ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('V')
    ax1.set_title('Vorticity V(t)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # (2) V² evolution
    ax2 = axes[0, 1]
    for l in oam_charges:
        times = results[l]['times']
        V2 = results[l]['V2']
        ax2.plot(times, V2, linewidth=2, label=f'l={l}', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('V²')
    ax2.set_title('V² Evolution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # (3) Mean entanglement
    ax3 = axes[0, 2]
    for l in oam_charges:
        times = results[l]['times']
        S_mean = results[l]['S_contour'].mean(axis=1)
        ax3.plot(times, S_mean, linewidth=2, label=f'l={l}', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mean S')
    ax3.set_title('Mean Entanglement')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # (4) Max |V| vs l
    ax4 = axes[1, 0]
    max_V = []
    for l in oam_charges:
        V = results[l]['V']
        max_V.append(np.max(np.abs(V)))
    
    ax4.plot(oam_charges, max_V, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('OAM charge l')
    ax4.set_ylabel('Max |V|')
    ax4.set_title('Max Vorticity vs l')
    ax4.grid(alpha=0.3)
    
    # (5) Topological order vs l
    ax5 = axes[1, 1]
    TOP_list = []
    pers_list = []
    
    for l in oam_charges:
        top = compute_topological_order_parameter(results[l])
        TOP_list.append(top['TOP'])
        pers_list.append(top['persistence'])
    
    ax5.plot(oam_charges, TOP_list, 'o-', linewidth=2, markersize=8, label='TOP')
    ax5.plot(oam_charges, pers_list, 's-', linewidth=2, markersize=8, label='Persistence')
    ax5.set_xlabel('OAM charge l')
    ax5.set_ylabel('Value')
    ax5.set_title('Topological Order vs l')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # (6) Summary table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "OAM Charge Scan Results\n" + "="*30 + "\n\n"
    for l, top_val, pers in zip(oam_charges, TOP_list, pers_list):
        max_v = np.max(np.abs(results[l]['V']))
        summary_text += f"l = {l}:\n"
        summary_text += f"  Max|V| = {max_v:.4f}\n"
        summary_text += f"  TOP    = {top_val:.4f}\n"
        summary_text += f"  Pers.  = {pers:.4f}\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Λ³ OAM Charge Scan', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


# =============================================================================
# Ladder Size Scan
# =============================================================================

def ladder_size_scan(
    L_values=(5, 10, 15, 20),
    oam_l=1,
    chirality=1,
    N_pump=200,
    N_free=200,
    dt=0.05,
    g=0.3,
    omega=0.3,
    chi_max=64,
    measure_every=5,
    verbose=True,
):
    """
    Scan ladder size L = 5, 10, 15, 20, ...
    
    Check finite-size scaling
    
    Returns
    -------
    dict with results for each L
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"LADDER SIZE SCAN (L = {L_values})")
        print("="*70)
        print(f"  l={oam_l}, χ={chirality}, N_pump={N_pump}, N_free={N_free}")
        print("="*70)
    
    results = {}
    timing = {}
    
    for L in L_values:
        if verbose:
            print(f"\n--- Running L = {L} ({2*L} sites) ---")
        
        t0 = time.time()
        res = run_ladder_tebd(
            L=L,
            N_pump=N_pump,
            N_free=N_free,
            dt=dt,
            g=g,
            omega=omega,
            oam_l=oam_l,
            chirality=chirality,
            chi_max=chi_max,
            measure_every=measure_every,
            make_plots=False,
            verbose=verbose,
        )
        elapsed = time.time() - t0
        
        results[L] = res
        timing[L] = elapsed
    
    results['_timing'] = timing
    
    return results


def compare_ladder_sizes(results, save_path=None, show=True):
    """
    Compare results from ladder size scan
    """
    
    timing = results.pop('_timing', {})
    L_values = sorted([k for k in results.keys() if isinstance(k, int)])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (1) V² at final time vs L
    ax1 = axes[0, 0]
    V2_final = []
    for L in L_values:
        V2_final.append(results[L]['V2'][-1])
    
    ax1.plot(L_values, V2_final, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Ladder size L')
    ax1.set_ylabel('V²(final)')
    ax1.set_title('Final V² vs System Size')
    ax1.grid(alpha=0.3)
    
    # (2) Persistence vs L
    ax2 = axes[0, 1]
    pers_list = []
    for L in L_values:
        top = compute_topological_order_parameter(results[L])
        pers_list.append(top['persistence'])
    
    ax2.plot(L_values, pers_list, 's-', linewidth=2, markersize=8, color='purple')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ladder size L')
    ax2.set_ylabel('Persistence')
    ax2.set_title('Persistence vs System Size')
    ax2.grid(alpha=0.3)
    
    # (3) Mean entanglement vs L
    ax3 = axes[0, 2]
    S_mean_list = []
    for L in L_values:
        S_mean = results[L]['S_contour'].mean()
        S_mean_list.append(S_mean)
    
    ax3.plot(L_values, S_mean_list, '^-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Ladder size L')
    ax3.set_ylabel('Mean S')
    ax3.set_title('Mean Entanglement vs System Size')
    ax3.grid(alpha=0.3)
    
    # (4) Computation time vs L
    ax4 = axes[1, 0]
    if timing:
        times = [timing.get(L, 0) for L in L_values]
        ax4.plot(L_values, times, 'o-', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Ladder size L')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Computation Time')
        ax4.grid(alpha=0.3)
    
    # (5) V² evolution for all L
    ax5 = axes[1, 1]
    for L in L_values:
        times = results[L]['times']
        V2 = results[L]['V2']
        ax5.plot(times, V2, linewidth=2, label=f'L={L}', alpha=0.7)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('V²')
    ax5.set_title('V² Evolution (All Sizes)')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # (6) Scaling summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "Finite-Size Scaling\n" + "="*30 + "\n\n"
    for L in L_values:
        N = 2 * L
        top = compute_topological_order_parameter(results[L])
        t = timing.get(L, 0)
        summary_text += f"L = {L} (N={N}):\n"
        summary_text += f"  V²_fin = {results[L]['V2'][-1]:.4f}\n"
        summary_text += f"  Pers.  = {top['persistence']:.4f}\n"
        summary_text += f"  Time   = {t:.1f}s\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Λ³ Ladder Size Scan', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    results['_timing'] = timing  # Restore
    
    return fig, axes


# =============================================================================
# chi_max Convergence Test
# =============================================================================

def chi_convergence_test(
    L=10,
    chi_values=(16, 32, 64, 128),
    oam_l=1,
    chirality=1,
    N_pump=200,
    N_free=200,
    dt=0.05,
    g=0.3,
    omega=0.3,
    measure_every=5,
    verbose=True,
):
    """
    Test convergence with respect to bond dimension chi_max
    
    Returns
    -------
    dict with results for each chi_max
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"BOND DIMENSION CONVERGENCE TEST (χ_max = {chi_values})")
        print("="*70)
        print(f"  L={L}, l={oam_l}, χ={chirality}")
        print("="*70)
    
    results = {}
    
    for chi in chi_values:
        if verbose:
            print(f"\n--- Running χ_max = {chi} ---")
        
        res = run_ladder_tebd(
            L=L,
            N_pump=N_pump,
            N_free=N_free,
            dt=dt,
            g=g,
            omega=omega,
            oam_l=oam_l,
            chirality=chirality,
            chi_max=chi,
            measure_every=measure_every,
            make_plots=False,
            verbose=verbose,
        )
        
        results[chi] = res
    
    return results


def compare_chi_convergence(results, save_path=None, show=True):
    """
    Compare convergence with chi_max
    """
    
    chi_values = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) V² evolution
    ax1 = axes[0, 0]
    for chi in chi_values:
        times = results[chi]['times']
        V2 = results[chi]['V2']
        ax1.plot(times, V2, linewidth=2, label=f'χ={chi}', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('V²')
    ax1.set_title('V² Evolution (Convergence)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # (2) V² at final time vs chi
    ax2 = axes[0, 1]
    V2_final = [results[chi]['V2'][-1] for chi in chi_values]
    ax2.plot(chi_values, V2_final, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('χ_max')
    ax2.set_ylabel('V²(final)')
    ax2.set_title('Final V² vs Bond Dimension')
    ax2.grid(alpha=0.3)
    
    # (3) Mean entanglement vs chi
    ax3 = axes[1, 0]
    S_mean_list = [results[chi]['S_contour'].mean() for chi in chi_values]
    ax3.plot(chi_values, S_mean_list, 's-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('χ_max')
    ax3.set_ylabel('Mean S')
    ax3.set_title('Mean Entanglement vs χ_max')
    ax3.grid(alpha=0.3)
    
    # (4) Convergence table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Convergence Summary\n" + "="*30 + "\n\n"
    for chi in chi_values:
        top = compute_topological_order_parameter(results[chi])
        V2_f = results[chi]['V2'][-1]
        S_m = results[chi]['S_contour'].mean()
        summary_text += f"χ_max = {chi}:\n"
        summary_text += f"  V²_fin = {V2_f:.6f}\n"
        summary_text += f"  Pers.  = {top['persistence']:.6f}\n"
        summary_text += f"  <S>    = {S_m:.6f}\n\n"
    
    # Convergence check
    if len(chi_values) >= 2:
        V2_diff = abs(V2_final[-1] - V2_final[-2]) / V2_final[-2] * 100
        summary_text += f"\nConvergence:\n"
        summary_text += f"  ΔV² = {V2_diff:.2f}%\n"
        if V2_diff < 1.0:
            summary_text += "  ✓ CONVERGED\n"
        else:
            summary_text += "  ⚠ Need higher χ\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Λ³ Bond Dimension Convergence Test', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


# =============================================================================
# Full Experiment Suite
# =============================================================================

def run_full_experiment_suite(
    output_dir='./experiments',
    L=10,
    verbose=True,
):
    """
    Run complete experiment suite:
    1. Chirality scan
    2. OAM charge scan
    3. Ladder size scan (smaller L values)
    4. chi convergence (if time permits)
    
    Parameters
    ----------
    output_dir : str
        Directory for outputs
    L : int
        Base ladder size
    verbose : bool
        Print progress
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Λ³ FULL EXPERIMENT SUITE")
    print("="*70)
    print(f"  Output directory: {output_dir}")
    print(f"  Base ladder size: L={L}")
    print("="*70)
    
    # 1. Chirality scan
    print("\n[1/4] CHIRALITY SCAN")
    res_chirality = chirality_scan(L=L, verbose=verbose)
    compare_chiralities(res_chirality,
                        save_path=f"{output_dir}/chirality_scan.png",
                        show=False)
    
    # 2. OAM charge scan
    print("\n[2/4] OAM CHARGE SCAN")
    res_oam = oam_charge_scan(L=L, oam_charges=(1, 2, 3), verbose=verbose)
    compare_oam_charges(res_oam,
                        save_path=f"{output_dir}/oam_charge_scan.png",
                        show=False)
    
    # 3. Ladder size scan
    print("\n[3/4] LADDER SIZE SCAN")
    L_vals = [5, 8, 10] if L >= 10 else [5, 7]
    res_size = ladder_size_scan(L_values=L_vals, verbose=verbose)
    compare_ladder_sizes(res_size,
                         save_path=f"{output_dir}/ladder_size_scan.png",
                         show=False)
    
    # 4. chi convergence (optional, can be slow)
    print("\n[4/4] BOND DIMENSION CONVERGENCE")
    res_chi = chi_convergence_test(L=8, chi_values=(32, 64), verbose=verbose)
    compare_chi_convergence(res_chi,
                            save_path=f"{output_dir}/chi_convergence.png",
                            show=False)
    
    print("\n" + "="*70)
    print("FULL EXPERIMENT SUITE COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("  - chirality_scan.png")
    print("  - oam_charge_scan.png")
    print("  - ladder_size_scan.png")
    print("  - chi_convergence.png")
    
    return {
        'chirality': res_chirality,
        'oam': res_oam,
        'size': res_size,
        'chi': res_chi,
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Λ³ Experiment Runner & Parameter Scans                         ║
║                                                                  ║
║  Usage:                                                          ║
║    from run_experiments import chirality_scan, oam_charge_scan  ║
║                                                                  ║
║    # Quick scans                                                ║
║    results = chirality_scan(L=10)                               ║
║    compare_chiralities(results)                                 ║
║                                                                  ║
║    # Full suite                                                 ║
║    run_full_experiment_suite(output_dir='./my_experiments')     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
