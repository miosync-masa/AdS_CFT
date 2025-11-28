"""
Λ³ Analysis & Visualization Tools
==================================

Advanced analysis and visualization for Λ³ TEBD simulations

Features:
1. 3D Bulk Geometry Visualization
2. Entanglement Spectrum Analysis
3. Curvature & Ricci Scalar
4. Topological Order Parameters
5. Persistence Analysis

Author: Masamichi & Tamaki (環)
Date: 2025-11-29
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 3D Bulk Geometry Visualization
# =============================================================================

def plot_bulk_3d(results, time_index=-1, save_path=None, show=True):
    """
    3D surface plot of bulk geometry at specific time
    
    Parameters
    ----------
    results : dict
        Output from run_ladder_tebd
    time_index : int
        Time index to plot (-1 = final)
    save_path : str, optional
        Save path for figure
    show : bool
        Show plot
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    
    S_contour = results['S_contour']
    times = results['times']
    N = S_contour.shape[1]
    
    t = times[time_index]
    S = S_contour[time_index]
    
    # Create spatial grid
    x = np.arange(N)
    X, Y = np.meshgrid(x, [0])
    Z = S.reshape(1, -1)
    
    fig = plt.figure(figsize=(14, 10))
    
    # (1) Entanglement contour
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Site i')
    ax1.set_ylabel('Replica')
    ax1.set_zlabel('S(i)')
    ax1.set_title(f'Entanglement Contour S(i) at t={t:.2f}')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # (2) Bulk metric g = 1/S
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    g = 1.0 / (S + 1e-8)
    Z_g = g.reshape(1, -1)
    surf2 = ax2.plot_surface(X, Y, Z_g, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Site i')
    ax2.set_ylabel('Replica')
    ax2.set_zlabel('g(i)')
    ax2.set_title(f'Bulk Metric g = 1/S(i) at t={t:.2f}')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # (3) Curvature R = -∇²S
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    curvature = results['curvature'][time_index]
    Z_R = curvature.reshape(1, -1)
    surf3 = ax3.plot_surface(X, Y, Z_R, cmap='RdBu_r', alpha=0.8)
    ax3.set_xlabel('Site i')
    ax3.set_ylabel('Replica')
    ax3.set_zlabel('R(i)')
    ax3.set_title(f'Curvature R = -∇²S at t={t:.2f}')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # (4) Combined visualization
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x, S, 'o-', label='S(i)', linewidth=2, markersize=6)
    ax4.plot(x, g / g.max() * S.max(), 's-', label='g(i) (scaled)', alpha=0.7)
    ax4.plot(x, curvature / np.abs(curvature).max() * S.max(), '^-', 
             label='R(i) (scaled)', alpha=0.7)
    ax4.set_xlabel('Site i')
    ax4.set_ylabel('Value')
    ax4.set_title('Combined Profile')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle(f'Λ³ Bulk Geometry Reconstruction (t={t:.2f})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, (ax1, ax2, ax3, ax4)


def plot_bulk_spacetime(results, save_path=None, show=True):
    """
    2D spacetime diagram of bulk geometry evolution
    
    Shows S(i,t) as 2D heatmap with proper spacetime structure
    """
    
    S_contour = results['S_contour']
    times = results['times']
    N = S_contour.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Entanglement spacetime
    im1 = axes[0, 0].imshow(
        S_contour.T,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, N-1],
        cmap='viridis'
    )
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Site i')
    axes[0, 0].set_title('Entanglement Contour S(i,t)')
    plt.colorbar(im1, ax=axes[0, 0], label='S')
    
    # (2) Bulk metric spacetime
    bulk_metric = results['bulk_metric']
    im2 = axes[0, 1].imshow(
        bulk_metric.T,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, N-1],
        cmap='plasma'
    )
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Site i')
    axes[0, 1].set_title('Bulk Metric g = 1/S(i,t)')
    plt.colorbar(im2, ax=axes[0, 1], label='g')
    
    # (3) Curvature spacetime
    curvature = results['curvature']
    im3 = axes[1, 0].imshow(
        curvature.T,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, N-1],
        cmap='RdBu_r',
        vmin=-np.percentile(np.abs(curvature), 95),
        vmax=np.percentile(np.abs(curvature), 95)
    )
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Site i')
    axes[1, 0].set_title('Curvature R(i,t) = -∇²S')
    plt.colorbar(im3, ax=axes[1, 0], label='R')
    
    # (4) Mean quantities
    S_mean = S_contour.mean(axis=1)
    S_std = S_contour.std(axis=1)
    R_mean = curvature.mean(axis=1)
    
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    ax4.plot(times, S_mean, 'b-', linewidth=2, label='<S>')
    ax4.fill_between(times, S_mean - S_std, S_mean + S_std, alpha=0.3, color='b')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('<S>', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax4_twin.plot(times, R_mean, 'r-', linewidth=2, label='<R>')
    ax4_twin.set_ylabel('<R>', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title('Mean Entanglement & Curvature')
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Λ³ Bulk Spacetime Evolution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def create_bulk_animation(results, save_path='bulk_evolution.mp4', fps=10):
    """
    Create animation of bulk geometry evolution
    
    Parameters
    ----------
    results : dict
        Output from run_ladder_tebd
    save_path : str
        Output video path
    fps : int
        Frames per second
    """
    
    S_contour = results['S_contour']
    times = results['times']
    N = S_contour.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': '3d'})
    
    x = np.arange(N)
    X, Y = np.meshgrid(x, [0])
    
    def update(frame):
        ax.clear()
        S = S_contour[frame]
        Z = S.reshape(1, -1)
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Site i')
        ax.set_ylabel('Replica')
        ax.set_zlabel('S(i)')
        ax.set_zlim(0, S_contour.max())
        ax.set_title(f'Entanglement Contour at t={times[frame]:.2f}')
        
        return surf,
    
    anim = FuncAnimation(fig, update, frames=len(times), interval=1000/fps, blit=False)
    
    try:
        anim.save(save_path, fps=fps, dpi=100)
        print(f"[Saved] {save_path}")
    except Exception as e:
        print(f"[Warning] Could not save animation: {e}")
        print("  Try installing ffmpeg: conda install -c conda-forge ffmpeg")
    
    plt.close()
    return anim


# =============================================================================
# Entanglement Spectrum Analysis
# =============================================================================

def analyze_entanglement_spectrum(results, time_indices=None):
    """
    Analyze entanglement spectrum evolution
    
    Parameters
    ----------
    results : dict
        Must contain 'S_contour'
    time_indices : list, optional
        Specific time points to analyze
    
    Returns
    -------
    dict with spectrum analysis
    """
    
    S_contour = results['S_contour']
    times = results['times']
    
    if time_indices is None:
        # Analyze initial, mid, final
        time_indices = [0, len(times)//2, -1]
    
    analysis = {
        'times': [times[i] for i in time_indices],
        'S_profiles': [S_contour[i] for i in time_indices],
        'max_S': [S_contour[i].max() for i in time_indices],
        'mean_S': [S_contour[i].mean() for i in time_indices],
        'total_S': [S_contour[i].sum() for i in time_indices],
    }
    
    # Compute spectrum gaps (difference between max and min)
    gaps = []
    for i in time_indices:
        S = S_contour[i]
        gap = S.max() - S.min()
        gaps.append(gap)
    
    analysis['gaps'] = gaps
    
    return analysis


def plot_entanglement_spectrum(results, time_indices=None, save_path=None, show=True):
    """
    Plot entanglement spectrum at multiple times
    """
    
    analysis = analyze_entanglement_spectrum(results, time_indices)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Spectrum profiles
    ax1 = axes[0, 0]
    for t, S in zip(analysis['times'], analysis['S_profiles']):
        ax1.plot(S, 'o-', label=f't={t:.2f}', alpha=0.7)
    ax1.set_xlabel('Site i')
    ax1.set_ylabel('S(i)')
    ax1.set_title('Entanglement Spectrum Profiles')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # (2) Spectrum evolution
    ax2 = axes[0, 1]
    S_contour = results['S_contour']
    times = results['times']
    for i in range(S_contour.shape[1]):
        ax2.plot(times, S_contour[:, i], alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('S(i)')
    ax2.set_title('All Site Entanglement Evolution')
    ax2.grid(alpha=0.3)
    
    # (3) Statistics
    ax3 = axes[1, 0]
    S_max = S_contour.max(axis=1)
    S_mean = S_contour.mean(axis=1)
    S_total = S_contour.sum(axis=1)
    
    ax3.plot(times, S_max, 'r-', label='Max S', linewidth=2)
    ax3.plot(times, S_mean, 'g-', label='Mean S', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('S')
    ax3.set_title('Entanglement Statistics')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(times, S_total, 'b--', label='Total S', linewidth=2)
    ax3_twin.set_ylabel('Total S', color='b')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    
    # (4) Gap evolution
    ax4 = axes[1, 1]
    gaps = S_max - S_contour.min(axis=1)
    ax4.plot(times, gaps, 'purple', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Gap')
    ax4.set_title('Entanglement Gap (Max - Min)')
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Λ³ Entanglement Spectrum Analysis', fontsize=14)
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
# Curvature & Ricci Scalar
# =============================================================================

def compute_ricci_scalar(S_contour, dx=1.0):
    """
    Compute Ricci scalar from entanglement contour
    
    R = g^{ij} R_{ij} ≈ -g ∇²S
    
    For 1D: R ≈ -(1/S) ∇²S
    
    Parameters
    ----------
    S_contour : np.ndarray
        Shape (T, N)
    dx : float
        Lattice spacing
    
    Returns
    -------
    R : np.ndarray
        Ricci scalar, shape (T, N)
    """
    
    T, N = S_contour.shape
    R = np.zeros_like(S_contour)
    
    for t in range(T):
        S = S_contour[t]
        g = 1.0 / (S + 1e-8)
        
        # ∇²S
        laplacian = np.zeros(N)
        for i in range(1, N-1):
            laplacian[i] = (S[i-1] - 2*S[i] + S[i+1]) / dx**2
        
        # Boundary
        laplacian[0] = laplacian[1]
        laplacian[-1] = laplacian[-2]
        
        # R = -g ∇²S
        R[t] = -g * laplacian
    
    return R


def plot_ricci_evolution(results, save_path=None, show=True):
    """
    Plot Ricci scalar evolution
    """
    
    S_contour = results['S_contour']
    times = results['times']
    
    R = compute_ricci_scalar(S_contour)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Ricci spacetime
    ax1 = axes[0, 0]
    im = ax1.imshow(
        R.T,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, S_contour.shape[1]-1],
        cmap='RdBu_r',
        vmin=-np.percentile(np.abs(R), 95),
        vmax=np.percentile(np.abs(R), 95)
    )
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Site i')
    ax1.set_title('Ricci Scalar R(i,t)')
    plt.colorbar(im, ax=ax1, label='R')
    
    # (2) Mean Ricci
    ax2 = axes[0, 1]
    R_mean = R.mean(axis=1)
    R_std = R.std(axis=1)
    ax2.plot(times, R_mean, 'r-', linewidth=2)
    ax2.fill_between(times, R_mean - R_std, R_mean + R_std, alpha=0.3, color='r')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('<R>')
    ax2.set_title('Mean Ricci Scalar')
    ax2.grid(alpha=0.3)
    
    # (3) Ricci profiles
    ax3 = axes[1, 0]
    time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    for ti in time_indices:
        ax3.plot(R[ti], 'o-', label=f't={times[ti]:.2f}', alpha=0.7)
    ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Site i')
    ax3.set_ylabel('R(i)')
    ax3.set_title('Ricci Profiles at Key Times')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # (4) Correlation: R vs Curvature
    ax4 = axes[1, 1]
    curvature = results['curvature']
    ax4.scatter(curvature.flatten(), R.flatten(), alpha=0.1, s=1)
    ax4.set_xlabel('Curvature (from ∇²S)')
    ax4.set_ylabel('Ricci Scalar R')
    ax4.set_title('R vs Curvature Correlation')
    ax4.grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(curvature.flatten(), R.flatten())[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.4f}',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Λ³ Ricci Scalar Analysis', fontsize=14)
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
# Topological Order Parameter
# =============================================================================

def compute_topological_order_parameter(results):
    """
    Compute topological order parameter from vorticity persistence
    
    TOP = <V²>_free / V²_pump
    
    Returns
    -------
    dict with TOP analysis
    """
    
    V2 = results['V2']
    times = results['times']
    
    # Find pump-off index
    if 'N_pump' in results:
        N_pump = results['N_pump']
        measure_every = results.get('measure_every', 5)
        pump_off_idx = N_pump // measure_every
    else:
        # Assume pump ends at half
        pump_off_idx = len(V2) // 2
    
    if pump_off_idx >= len(V2):
        pump_off_idx = len(V2) - 1
    
    V2_pump = V2[pump_off_idx]
    V2_free = V2[pump_off_idx:]
    
    if V2_pump > 1e-8:
        TOP = V2_free.mean() / V2_pump
        persistence = V2[-1] / V2_pump
    else:
        TOP = 0.0
        persistence = 0.0
    
    return {
        'TOP': TOP,
        'persistence': persistence,
        'V2_pump': V2_pump,
        'V2_final': V2[-1],
        'V2_free_mean': V2_free.mean(),
        'V2_free_std': V2_free.std(),
        'pump_off_idx': pump_off_idx,
    }


def plot_topological_analysis(results, save_path=None, show=True):
    """
    Comprehensive topological order analysis
    """
    
    top = compute_topological_order_parameter(results)
    
    V = results['V']
    V2 = results['V2']
    times = results['times']
    pump_off_idx = top['pump_off_idx']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) V evolution
    ax1 = axes[0, 0]
    ax1.plot(times, V, 'orange', linewidth=2)
    ax1.axvline(times[pump_off_idx], color='r', linestyle='--', label='Pump OFF')
    ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('V')
    ax1.set_title('Vorticity V(t)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # (2) V² evolution
    ax2 = axes[0, 1]
    ax2.plot(times, V2, 'g-', linewidth=2)
    ax2.axvline(times[pump_off_idx], color='r', linestyle='--', label='Pump OFF')
    ax2.axhline(top['V2_pump'], color='gray', linestyle=':', alpha=0.5, label='V²(pump)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('V²')
    ax2.set_title('V² Evolution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # (3) Persistence
    ax3 = axes[1, 0]
    V2_free = V2[pump_off_idx:]
    times_free = times[pump_off_idx:]
    
    ax3.plot(times_free, V2_free, 'b-', linewidth=2, label='V²(t)')
    ax3.axhline(top['V2_free_mean'], color='purple', linestyle='--',
                label=f"Mean = {top['V2_free_mean']:.4f}")
    ax3.fill_between(times_free,
                     top['V2_free_mean'] - top['V2_free_std'],
                     top['V2_free_mean'] + top['V2_free_std'],
                     alpha=0.3, color='purple')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('V²')
    ax3.set_title('Free Evolution Persistence')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # (4) Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Topological Order Parameter (TOP)
{'='*40}

Topological Order Parameter:
    TOP = <V²>_free / V²_pump = {top['TOP']:.4f}

Persistence Ratio:
    η = V²_final / V²_pump = {top['persistence']:.4f}

V² Values:
    V²(pump-off) = {top['V2_pump']:.4f}
    V²(final)    = {top['V2_final']:.4f}
    <V²>_free    = {top['V2_free_mean']:.4f} ± {top['V2_free_std']:.4f}

Classification:
    {'🏆 PERSISTENT TOPOLOGICAL ORDER!' if top['persistence'] > 0.5 else 
     '⚡ Partial persistence' if top['persistence'] > 0.1 else
     '⚠️ Decay after pump off'}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Λ³ Topological Order Analysis', fontsize=14)
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
# Quick Analysis Suite
# =============================================================================

def full_analysis(results, output_dir='.', prefix='analysis'):
    """
    Run complete analysis suite and save all plots
    
    Parameters
    ----------
    results : dict
        Output from run_ladder_tebd
    output_dir : str
        Directory for output files
    prefix : str
        Prefix for output files
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Λ³ FULL ANALYSIS SUITE")
    print("="*70)
    
    # 1. 3D Bulk
    print("\n[1/5] 3D Bulk Geometry...")
    plot_bulk_3d(results, 
                 save_path=f"{output_dir}/{prefix}_bulk_3d.png",
                 show=False)
    
    # 2. Bulk Spacetime
    print("[2/5] Bulk Spacetime...")
    plot_bulk_spacetime(results,
                        save_path=f"{output_dir}/{prefix}_bulk_spacetime.png",
                        show=False)
    
    # 3. Entanglement Spectrum
    print("[3/5] Entanglement Spectrum...")
    plot_entanglement_spectrum(results,
                                save_path=f"{output_dir}/{prefix}_spectrum.png",
                                show=False)
    
    # 4. Ricci Scalar
    print("[4/5] Ricci Scalar...")
    plot_ricci_evolution(results,
                          save_path=f"{output_dir}/{prefix}_ricci.png",
                          show=False)
    
    # 5. Topological Order
    print("[5/5] Topological Order...")
    plot_topological_analysis(results,
                               save_path=f"{output_dir}/{prefix}_topological.png",
                               show=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved in: {output_dir}/")
    print(f"  - {prefix}_bulk_3d.png")
    print(f"  - {prefix}_bulk_spacetime.png")
    print(f"  - {prefix}_spectrum.png")
    print(f"  - {prefix}_ricci.png")
    print(f"  - {prefix}_topological.png")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Λ³ Analysis & Visualization Tools                              ║
║                                                                  ║
║  Usage:                                                          ║
║    from lambda3_ladder_tebd import run_ladder_tebd              ║
║    from analysis_tools import full_analysis                     ║
║                                                                  ║
║    results = run_ladder_tebd(L=10, ...)                         ║
║    full_analysis(results, output_dir='./figures')               ║
╚══════════════════════════════════════════════════════════════════╝
    """)
