"""
lambda3_holo/run_experiment.py - Improved runner with burn-in support
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .automaton import Automaton
from .plotters import plot_rt_timeseries, plot_crosscorr, plot_transfer_entropy
from .metrics import spearman_corr

def run_experiment(args):
    """Run experiment with burn-in period for stability"""
    os.makedirs(args.outdir, exist_ok=True)
    
    # Initialize automaton with full parameter set
    ua = Automaton(
        H=args.H, W=args.W, Z=args.Z, 
        L_ads=args.L_ads, alpha=args.alpha,
        c0=args.c0, gamma=args.gamma, c_eff_max=args.c_eff_max,
        gate_delay=args.gate_delay, gate_strength=args.gate_strength,
        soc_rate=args.soc_rate, seed=args.seed
    )
    
    # Reset state explicitly for reproducibility
    ua.reset_state()
    
    # Run with burn-in period
    print(f"[INFO] Running {args.preset}: burn-in={args.burnin}, measure={args.steps}")
    rows = ua.run_with_burnin(
        burn_in=args.burnin,
        measure_steps=args.steps
    )
    
    # Add derived metrics
    prev = None
    for rec in rows:
        rec["dS_RT_mo"] = 0.0 if prev is None else rec["entropy_RT_mo"] - prev
        prev = rec["entropy_RT_mo"]
    
    # Save to DataFrame
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate plots
    f1 = os.path.join(args.outdir, "fig_rt_timeseries.png")
    plot_rt_timeseries(df, f1)
    
    f2 = os.path.join(args.outdir, "fig_xcorr.png")
    best_lag, best_corr = plot_crosscorr(
        df, "lambda_p99_A_out_pre", "entropy_RT_mo", f2, maxlag=args.maxlag
    )
    
    # Compute Spearman correlation at best lag
    def shift_for_lag(a, b, lag):
        if lag == 0: 
            return a, b
        elif lag > 0: 
            return a[lag:], b[:-lag]
        else: 
            return a[:lag], b[-lag:]
    
    S = df["entropy_RT_mo"].values
    lam = df["lambda_p99_A_out_pre"].values
    
    # Handle NaN interpolation
    if np.isnan(lam).any():
        idx = np.arange(len(lam))
        m = ~np.isnan(lam)
        if m.sum() >= 2: 
            lam = np.interp(idx, idx[m], lam[m])
        else: 
            lam = np.nan_to_num(lam, nan=np.nanmean(lam) if np.isfinite(np.nanmean(lam)) else 0.0)
    
    S_lag, lam_lag = shift_for_lag(S, lam, best_lag)
    rho_s = spearman_corr(S_lag, lam_lag) if len(S_lag) > 3 else float("nan")
    
    # Transfer entropy analysis
    f3 = os.path.join(args.outdir, "fig_te.png")
    TE_x_to_y, TE_y_to_x = plot_transfer_entropy(
        df, "lambda_p99_A_out_pre", "entropy_RT_mo", f3, n_bins=3
    )
    
    # Write summary
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== PhaseShift-{args.preset.upper()} Results ===\n")
        f.write(f"Configuration:\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  Gate delay: {args.gate_delay}\n")
        f.write(f"  Gate strength: {args.gate_strength}\n")
        f.write(f"  c_eff_max: {args.c_eff_max}\n")
        f.write(f"  Burn-in: {args.burnin} steps\n")
        f.write(f"  Measurement: {args.steps} steps\n")
        f.write(f"\nResults:\n")
        f.write(f"  Best Pearson cross-corr (S vs λ_pre): {best_corr:.3f} at lag {best_lag}\n")
        f.write(f"  Spearman at best lag: {rho_s:.3f}\n")
        f.write(f"  Transfer Entropy (λ_pre→S): {TE_x_to_y:.4f} nats\n")
        f.write(f"  Transfer Entropy (S→λ_pre): {TE_y_to_x:.4f} nats\n")
    
    # Save metadata as JSON for easy parsing
    metadata = {
        "preset": args.preset,
        "seed": args.seed,
        "burnin": args.burnin,
        "steps": args.steps,
        "gate_delay": args.gate_delay,
        "gate_strength": args.gate_strength,
        "c0": args.c0,
        "gamma": args.gamma,
        "c_eff_max": args.c_eff_max,
        "alpha": args.alpha,
        "soc_rate": args.soc_rate,
        "results": {
            "best_lag": int(best_lag),
            "pearson_corr": float(best_corr),
            "spearman_corr": float(rho_s),
            "te_lambda_to_s": float(TE_x_to_y),
            "te_s_to_lambda": float(TE_y_to_x)
        }
    }
    
    json_path = os.path.join(args.outdir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print results
    print(f"[DONE] {args.preset} - outdir={args.outdir}")
    print(f"  Best Pearson corr={best_corr:.3f} at lag={best_lag}")
    print(f"  Spearman@bestlag={rho_s:.3f}")
    print(f"  TE λ→S={TE_x_to_y:.4f}, S→λ={TE_y_to_x:.4f}")
    
    # Success indicator
    if best_lag > 0 and rho_s > 0.4:
        print(f"  ✓ SUCCESS: Positive lag with strong monotonic correlation!")
    elif best_lag > 0:
        print(f"  ⚠ PARTIAL: Positive lag but weak correlation (ρ={rho_s:.3f})")
    else:
        print(f"  ✗ FAIL: No causal lag detected")
    
    import matplotlib.pyplot as plt
    
    # 詳細な相関プロット
    print("[DEBUG] Generating detailed correlation analysis...")
    lags = np.arange(-50, 51)
    corrs = []
    for lag in lags:
        try:
            if lag >= 0:
                a = df['lambda_p99_A_out_pre'].values[lag:]
                b = df['entropy_RT_mo'].values[:len(df)-lag]
            else:
                a = df['lambda_p99_A_out_pre'].values[:lag]
                b = df['entropy_RT_mo'].values[-lag:]
            
            if len(a) > 10:  # 十分なデータがある場合のみ
                corr = np.corrcoef(a, b)[0, 1]
                corrs.append(corr)
            else:
                corrs.append(np.nan)
        except:
            corrs.append(np.nan)
    
    plt.figure(figsize=(12, 6))
    plt.plot(lags, corrs, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=best_lag, color='r', linestyle='--', alpha=0.7, label=f'Best lag={best_lag}')
    plt.xlabel('Lag (steps)')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Cross-correlation: λ_p99(t) vs S_RT(t+lag)\nBest: {best_corr:.3f} at lag {best_lag}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    debug_plot_path = os.path.join(args.outdir, "debug_correlation.png")
    plt.savefig(debug_plot_path, dpi=150)
    plt.close()
    print(f"[DEBUG] Saved correlation plot to {debug_plot_path}")
    
    # λとS_RTの時系列を並べて表示
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # λの時系列
    axes[0].plot(df['t'], df['lambda_p99_A_out_pre'], 'b-', alpha=0.7)
    axes[0].set_ylabel('λ_p99')
    axes[0].set_title('Lambda p99 (outer band, pre)')
    axes[0].grid(True, alpha=0.3)
    
    # S_RTの時系列
    axes[1].plot(df['t'], df['entropy_RT_mo'], 'r-', alpha=0.7)
    axes[1].set_ylabel('S_RT')
    axes[1].set_title('RT Entropy (multi-objective)')
    axes[1].grid(True, alpha=0.3)
    
    # ゲート適用のタイミング
    axes[2].plot(df['t'], df['gate_applied_px'], 'g-', alpha=0.7)
    axes[2].set_ylabel('Gate pixels')
    axes[2].set_xlabel('Time step')
    axes[2].set_title('Gate application timing')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    debug_ts_path = os.path.join(args.outdir, "debug_timeseries.png")
    plt.savefig(debug_ts_path, dpi=150)
    plt.close()
    print(f"[DEBUG] Saved timeseries plot to {debug_ts_path}")
    
    # 統計情報を出力
    print("\n[DEBUG] Statistics:")
    print(f"  λ_p99 mean: {df['lambda_p99_A_out_pre'].mean():.3f}")
    print(f"  λ_p99 std: {df['lambda_p99_A_out_pre'].std():.3f}")
    print(f"  S_RT mean: {df['entropy_RT_mo'].mean():.3f}")
    print(f"  S_RT std: {df['entropy_RT_mo'].std():.3f}")
    print(f"  Gate applications: {(df['gate_applied_px'] > 0).sum()} times")
    print(f"  c_eff range: [{df['c_eff'].min():.3f}, {df['c_eff'].max():.3f}]")
    
    return metadata

def parse_args():
    p = argparse.ArgumentParser(
        description="Lambda3 Holography — PhaseShift runner with burn-in support"
    )
    
    # Preset configurations
    p.add_argument("--preset", type=str, default="ixb", 
                   choices=["ixb", "ixc", "custom"],
                   help="Preset: ixb (delay=1), ixc (delay=2), custom (use args)")
    
    # Output
    p.add_argument("--outdir", type=str, default="out",
                   help="Output directory for results")
    
    # Simulation parameters
    p.add_argument("--burnin", type=int, default=200,
                   help="Burn-in steps before measurement")
    p.add_argument("--steps", type=int, default=300,
                   help="Measurement steps after burn-in")
    
    # Grid geometry
    p.add_argument("--H", type=int, default=44,
                   help="Grid height")
    p.add_argument("--W", type=int, default=44,
                   help="Grid width")
    p.add_argument("--Z", type=int, default=24,
                   help="Bulk depth (z-layers)")
    
    # AdS/CFT parameters
    p.add_argument("--L_ads", type=float, default=1.0,
                   help="AdS radius")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="HR decay rate")
    p.add_argument("--c0", type=float, default=0.08,
                   help="Base coupling strength")
    p.add_argument("--gamma", type=float, default=0.8,
                   help="Coupling amplification")
    p.add_argument("--c_eff_max", type=float, default=0.18,
                   help="Maximum HR coupling (zero-lag suppression)")
    
    # Geodesic gating
    p.add_argument("--gate_delay", type=int, default=1,
                   help="Gate application delay (steps)")
    p.add_argument("--gate_strength", type=float, default=0.15,
                   help="Gate boost magnitude")
    
    # SOC
    p.add_argument("--soc_rate", type=float, default=0.01,
                   help="SOC tuning rate")
    
    # Random seed
    p.add_argument("--seed", type=int, default=913,
                   help="Random seed for reproducibility")
    
    # RT weights
    p.add_argument("--w_len", type=float, default=1.0,
                   help="Weight for perimeter term")
    p.add_argument("--w_hole", type=float, default=2.0,
                   help="Weight for hole count")
    p.add_argument("--w_curv", type=float, default=0.5,
                   help="Weight for curvature")
    
    # Analysis
    p.add_argument("--maxlag", type=int, default=40,
                   help="Maximum lag for cross-correlation")
    
    args = p.parse_args()
    
    # Apply preset configurations
    if args.preset == "ixb":
        print("[CONFIG] Using preset IXb: gate_delay=1")
        args.gate_delay = 1
        args.gate_strength = 0.15
        args.steps = max(args.steps, 300)
        args.burnin = max(args.burnin, 200)
        
    elif args.preset == "ixc":
        print("[CONFIG] Using preset IXc: gate_delay=2")
        args.gate_delay = 2
        args.gate_strength = 0.15
        args.steps = max(args.steps, 300)
        args.burnin = max(args.burnin, 200)
        
    return args

def main():
    """Main entry point"""
    args = parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()
