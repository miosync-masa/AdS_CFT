import argparse, os
import numpy as np
import pandas as pd
from .automaton import Automaton
from .plotters import plot_rt_timeseries, plot_crosscorr, plot_transfer_entropy
from .metrics import spearman_corr

def run_once(args):
    os.makedirs(args.outdir, exist_ok=True)
    ua = Automaton(
        H=args.H, W=args.W, Z=args.Z, L_ads=args.L_ads, alpha=args.alpha,
        c0=args.c0, gamma=args.gamma, c_eff_max=args.c_eff_max,
        gate_delay=args.gate_delay, gate_strength=args.gate_strength,
        soc_rate=args.soc_rate, seed=args.seed
    )
    ua.boundary = ua.coop_field()
    rows=[]; prev=None
    for t in range(1, args.steps+1):
        rec = ua.step_once(t, weights=(args.w_len, args.w_hole, args.w_curv))
        rec["dS_RT_mo"] = 0.0 if prev is None else rec["entropy_RT_mo"] - prev
        prev = rec["entropy_RT_mo"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    f1 = os.path.join(args.outdir, "fig_rt_timeseries.png")
    plot_rt_timeseries(df, f1)

    f2 = os.path.join(args.outdir, "fig_xcorr.png")
    best_lag, best_corr = plot_crosscorr(df, "lambda_p99_A_out_pre", "entropy_RT_mo", f2, maxlag=args.maxlag)

    def shift_for_lag(a, b, lag):
        if lag == 0: return a, b
        elif lag > 0: return a[lag:], b[:-lag]
        else: return a[:lag], b[-lag:]
    S = df["entropy_RT_mo"].values
    lam = df["lambda_p99_A_out_pre"].values
    if np.isnan(lam).any():
        idx = np.arange(len(lam)); m = ~np.isnan(lam)
        if m.sum()>=2: lam = np.interp(idx, idx[m], lam[m])
        else: lam = np.nan_to_num(lam, nan=np.nanmean(lam) if np.isfinite(np.nanmean(lam)) else 0.0)
    S_lag, lam_lag = shift_for_lag(S, lam, best_lag)
    rho_s = spearman_corr(S_lag, lam_lag) if len(S_lag)>3 else float("nan")

    f3 = os.path.join(args.outdir, "fig_te.png")
    TE_x_to_y, TE_y_to_x = plot_transfer_entropy(df, "lambda_p99_A_out_pre", "entropy_RT_mo", f3, n_bins=3)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"Best Pearson cross-corr (S vs λ_pre): {best_corr:.3f} at lag {best_lag}\n")
        f.write(f"Spearman at best lag: {rho_s:.3f}\n")
        f.write(f"Transfer Entropy (λ_pre→S): {TE_x_to_y:.4f} nats\n")
        f.write(f"Transfer Entropy (S→λ_pre): {TE_y_to_x:.4f} nats\n")

    print(f"[DONE] outdir={args.outdir}")
    print(f"  best Pearson corr={best_corr:.3f} at lag={best_lag}")
    print(f"  Spearman@bestlag={rho_s:.3f}")
    print(f"  TE λ→S={TE_x_to_y:.4f}, S→λ={TE_y_to_x:.4f}")

def parse_args():
    p = argparse.ArgumentParser(description="Lambda3 Holography — PhaseShift runner")
    p.add_argument("--preset", type=str, default="ixb", choices=["ixb","ixc","custom"],
                   help="ixb: delay=1, ixc: delay=2, custom: use args")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--steps", type=int, default=280)
    p.add_argument("--H", type=int, default=44)
    p.add_argument("--W", type=int, default=44)
    p.add_argument("--Z", type=int, default=24)
    p.add_argument("--L_ads", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--c0", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--c_eff_max", type=float, default=0.18)
    p.add_argument("--gate_delay", type=int, default=1)
    p.add_argument("--gate_strength", type=float, default=0.15)
    p.add_argument("--soc_rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--w_len", type=float, default=1.0)
    p.add_argument("--w_hole", type=float, default=2.0)
    p.add_argument("--w_curv", type=float, default=0.5)
    p.add_argument("--maxlag", type=int, default=16)
    args = p.parse_args()

    if args.preset == "ixb":
        args.gate_delay = 1
        args.steps = max(args.steps, 280)
    elif args.preset == "ixc":
        args.gate_delay = 2
        args.steps = max(args.steps, 300)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_once(args)
