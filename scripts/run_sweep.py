#!/usr/bin/env python3
"""
Sweep multiple seeds to verify robustness
"""

import sys
import pathlib
import subprocess
import json

# Seeds to test
seeds = [913, 123, 456, 789, 2025]

results = []
for seed in seeds:
    outdir = f"out_sweep/seed_{seed}"
    pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
    
    # Run experiment
    ret = subprocess.call([
        sys.executable, "-m", "lambda3_holo.run_experiment",
        "--preset", "ixb",
        "--outdir", outdir,
        "--burnin", "200",
        "--steps", "300",
        "--seed", str(seed)
    ])
    
    # Read results
    metadata_path = pathlib.Path(outdir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            results.append({
                "seed": seed,
                "lag": meta["results"]["best_lag"],
                "spearman": meta["results"]["spearman_corr"]
            })
            print(f"Seed {seed}: lag={meta['results']['best_lag']}, ρ={meta['results']['spearman_corr']:.3f}")

# Summary
with open("out_sweep/summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSummary:")
lags = [r["lag"] for r in results]
rhos = [r["spearman"] for r in results]
print(f"  Lag range: {min(lags)} to {max(lags)}")
print(f"  Spearman range: {min(rhos):.3f} to {max(rhos):.3f}")
