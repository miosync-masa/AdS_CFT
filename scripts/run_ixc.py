#!/usr/bin/env python3
"""
Run PhaseShift-IXc experiment (gate_delay=2)
Expected: lag≈12, Spearman≈0.35
"""

import sys
import pathlib
import subprocess

# Create output directory
pathlib.Path("out_ixc").mkdir(exist_ok=True, parents=True)

# Run experiment with burn-in
sys.exit(subprocess.call([
    sys.executable, "-m", "lambda3_holo.run_experiment",
    "--preset", "ixc",
    "--outdir", "out_ixc",
    "--burnin", "200",
    "--steps", "300",
    "--seed", "913"
]))
