#!/usr/bin/env python3
"""
Run PhaseShift-IXb experiment (gate_delay=1)
Expected: lag≈6, Spearman≈0.45
"""

import sys
import pathlib
import subprocess

# Create output directory
pathlib.Path("out_ixb").mkdir(exist_ok=True, parents=True)

# Run experiment with burn-in
sys.exit(subprocess.call([
    sys.executable, "-m", "lambda3_holo.run_experiment",
    "--preset", "ixb",
    "--outdir", "out_ixb",
    "--burnin", "200",
    "--steps", "300",
    "--seed", "913"
]))
