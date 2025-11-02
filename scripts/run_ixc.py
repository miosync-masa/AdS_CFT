import sys, pathlib, subprocess
pathlib.Path("out_ixc").mkdir(exist_ok=True, parents=True)
sys.exit(subprocess.call([sys.executable, "-m", "lambda3_holo.run_experiment", "--preset", "ixc", "--outdir", "out_ixc"]))