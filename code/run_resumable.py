"""Resumable runner for the frozen simulation scripts (their code is not changed).
Runs the same grid and the same per-dataset function, writes results in batches, and on restart
skips the datasets already saved. Usage:
  python3 run_resumable.py main  <seed> <reps> <outdir>
  python3 run_resumable.py wald  <seed> <reps> <outdir>"""
import sys, os, time, importlib
import pandas as pd
from multiprocessing import Pool

which, seed, reps, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.environ["SIM_SEED"] = seed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(outdir, exist_ok=True)
if which == "main":
    mod = importlib.import_module("multinom_sim"); grid = mod.grid("full", reps)
    keycols = ["K", "n", "rho", "imb", "same", "corr", "rep"]; fname = "multinom_results.csv"
elif which == "sens":
    import itertools
    mod = importlib.import_module("sensitivity_sim")
    grid = [(t,) + g for t in ["two_cluster", "dense"]
            for g in itertools.product([10, 40], [500, 2000], [0.0], [False], [False], [0.0], range(reps))]
    keycols = None; fname = "sensitivity_results.csv"
else:
    import itertools
    mod = importlib.import_module("wald_sim")
    grid = list(itertools.product([5, 10, 20], [1000, 5000], [0.1, 0.3, 0.5], [False, True], [0.0, 0.5], range(reps)))
    keycols = ["K", "n", "rho", "imb", "corr", "rep"]; fname = "wald_results.csv"
path = os.path.join(outdir, fname); donef = os.path.join(outdir, "done_keys.txt")
done = set(open(donef).read().split()) if os.path.exists(donef) else set()
key = lambda g: "|".join(str(x) for x in g)
todo = [g for g in grid if key(g) not in done]
print(f"{which}: {len(grid)} datasets, {len(done)} done, {len(todo)} to run", flush=True)
B = 8 * os.cpu_count(); t0 = time.time()
with Pool(os.cpu_count()) as pool:
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        res = pool.map(mod.one, batch, chunksize=1)
        rows = [r for rr in res for r in rr]
        if rows:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        with open(donef, "a") as f:
            f.write("\n".join(key(g) for g in batch) + "\n")
        print(f"{which}: {i + len(batch)}/{len(todo)} in {round(time.time() - t0)} s", flush=True)
print("done", which, flush=True)
