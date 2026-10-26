"""Checkpointing runner for Addendum B (same grid, seeds and functions as addB_sim.py; results identical).
Usage: python3 addB_resumable.py b1|b2 <seed> <reps> <outdir>"""
import sys, os, itertools, time, warnings
import pandas as pd
from multiprocessing import Pool
from functools import partial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import addB_sim as ab
warnings.filterwarnings("ignore")

def b1_safe(args, seed):
    """As in Experiment 2 (wald_sim.py), a data set is skipped when any unpenalized fit fails."""
    try: return ab.b1_one(args, seed)
    except Exception:
        K, n, rho, imb, corr, rep = args
        return [dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref="skipped")]

if __name__ == "__main__":
    which, seed, R, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    if which == "b1":
        grid = list(itertools.product([5, 10, 20], [1000, 5000], [0.1, 0.3], [False, True], [0.0], range(R))); fn = b1_safe
    else:
        grid = list(itertools.product([10, 40], [500, 2000], [0.1, 0.3], [False, True], [0.0, 0.5], range(R))); fn = ab.b2_one
    grid.sort(key=lambda g: -g[0])
    final = os.path.join(out, f"addB_{which}.csv"); keyf = os.path.join(out, f"done_{which}.txt")
    if os.path.exists(final): print("already done", which); sys.exit()
    done = set(open(keyf).read().split()) if os.path.exists(keyf) else set()
    key = lambda g: "_".join(map(str, g))
    todo = [g for g in grid if key(g) not in done]
    t0 = time.time(); B = 4 * os.cpu_count()
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            res = pool.map(partial(fn, seed=seed), batch, chunksize=1)
            df = pd.DataFrame([r for rr in res for r in rr])
            pdir = final + ".parts"; os.makedirs(pdir, exist_ok=True)
            df.to_csv(os.path.join(pdir, f"{len(done) + i:06d}.csv"), index=False)
            with open(keyf, "a") as f: f.write("\n".join(key(g) for g in batch) + "\n")
            print(f"{which}: {len(done) + i + len(batch)}/{len(grid)} in {round(time.time() - t0)} s", flush=True)
    pdir = final + ".parts"
    pd.concat([pd.read_csv(os.path.join(pdir, f)) for f in sorted(os.listdir(pdir))]).to_csv(final, index=False); print("done", which, len(grid), round(time.time() - t0), "s")
