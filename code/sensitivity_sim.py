"""Addendum A (sensitivity): Experiment 1 under truths that are not mode-sparse.
Reuses the frozen per-dataset function of multinom_sim.py with a replaced truth generator.
truth = "two_cluster": per predictor, 60% of categories share slope c1 and 40% share c2 (|c2 - c1| = 0.6);
                       scored against the representation centred at the larger cluster.
truth = "dense":       every category has its own slope N(0, 0.5^2); there is no shared value (no true zeros).
Usage: SIM_SEED=<int> python3 sensitivity_sim.py <reps> <outdir>   (output folder created if missing)"""
import sys, os, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multinom_sim as ms

TRUTH = {"t": None}

def gen_truth(rng, K, rho, imb, same):
    P = ms.P_DIM
    base = rng.normal(0, 0.5, (P, 1))
    if TRUTH["t"] == "two_cluster":
        second = np.zeros((P, K), bool)
        for j in range(P):
            second[j, rng.choice(K, int(round(0.4 * K)), replace=False)] = True
        D = np.where(second, rng.choice([-1, 1], (P, 1)) * 0.6, 0.0)
    else:
        D = rng.normal(0, 0.5, (P, K)); D = D - np.median(D, 1, keepdims=True)
    a = rng.normal(0, 0.2, K)
    return D, base + D, a

def one(args):
    truth = args[0]; TRUTH["t"] = truth
    ms.gen_truth = gen_truth
    rows = ms.one(args[1:])
    for r in rows: r["truth"] = truth
    return rows

if __name__ == "__main__":
    R = int(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    grid = [(t,) + g for t in ["two_cluster", "dense"]
            for g in itertools.product([10, 40], [500, 2000], [0.0], [False], [False], [0.0], range(R))]
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        res = pool.map(one, grid, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(out, "sensitivity_results.csv"), index=False)
    print("done", len(grid), round(time.time() - t0), "s")
