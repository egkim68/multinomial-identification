"""Reference-category simulation (one experiment, four references).
Design: K = 5, 10, 20; n = 1,000, 5,000; rho = 0.1, 0.3, 0.5; balanced or imbalanced intercepts;
predictor correlation 0 or 0.5; 100 data sets per cell. References: most frequent, least frequent,
first label, smallest estimated atypicality. Unpenalized fits (statsmodels MNLogit) and Wald tests at 0.05.
Each design cell is saved as it finishes, and a rerun skips finished cells.
Usage: python3 reference_sim.py <seed> <reps> <outdir>   (output folder created if missing)
The paper uses: python3 reference_sim.py 20261125 100 output/reference"""
import sys, os, itertools, time, glob, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from functools import partial
from scipy.stats import norm
warnings.filterwarnings("ignore")
P_DIM = 10

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def ar1(corr): return np.linalg.cholesky(corr ** np.abs(np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM))))

def one(args, seed):
    import statsmodels.api as sm
    K, n, rho, imb, corr, rep = args
    rng = np.random.default_rng([seed, K, n, int(rho * 100), int(imb), int(corr * 100), rep])
    D = np.where(rng.random((P_DIM, K)) < rho, rng.choice([-1, 1], (P_DIM, K)) * rng.uniform(0.3, 0.9, (P_DIM, K)), 0.0)
    a = rng.permutation(np.linspace(0, -3, K)) if imb else np.zeros(K)
    X = rng.normal(size=(n, P_DIM)) @ ar1(corr).T
    y = np.minimum((softmax(X @ D + a).cumsum(1) < rng.random((n, 1))).sum(1), K - 1)
    counts = np.bincount(y, minlength=K)
    if counts.min() < 5: return [dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref="skipped")]
    Xc = sm.add_constant(X)
    def mle(r):
        order = [r] + [k for k in range(K) if k != r]; ymap = np.empty(K, int); ymap[order] = np.arange(K)
        res = sm.MNLogit(ymap[y], Xc).fit(method="newton", maxiter=100, disp=0)
        return order, res
    # estimated atypicality from the most-frequent-reference fit (q is invariant to the reference)
    r0 = int(np.argmax(counts)); order0, res0 = mle(r0)
    Bfull = np.zeros((P_DIM, K)); afull = np.zeros(K)
    Bfull[:, order0[1:]] = np.asarray(res0.params)[1:, :]; afull[order0[1:]] = np.asarray(res0.params)[0, :]
    pi0 = softmax(afull[None, :])[0]; S = np.cov(X, rowvar=False)
    dv = Bfull - (Bfull @ pi0)[:, None]; qhat = np.einsum("jk,jl,lk->k", dv, S, dv)
    refs = {"freq": r0, "rare": int(np.argmin(counts)), "minq": int(np.argmin(qhat)), "first": 0}
    pervasive_true = ((D != 0).mean(1) >= 0.5)
    rows = []
    for rname, r in refs.items():
        order, res = (order0, res0) if r == r0 else mle(r)
        B = np.asarray(res.params)[1:, :]; SE = np.asarray(res.bse)[1:, :]
        sig = 2 * norm.sf(np.abs(B / SE)) < 0.05
        tnz = np.abs(D[:, order[1:]] - D[:, [r]]) > 1e-12
        rowfrac = sig.mean(1)
        rows.append(dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref=rname, ref_deviates=bool((D[:, r] != 0).any()),
                         true_density=tnz.mean(), sig_frac=sig.mean(),
                         typeI=sig[~tnz].mean() if (~tnz).any() else np.nan, power=sig[tnz].mean() if tnz.any() else np.nan,
                         median_se=np.median(SE), apparent_pervasive=int(((rowfrac >= 0.5) & ~pervasive_true).sum()),
                         converged=bool(res.mle_retvals.get("converged", True))))
    return rows

def safe_one(args, seed):
    try:
        return one(args, seed)
    except Exception as e:          # failed fit (e.g. singular information matrix): recorded and excluded
        K, n, rho, imb, corr, rep = args
        return [dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref="failed", error=type(e).__name__)]

if __name__ == "__main__":
    seed, R, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    cells_dir = os.path.join(out, "cells"); os.makedirs(cells_dir, exist_ok=True)
    cells = list(itertools.product([20, 10, 5], [5000, 1000], [0.1, 0.3, 0.5], [False, True], [0.0, 0.5]))
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for c in cells:
            f = os.path.join(cells_dir, "K%d_n%d_rho%d_imb%d_corr%d.csv" % (c[0], c[1], int(c[2] * 100), int(c[3]), int(c[4] * 100)))
            if os.path.exists(f): continue
            res = pool.map(partial(safe_one, seed=seed), [c + (r,) for r in range(R)], chunksize=1)
            pd.DataFrame([r for rr in res for r in rr]).to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)
            print("cell", c, round(time.time() - t0), "s", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(cells_dir, "*.csv")))])
    df.to_csv(os.path.join(out, "reference_results.csv"), index=False)
    print("done", len(cells), "cells", round(time.time() - t0), "s")
