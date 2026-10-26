"""Addendum C (written before running; see PROTOCOL_multinomial_v1.md). Descriptive only.
C1: Experiment 1 under continuous, heavy-tailed deviations that include small ones. Each category deviates with
    probability rho by |0.2 * t_3| (median about 0.15, a quarter below about 0.07), with mixed or one-sided signs.
    Uses the frozen per-data-set function of multinom_sim.py with a replaced truth generator.
C2: the Wald design of Addendum B1 (four references, including the smallest estimated atypicality) with one-sided
    deviations, the condition B1 did not cover.
Both parts save results in batches and resume after an interruption.
Usage: SIM_SEED=<int> python3 addC_sim.py c1|c2 <reps> <outdir>   (output folder created if missing)"""
import sys, os, itertools, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import multinom_sim as ms
import addB_sim as ab

def gen_truth_c1(rng, K, rho, imb, same):
    P = ms.P_DIM
    signs = (np.where(rng.random((P, 1)) < .5, -1, 1) * np.ones((P, K))) if same else rng.choice([-1, 1], (P, K))
    mag = np.abs(0.2 * rng.standard_t(3, (P, K)))
    D = np.where(rng.random((P, K)) < rho, signs * mag, 0.0)
    base = rng.normal(0, 0.5, (P, 1))
    a = rng.normal(0, 0.2, K)
    return D, base + D, a

def c1_one(args):
    ms.gen_truth = gen_truth_c1
    return ms.one(args)

def c2_one(args, seed):
    """Copy of addB_sim.b1_one with one-sided deviations (same sign within a predictor)."""
    import statsmodels.api as sm
    from scipy.stats import norm
    from mnl import softmax
    K, n, rho, imb, corr, rep = args
    P_DIM = ab.P_DIM
    rng = np.random.default_rng([seed, K, n, int(rho * 100), int(imb), int(corr * 100), rep])
    sgn = np.where(rng.random((P_DIM, 1)) < .5, -1, 1) * np.ones((P_DIM, K))
    D = np.where(rng.random((P_DIM, K)) < rho, sgn * rng.uniform(0.3, 0.9, (P_DIM, K)), 0.0)
    a = rng.permutation(np.linspace(0, -3, K)) if imb else np.zeros(K)
    X = rng.normal(size=(n, P_DIM)) @ ab.ar1(corr).T
    y = np.minimum((softmax(X @ D + a).cumsum(1) < rng.random((n, 1))).sum(1), K - 1)
    counts = np.bincount(y, minlength=K)
    skip = [dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref="skipped")]
    if counts.min() < 5: return skip
    Xc = sm.add_constant(X)
    def mle(r):
        order = [r] + [k for k in range(K) if k != r]; ymap = np.empty(K, int); ymap[order] = np.arange(K)
        return order, sm.MNLogit(ymap[y], Xc).fit(method="newton", maxiter=100, disp=0)
    try:
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
    except Exception:
        return skip

def _c2(args): return c2_one(args, int(os.environ.get("SIM_SEED", "9921")))

if __name__ == "__main__":
    which, R, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    if which == "c1":
        grid = list(itertools.product([10, 40], [500, 2000], [0.1, 0.3], [False], [False, True], [0.0], range(R))); fn = c1_one
    else:
        grid = list(itertools.product([5, 10, 20], [1000, 5000], [0.1, 0.3], [False, True], [0.0], range(R))); fn = _c2
    grid.sort(key=lambda g: -g[0])
    final = os.path.join(out, f"addC_{which}.csv"); keyf = os.path.join(out, f"done_{which}.txt"); pdir = final + ".parts"
    if os.path.exists(final): print("already done", which); sys.exit()
    done = set(open(keyf).read().split()) if os.path.exists(keyf) else set()
    key = lambda g: "_".join(map(str, g))
    todo = [g for g in grid if key(g) not in done]
    t0 = time.time(); Bt = 4 * os.cpu_count(); os.makedirs(pdir, exist_ok=True)
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(todo), Bt):
            batch = todo[i:i + Bt]
            res = pool.map(fn, batch, chunksize=1)
            pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(pdir, f"{len(done) + i:06d}.csv"), index=False)
            with open(keyf, "a") as f: f.write("\n".join(key(g) for g in batch) + "\n")
            print(f"{which}: {len(done) + i + len(batch)}/{len(grid)} in {round(time.time() - t0)} s", flush=True)
    pd.concat([pd.read_csv(os.path.join(pdir, f)) for f in sorted(os.listdir(pdir))]).to_csv(final, index=False)
    print("done", which, len(grid), round(time.time() - t0), "s")
