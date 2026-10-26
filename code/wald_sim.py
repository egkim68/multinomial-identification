"""Study 3 (wald_sim.py; generalized from exploratory explore3_wald.py): unpenalized reference-coded MLE (as in Stata/SPSS/R nnet) and Wald tables.
How does the reference choice (most frequent = Stata default, rarest, arbitrary first) change what the
coefficient table appears to say about which predictors matter for which categories?"""
import sys, os, itertools, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import norm
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnl import softmax
warnings.filterwarnings("ignore")
P_DIM = 10

def gen(rng, K, rho, imb):
    D = np.where(rng.random((P_DIM, K)) < rho,
                 rng.choice([-1, 1], (P_DIM, K)) * rng.uniform(0.3, 0.9, (P_DIM, K)), 0.0)
    a = rng.permutation(np.linspace(0, -3, K)) if imb else np.zeros(K)
    return D, a

SEED = int(os.environ.get('SIM_SEED', '3'))

def one(args):
    K, n, rho, imb, corr, rep = args
    rng = np.random.default_rng([SEED, K, n, int(rho * 100), int(imb), int(corr * 100), rep])
    D, a = gen(rng, K, rho, imb)
    Lc = np.linalg.cholesky(corr ** np.abs(np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM))))
    X = rng.normal(size=(n, P_DIM)) @ Lc.T; P = softmax(X @ D + a)
    y = np.minimum((P.cumsum(1) < rng.random((n, 1))).sum(1), K - 1)
    counts = np.bincount(y, minlength=K)
    if counts.min() < 5: return []
    refs = {"freq": int(np.argmax(counts)), "rare": int(np.argmin(counts)), "first": 0}
    Xc = sm.add_constant(X)
    rows = []
    pervasive_true = ((D != 0).mean(1) >= 0.5)            # predictor truly affects >= half of categories
    for rname, r in refs.items():
        # relabel so that r becomes category 0 (statsmodels uses the lowest code as base)
        order = [r] + [k for k in range(K) if k != r]
        ymap = np.empty(K, int); ymap[order] = np.arange(K)
        try:
            res = sm.MNLogit(ymap[y], Xc).fit(method="newton", maxiter=100, disp=0)
        except Exception:
            return []
        B = np.asarray(res.params)[1:, :]; SE = np.asarray(res.bse)[1:, :]
        pv = 2 * norm.sf(np.abs(B / SE))
        sig = pv < 0.05
        Dr = D[:, order[1:]] - D[:, [r]]                     # true contrasts in this coding
        tnz = np.abs(Dr) > 1e-12
        rowfrac = sig.mean(1)
        rows.append(dict(K=K, n=n, rho=rho, imb=imb, corr=corr, rep=rep, ref=rname,
                         ref_deviates=bool((D[:, r] != 0).any()), ref_count=int(counts[r]),
                         true_density=tnz.mean(), sig_frac=sig.mean(),
                         power=sig[tnz].mean() if tnz.any() else np.nan,
                         typeI=sig[~tnz].mean() if (~tnz).any() else np.nan,
                         median_se=np.median(SE),
                         pervasive_flagged=int((rowfrac >= 0.5).sum()),
                         pervasive_spurious=int(((rowfrac >= 0.5) & ~pervasive_true).sum()),
                         converged=bool(res.mle_retvals.get("converged", True))))
    return rows

if __name__ == "__main__":
    R = int(os.environ.get("SIM_REPS", "50"))
    outdir = os.environ.get("SIM_OUT", "output/wald"); os.makedirs(outdir, exist_ok=True)
    grid = list(itertools.product([5, 10, 20], [1000, 5000], [0.1, 0.3, 0.5], [False, True], [0.0, 0.5], range(R)))
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        res = pool.map(one, grid, chunksize=4)
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(os.path.join(outdir, "wald_results.csv"), index=False)
    print("done", len(df), round(time.time() - t0), "s")
