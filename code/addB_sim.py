"""Addendum B (written before running; see PROTOCOL_multinomial_v1.md).
B1 (Wald): adds a data-driven reference, the category with the smallest estimated atypicality q_k, to the Wald
   design and records Type I error, significance share, apparently pervasive predictors, and truth density.
B2 (lasso comparators): on a subset of Experiment 1 cells, compares shorth thresholding with symmetric lasso tuned
   by validation minimum, validation 1-SE rule, BIC, adaptive lasso, relaxed lasso, and lasso + shorth thresholding.
Usage: python3 addB_sim.py b1|b2 <seed> <reps> <outdir>   (output folder created if missing)"""
import sys, os, itertools, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnl import softmax, logloss, predict, fit, center, threshold
warnings.filterwarnings("ignore")
P_DIM = 10

def ar1(corr): return np.linalg.cholesky(corr ** np.abs(np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM))))

# ---------------- B1 ----------------
def b1_one(args, seed):
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

# ---------------- B2 ----------------
def fit_wl1(X, y, K, lam, Wt, W0=None, b0=None, max_iter=3000, tol=1e-7):
    """Adaptive lasso: penalty lam * sum Wt * |W|, symmetric parameterization."""
    n, p = X.shape; Y = np.eye(K)[y]
    W = np.zeros((p, K)) if W0 is None else W0.copy()
    b = np.log(np.clip(Y.mean(0), 1e-4, None)) if b0 is None else b0.copy(); b = b - b.mean()
    t = 1.0 / (0.5 * (np.linalg.norm(X, 2) ** 2 / n + 1.0)); Wy, by, tk = W.copy(), b.copy(), 1.0
    obj = lambda W, b: logloss(predict(X, W, b), y) + lam * (Wt * np.abs(W)).sum(); prev = obj(W, b)
    for it in range(max_iter):
        P = predict(X, Wy, by); G = X.T @ (P - Y) / n; gb = (P - Y).mean(0)
        Wn = Wy - t * G; bn = by - t * gb
        Wn = np.sign(Wn) * np.maximum(np.abs(Wn) - t * lam * Wt, 0)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Wy = Wn + (tk - 1) / tn * (Wn - W); by = bn + (tk - 1) / tn * (bn - b); W, b, tk = Wn, bn, tn
        if it % 25 == 24:
            cur = obj(W, b)
            if cur > prev: Wy, by, tk = W.copy(), b.copy(), 1.0
            if abs(prev - cur) < tol * max(1, abs(cur)): break
            prev = cur
    return W, b

def b2_one(args, seed):
    import multinom_sim as ms
    K, n, rho, same, corr, rep = args
    rng = np.random.default_rng([seed, K, n, int(rho * 100), int(same), int(corr * 100), rep])
    L = ar1(corr); D, B, a = ms.gen_truth(rng, K, rho, False, same)
    X, y = ms.draw(rng, B, a, n, L); V, v = ms.draw(rng, B, a, 2000, L); T, t = ms.draw(rng, B, a, 5000, L)
    mu, sd = X.mean(0), X.std(0); Xs, Vs, Ts = (X - mu) / sd, (V - mu) / sd, (T - mu) / sd
    tnz = (D * sd[:, None]) != 0; oracle = logloss(softmax(T @ B + a), t)
    rows = []
    def rec(method, W, b, extra=None):
        nz = W != 0; tp = (nz & tnz).sum(); f1 = 2 * tp / max(2 * tp + (nz & ~tnz).sum() + (~nz & tnz).sum(), 1)
        rows.append(dict(K=K, n=n, rho=rho, same=same, corr=corr, rep=rep, method=method, f1=f1, frac_nz=nz.mean(),
                         excess=logloss(predict(Ts, W, b), t) - oracle, **(extra or {})))
    vloss = lambda W, b: logloss(predict(Vs, W, b), v)
    # ridge + shorth threshold (as in Experiment 1)
    rp = []; W_ = b_ = None
    for lam in np.logspace(-1, -4.5, 9):
        W_, b_ = fit(Xs, y, K, lam, "l2", W0=W_, b0=b_); rp.append((W_.copy(), b_.copy()))
    Wr, br = min(rp, key=lambda z: vloss(*z))
    def thr_best(W, b):
        Wc = center(W, "shorth"); best = None
        for q in np.round(np.arange(0, 0.96, 0.05), 2):
            Wt_ = threshold(Wc, q); vl = vloss(Wt_, b)
            if best is None or vl < best[0] - 1e-12: best = (vl, Wt_)
        return best[1]
    rec("thr_shorth", thr_best(Wr, br), br)
    # lasso path
    Y = np.eye(K)[y]; lmax = np.abs(Xs.T @ (Y - Y.mean(0))).max() / n; lams = lmax * np.logspace(0, -3, 20)
    path = []; W_ = b_ = None
    for lam in lams:
        W_, b_ = fit(Xs, y, K, lam, "l1", W0=W_, b0=b_); path.append((lam, W_.copy(), b_.copy()))
    vl = np.array([vloss(W_, b_) for _, W_, b_ in path])
    i_min = int(np.argmin(vl)); rec("lasso_valmin", path[i_min][1], path[i_min][2])
    Pm = predict(Vs, path[i_min][1], path[i_min][2]); per = -np.log(np.clip(Pm[np.arange(len(v)), v], 1e-15, None))
    se = per.std(ddof=1) / np.sqrt(len(v)); i_1se = min(i for i in range(len(path)) if vl[i] <= vl[i_min] + se)
    rec("lasso_1se", path[i_1se][1], path[i_1se][2])
    bic = [2 * n * logloss(predict(Xs, W_, b_), y) + np.log(n) * ((W_ != 0).sum() + K - 1) for _, W_, b_ in path]
    i_b = int(np.argmin(bic)); rec("lasso_bic", path[i_b][1], path[i_b][2])
    # lasso + shorth threshold
    rec("lasso_thr_shorth", thr_best(path[i_min][1], path[i_min][2]), path[i_min][2])
    # relaxed lasso: refit on the support (tiny ridge), mix with lasso, choose (lambda, phi) on validation
    best = None
    for lam, W_, b_ in path:
        sup = (W_ != 0).astype(float)
        if sup.sum() == 0: continue
        Wf, bf = fit(Xs, y, K, 1e-4, "l2", support=sup, W0=W_, b0=b_)
        for phi in [0, .25, .5, .75, 1]:
            Wm, bm = phi * W_ + (1 - phi) * Wf, phi * b_ + (1 - phi) * bf; vv = vloss(Wm, bm)
            if best is None or vv < best[0] - 1e-12: best = (vv, Wm, bm)
    rec("lasso_relaxed", best[1], best[2])
    # adaptive lasso: weights 1/|median-centred ridge coefficient|
    Wt = 1.0 / np.maximum(np.abs(center(Wr, "median")), 1e-3)
    lmax_a = np.max(np.abs(Xs.T @ (Y - Y.mean(0))) / n / Wt); best = None; W_ = b_ = None
    for lam in lmax_a * np.logspace(0, -3, 20):
        W_, b_ = fit_wl1(Xs, y, K, lam, Wt, W0=W_, b0=b_); vv = vloss(W_, b_)
        if best is None or vv < best[0] - 1e-12: best = (vv, W_.copy(), b_.copy())
    rec("lasso_adaptive", best[1], best[2])
    return rows

if __name__ == "__main__":
    which, seed, R, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    if which == "b1":
        grid = list(itertools.product([5, 10, 20], [1000, 5000], [0.1, 0.3], [False, True], [0.0], range(R))); fn = b1_one
    else:
        grid = list(itertools.product([10, 40], [500, 2000], [0.1, 0.3], [False, True], [0.0, 0.5], range(R))); fn = b2_one
    grid.sort(key=lambda g: -g[0])
    from functools import partial
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        res = pool.map(partial(fn, seed=seed), grid, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(out, f"addB_{which}.csv"), index=False)
    print("done", which, len(grid), round(time.time() - t0), "s")
