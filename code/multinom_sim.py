"""Studies 1-2: identification shift before post-estimation thresholding; reference-coded lasso.
Usage: SIM_MODE=pilot|full SIM_SEED=<int> SIM_REPS=<int> SIM_OUT=<dir> python3 multinom_sim.py
The output folder is created if it does not exist."""
import sys, os, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnl import softmax, logloss, predict, fit, center, threshold

P_DIM = 10
QGRID = np.round(np.arange(0, 0.96, 0.05), 2)
SEED = int(os.environ.get("SIM_SEED", "5501"))

def gen_truth(rng, K, rho, imb, same):
    """Mode-sparse truth: common (unidentified) effect plus deviations for a rho share of categories."""
    signs = (np.where(rng.random((P_DIM, 1)) < .5, -1, 1) * np.ones((P_DIM, K))) if same \
        else rng.choice([-1, 1], (P_DIM, K))
    D = np.where(rng.random((P_DIM, K)) < rho, signs * rng.uniform(0.3, 0.9, (P_DIM, K)), 0.0)
    base = rng.normal(0, 0.5, (P_DIM, 1))
    a = rng.permutation(np.linspace(0, -3, K)) if imb else rng.normal(0, 0.2, K)
    return D, base + D, a

def draw(rng, B, a, m, L):
    X = rng.normal(size=(m, P_DIM)) @ L.T
    P = softmax(X @ B + a)
    y = np.minimum((P.cumsum(1) < rng.random((m, 1))).sum(1), B.shape[1] - 1)
    return X, y

def f1(est, tru):
    tp = (est & tru).sum(); fp = (est & ~tru).sum(); fn = (~est & tru).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

def path(X, y, K, pen, lams, ref=None):
    out, W, b = [], None, None
    for lam in lams:
        W, b = fit(X, y, K, lam, pen, ref=ref, W0=W, b0=b)
        out.append((lam, W.copy(), b.copy()))
    return out

def one(args):
    K, n, rho, imb, same, corr, rep = args
    rng = np.random.default_rng([SEED, K, n, int(rho * 100), int(imb), int(same), int(corr * 100), rep])
    S = corr ** np.abs(np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM)))
    L = np.linalg.cholesky(S)
    D, B, a = gen_truth(rng, K, rho, imb, same)
    X, y = draw(rng, B, a, n, L); V, v = draw(rng, B, a, 2000, L); T, t = draw(rng, B, a, 5000, L)
    mu, sd = X.mean(0), X.std(0)
    Xs, Vs, Ts = (X - mu) / sd, (V - mu) / sd, (T - mu) / sd
    Dstd = D * sd[:, None]
    true_nz = Dstd != 0
    counts = np.bincount(y, minlength=K)
    refs = {"first": 0, "freq": int(np.argmax(counts)), "rare": int(np.argmin(counts))}
    oracle = logloss(softmax(T @ B + a), t)
    rows = []
    def rec(method, W, b, ref=None, q=np.nan):
        Pt = predict(Ts, W, b)
        if ref is None:
            nz, tnz = W != 0, true_nz
        else:
            nz = np.delete(W != 0, ref, 1)
            tnz = np.delete(np.abs(Dstd - Dstd[:, [ref]]) > 1e-12, ref, 1)
        rows.append(dict(K=K, n=n, rho=rho, imb=imb, same=same, corr=corr, rep=rep, method=method,
                         test_loss=logloss(Pt, t), excess=logloss(Pt, t) - oracle,
                         frac_nz=nz.mean(), true_frac_nz=tnz.mean(), f1=f1(nz, tnz),
                         ref_count=int(counts[ref]) if ref is not None else -1,
                         ref_deviates=bool((D[:, ref] != 0).any()) if ref is not None else None, q=q))
    # ridge, validation-tuned (symmetric; the penalty centres each row at its mean)
    rp = path(Xs, y, K, "l2", np.logspace(-1, -4.5, 9))
    _, Wr, br = min(rp, key=lambda z: logloss(predict(Vs, z[1], z[2]), v))
    rec("ridge", Wr, br)
    for how, rname in [("mean", None), ("median", None), ("shorth", None),
                       ("ref", "first"), ("ref", "freq"), ("ref", "rare")]:
        ref = refs[rname] if rname else None
        Wc = center(Wr, how, ref)
        best = None
        for q in QGRID:
            Wt = threshold(Wc, q, keep_col=ref)
            vl = logloss(predict(Vs, Wt, br), v)
            if best is None or vl < best[0] - 1e-12:
                best = (vl, q, Wt)
        rec(f"thr_{how}" if rname is None else f"thr_ref_{rname}", best[2], br, ref=ref, q=best[1])
    Y = np.eye(K)[y]
    lmax = np.abs(Xs.T @ (Y - Y.mean(0))).max() / n
    lams = lmax * np.logspace(0, -3, 12)
    for rname in [None, "first", "freq", "rare"]:
        ref = refs[rname] if rname else None
        lp = path(Xs, y, K, "l1", lams, ref=ref)
        _, Wl, bl = min(lp, key=lambda z: logloss(predict(Vs, z[1], z[2]), v))
        rec("lasso_sym" if rname is None else f"lasso_ref_{rname}", Wl, bl, ref=ref)
    return rows

def grid(mode, R):
    if mode == "pilot":   # new conditions not covered by the exploratory runs
        g = [c for c in itertools.product([10, 40], [500, 2000], [0.1, 0.3, 0.5], [False], [False, True], [0.0, 0.5], range(R))
             if c[2] == 0.5 or c[5] == 0.5]
    else:                 # full factorial
        g = list(itertools.product([5, 10, 20, 40], [500, 2000], [0.1, 0.3, 0.5], [False, True], [False, True], [0.0, 0.5], range(R)))
    return sorted(g, key=lambda c: -c[0])     # long jobs first

if __name__ == "__main__":
    mode = os.environ.get("SIM_MODE", "pilot"); R = int(os.environ.get("SIM_REPS", "8"))
    outdir = os.environ.get("SIM_OUT", f"output/{mode}"); os.makedirs(outdir, exist_ok=True)
    g = grid(mode, R); t0 = time.time()
    with Pool(int(os.environ.get("SIM_CORES", os.cpu_count()))) as pool:
        res = pool.map(one, g, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(outdir, "multinom_results.csv"), index=False)
    print("done", mode, len(g), "datasets", round(time.time() - t0), "s")
