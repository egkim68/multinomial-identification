"""Numerical checks for the theory note.
Check 1: second-order frequency formula  f_k ~= pi0_k [1 + (d_k' S d_k - Dbar)/2]  vs exact E[softmax] (large MC).
Check 2: truth density of the reference-coded contrast matrix: formula vs direct count; freq / rare / random reference.
Check 3: centring bias of mean, median, shorth for one row with one-sided deviations, as n grows."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnl import softmax, shorth_center

rng = np.random.default_rng(20261016)
out = []

# ---- Check 1 ----
for K, rho, scale, corr in [(5, .2, 1, 0), (10, .1, 1, 0), (20, .1, 1, 0), (40, .1, 1, 0), (40, .3, 1, 0),
                            (20, .1, 1, .5), (20, .1, .5, 0), (20, .1, 1.5, 0)]:
    p = 10
    S = corr ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Lc = np.linalg.cholesky(S)
    errs, cors, top_dev, bot_dev = [], [], [], []
    for r in range(40):
        D = np.where(rng.random((p, K)) < rho, rng.choice([-1, 1], (p, K)) * rng.uniform(.3, .9, (p, K)), 0) * scale
        a = np.zeros(K)
        X = rng.normal(size=(200000, p)) @ Lc.T
        f = softmax(X @ D + a).mean(0)
        pi0 = softmax(a[None, :])[0]
        bbar = D @ pi0
        dk = D - bbar[:, None]
        q = np.einsum("jk,jl,lk->k", dk, S, dk)
        Dbar = pi0 @ q
        approx = pi0 * (1 + 0.5 * (q - Dbar))
        errs.append(np.abs(np.log(f) - np.log(approx)).max())
        cors.append(np.corrcoef(np.log(f), q)[0, 1])
        top_dev.append((D[:, np.argmax(f)] != 0).any()); bot_dev.append((D[:, np.argmin(f)] != 0).any())
    out.append(dict(check="1_frequency", K=K, rho=rho, scale=scale, corr=corr,
                    max_abs_log_error=np.mean(errs), corr_logf_q=np.mean(cors),
                    p_most_frequent_deviates=np.mean(top_dev), p_rarest_deviates=np.mean(bot_dev)))

# ---- Check 2 ----
def density_formula(D, r):
    p, K = D.shape
    s = (D != 0).sum(1)
    dev_r = D[:, r] != 0
    nz = np.where(dev_r, K - 1, s)        # continuous deviations: no ties among deviating entries
    return nz.sum() / (p * (K - 1))
for K, rho in [(10, .1), (40, .1), (40, .3)]:
    rows = []
    for r_ in range(200):
        D = np.where(rng.random((10, K)) < rho, rng.choice([-1, 1], (10, K)) * rng.uniform(.3, .9, (10, K)), 0)
        X = rng.normal(size=(50000, 10)); f = softmax(X @ D).mean(0)
        direct = lambda r: (np.abs(np.delete(D - D[:, [r]], r, 1)) > 0).mean()
        assert all(abs(direct(r) - density_formula(D, r)) < 1e-12 for r in range(K))
        rows.append((density_formula(D, int(np.argmax(f))), density_formula(D, int(np.argmin(f))),
                     np.mean([density_formula(D, r) for r in range(K)])))
    m = np.mean(rows, 0)
    out.append(dict(check="2_density", K=K, rho=rho, density_freq=m[0], density_rare=m[1], density_random=m[2],
                    random_formula_rho2mrho=rho * (2 - rho)))

# ---- Check 3 ----
K = 40
for rho in [.1, .3]:
    for sigma in [.2, .1, .05, .025]:          # estimation noise, shrinking like n^-1/2
        bias = {"mean": [], "median": [], "shorth": []}
        for r_ in range(4000):
            dev = (rng.random(K) < rho) * rng.uniform(.3, .9, K)       # one-sided deviations, mode = 0
            b = dev + rng.normal(0, sigma, K)
            bias["mean"].append(b.mean()); bias["median"].append(np.median(b)); bias["shorth"].append(shorth_center(b))
        out.append(dict(check="3_centre_bias", K=K, rho=rho, sigma=sigma,
                        **{f"bias_{k}": np.mean(v) for k, v in bias.items()},
                        median_theory=sigma * __import__("scipy.stats").stats.norm.ppf(0.5 / (1 - rho)),
                        mean_theory=rho * 0.6))

df = pd.DataFrame(out)
os.makedirs("output", exist_ok=True)
df.to_csv("output/theory_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for c, g in df.groupby("check"):
    print(g.dropna(axis=1, how="all").round(4).to_string(index=False))
