"""Check the FISTA solver in mnl.py against scikit-learn (ridge, lasso) and statsmodels (unpenalized).
Usage: python3 solver_check.py"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from mnl import fit
rng = np.random.default_rng(2026)
out = {"ridge": 0, "lasso": 0, "mle": 0}
for rep in range(5):
    n, p, K = 1000, 6, 5   # odd K: the symmetric lasso solution is unique (even K can leave a flat set of row shifts)
    X = rng.normal(size=(n, p)); B = rng.normal(0, .6, (p, K)); a = rng.normal(0, .3, K)
    Z = X @ B + a; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) < rng.random((n, 1))).sum(1)
    for pen, lam in [("l2", 0.01), ("l1", 0.01)]:
        W, b = fit(X, y, K, lam, pen, max_iter=20000, tol=1e-10)
        # mnl.py minimizes mean log loss + lam*|W|_1 (l1) or 0.5*lam*|W|^2 (l2); scikit-learn's equivalent is C = 1/(n*lam)
        C = 1 / (n * lam)
        sk = LogisticRegression(penalty=pen, C=C, solver="saga" if pen == "l1" else "lbfgs", max_iter=100000, tol=1e-12).fit(X, y)
        Ws = sk.coef_.T
        out["ridge" if pen == "l2" else "lasso"] = max(out["ridge" if pen == "l2" else "lasso"], np.abs(W - Ws).max())
    # unpenalized, reference category 0
    W, b = fit(X, y, K, 0.0, "l2", ref=0, max_iter=50000, tol=1e-12)
    res = sm.MNLogit(y, sm.add_constant(X)).fit(method="newton", disp=0)
    out["mle"] = max(out["mle"], np.abs(W[:, 1:] - np.asarray(res.params)[1:, :]).max())
print({k: f"{v:.1e}" for k, v in out.items()})
