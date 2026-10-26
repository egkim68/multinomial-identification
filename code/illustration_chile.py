"""Real-data illustration: voting intention in the 1988 Chilean plebiscite (carData::Chile; Fox and Weisberg 2019),
a nominal outcome with four categories. Shows how the identification changes the sparse table.
Usage: python3 illustration_chile.py <Chile.csv> <outdir>   (output folder created if missing)"""
import sys, os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnl import fit, predict, logloss, center, threshold

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src).dropna()
labels = ["Abstain", "No", "Undecided", "Yes"]; codes = ["A", "N", "U", "Y"]
d["log_population"] = np.log(d.population); d["log_income"] = np.log(d.income)
d["male"] = (d.sex == "M").astype(float)
for r in ["M", "N", "S", "SA"]: d[f"region_{r}"] = (d.region == r).astype(float)
for e in ["PS", "S"]: d[f"educ_{e}"] = (d.education == e).astype(float)
preds = ["statusquo", "age", "log_income", "log_population", "male", "educ_PS", "educ_S", "region_M", "region_N", "region_S", "region_SA"]
X = d[preds].values.astype(float); X = (X - X.mean(0)) / X.std(0)
y = d.vote.map({c: i for i, c in enumerate(codes)}).values.astype(int); K = 4; n = len(y)
counts = np.bincount(y, minlength=K)

# ridge with 5-fold cross-validated penalty
rng = np.random.default_rng(1988); folds = rng.permutation(n) % 5
lams = np.logspace(-1, -4, 10)
cv = []
for lam in lams:
    ll = 0
    for f in range(5):
        tr, te = folds != f, folds == f
        W, b = fit(X[tr], y[tr], K, lam, "l2"); ll += logloss(predict(X[te], W, b), y[te]) * te.sum()
    cv.append(ll / n)
lam = lams[int(np.argmin(cv))]
W, b = fit(X, y, K, lam, "l2")

# atypicality q_k (Sigma = identity on the standardized scale, pi0 from intercepts)
pi0 = np.exp(b - b.max()); pi0 /= pi0.sum()
dev = W - (W @ pi0)[:, None]
q = (dev ** 2).sum(0)

# thresholding under each identification; removal share by 5-fold CV of the thresholded fit
def cv_threshold(how, ref=None):
    best = None
    for qq in np.round(np.arange(0, 0.96, 0.05), 2):
        ll = 0
        for f in range(5):
            tr, te = folds != f, folds == f
            Wf, bf = fit(X[tr], y[tr], K, lam, "l2")
            Wt = threshold(center(Wf, how, ref), qq, keep_col=ref)
            ll += logloss(predict(X[te], Wt, bf), y[te]) * te.sum()
        if best is None or ll < best[0] - 1e-9: best = (ll, qq)
    return best[1]
rows = []; tables = {}
for how, ref, lab in [("mean", None, "Mean (ridge output)"), ("median", None, "Median"), ("shorth", None, "Shorth"),
                      ("ref", int(np.argmax(counts)), f"Reference: most frequent ({labels[int(np.argmax(counts))]})"),
                      ("ref", int(np.argmin(counts)), f"Reference: least frequent ({labels[int(np.argmin(counts))]})")]:
    qq = cv_threshold(how, ref)
    Wt = threshold(center(W, how, ref), qq, keep_col=ref)
    nz = (Wt != 0) if ref is None else np.delete(Wt != 0, ref, 1)
    per_pred = nz.sum(1)
    rows.append(dict(identification=lab, removed_share=qq, nonzero=int(nz.sum()), cells=int(nz.size),
                     **{f"nz_{p}": int(v) for p, v in zip(preds, per_pred)}))
    tables[lab] = pd.DataFrame(Wt.round(2), index=preds, columns=labels)
T = pd.DataFrame(rows)

# Wald tables from unpenalized fits under the two frequency-based references
wald = []
for rname, r in [("most frequent", int(np.argmax(counts))), ("least frequent", int(np.argmin(counts))), ("smallest q", int(np.argmin(q)))]:
    order = [r] + [k for k in range(K) if k != r]; ymap = np.empty(K, int); ymap[order] = np.arange(K)
    res = sm.MNLogit(ymap[y], sm.add_constant(X)).fit(disp=0, method="newton", maxiter=100)
    B = np.asarray(res.params)[1:]; SE = np.asarray(res.bse)[1:]
    sig = 2 * norm.sf(np.abs(B / SE)) < .05
    wald.append(dict(reference=f"{rname} ({labels[r]})", rule=rname, category=labels[r], n_cat=int(counts[r]), q=float(q[r]),
                     significant=int(sig.sum()), cells=int(sig.size),
                     pervasive=int((sig.mean(1) >= .5).sum()), median_se=float(np.median(SE)),
                     **{f"sig_{p}": int(v) for p, v in zip(preds, sig.sum(1))}))
Wd = pd.DataFrame(wald)
summary = dict(n=n, lam=float(lam), counts=dict(zip(labels, map(int, counts))), q=dict(zip(labels, map(float, q.round(3)))))
T.to_csv(os.path.join(out, "chile_threshold.csv"), index=False); Wd.to_csv(os.path.join(out, "chile_wald.csv"), index=False)
json.dump(summary, open(os.path.join(out, "chile_summary.json"), "w"), indent=1)
with open(os.path.join(out, "chile_tables.txt"), "w") as f:
    for k_, v in tables.items(): f.write(k_ + "\n" + v.to_string() + "\n\n")
pd.set_option("display.width", 200)
print(json.dumps(summary, indent=1)); print(T.to_string(index=False)); print(Wd.to_string(index=False))
for k_ in list(tables): print(k_); print(tables[k_].to_string())
