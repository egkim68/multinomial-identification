"""Second survey example (Supplementary Material S2): party identification in the 1996 American National
Election Study, as distributed with statsmodels. Analyzed as in chile.py.
Usage: python3 anes.py <outdir>   (output folder created if missing)"""
import sys, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
d = sm.datasets.anes96.load_pandas().data
labels = ["Strong Democrat", "Weak Democrat", "Independent Democrat", "Independent", "Independent Republican",
          "Weak Republican", "Strong Republican"]
preds = ["logpopul", "TVnews", "selfLR", "ClinLR", "DoleLR", "age", "educ", "income"]
Xraw = d[preds].values.astype(float); X = (Xraw - Xraw.mean(0)) / Xraw.std(0)
y = d.PID.values.astype(int); K = len(labels); counts = np.bincount(y, minlength=K); Xc = sm.add_constant(X)

def mle(r):
    order = [r] + [k for k in range(K) if k != r]; ymap = np.empty(K, int); ymap[order] = np.arange(K)
    return order, sm.MNLogit(ymap[y], Xc).fit(method="newton", maxiter=100, disp=0)

# atypicality from the unpenalized fit (does not depend on the reference)
order0, res0 = mle(int(np.argmax(counts)))
B = np.zeros((len(preds), K)); a = np.zeros(K)
B[:, order0[1:]] = np.asarray(res0.params)[1:, :]; a[order0[1:]] = np.asarray(res0.params)[0, :]
pi0 = np.exp(a - a.max()); pi0 /= pi0.sum()
dv = B - (B @ pi0)[:, None]; q = np.einsum("jk,jl,lk->k", dv, np.cov(X, rowvar=False), dv)

rules = {}
for r, lab in [(int(np.argmin(counts)), "Least frequent"), (int(np.argmin(q)), "Smallest atypicality"), (int(np.argmax(counts)), "Most frequent")]:
    rules[r] = lab if r not in rules else rules[r] + " / " + lab
rows = []
for r in sorted(range(K), key=lambda k: -counts[k]):
    order, res = mle(r)
    SE = np.asarray(res.bse)[1:]; sig = 2 * norm.sf(np.abs(np.asarray(res.params)[1:] / SE)) < 0.05
    rows.append(dict(rule=rules.get(r, "No rule"), reference=labels[r], respondents=int(counts[r]),
                     observed_share=round(float(counts[r] / len(y)), 3), baseline_share=round(float(pi0[r]), 3), atypicality=round(float(q[r]), 3),
                     significant=int(sig.sum()), contrasts=int(sig.size), pervasive=int((sig.mean(1) >= 0.5).sum()),
                     median_se=round(float(np.median(SE)), 3)))
W = pd.DataFrame(rows); W.to_csv(os.path.join(out, "anes_wald.csv"), index=False)
print(W.to_string(index=False))
