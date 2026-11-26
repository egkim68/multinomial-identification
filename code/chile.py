"""Survey illustration: voting intention before the 1988 Chilean plebiscite (carData::Chile; Fox et al. 2026).
Fits the unpenalized multinomial logit under each reference, estimates the atypicality and baseline share (probability at the predictor means)
of every category (as in the simulation), and records the Wald table under each reference.
Usage: python3 chile.py <Chile.csv> <outdir>   (output folder created if missing)
Export the data in R with write.csv(carData::Chile, "Chile.csv")."""
import sys, os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src).dropna()
labels = ["Abstain", "No", "Undecided", "Yes"]; codes = ["A", "N", "U", "Y"]
d["log_population"] = np.log(d.population); d["log_income"] = np.log(d.income)
d["male"] = (d.sex == "M").astype(float)
for r in ["M", "N", "S", "SA"]: d[f"region_{r}"] = (d.region == r).astype(float)
for e in ["PS", "S"]: d[f"educ_{e}"] = (d.education == e).astype(float)
preds = ["statusquo", "age", "log_income", "log_population", "male", "educ_PS", "educ_S",
         "region_M", "region_N", "region_S", "region_SA"]
Xraw = d[preds].values.astype(float); sd = Xraw.std(0); X = (Xraw - Xraw.mean(0)) / sd
y = d.vote.map({c: i for i, c in enumerate(codes)}).values.astype(int); K = 4; n = len(y)
counts = np.bincount(y, minlength=K)
Xc = sm.add_constant(X)

def mle(r):
    order = [r] + [k for k in range(K) if k != r]; ymap = np.empty(K, int); ymap[order] = np.arange(K)
    res = sm.MNLogit(ymap[y], Xc).fit(method="newton", maxiter=100, disp=0)
    return order, res

# atypicality from the unpenalized fit (does not depend on the reference)
order0, res0 = mle(int(np.argmax(counts)))
B = np.zeros((len(preds), K)); a = np.zeros(K)
B[:, order0[1:]] = np.asarray(res0.params)[1:, :]; a[order0[1:]] = np.asarray(res0.params)[0, :]
pi0 = np.exp(a - a.max()); pi0 /= pi0.sum()
S = np.cov(X, rowvar=False)
dv = B - (B @ pi0)[:, None]; q = np.einsum("jk,jl,lk->k", dv, S, dv)

rows, sexrows = [], []
rules = {int(np.argmax(counts)): "Most frequent", int(np.argmin(counts)): "Least frequent", int(np.argmin(q)): "Smallest atypicality"}
for r in sorted(range(K), key=lambda k: -counts[k]):
    rule = rules.get(r, "No rule")
    order, res = mle(r)
    Bm = np.asarray(res.params)[1:]; SE = np.asarray(res.bse)[1:]; pv = 2 * norm.sf(np.abs(Bm / SE))
    sig = pv < 0.05
    rows.append(dict(rule=rule, reference=labels[r], respondents=int(counts[r]), observed_share=float(counts[r] / n),
                     baseline_share=float(pi0[r]), atypicality=float(q[r]),
                     significant=int(sig.sum()), cells=int(sig.size), pervasive=int((sig.mean(1) >= 2 / 3 - 1e-9).sum()),
                     median_se=float(np.median(SE)), **{f"sig_{p}": int(v) for p, v in zip(preds, sig.sum(1))}))
    j = preds.index("male")
    for c_i, k in enumerate(order[1:]):
        # coefficients on the men-versus-women scale (the standardized indicator divided by its SD)
        sexrows.append(dict(reference=labels[r], contrast=f"{labels[k]} vs {labels[r]}", coef=float(Bm[j, c_i] / sd[j]),
                            se=float(SE[j, c_i] / sd[j]), p=float(pv[j, c_i])))
W = pd.DataFrame(rows); Sx = pd.DataFrame(sexrows)
summary = dict(n=n, counts=dict(zip(labels, map(int, counts))), atypicality=dict(zip(labels, map(float, q.round(3)))),
               baseline_share=dict(zip(labels, map(float, pi0.round(3)))),
               male_raw_coef_by_category=dict(zip(labels, map(float, (B[preds.index('male')] - B[preds.index('male')].mean()).round(3)))))
W.to_csv(os.path.join(out, "chile_wald.csv"), index=False); Sx.to_csv(os.path.join(out, "chile_sex.csv"), index=False)
json.dump(summary, open(os.path.join(out, "chile_summary.json"), "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(summary, indent=1)); print(W.to_string(index=False)); print(Sx.round(3).to_string(index=False))
