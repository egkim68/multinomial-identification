"""Tables and hypothesis tests for the reference-category simulation.
Usage: python3 analysis.py <reference_results.csv> <outdir>   (output folder created if missing)"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
raw = pd.read_csv(src)
key = ["K", "n", "rho", "imb", "corr", "rep"]
skipped = raw[raw.ref == "skipped"]
failed = raw[raw.ref == "failed"]
df = raw[~raw.ref.isin(["skipped", "failed"])].copy()
df["imb"] = df["imb"].astype(bool)
REFS = ["freq", "first", "rare", "minq"]
LAB = {"freq": "Most frequent", "first": "First label", "rare": "Least frequent", "minq": "Smallest atypicality"}
M = ["ref_deviates", "true_density", "sig_frac", "power", "typeI", "median_se", "apparent_pervasive"]
rng = np.random.default_rng(20261125)

# ---- counts ----
counts = dict(total=int(len(raw.drop_duplicates(key))), skipped=int(len(skipped)), failed=int(len(failed)),
              failed_by_cell={str(k): int(v) for k, v in failed.groupby(["K", "n", "rho", "imb"]).size().items()},
              retained=int(len(df.drop_duplicates(key))),
              skipped_by_cell=skipped.groupby(["K", "n", "imb"]).size().astype(int).to_dict(),
              nonconverged=int((~df.converged.astype(bool)).sum()))
counts["skipped_by_cell"] = {str(k): v for k, v in counts["skipped_by_cell"].items()}

# ---- main table: by intercepts x reference, rho <= 0.3 ----
def summ(d, by):
    return d.groupby(by)[M].mean().reset_index()
main = df[df.rho <= 0.3]
T_main = summ(main, ["imb", "ref"])
T_main["ref"] = pd.Categorical(T_main.ref, REFS); T_main = T_main.sort_values(["imb", "ref"])
T_main.to_csv(os.path.join(out, "table_main.csv"), index=False)

# by K (for figure / text), by rho, by n, by corr
for name, by in [("byK", ["imb", "K", "ref"]), ("byrho", ["imb", "rho", "ref"]), ("byn", ["imb", "n", "ref"]),
                 ("bycorr", ["imb", "corr", "ref"])]:
    d = df if name == "byrho" else main
    summ(d, by).to_csv(os.path.join(out, f"table_{name}.csv"), index=False)

# Type I by K and intercepts (all rho <= 0.3)
summ(main, ["imb", "K", "ref"])[["imb", "K", "ref", "typeI"]].to_csv(os.path.join(out, "typeI_byK.csv"), index=False)

# share of data sets where minq gives lowest density / fewest apparent
w = main.pivot_table(index=key, columns="ref", values=["true_density", "apparent_pervasive"])
extra = {}
for imb in [False, True]:
    ww = w[w.index.get_level_values("imb") == imb]
    dens = ww["true_density"]
    extra[f"minq_lowest_density_share_imb{int(imb)}"] = float((dens["minq"] <= dens[REFS].min(axis=1) + 1e-12).mean())
    app = ww["apparent_pervasive"]
    extra[f"minq_fewest_apparent_share_imb{int(imb)}"] = float((app["minq"] <= app[REFS].min(axis=1)).mean())

# ---- hypotheses ----
bal = main[~main.imb]; imb = main[main.imb]
def per(d, ref, col):
    return d[d.ref == ref].set_index(key)[col]

def share_test(x, thr, direction):
    x = x.dropna().astype(float); k, n_ = int(x.sum()), len(x); est = k / n_
    alt = "greater" if direction == ">=" else "less"
    p = stats.binomtest(k, n_, thr, alternative=alt).pvalue
    return est, p

def mean_test(x, thr, direction):
    x = x.dropna().astype(float); est = x.mean()
    t = stats.ttest_1samp(x, thr, alternative="greater" if direction == ">=" else "less")
    return est, t.pvalue

def ratio_boot(a, b, thr, direction, B=2000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    est = j.a.mean() / j.b.mean()
    idx = rng.integers(0, len(j), (B, len(j))); A, Bv = j.a.values[idx].mean(1), j.b.values[idx].mean(1)
    r = A / Bv
    p = ((r <= thr).mean() if direction in (">=", ">") else (r >= thr).mean())
    return est, max(p, 1 / B) if p == 0 else p

def ratio_diff_boot(a1, b1, a2, b2, thr, direction, B=2000):
    j1 = pd.concat([a1.rename("a"), b1.rename("b")], axis=1).dropna()
    j2 = pd.concat([a2.rename("a"), b2.rename("b")], axis=1).dropna()
    est = j2.a.mean() / j2.b.mean() - j1.a.mean() / j1.b.mean()
    i1 = rng.integers(0, len(j1), (B, len(j1))); i2 = rng.integers(0, len(j2), (B, len(j2)))
    r = j2.a.values[i2].mean(1) / j2.b.values[i2].mean(1) - j1.a.values[i1].mean(1) / j1.b.values[i1].mean(1)
    p = (r >= thr).mean() if direction == "<" else (r <= thr).mean()
    return est, max(p, 1 / B) if p == 0 else p

H = []
def add(h, crit, typ, thr_txt, est, p, ok):
    H.append(dict(H=h, criterion=crit, type=typ, threshold=thr_txt, estimate=est, p=p, decision="Supported" if ok else "Not supported"))

# H1: most frequent reference is atypical (balanced intercepts)
e, p = share_test(per(bal, "freq", "ref_deviates"), 0.90, ">="); add("H1", "Most frequent reference deviates", "Share", "≥ 0.90", e, p, e >= 0.90)
e, p = share_test(per(bal, "rare", "ref_deviates"), 0.75, "<="); add("H1", "Least frequent reference deviates", "Share", "≤ 0.75", e, p, e <= 0.75)
e, p = ratio_boot(per(bal, "freq", "true_density"), per(bal, "rare", "true_density"), 1.25, ">="); add("H1", "True density, most / least frequent", "Ratio", "≥ 1.25", e, p, e >= 1.25)
# H2: denser Wald tables under most frequent, Type I near nominal (balanced)
e, p = ratio_boot(per(bal, "freq", "sig_frac"), per(bal, "rare", "sig_frac"), 1.25, ">="); add("H2", "Share significant, most / least frequent", "Ratio", "≥ 1.25", e, p, e >= 1.25)
e, p = ratio_boot(per(bal, "freq", "apparent_pervasive"), per(bal, "rare", "apparent_pervasive"), 1.5, ">="); add("H2", "Apparently pervasive, most / least frequent", "Ratio", "≥ 1.50", e, p, e >= 1.5)
e, p = ratio_boot(per(bal, "rare", "median_se"), per(bal, "freq", "median_se"), 1.25, "<="); add("H2", "Median SE, least / most frequent", "Ratio", "≤ 1.25", e, p, e <= 1.25)
t1 = {r: per(bal, r, "typeI") for r in ["freq", "first", "rare"]}
lo = min(t1, key=lambda r: t1[r].mean()); hi = max(t1, key=lambda r: t1[r].mean())
e, p = mean_test(t1[lo], 0.04, ">="); add("H2", "Type I error, lowest of three fixed references", "Mean", "≥ 0.04", e, p, e >= 0.04)
e, p = mean_test(t1[hi], 0.06, "<="); add("H2", "Type I error, highest of three fixed references", "Mean", "≤ 0.06", e, p, e <= 0.06)
# H3: differences shrink with imbalanced intercepts
e, p = ratio_diff_boot(per(bal, "freq", "sig_frac"), per(bal, "rare", "sig_frac"), per(imb, "freq", "sig_frac"), per(imb, "rare", "sig_frac"), 0, "<")
add("H3", "Share-significant ratio, imbalanced minus balanced", "Ratio diff.", "< 0", e, p, e < 0)
e, p = ratio_diff_boot(per(bal, "rare", "median_se"), per(bal, "freq", "median_se"), per(imb, "rare", "median_se"), per(imb, "freq", "median_se"), 0, ">")
add("H3", "Median-SE ratio, imbalanced minus balanced", "Ratio diff.", "> 0", e, p, e > 0)
# H4: smallest-atypicality reference
for lab, d in [("balanced", bal), ("imbalanced", imb)]:
    e, p = ratio_boot(per(d, "freq", "apparent_pervasive"), per(d, "minq", "apparent_pervasive"), 1.5, ">=")
    add("H4", f"Apparently pervasive, most frequent / smallest atypicality ({lab})", "Ratio", "≥ 1.50", e, p, e >= 1.5)
e, p = ratio_boot(per(imb, "minq", "median_se"), per(imb, "freq", "median_se"), 1.25, "<=")
add("H4", "Median SE, smallest atypicality / most frequent (imbalanced)", "Ratio", "≤ 1.25", e, p, e <= 1.25)
e, p = mean_test(per(bal, "minq", "typeI"), 0.06, "<="); add("H4", "Type I error, smallest atypicality (balanced)", "Mean", "≤ 0.06", e, p, e <= 0.06)
Hd = pd.DataFrame(H); Hd.to_csv(os.path.join(out, "hypotheses.csv"), index=False)

json.dump(dict(counts=counts, extra=extra), open(os.path.join(out, "summary.json"), "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(counts, indent=1)); print(json.dumps(extra, indent=1))
print(T_main.round(3).to_string(index=False)); print(Hd.round(4).to_string(index=False))
