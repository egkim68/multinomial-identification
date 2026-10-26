"""Summaries, hypothesis checks and tables for the manuscript.
Usage: python3 analysis.py <main_results.csv> <wald_results.csv> <outdir>
Writes stats.json, hypotheses.csv and table CSVs to <outdir> (created if missing)."""
import sys, os, json
import numpy as np, pandas as pd

main_path, wald_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
d = pd.read_csv(main_path); w3 = pd.read_csv(wald_path)
for c, v in [("same", False), ("corr", 0.0)]:
    if c not in d: d[c] = v
    if c not in w3 and c == "corr": w3[c] = v
d["imb"] = d["imb"].astype(bool); d["same"] = d["same"].astype(bool); w3["imb"] = w3["imb"].astype(bool)
key = ["K", "n", "rho", "imb", "same", "corr", "rep"]
L = d.pivot_table(index=key, columns="method", values="test_loss")
F = d.pivot_table(index=key, columns="method", values="f1")
E = d.pivot_table(index=key, columns="method", values="excess")
NZ = d.pivot_table(index=key, columns="method", values="frac_nz")
TD = d.pivot_table(index=key, columns="method", values="true_frac_nz")
k = L.index.to_frame(index=False); k.index = L.index.rename([None] * len(key))
S = {}
S["n_datasets"] = int(len(L)); S["n_cells"] = int(k.drop(columns="rep").drop_duplicates().shape[0])
sparse = k.rho <= .3
def win(a, b, m): return float((L[a] < L[b])[m].mean())
def held(m): return m & (k.rep % 4 == 0)

from scipy import stats
RNG = np.random.default_rng(2026)
B_BOOT = 2000
rows = []
def crit(h, code, text, kind, op, thr, value, p, ok=None):
    if ok is None: ok = value >= thr if op == ">=" else value <= thr if op == "<=" else value < thr if op == "<" else value > thr
    rows.append(dict(hypothesis=h, criterion=code, text=text, kind=kind, op=op, threshold=thr,
                     value=round(float(value), 4), p=float(p), met=bool(ok)))

def p_share(x, thr, op):
    x = np.asarray(x, bool); alt = "greater" if op == ">=" else "less"
    return stats.binomtest(int(x.sum()), len(x), thr, alternative=alt).pvalue
def p_mean(x, thr, op):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return stats.ttest_1samp(x, thr, alternative="greater" if op == ">=" else "less").pvalue
def boot(fn, *arrs):
    n = len(arrs[0]); out = np.empty(B_BOOT)
    for b in range(B_BOOT):
        i = RNG.integers(0, n, n); out[b] = fn(*[a[i] for a in arrs])
    return out
def p_boot(draws, thr, op):
    wrong = (draws < thr) if op in (">=", ">") else (draws > thr)
    return (wrong.sum() + 1) / (len(draws) + 1)

def evaluate(tag, base):
    m_s = sparse & base
    x = (L.thr_median < L.thr_mean)[m_s]
    crit("H1" + tag, "win median<mean, sparse >= .65", "Median beats mean centering in test loss (ρ ≤ .3)", "Share", ">=", .65, x.mean(), p_share(x, .65, ">="))
    for sm, lab in [(False, "mixed signs"), (True, "one-sided signs")]:
        m = m_s & (k.n == 2000) & (k.same == sm); x = (F.thr_median - F.thr_mean)[m]
        crit("H1" + tag, f"F1 median - mean >= .08 (same={sm})", f"F1 gain of median over mean (n = 2,000, {lab})", "Mean difference", ">=", .08, x.mean(), p_mean(x, .08, ">="))
    m = base & k.same & (k.rho == .3) & (k.n == 2000)
    x = F.thr_shorth[m]; crit("H2" + tag, "F1 shorth >= .75", "F1 of shorth (one-sided, ρ = .3, n = 2,000)", "Mean", ">=", .75, x.mean(), p_mean(x, .75, ">="))
    x = (F.thr_shorth - F.thr_mean)[m]; crit("H2" + tag, "F1 shorth - mean >= .25", "F1 gain of shorth over mean (same cells)", "Mean difference", ">=", .25, x.mean(), p_mean(x, .25, ">="))
    x = (L.thr_shorth < L.thr_median)[m]; crit("H2" + tag, "win shorth<median >= .75", "Shorth beats median in test loss (same cells)", "Share", ">=", .75, x.mean(), p_share(x, .75, ">="))
    for c in [0.0, 0.5]:
        mm = m_s & (k.n == 2000) & (k["corr"] == c); x = (F.thr_shorth - F.lasso_sym)[mm]
        crit("H3" + tag, f"F1 shorth - lasso >= .20 (corr={c})", f"F1 gain of shorth over lasso (n = 2,000, γ = {c:g})".replace("γ = 0.5", "γ = .5"), "Mean difference", ">=", .20, x.mean(), p_mean(x, .20, ">="))
    mm = m_s & (k.n == 2000) & (k["corr"] == 0); x = (E.thr_shorth - E.lasso_sym)[mm]
    crit("H3" + tag, "excess shorth - lasso <= .002 (corr 0)", "Excess loss of shorth minus lasso (n = 2,000, γ = 0)", "Mean difference", "<=", .002, x.mean(), p_mean(x, .002, "<="))
    mb = m_s & ~k.imb
    if "ref_deviates" in d and d.ref_deviates.notna().any():
        RD = d.pivot_table(index=key, columns="method", values="ref_deviates", aggfunc="first").astype(float)
        RD.index = RD.index.rename([None] * len(key)); RD = RD.reindex(k.index)
        x = RD.lasso_ref_freq[mb].dropna().astype(bool)
        crit("H4" + tag, "freq ref deviates >= .90", "Most frequent reference deviates (balanced)", "Share", ">=", .90, x.mean(), p_share(x, .90, ">="))
        x = RD.lasso_ref_rare[mb].dropna().astype(bool)
        crit("H4" + tag, "rare ref deviates <= .75", "Least frequent reference deviates (balanced)", "Share", "<=", .75, x.mean(), p_share(x, .75, "<="))
    a, b_ = TD.lasso_ref_freq[mb].values, TD.lasso_ref_rare[mb].values
    v = a.mean() / b_.mean(); draws = boot(lambda u, w: u.mean() / w.mean(), a, b_)
    crit("H4" + tag, "density ratio freq/rare >= 1.25", "Truth density, most / least frequent (balanced)", "Ratio", ">=", 1.25, v, p_boot(draws, 1.25, ">="))
    x = (L.lasso_ref_rare < L.lasso_ref_freq)[base & ~k.imb]
    crit("H5" + tag, "win rare<freq balanced >= .70", "Lasso: least beats most frequent reference (balanced)", "Share", ">=", .70, x.mean(), p_share(x, .70, ">="))
    x = (L.lasso_ref_rare < L.lasso_ref_freq)[base]
    crit("H5" + tag, "win rare<freq overall >= .60", "Lasso: least beats most frequent reference (all)", "Share", ">=", .60, x.mean(), p_share(x, .60, ">="))

def evaluate_wald(tag, wd):
    wd = wd[wd.rho <= .3]
    kk = ["K", "n", "rho", "imb", "corr", "rep"]
    P = {v: wd.pivot_table(index=kk, columns="ref", values=v, dropna=False) for v in ["sig_frac", "pervasive_spurious", "median_se", "typeI"]}
    idx = P["sig_frac"].index
    P = {v: p_.reindex(idx) for v, p_ in P.items()}
    imbv = P["sig_frac"].index.get_level_values("imb").astype(bool)
    def arr(v, r, im): return P[v][r].values[imbv == im]
    ratio = lambda u, w: np.nanmean(u) / np.nanmean(w)
    for v, r1, r2, code, text, op, thr in [
        ("sig_frac", "freq", "rare", "sig share ratio freq/rare >= 1.25", "Share significant, most / least frequent (balanced)", ">=", 1.25),
        ("pervasive_spurious", "freq", "rare", "spurious pervasive ratio >= 1.5", "Apparently pervasive, most / least frequent (balanced)", ">=", 1.5),
        ("median_se", "rare", "freq", "median SE ratio rare/freq <= 1.25", "Median SE, least / most frequent (balanced)", "<=", 1.25)]:
        a, b_ = arr(v, r1, False), arr(v, r2, False)
        crit("H6" + tag, code, text, "Ratio", op, thr, ratio(a, b_), p_boot(boot(ratio, a, b_), thr, op))
    t = {r: arr("typeI", r, False) for r in ["freq", "first", "rare"]}
    tm = {r: np.nanmean(x) for r, x in t.items()}
    rmin = min(tm, key=tm.get); rmax = max(tm, key=tm.get)
    crit("H6" + tag, "Type I in [.04,.06], all refs (min)", "Type I error, lowest of the three references", "Mean", ">=", .04, tm[rmin], p_mean(t[rmin], .04, ">="))
    crit("H6" + tag, "Type I in [.04,.06], all refs (max)", "Type I error, highest of the three references", "Mean", "<=", .06, tm[rmax], p_mean(t[rmax], .06, "<="))
    for v, r1, r2, code, text, op in [
        ("sig_frac", "freq", "rare", "sig ratio imbalanced < balanced (imbalanced value)", "Share-significant ratio, imbalanced minus balanced", "<"),
        ("median_se", "rare", "freq", "SE ratio imbalanced > balanced (imbalanced value)", "Median-SE ratio, imbalanced minus balanced", ">")]:
        ab, bb = arr(v, r1, False), arr(v, r2, False); ai, bi = arr(v, r1, True), arr(v, r2, True)
        diff = ratio(ai, bi) - ratio(ab, bb)
        draws = boot(ratio, ai, bi) - boot(ratio, ab, bb)
        crit("H7" + tag, code, text, "Difference of ratios", op, 0.0, diff, p_boot(draws, 0.0, op))

allm = pd.Series(True, index=k.index)
evaluate("", allm); evaluate(" (held-out)", held(allm))
evaluate_wald("", w3); evaluate_wald(" (held-out)", w3[w3.rep % 4 == 0])
H = pd.DataFrame(rows); H.to_csv(os.path.join(out, "hypotheses.csv"), index=False)

# ---- descriptive tables ----
meth = ["ridge", "thr_mean", "thr_median", "thr_shorth", "lasso_sym"]
t2 = []
for (sm, rho), g in k[k.n == 2000].groupby(["same", "rho"]):
    idx = g.index
    t2.append(dict(signs="one-sided" if sm else "mixed", rho=rho,
                   **{f"F1_{m}": F[m][idx].mean() for m in meth[1:]},
                   **{f"nz_{m}": NZ[m][idx].mean() for m in meth[1:]},
                   **{f"ex_{m}": E[m][idx].mean() for m in meth}))
T2 = pd.DataFrame(t2); T2.to_csv(os.path.join(out, "table2_centring_n2000.csv"), index=False)
t2b = []
for (c, nn), g in k[sparse].groupby(["corr", "n"]):
    idx = g.index
    t2b.append(dict(corr=c, n=nn, **{f"F1_{m}": F[m][idx].mean() for m in ["thr_shorth", "lasso_sym"]},
                    **{f"ex_{m}": E[m][idx].mean() for m in ["ridge", "thr_shorth", "lasso_sym"]},
                    win_shorth_lasso=float((L.thr_shorth < L.lasso_sym)[idx].mean())))
pd.DataFrame(t2b).to_csv(os.path.join(out, "table2b_vs_lasso.csv"), index=False)
t3 = []
for imb, g in k.groupby("imb"):
    idx = g.index
    t3.append(dict(intercepts="imbalanced" if imb else "balanced",
                   **{f"dens_{r}": TD[f"lasso_ref_{r}"][idx].mean() for r in ["first", "freq", "rare"]},
                   **{f"ex_{r}": E[f"lasso_ref_{r}"][idx].mean() for r in ["first", "freq", "rare"]},
                   ex_sym=E.lasso_sym[idx].mean(),
                   win_rare_freq=float((L.lasso_ref_rare < L.lasso_ref_freq)[idx].mean())))
pd.DataFrame(t3).to_csv(os.path.join(out, "table3_reference_lasso.csv"), index=False)
wt = w3.groupby(["imb", "ref"])[["true_density", "sig_frac", "power", "typeI", "median_se", "pervasive_spurious", "ref_deviates"]].mean().reset_index()
wt.to_csv(os.path.join(out, "table4_wald.csv"), index=False)
S["wald_n_datasets"] = int(w3.groupby(["K", "n", "rho", "imb", "corr", "rep"]).ngroups)
S["wald_skipped"] = None
S["hypotheses_all_met"] = {h: bool(g.met.all()) for h, g in H.assign(h=H.hypothesis.str.slice(0, 2)).groupby("h")}
# headline numbers used in text
S["win_median_mean_sparse"] = win("thr_median", "thr_mean", sparse)
S["win_shorth_mean_all"] = win("thr_shorth", "thr_mean", allm)
S["win_rare_freq_bal"] = win("lasso_ref_rare", "lasso_ref_freq", ~k.imb)
S["win_rare_freq_imb"] = win("lasso_ref_rare", "lasso_ref_freq", k.imb)
S["q_median_sparse_n2000"] = float(d[(d.method == "thr_shorth") & (d.rho <= .3) & (d.n == 2000)].q.median())
json.dump(S, open(os.path.join(out, "stats.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print(H.to_string(index=False)); print(json.dumps(S, indent=1, default=float))
print(T2.round(3).to_string(index=False)); print(pd.DataFrame(t2b).round(4).to_string(index=False))
print(pd.DataFrame(t3).round(4).to_string(index=False)); print(wt.round(3).to_string(index=False))
