"""Descriptive summaries for Addendum C. Usage: python3 analysis_addC.py <addC_c1.csv> <addC_c2.csv> <outdir>"""
import sys, os, json
import pandas as pd
c1p, c2p, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
pd.set_option("display.width", 220)
S = {}
if os.path.exists(c1p):
    c = pd.read_csv(c1p); c = c[c.method.isin(["thr_mean", "thr_median", "thr_shorth", "lasso_sym", "ridge"])]
    t = c.groupby(["same", "n", "method"])[["f1", "frac_nz", "excess", "true_frac_nz"]].mean().unstack("method")
    t.to_csv(os.path.join(out, "c1_summary.csv")); print(t.round(3))
    w = c.pivot_table(index=["K", "n", "rho", "same", "rep"], columns="method", values="test_loss")
    S["c1_win_sh_med_onesided"] = float((w.thr_shorth < w.thr_median)[w.index.get_level_values("same")].mean())
    S["c1_n"] = int(c[["K", "n", "rho", "same", "rep"]].drop_duplicates().shape[0])
if os.path.exists(c2p):
    b = pd.read_csv(c2p); S["c2_skipped"] = int((b.ref == "skipped").sum()); b = b[b.ref != "skipped"]
    g = b.groupby(["imb", "ref"])[["true_density", "ref_deviates", "sig_frac", "typeI", "power", "median_se", "apparent_pervasive"]].mean()
    g.to_csv(os.path.join(out, "c2_summary.csv")); print(g.round(3))
    kc = ["K", "n", "rho", "imb", "rep"]
    wd = b.pivot_table(index=kc, columns="ref", values="true_density")
    S["c2_minq_lowest_density"] = (wd["minq"] <= wd.min(axis=1) + 1e-12).groupby(level="imb").mean().to_dict()
    print(b.groupby(["imb", "K", "ref"]).typeI.mean().unstack().round(3))
json.dump(S, open(os.path.join(out, "addC_stats.json"), "w"), indent=1, default=str); print(json.dumps(S, indent=1, default=str))
