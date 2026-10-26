"""Descriptive summaries for Addendum A (sensitivity) and Addendum B (B1 Wald reference, B2 lasso comparators).
Usage: python3 analysis_addenda.py <sensitivity_results.csv> <addB_b1.csv|-> <addB_b2.csv|-> <outdir>"""
import sys, os, json
import numpy as np, pandas as pd

sens, b1f, b2f, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
S = {}
d = pd.read_csv(sens)
m = ["thr_mean", "thr_median", "thr_shorth", "lasso_sym"]
d = d[d.method.isin(m + ["ridge"])]
tA = d[d.method.isin(m)].pivot_table(index=["truth", "n"], columns="method", values=["f1", "frac_nz", "excess"], aggfunc="mean")
tA.columns = [f"{a}_{b}" for a, b in tA.columns]; tA = tA.reset_index()
ex_r = d[d.method == "ridge"].groupby(["truth", "n"]).excess.mean().rename("excess_ridge").reset_index()
tA = tA.merge(ex_r, on=["truth", "n"]); tA.to_csv(os.path.join(out, "tableA_sensitivity.csv"), index=False)
S["sens_n"] = int(d[["truth", "K", "n", "rep"]].drop_duplicates().shape[0])
S["true_frac_nz"] = d.groupby("truth").true_frac_nz.mean().to_dict()
print(tA.round(4).to_string())

if b1f != "-" and os.path.exists(b1f):
    b = pd.read_csv(b1f); S["b1_skipped"] = int((b.ref == "skipped").sum()); b = b[b.ref != "skipped"]
    S["b1_n"] = int(b[["K", "n", "rho", "imb", "rep"]].drop_duplicates().shape[0])
    t1 = b.groupby(["imb", "ref"])[["true_density", "sig_frac", "typeI", "power", "median_se", "apparent_pervasive", "ref_deviates"]].mean().reset_index()
    t1.to_csv(os.path.join(out, "tableB1_wald_minq.csv"), index=False); print(t1.round(4).to_string())
    # how often minq equals freq / rare
    w = b.pivot_table(index=["K", "n", "rho", "imb", "rep"], columns="ref", values="true_density")
    S["b1_lowest_density_share"] = b.loc[b.groupby(["K", "n", "rho", "imb", "rep"]).true_density.idxmin()].ref.value_counts(normalize=True).to_dict()
if b2f != "-" and os.path.exists(b2f):
    c = pd.read_csv(b2f)
    S["b2_n"] = int(c[["K", "n", "rho", "same", "corr", "rep"]].drop_duplicates().shape[0])
    t2 = c.groupby("method")[["f1", "frac_nz", "excess"]].mean()
    t2n = c.pivot_table(index="method", columns="n", values=["f1", "excess"], aggfunc="mean")
    t2n.columns = [f"{a}_{b}" for a, b in t2n.columns]
    t2 = t2.join(t2n).reset_index(); t2.to_csv(os.path.join(out, "tableB2_lasso_comparators.csv"), index=False)
    print(t2.round(4).to_string())
    t2c = c.pivot_table(index="method", columns=["corr", "n"], values="excess", aggfunc="mean"); print((100 * t2c).round(3))
    t2s = c.pivot_table(index="method", columns=["same", "rho"], values="f1", aggfunc="mean"); print(t2s.round(3))
json.dump(S, open(os.path.join(out, "addenda_stats.json"), "w"), indent=1, default=str); print(json.dumps(S, indent=1, default=str))
