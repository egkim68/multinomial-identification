"""Figures for the revised paper, as PNG files (separate from the simulations, so they can be redrawn alone).
Usage: python3 make_figures.py <frequency_check folder> <analysis folder> <outdir>   (output folder created if missing)"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fc, an, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "axes.titlecolor": "black"})
GRID = dict(color="#d9d9d9", linewidth=0.5)
INK = "#000000"
REF = [("freq", "Most frequent", "#257de0", "o", "-"), ("first", "First label", "#f47230", "s", "--"),
       ("rare", "Least frequent", "#00b894", "^", "-."), ("minq", "Smallest atypicality", "#f2a600", "D", ":")]
DPI = 1200

# ---- Figure 1: one predictor reported under two references (worked example) ----
slopes = np.array([0.5, 0.5, 0.5, 1.2]); cats = np.arange(1, 5)
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6.5, 4.7))
gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.55, wspace=0.6)
ax = fig.add_subplot(gs[0, 1:3])
ax.axhline(0.5, color=INK, linewidth=0.8, linestyle="--", zorder=1)
ax.scatter(cats, slopes, s=60, color=["#257de0"] * 3 + ["#f47230"], edgecolor="white", linewidth=0.6, zorder=3)
ax.text(1, 0.62, "common slope", fontsize=10, color=INK)
ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("Slope"); ax.set_title("a. Slopes", loc="left", fontsize=11)
axes = [ax]
axb = fig.add_subplot(gs[1, 0:2]); axc = fig.add_subplot(gs[1, 2:4], sharey=axb)
for ax, ref, title in [(axb, 1, "b. Reference: category 1"), (axc, 4, "c. Reference: category 4")]:
    others = [c for c in cats if c != ref]; vals = [slopes[c - 1] - slopes[ref - 1] for c in others]
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.bar(others, vals, width=0.6, color=["#f47230" if c == 4 else "#257de0" for c in others], zorder=2)
    ax.text(ref, 0.06, "ref.", ha="center", va="bottom", fontsize=10, color=INK)
    for c, v in zip(others, vals):
        ax.text(c, v + (0.06 if v >= 0 else -0.06), f"{v:.1f}".replace("-", "\u2212") if abs(v) > 1e-9 else "0",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
    ax.set_ylim(-1.0, 1.0); ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0]); ax.set_title(title, loc="left", fontsize=11)
    axes.append(ax)
axb.set_ylabel("Contrast"); plt.setp(axc.get_yticklabels(), visible=False)
for ax in axes:
    ax.set_xticks(cats); ax.set_xlim(0.5, 4.5); ax.set_xlabel("Category")
from matplotlib.patches import Patch
fig.legend([Patch(color="#257de0"), Patch(color="#f47230")], ["Category with the common slope", "Deviating category"],
           loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.01), handlelength=1.2, columnspacing=2.0)
fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17)
fig.savefig(os.path.join(out, "Figure_1.png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---- Figure 2: category frequencies, sorted, with the frequency-atypicality approximation ----
p = pd.read_csv(os.path.join(fc, "figure1_points.csv")).sort_values(["frequency", "atypicality"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.grid(True, axis="y", **GRID); ax.set_axisbelow(True)
x = np.arange(1, len(p) + 1)
for dev, lab, col in [(False, "Common slope on\nevery predictor", "#257de0"), (True, "Deviates on at least\none predictor", "#f47230")]:
    s = p.deviates == dev
    ax.bar(x[s], 100 * p.frequency[s], width=0.8, color=col, edgecolor="white", linewidth=0.4, label=lab, zorder=2)
ax.scatter(x, 100 * p.approximation, marker="_", s=60, color=INK, linewidths=1.4, label="Approximation", zorder=3)
ax.set_xlabel("Categories, sorted from least to most frequent", fontsize=12); ax.set_ylabel("Frequency (%)", fontsize=12)
ax.tick_params(labelsize=11)
ax.set_xlim(0.3, len(p) + 0.7); ax.set_xticks([1, 10, 20, 30, 40])
h, l = ax.get_legend_handles_labels(); order = [l.index(k) for k in l if k != "Approximation"] + [l.index("Approximation")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, labelspacing=1.0)
fig.tight_layout(); fig.savefig(os.path.join(out, "Figure_2.png"), dpi=DPI); plt.close(fig)

# ---- Figure 3: share of significant contrasts by K and reference ----
t = pd.read_csv(os.path.join(an, "table_byK.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
for ax, imb, title in [(axes[0], False, "a. Balanced intercepts"), (axes[1], True, "b. Imbalanced intercepts")]:
    ax.grid(True, axis="y", **GRID); ax.set_axisbelow(True)
    for key, lab, col, mk, ls in REF:
        s = t[(t.imb == imb) & (t.ref == key)].sort_values("K")
        ax.plot(s.K, 100 * s.sig_frac, color=col, marker=mk, linestyle=ls, linewidth=1.6, markersize=6,
                markeredgecolor="white", markeredgewidth=0.6, label=lab)
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_xticks([5, 10, 20]); ax.set_xlabel("Number of categories (K)")
axes[0].set_ylabel("Significant contrasts (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(out, "Figure_3.png"), dpi=DPI); plt.close(fig)
print("figures written to", out)
