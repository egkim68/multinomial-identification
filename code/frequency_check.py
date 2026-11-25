"""Numerical check of the frequency approximation f_k ~= pi0_k {1 + (q_k - Qbar)/2} (Equation 4).
For random coefficient structures, compares the exact category frequencies (Monte Carlo, 200,000 draws)
with the approximation, and records how often the most and least frequent categories deviate.
Also saves one structure (K = 40, rho = 0.1) for Figure 1.
Usage: python3 frequency_check.py <outdir>   (output folder created if missing)"""
import sys, os
import numpy as np, pandas as pd

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261016)
P_DIM, DRAWS, STRUCT = 10, 200000, 40

def structure(K, rho, scale):
    return np.where(rng.random((P_DIM, K)) < rho,
                    rng.choice([-1, 1], (P_DIM, K)) * rng.uniform(.3, .9, (P_DIM, K)), 0) * scale

def check(D, S, Lc):
    K = D.shape[1]; a = np.zeros(K)
    X = rng.normal(size=(DRAWS, P_DIM)) @ Lc.T
    f = softmax(X @ D + a).mean(0)
    pi0 = softmax(a[None, :])[0]
    dk = D - (D @ pi0)[:, None]
    q = np.einsum("jk,jl,lk->k", dk, S, dk)
    approx = pi0 * (1 + 0.5 * (q - pi0 @ q))
    return f, q, approx

rows = []
for K, rho, scale, corr in [(5, .2, 1, 0), (10, .1, 1, 0), (20, .1, 1, 0), (40, .1, 1, 0), (40, .3, 1, 0),
                            (20, .1, 1, .5), (20, .1, .5, 0), (20, .1, 1.5, 0)]:
    S = corr ** np.abs(np.subtract.outer(np.arange(P_DIM), np.arange(P_DIM))); Lc = np.linalg.cholesky(S)
    err, cor, top, bot = [], [], [], []
    for _ in range(STRUCT):
        D = structure(K, rho, scale); f, q, approx = check(D, S, Lc)
        err.append(np.abs(np.log(f) - np.log(approx)).max()); cor.append(np.corrcoef(np.log(f), q)[0, 1])
        top.append((D[:, np.argmax(f)] != 0).any()); bot.append((D[:, np.argmin(f)] != 0).any())
    rows.append(dict(K=K, rho=rho, scale=scale, corr=corr, structures=STRUCT, max_abs_log_error=np.mean(err),
                     corr_logf_q=np.mean(cor), min_corr_logf_q=np.min(cor),
                     share_most_frequent_deviates=np.mean(top), share_least_frequent_deviates=np.mean(bot)))
pd.DataFrame(rows).to_csv(os.path.join(out, "frequency_check.csv"), index=False)

# one structure for Figure 1
S = np.eye(P_DIM); D = structure(40, .1, 1); f, q, approx = check(D, S, np.eye(P_DIM))
pd.DataFrame(dict(category=np.arange(40), frequency=f, atypicality=q, approximation=approx,
                  deviates=(D != 0).any(0))).to_csv(os.path.join(out, "figure1_points.csv"), index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(4).to_string(index=False))
