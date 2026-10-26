"""Multinomial logistic solver (FISTA) and helpers for the thresholding study."""
import numpy as np

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def logloss(P, y):
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)))

def predict(X, W, b):
    return softmax(X @ W + b)

def fit(X, y, K, lam, penalty="l1", ref=None, W0=None, b0=None, max_iter=3000, tol=1e-7, support=None):
    """Penalized multinomial logit. penalty 'l1' or 'l2' on W only (intercepts free).
    ref=None: symmetric K-column parameterization. ref=r: column r fixed at zero."""
    n, p = X.shape
    Y = np.zeros((n, K)); Y[np.arange(n), y] = 1
    W = np.zeros((p, K)) if W0 is None else W0.copy()
    b = np.log(np.clip(Y.mean(0), 1e-4, None)) if b0 is None else b0.copy()
    b = b - b.mean()
    mask = np.ones(K)
    if ref is not None:
        mask[ref] = 0; W[:, ref] = 0; b = b - b[ref]
    Mw = np.ones((p, K)) * mask if support is None else support * mask
    W = W * Mw
    L = 0.5 * (np.linalg.norm(X, 2) ** 2 / n + 1.0) + (lam if penalty == "l2" else 0)
    t = 1.0 / L
    Wy, by, tk = W.copy(), b.copy(), 1.0
    def obj(W, b):
        P = predict(X, W, b)
        f = logloss(P, y)
        return f + (lam * np.abs(W).sum() if penalty == "l1" else 0.5 * lam * (W ** 2).sum())
    prev = obj(W, b)
    for it in range(max_iter):
        P = predict(X, Wy, by)
        G = X.T @ (P - Y) / n
        gb = (P - Y).mean(0)
        if penalty == "l2":
            G = G + lam * Wy
        Wn = Wy - t * G; bn = by - t * gb
        if penalty == "l1":
            Wn = np.sign(Wn) * np.maximum(np.abs(Wn) - t * lam, 0)
        Wn *= Mw; bn *= mask
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Wy = Wn + (tk - 1) / tn * (Wn - W); by = bn + (tk - 1) / tn * (bn - b)
        W, b, tk = Wn, bn, tn
        if it % 25 == 24:
            cur = obj(W, b)
            if cur > prev:  # restart momentum
                Wy, by, tk = W.copy(), b.copy(), 1.0
            if abs(prev - cur) < tol * max(1, abs(cur)):
                break
            prev = cur
    return W, b

# ---- identification shifts (per predictor row; probabilities unchanged) ----
def shorth_center(row):
    """Midpoint of the shortest interval containing ceil(K/2) entries (half-sample mode)."""
    s = np.sort(row); K = len(s); h = int(np.ceil(K / 2))
    widths = s[h - 1:] - s[:K - h + 1]
    i = int(np.argmin(widths))
    return 0.5 * (s[i] + s[i + h - 1])

def center(W, how, ref=None):
    if how == "mean":   c = W.mean(1)
    elif how == "median": c = np.median(W, 1)
    elif how == "shorth": c = np.array([shorth_center(r) for r in W])
    elif how == "ref":  c = W[:, ref]
    else: raise ValueError(how)
    return W - c[:, None]

def threshold(W, q, keep_col=None):
    """Zero the q-fraction of smallest |W| entries (global quantile). keep_col: reference column excluded from count."""
    A = np.abs(W) if keep_col is None else np.abs(np.delete(W, keep_col, 1))
    if q <= 0: return W.copy()
    tau = np.quantile(A, q)
    Wt = W.copy(); Wt[np.abs(Wt) <= tau] = 0
    return Wt
