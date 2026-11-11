"""Statistics for the label-budget study (numpy + scipy only).

All differences are paired by partition: d[i] = OA(arm A, partition i) - OA(arm B, partition i),
in OA points. Verdict rules follow the blueprint:
  HELPS        mean >= 1.0, 95 % CI above 0, wins >= n - 1
  EQUIVALENT   TOST: 90 % CI inside +/- margin
  NO BENEFIT   mean < 0.5 and not equivalent
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired(d):
    d = np.asarray(d, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p_t = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 0.0
    p_w = stats.wilcoxon(d, method="exact").pvalue if np.any(d != 0) else 1.0
    wins = int((d > 0).sum())
    p_sign = stats.binomtest(wins, int((d != 0).sum()) or 1).pvalue
    return {"n": n, "mean": m, "sd": sd, "ci95": (m - t * se, m + t * se), "wins": wins,
            "p_t": p_t, "p_wilcoxon": p_w, "p_sign": p_sign}


def tost(d, margin):
    """Two one-sided t-tests for |mean| < margin. Returns the 90 % CI and the TOST p (max of the two)."""
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t90 = stats.t.ppf(0.95, n - 1)
    p_lo = stats.t.sf((m + margin) / se, n - 1)       # H0: mean <= -margin
    p_hi = stats.t.cdf((m - margin) / se, n - 1)      # H0: mean >= +margin
    return {"ci90": (m - t90 * se, m + t90 * se), "p_tost": max(p_lo, p_hi),
            "equivalent": (m - t90 * se > -margin) and (m + t90 * se < margin)}


def verdict(d, margin=1.5):
    p, e = paired(d), tost(d, margin)
    if p["mean"] >= 1.0 and p["ci95"][0] > 0 and p["wins"] >= p["n"] - 1:
        return "HELPS"
    if e["equivalent"]:
        return "EQUIVALENT"
    if p["mean"] < 0.5:
        return "NO BENEFIT SHOWN"
    return "INCONCLUSIVE"


def iut(pvalues):
    """Intersection-union test: a claim that must hold in every task has p = max of the per-task p."""
    return max(pvalues)


def page_trend(matrix, decreasing=False):
    """Page's L test. matrix: partitions x ordered levels (lowest budget first)."""
    m = np.asarray(matrix, float)
    if decreasing:
        m = -m
    return stats.page_trend_test(m).pvalue


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def hksj(means, ses):
    """Random-effects pool (DerSimonian-Laird tau^2) with the Hartung-Knapp-Sidik-Jonkman interval."""
    y, s2 = np.asarray(means, float), np.asarray(ses, float) ** 2
    k = len(y)
    w = 1 / s2
    yw = (w * y).sum() / w.sum()
    q = (w * (y - yw) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (s2 + tau2)
    mu = (ws * y).sum() / ws.sum()
    var = (ws * (y - mu) ** 2).sum() / ((k - 1) * ws.sum())
    t = stats.t.ppf(0.975, k - 1)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    pred = t * np.sqrt(tau2 + var)
    return {"mu": mu, "ci95": (mu - t * np.sqrt(var), mu + t * np.sqrt(var)), "tau": np.sqrt(tau2), "I2": i2,
            "pred95": (mu - pred, mu + pred), "p": 2 * stats.t.sf(abs(mu / np.sqrt(var)), k - 1) if var > 0 else 0.0}


def reversal_probability(a_by_part, b_by_part, n_boot=10_000, seed=0):
    """Probability that the order of two configurations flips between two protocols.
    a_by_part, b_by_part: arrays (partitions x 2) of OA under (leaky, disjoint) for configurations A and B.
    Returns P(sign of mean(A - B) differs between the two protocols) over a partition bootstrap."""
    a, b = np.asarray(a_by_part, float), np.asarray(b_by_part, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diff = a[idx].mean(1) - b[idx].mean(1)          # (n_boot, 2): A - B under each protocol
    return float(np.mean(np.sign(diff[:, 0]) != np.sign(diff[:, 1])))
