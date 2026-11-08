"""Evaluation for probabilistic classifiers: discrimination, calibration, decisions and fairness.

All implemented directly (and unit-tested against scikit-learn where an equivalent exists).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import rankdata

from nhsread.models.logit_scratch import logit


# ----------------------------------------------------------------------------- discrimination
def auc(y, p) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half)."""
    y = np.asarray(y)
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def roc_curve(y, p):
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    return fp / fp[-1], tp / tp[-1]


def auc_ci_delong(y, p, alpha: float = 0.05) -> tuple[float, float, float]:
    """AUC with a DeLong (1988) standard error."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    allr = rankdata(np.concatenate([pos, neg]))
    v10 = (allr[:m] - rankdata(pos)) / n
    v01 = 1 - (allr[m:] - rankdata(neg)) / m
    a = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(a), float(a - z * se), float(a + z * se)


# ----------------------------------------------------------------------------- probability accuracy
def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Calibration slope: logistic regression of y on logit(p); ideal slope 1.
    Calibration-in-the-large: intercept with logit(p) as an offset; ideal 0."""
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(lp), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(citl), float(slope)


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Observed vs mean predicted rate by decile of predicted risk."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    se = np.sqrt(t["obs"] * (1 - t["obs"]) / t["n"])
    t["obs_lo"], t["obs_hi"] = t["obs"] - 1.96 * se, t["obs"] + 1.96 * se
    return t.reset_index()


def hosmer_lemeshow(y, p, bins: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow chi2 over deciles of risk (with large n this rejects for tiny miscalibration)."""
    t = reliability_table(y, p, bins)
    o1 = t["obs"] * t["n"]
    e1 = t["pred"] * t["n"]
    hl = float((((o1 - e1) ** 2) / (e1 * (1 - t["pred"]))).sum())
    return hl, float(stats.chi2.sf(hl, bins - 2))


# ----------------------------------------------------------------------------- decisions
def confusion_at(y, p, threshold: float) -> dict[str, float]:
    y = np.asarray(y)
    flag = np.asarray(p) >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged_pct": 100 * flag.mean(),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
    }


def net_benefit(y, p, thresholds) -> pd.DataFrame:
    """Decision-curve analysis (Vickers & Elkin 2006): NB = TP/N - FP/N * t/(1-t)."""
    y = np.asarray(y)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        c = confusion_at(y, p, t)
        nb = c["tp"] / n - c["fp"] / n * t / (1 - t)
        rows.append({"threshold": t, "model": nb, "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0})
    return pd.DataFrame(rows)


def expected_cost(y, p, threshold, cost_intervention: float, cost_readmission: float,
                  effectiveness: float) -> float:
    """Average cost per discharge of a follow-up programme offered to everyone above the threshold.

    Flagged patients cost `cost_intervention`; readmissions cost `cost_readmission`; the programme
    prevents a fraction `effectiveness` of readmissions among flagged patients.
    """
    y = np.asarray(y)
    flag = np.asarray(p) >= threshold
    readmissions = y.sum() - effectiveness * (y[flag]).sum()
    return float((flag.sum() * cost_intervention + readmissions * cost_readmission) / len(y))


def cost_optimal_threshold(y, p, cost_intervention, cost_readmission, effectiveness, grid=None):
    grid = np.linspace(0.02, 0.6, 117) if grid is None else grid
    costs = np.array([expected_cost(y, p, t, cost_intervention, cost_readmission, effectiveness) for t in grid])
    return float(grid[costs.argmin()]), pd.DataFrame({"threshold": grid, "cost_per_discharge": costs})


# ----------------------------------------------------------------------------- fairness
def group_report(y, p, group, threshold: float) -> pd.DataFrame:
    """Per-group discrimination, calibration and error rates at a common threshold."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(group)})
    rows = []
    for g, s in d.groupby("g", sort=True):
        c = confusion_at(s["y"], s["p"], threshold)
        rows.append({
            "group": g, "n": len(s), "obs_rate": s["y"].mean(), "mean_pred": s["p"].mean(),
            "o_to_e": s["y"].mean() / s["p"].mean(),
            "auc": auc(s["y"], s["p"]) if 0 < s["y"].sum() < len(s) else np.nan,
            "flagged_pct": c["flagged_pct"], "sensitivity": c["sensitivity"], "fpr": 1 - c["specificity"],
            "ppv": c["ppv"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- LACE
def lace_score(los_days, emergency, charlson, ed_visits_6m) -> np.ndarray:
    """LACE index (van Walraven et al. 2010): Length of stay, Acuity, Comorbidity, ED visits. Range 0-19."""
    los = np.asarray(los_days)
    L = np.select([los < 1, los == 1, los == 2, los == 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
    A = 3 * np.asarray(emergency).astype(int)
    ch = np.asarray(charlson)
    C = np.select([ch == 0, ch == 1, ch == 2, ch == 3], [0, 1, 2, 3], 5)
    E = np.minimum(np.asarray(ed_visits_6m), 4)
    return (L + A + C + E).astype(int)
