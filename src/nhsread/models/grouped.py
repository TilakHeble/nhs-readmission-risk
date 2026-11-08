"""Grouped-binomial logistic regression on the real NHS cells (IMD quintile x age x sex x year).

Each row is a cell with `num` readmissions out of `den` discharges. With ~65 million discharges the
standard binomial errors are absurdly small; real cells vary more than the binomial allows
(over-dispersion), so we also report quasi-binomial errors (scaled by Pearson chi2 / df).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from nhsread.config import AGE_BANDS


def prepare(cells: pd.DataFrame) -> pd.DataFrame:
    d = cells.copy()
    d["fail"] = d["den"] - d["num"]
    d["age"] = pd.Categorical(d["age"], categories=["16-74", "<16", "75+"])  # working age = reference
    d["sex"] = pd.Categorical(d["sex"], categories=["Female", "Male"])
    d["imd_c"] = d["imd"].astype(int)
    d["deprived"] = 6 - d["imd_c"]  # 1 = least deprived ... 5 = most deprived (higher = worse)
    d["t"] = d["year_start"] - 2014
    d["covid"] = (d["year"] == "2020/21").astype(int)
    return d


def fit(cells: pd.DataFrame, formula: str, scale: str | float | None = None):
    """Fit a binomial GLM with `cbind(num, den-num)` response via the statsmodels formula API.

    scale=None  -> binomial errors (dispersion fixed at 1)
    scale="X2"  -> quasi-binomial (dispersion = Pearson chi2 / df)

    Pitfall: with a two-column (successes, failures) response, statsmodels' own `fit(scale="X2")`
    estimates the dispersion on the proportion scale (about 0.0005 here instead of about 70),
    which *shrinks* the standard errors. We therefore fit once, compute Pearson chi2 / df
    ourselves, and refit with that fixed scale.
    """
    d = prepare(cells) if "fail" not in cells else cells
    model = smf.glm(f"num + fail ~ {formula}", data=d, family=sm.families.Binomial())
    res = model.fit()
    if scale is None:
        return res
    phi = res.pearson_chi2 / res.df_resid if scale == "X2" else float(scale)
    return model.fit(scale=phi)


def odds_ratio_table(res, alpha: float = 0.05) -> pd.DataFrame:
    ci = res.conf_int(alpha)
    out = pd.DataFrame({
        "coef": res.params,
        "se": res.bse,
        "odds_ratio": np.exp(res.params),
        "or_lo": np.exp(ci[0]),
        "or_hi": np.exp(ci[1]),
        "p": res.pvalues,
    })
    return out


def dispersion(res) -> float:
    return float(res.pearson_chi2 / res.df_resid)


def lr_test(restricted, full) -> tuple[float, int, float]:
    lr = 2 * (full.llf - restricted.llf)
    df = int(full.df_model - restricted.df_model)
    return float(lr), df, float(stats.chi2.sf(lr, df))


def quasi_f_test(restricted, full) -> tuple[float, int, int, float]:
    """F-test for nested quasi-binomial models: (D_r - D_f)/df / phi_full ~ F(df, df_resid)."""
    phi = dispersion(full)
    df = int(full.df_model - restricted.df_model)
    f = (restricted.deviance - full.deviance) / df / phi
    return float(f), df, int(full.df_resid), float(stats.f.sf(f, df, full.df_resid))


def average_marginal_effect(res, data: pd.DataFrame, var: str, levels: tuple) -> float:
    """Average change in predicted probability (percentage points) moving `var` from levels[0] to levels[1],
    weighted by discharges (so it is the effect for the average discharged patient)."""
    d0, d1 = data.copy(), data.copy()
    d0[var] = levels[0]
    d1[var] = levels[1]
    if isinstance(data[var].dtype, pd.CategoricalDtype):
        d0[var] = pd.Categorical(d0[var], categories=data[var].cat.categories)
        d1[var] = pd.Categorical(d1[var], categories=data[var].cat.categories)
    p0 = res.predict(d0)
    p1 = res.predict(d1)
    return float(100 * np.average(p1 - p0, weights=data["den"]))


def inequality_indices(year_cells: pd.DataFrame) -> dict[str, float]:
    """Slope and Relative Index of Inequality (SII / RII), the standard UK public-health measures.

    Quintiles are placed at the midpoint of their cumulative population share (ridit), from least
    deprived (0) to most deprived (1). A weighted logistic fit of rate on ridit gives the RII as the
    odds ratio between the hypothetical most- and least-deprived person; the SII is the
    absolute gap in percentage points from a weighted linear fit.
    """
    g = year_cells.groupby("imd")[["num", "den"]].sum().sort_index(ascending=False)  # 5 (least) -> 1
    share = g["den"] / g["den"].sum()
    ridit = (share.cumsum() - share / 2).to_numpy()
    rate = (g["num"] / g["den"]).to_numpy()
    X = sm.add_constant(ridit)
    lin = sm.WLS(rate, X, weights=g["den"].to_numpy()).fit()
    lg = sm.GLM(np.column_stack([g["num"], g["den"] - g["num"]]), X, family=sm.families.Binomial()).fit()
    return {
        "sii_pp": float(100 * lin.params[1]),
        "rii_or": float(np.exp(lg.params[1])),
        "rate_most": float(100 * rate[-1]),
        "rate_least": float(100 * rate[0]),
        "gap_pp": float(100 * (rate[-1] - rate[0])),
        "ratio": float(rate[-1] / rate[0]),
    }


def predicted_grid(res, years: list[str], year_start: dict[str, int]) -> pd.DataFrame:
    """Predicted readmission rate (%) for every IMD x age x sex x year combination."""
    rows = [
        {"year": y, "year_start": year_start[y], "imd": q, "age": a, "sex": s}
        for y in years for q in range(1, 6) for a in AGE_BANDS for s in ["Female", "Male"]
    ]
    g = pd.DataFrame(rows)
    g["num"], g["den"] = 0, 1
    g = prepare(g)
    g["pred_pct"] = 100 * res.predict(g)
    return g
