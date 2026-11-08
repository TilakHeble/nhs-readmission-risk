# %% [markdown]
# # 02 · Grouped logistic regression: who is readmitted, and is the deprivation gap closing?
#
# **Data:** 330 real NHS cells (deprivation quintile x age band x sex x year, 2014/15 to 2024/25),
# each with readmissions out of discharges: 64.7 million discharges in total.
#
# **Model:** binomial logistic regression on grouped data,
#
# $$\text{readmissions}_c \sim \text{Binomial}(n_c, p_c), \qquad \log\frac{p_c}{1-p_c} = x_c^\top\beta .$$
#
# Fitting to the cell counts gives exactly the same estimates as fitting to 64.7 million
# patient rows with the same covariates: the cell totals are *sufficient statistics*.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nhsread import viz
from nhsread.config import CELLS_FILE, YEARS
from nhsread.models import grouped as G
from nhsread.models.logit_scratch import fit_logit

viz.set_style()
pd.set_option("display.width", 140)
cells = pd.read_parquet(CELLS_FILE)
d = G.prepare(cells)
print(f"{len(d)} cells, {d['den'].sum():,} discharges, overall rate {d['num'].sum() / d['den'].sum():.2%}")

# %% [markdown]
# ## 1. Our own IRLS implementation matches statsmodels
# Before trusting a library, check it against first principles. `fit_logit` is Newton-Raphson /
# IRLS written in NumPy (see `src/nhsread/models/logit_scratch.py`).

# %%
from patsy import dmatrices

y, X = dmatrices("num ~ C(imd_c) + age + sex", d, return_type="dataframe")
mine = fit_logit(X.to_numpy(), d["num"].to_numpy(), n=d["den"].to_numpy(), names=list(X.columns))
ref = G.fit(d, "C(imd_c) + age + sex")
pd.DataFrame({"scratch coef": mine.params, "statsmodels coef": ref.params.to_numpy(),
              "scratch se": mine.bse, "statsmodels se": ref.bse.to_numpy()}, index=X.columns).round(6)

# %%
print(f"iterations to converge: {mine.n_iter}; max |coef difference| = {np.max(np.abs(mine.params - ref.params.to_numpy())):.1e}")
print(f"deviance: scratch {mine.deviance:,.1f} vs statsmodels {ref.deviance:,.1f}")

# %% [markdown]
# ## 2. Building the model step by step
#
# Each step adds one idea. We compare by deviance, AIC and, because these data are over-dispersed
# (section 3), by a **quasi-likelihood F-test** that scales the deviance change by the dispersion
# of the larger model.

# %%
ladder = {
    "M0 intercept": "1",
    "M1 + age + sex": "age + sex",
    "M2 + deprivation": "age + sex + C(imd_c)",
    "M3 + year": "age + sex + C(imd_c) + C(year)",
    "M4 + deprivation x age": "age + sex + C(imd_c) * age + C(year)",
    "M5 + age x sex": "C(imd_c) * age + age * sex + C(year)",
    "M6 + year x age": "C(imd_c) * age + age * sex + C(year) * age",
}
fits = {k: G.fit(d, f) for k, f in ladder.items()}
rows, prev = [], None
for k, r in fits.items():
    row = {"model": k, "params": int(r.df_model + 1), "deviance": r.deviance, "resid df": int(r.df_resid),
           "dispersion": G.dispersion(r), "AIC": r.aic}
    if prev is not None:
        f, df1, df2, p = G.quasi_f_test(prev, r)
        row.update({"quasi F": f, "F p-value": p})
    rows.append(row)
    prev = r
ladder_tbl = pd.DataFrame(rows).set_index("model")
ladder_tbl.style.format({"deviance": "{:,.0f}", "dispersion": "{:.1f}", "AIC": "{:,.0f}", "quasi F": "{:.1f}",
                         "F p-value": "{:.2g}"})

# %% [markdown]
# Every step improves the fit significantly, even after allowing for over-dispersion:
#
# * **Deprivation** is a large effect on top of age and sex.
# * **Year** removes a lot of deviance: the national rate drifted up over the decade.
# * **Deprivation x age**: the gradient really does differ by age (notebook 01).
# * **Year x age** is the biggest late gain: the upward drift was not the same for every age group.
#
# We use **M5** as the *interpretation* model (clean deprivation and age effects with a common year
# effect) and keep M6 as a check that the deprivation conclusions do not depend on it.

# %% [markdown]
# ## 3. Over-dispersion: why the textbook standard errors are wrong here
#
# For a correct binomial model, Pearson chi-square / residual df should be about 1.
# Here it is far larger even for the richest model: real cells are aggregates of very
# different places and hospitals, so they vary much more than binomial sampling allows.
#
# The fix is **quasi-binomial** inference: keep the same coefficients but multiply the variance
# by the dispersion $\hat\phi$ (standard errors grow by $\sqrt{\hat\phi}$).
#
# > **Pitfall found while building this:** with a two-column (successes, failures) response,
# > `statsmodels` `fit(scale="X2")` estimates the dispersion on the *proportion* scale and returns
# > a tiny value, which shrinks the standard errors instead of inflating them. `grouped.fit`
# > computes Pearson chi-square / df itself and refits with that scale (there is a unit test).

# %%
m5 = fits["M5 + age x sex"]
q5 = G.fit(d, ladder["M5 + age x sex"], scale="X2")
phi = G.dispersion(m5)
cmp = pd.DataFrame({"binomial SE": m5.bse, "quasi SE": q5.bse, "ratio": q5.bse / m5.bse}).iloc[:6]
print(f"dispersion phi = {phi:.1f}  ->  SEs inflate by sqrt(phi) = {np.sqrt(phi):.2f}")
cmp.round(5)

# %% [markdown]
# With 64.7 million discharges, the ordinary binomial errors are microscopic and every effect looks
# "certain". The quasi-binomial errors are about 7 times bigger, which is still small, because the
# data really are huge, but honest.

# %% [markdown]
# ## 4. Interpreting the coefficients
#
# A logistic coefficient is a change in **log-odds**; $e^\beta$ is an **odds ratio** (OR).
# With a 14% outcome, odds ratios overstate relative risks a little (OR 1.20 is roughly RR 1.17),
# so for communication we also report **average marginal effects** in percentage points.

# %%
ors = G.odds_ratio_table(q5)
ors = ors.loc[[i for i in ors.index if not i.startswith("C(year)")]]
ors[["odds_ratio", "or_lo", "or_hi", "p"]].style.format({"odds_ratio": "{:.3f}", "or_lo": "{:.3f}", "or_hi": "{:.3f}",
                                                         "p": "{:.1e}"})

# %% [markdown]
# Reference category: most deprived quintile (1), age 16-74, female, 2014/15. Because deprivation
# interacts with age, the `C(imd_c)[T.k]` rows are the deprivation effects **for 16-74 year olds**;
# the interaction rows say how much flatter (or steeper) the gradient is for children and 75+.

# %% [markdown]
# M5 assumes one deprivation effect for the whole decade. Section 6 shows the gap has narrowed, so
# for *current* effects we refit the same structure on 2024/25 cells only, then average the
# predicted change over that year's discharges.

# %%
lat = d[d["year"] == "2024/25"].copy()
m_lat = G.fit(lat, "C(imd_c) * age + age * sex")
ame = {
    "most (1) vs least (5) deprived": G.average_marginal_effect(m_lat, lat, "imd_c", (5, 1)),
    "age 75+ vs 16-74": G.average_marginal_effect(m_lat, lat, "age", ("16-74", "75+")),
    "male vs female": G.average_marginal_effect(m_lat, lat, "sex", ("Female", "Male")),
    "(decade-average M5) most vs least deprived": G.average_marginal_effect(m5, lat, "imd_c", (5, 1)),
}
ame_s = pd.Series(ame, name="average marginal effect, percentage points (2024/25 discharges)")
ame_s.round(2).to_frame()

# %%
extra = ame["most (1) vs least (5) deprived"] / 100 * lat["den"].sum()
print(f"implied extra readmissions if every 2024/25 discharge had most- rather than least-deprived risk: {extra:,.0f}")

# %% [markdown]
# For the 2024/25 discharge population, moving everyone from the least to the most deprived
# quintile adds about **2.4 percentage points**, roughly 156,000 extra readmissions a year on
# 6.4 million discharges. Being 75+ rather than 16-74 adds about 4.4 points.
#
# Two lessons sit in this table:
#
# * The adjusted deprivation effect (2.4 points) is **bigger than the crude gap** (1.7 points in
#   notebook 01) because discharges from deprived areas are younger on average, and younger
#   patients have lower risk. Age partly masks the inequality: classic confounding.
# * The decade-average model says 3.1 points. Using a pooled model for a *current* estimate when the
#   effect is changing would overstate today's gap.

# %% [markdown]
# ## 5. The fitted interaction, as a picture

# %%
pred = G.predicted_grid(m5, ["2024/25"], {"2024/25": 2024})
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for a, col in viz.AGE_COLOURS.items():
    s = pred[pred["age"] == a].groupby("imd")["pred_pct"].mean()
    ax.plot(s.index, s.values, marker="o", color=col, label=f"age {a}")
    ax.annotate(f"{s.iloc[0]:.1f}%", (1, s.iloc[0]), xytext=(-34, 0), textcoords="offset points", va="center",
                fontsize=9, color=col)
ax.set_xticks(range(1, 6), ["1 most\ndeprived", "2", "3", "4", "5 least\ndeprived"])
ax.set_xlim(0.6, 5.3)
ax.set_ylabel("predicted readmission rate (%)")
ax.legend(loc="upper right")
ax.set_title("Model M5: predicted 2024/25 readmission rate by deprivation and age")
viz.subtitle(ax, "Average of male and female predictions; the gradient is far flatter for children")
viz.save(fig, "02_interaction")
fig

# %% [markdown]
# ## 6. Is the deprivation gap closing?
#
# UK public-health reporting summarises inequality with the **Slope Index of Inequality (SII)**:
# the modelled absolute gap (percentage points) between the most and least deprived ends of the
# population, and the **Relative Index of Inequality (RII)**: the same thing as a ratio, here as an
# odds ratio from a logistic fit on the population *ridit* (cumulative share) of deprivation.

# %%
ineq = pd.DataFrame({y: G.inequality_indices(cells[cells["year"] == y]) for y in YEARS}).T
ineq[["rate_most", "rate_least", "gap_pp", "sii_pp", "rii_or"]].round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(YEARS))
ax = axes[0]
ax.plot(x, ineq["rate_most"], marker="o", color=viz.IMD_COLOURS[1], label="most deprived fifth")
ax.plot(x, ineq["rate_least"], marker="o", color=viz.IMD_COLOURS[5], label="least deprived fifth")
ax.set_xticks(x[::2], [YEARS[i] for i in x[::2]], fontsize=8.5)
ax.set_ylabel("crude readmission rate (%)")
ax.legend(loc="lower right")
ax.set_title("Rates by deprivation fifth", fontsize=11, pad=6)
ax = axes[1]
ax.plot(x, ineq["rii_or"], marker="o", color=viz.BLUE)
ax.set_xticks(x[::2], [YEARS[i] for i in x[::2]], fontsize=8.5)
ax.set_ylabel("Relative Index of Inequality (odds ratio)")
ax.axhline(1, color=viz.GREY, lw=1, ls="--")
ax.set_ylim(0.95, 1.45)
ax.set_title("Relative inequality", fontsize=11, pad=6)
for i in [0, len(YEARS) - 1]:
    ax.annotate(f"{ineq['rii_or'].iloc[i]:.2f}", (i, ineq["rii_or"].iloc[i]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9)
viz.fig_title(fig, "The deprivation gap in readmissions has narrowed since 2014/15",
              "Mostly because rates rose faster in the least deprived areas, not because they fell in the most deprived")
viz.save(fig, "02_inequality_trend")
fig

# %% [markdown]
# ### A formal test of the narrowing
# Replace the deprivation categories with a linear deprivation score (1 = least ... 5 = most
# deprived) and let its slope change over time. The `deprived:t` coefficient is the yearly change
# in the log-odds gradient; quasi-binomial errors keep the test honest.

# %%
trend = G.fit(d, "deprived + deprived:t + deprived:age + age * sex + C(year)", scale="X2")
tt = G.odds_ratio_table(trend).loc[["deprived", "deprived:t"]]
tt[["coef", "se", "p"]].style.format({"coef": "{:.4f}", "se": "{:.4f}", "p": "{:.2g}"})

# %%
b0, b1 = trend.params["deprived"], trend.params["deprived:t"]
print(f"odds ratio per deprivation step (16-74s): 2014/15 {np.exp(b0):.3f} -> 2024/25 {np.exp(b0 + 10 * b1):.3f}")

# %%
trend6 = G.fit(d, "deprived + deprived:t + deprived:age + age * sex + C(year) * age", scale="X2")
r6 = G.odds_ratio_table(trend6).loc["deprived:t"]
print(f"robustness (year x age effects as in M6): deprived:t = {r6['coef']:.4f} (quasi SE {r6['se']:.4f}, p = {r6['p']:.1g})")

# %% [markdown]
# The gradient shrinks year on year by about five quasi-binomial standard errors' worth, and the
# result survives letting every age band have its own year effects (as in M6), so it is not an
# artefact of how the time trend is modelled.
#
# **Caution on interpretation.** This is an *association* in aggregated data. A narrowing gap could
# reflect changes in who is admitted (for example, more short-stay emergency care in less deprived
# areas), coding practice, or genuine improvement. The data cannot separate these: an ecological
# analysis says where to look, not why.

# %% [markdown]
# ## 7. Where does the model fit badly?
# Pearson residuals by cell (from M5). Large values show the cells the model cannot explain.

# %%
d["fitted"] = m5.fittedvalues
d["pearson"] = m5.resid_pearson
worst = d.reindex(d["pearson"].abs().sort_values(ascending=False).index).head(8)
worst[["year", "imd", "age", "sex", "rate", "fitted", "pearson"]].round(4)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for a, col in viz.AGE_COLOURS.items():
    s = d[d["age"] == a]
    ax.scatter(s["year_start"] + {"<16": -0.2, "16-74": 0, "75+": 0.2}[a], s["pearson"], s=14, color=col,
               alpha=0.7, label=f"age {a}")
ax.axhline(0, color=viz.GREY, lw=1)
ax.set_xticks(range(2014, 2025), YEARS, fontsize=8)
ax.set_ylabel("Pearson residual")
ax.legend(ncol=3, loc="lower left")
ax.set_title("Model M5 residuals by year and age band")
viz.subtitle(ax, "Residuals move together by age within a year: the age-specific trend that M6 adds")
viz.save(fig, "02_residuals")
fig

# %% [markdown]
# The residuals cluster by age within years: the national drift was different for children,
# working-age adults and the over-75s. That is exactly what the `year x age` term in M6 captures,
# and why its deviance gain was so large.
#
# **Takeaways**
#
# * Grouped logistic regression on 330 real cells reproduces patient-level estimates at a fraction
#   of the cost, and our from-scratch IRLS matches statsmodels to machine precision.
# * Deprivation, age, sex and year all matter; deprivation matters **more for adults than children**.
# * The data are heavily over-dispersed, so binomial standard errors overstate certainty about
#   7-fold; quasi-binomial inference fixes this (and exposed a statsmodels pitfall).
# * The deprivation gap in readmissions has **narrowed** since 2014/15 (RII 1.38 to 1.19),
#   mainly because rates rose faster in the least deprived areas.
