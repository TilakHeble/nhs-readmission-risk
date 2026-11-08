# 04 · Inference: standard errors, tests and model comparison

🟡 Intermediate · [Course index](README.md) · Previous: [03](03_interpretation.md) · Next: [05 · Grouped data and over-dispersion](05_grouped_overdispersion.md)

## The question

Is the effect of a weekend discharge real, or noise? Does adding an interaction improve the model enough to justify it?

## Setup

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
sample = cohort.sample(20_000, random_state=3)   # a hospital-sized sample makes tests less trivially significant
base = "emergency + charlson + np.log1p(los_days) + np.log1p(prior_adm_12m) + C(age_band) + sex"
m_small = smf.logit(f"readmit_30d ~ {base}", data=sample).fit(disp=0)
m_big = smf.logit(f"readmit_30d ~ {base} + weekend_discharge + ward_occupancy + discharge_hour", data=sample).fit(disp=0)
print(m_big.summary2().tables[1].round(3))
```

## Standard errors come from the curvature

The inverse of the Fisher information $(X^\top W X)^{-1}$ (lesson 02) estimates the covariance of $\hat\beta$. Its square-rooted diagonal gives the standard errors. These are **asymptotic**: they rely on a large sample and a correctly specified model.

## Three classic tests

| Test | Idea | Needs |
|---|---|---|
| **Wald** | is $\hat\beta$ far from 0 relative to its SE? $z = \hat\beta / SE$ | the big model only |
| **Likelihood ratio (LR)** | does the log-likelihood drop a lot when we remove the term? $LR = 2(\ell_{big} - \ell_{small}) \sim \chi^2_{df}$ | both models |
| **Score (Lagrange multiplier)** | is the slope of the log-likelihood at the restricted estimate far from 0? | the small model only |

They agree in large samples. When they differ, prefer the **LR test**: it does not depend on how the model is parameterised and behaves better near boundaries and with sparse data.

```python
lr = 2 * (m_big.llf - m_small.llf)
df = m_big.df_model - m_small.df_model
print(f"LR test for the 3 added variables: LR = {lr:.2f}, df = {df:.0f}, p = {stats.chi2.sf(lr, df):.3f}")

R = np.zeros((3, len(m_big.params)))
for i, name in enumerate(["weekend_discharge", "ward_occupancy", "discharge_hour"]):
    R[i, list(m_big.params.index).index(name)] = 1
print(m_big.wald_test(R, scalar=True))
```

### A false positive, caught in the act

Of the three added variables, only `weekend_discharge` has a real (small) effect. `ward_occupancy` and `discharge_hour` have **no effect at all**: the synthetic data were generated without them ([notebook 04](../notebooks/04_synthetic_cohort.ipynb)). Test those two on their own, in this sample and in a fresh one:

```python
def noise_test(data):
    keep = smf.logit(f"readmit_30d ~ {base} + weekend_discharge", data=data).fit(disp=0)
    add = smf.logit(f"readmit_30d ~ {base} + weekend_discharge + ward_occupancy + discharge_hour", data=data).fit(disp=0)
    lr = 2 * (add.llf - keep.llf)
    return lr, stats.chi2.sf(lr, 2)


fresh = cohort.drop(sample.index).sample(20_000, random_state=11)
for name, data in [("this sample", sample), ("fresh sample", fresh)]:
    lr, p = noise_test(data)
    print(f"{name}: LR = {lr:.2f}, p = {p:.3f}")
```

In our sample the two pure-noise variables come out "significant" (p ≈ 0.01); in a fresh sample of the same size they are not (p ≈ 0.1). That is what a p-value means: with no real effect, results this extreme still turn up about 1 time in 100, and the more variables and samples you try, the more often it happens. A single significant p-value for a variable with no prior reason to matter is weak evidence: confirm on independent data, correct for multiple testing, or let cross-validation decide.

## Confidence intervals

The default interval is the **Wald interval**, $\hat\beta \pm 1.96\,SE$, exponentiated for an odds ratio (giving an asymmetric interval on the OR scale). **Profile-likelihood** intervals invert the LR test and are more accurate in small samples or near separation.

```python
ci = m_big.conf_int().loc[["weekend_discharge", "emergency"]]
print(np.exp(ci).rename(columns={0: "OR low", 1: "OR high"}).round(3))
```

## Deviance

The **deviance** is $-2\times$ the log-likelihood (relative to a perfect "saturated" model). Differences in deviance between nested models are LR statistics. For individual 0/1 data the deviance itself is **not** a goodness-of-fit test (it has no useful reference distribution); for grouped data it is (lesson 05).

## Comparing non-nested models: AIC and BIC

$$\text{AIC} = -2\ell + 2k, \qquad \text{BIC} = -2\ell + k\log n$$

Lower is better. BIC penalises extra parameters more once $n > 7$, so it tends to choose smaller models.

```python
log_los = smf.logit(f"readmit_30d ~ {base}", data=sample).fit(disp=0)
raw_los = smf.logit(f"readmit_30d ~ {base.replace('np.log1p(los_days)', 'los_days')}", data=sample).fit(disp=0)
print(pd.DataFrame({"AIC": [log_los.aic, raw_los.aic], "BIC": [log_los.bic, raw_los.bic]},
                   index=["log(1 + LOS)", "raw LOS"]).round(1))
```

## Statistical vs practical significance

With 240,000 patients almost any effect is "significant". With 20,000 some are not. Always ask two questions: is the effect **distinguishable from zero**, and is it **big enough to matter**? Report effect sizes with intervals, not just p-values.

## Practice

**Q1.** A Wald z-statistic is 2.4. What is the two-sided p-value approximately, and what is the LR statistic likely to be?

<details><summary>Answer</summary>

p ≈ 0.016. In large samples the LR statistic is close to z² ≈ 5.8 (chi-square with 1 df).
</details>

**Q2.** Why can't you use an LR test to choose between "log length of stay" and "raw length of stay"?

<details><summary>Answer</summary>

The models are not nested: neither is a special case of the other. Compare them with AIC/BIC or cross-validated log-loss instead.
</details>

**Q3.** An odds ratio's Wald 95% interval is (0.98, 1.31). A colleague reports "no effect". Is that right?

<details><summary>Answer</summary>

Not quite. The data are compatible with anything from a 2% reduction to a 31% increase in odds. "No significant effect" is not "no effect"; the interval shows a meaningful increase is plausible and a larger sample might detect it.
</details>

**Q4.** Your model's standard errors look tiny. Name two reasons they might be too small.

<details><summary>Answer</summary>

(1) Clustering: patients in the same hospital are correlated, so the effective sample is smaller (use cluster-robust errors or a random effect). (2) Over-dispersion in grouped data (lesson 05). Also model misspecification and treating repeat admissions of the same patient as independent.
</details>
