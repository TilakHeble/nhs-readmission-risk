# 06 · Functional form, diagnostics and things that break

🟡 Intermediate · [Course index](README.md) · Previous: [05](05_grouped_overdispersion.md) · Next: [07 · Regularisation and imbalance](07_regularisation_imbalance.md)

## The question

Logistic regression assumes each predictor acts **linearly on the log-odds**. How do you check that, and what else can go wrong?

## Setup

```python
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.models.logit_scratch import fit_logit
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
```

## 1. Empirical logits: look before you model

Group a predictor into bins, compute log(events / non-events) in each bin, and plot against the predictor. A straight line means a linear term is fine.

```python
def empirical_logit(col, cap):
    g = cohort.assign(x=cohort[col].clip(upper=cap)).groupby("x")["readmit_30d"].agg(["sum", "size"])
    return np.log((g["sum"] + 0.5) / (g["size"] - g["sum"] + 0.5))


el = empirical_logit("los_days", 20)
print("change in log-odds, LOS 0->1, 1->5, 5->20 days:",
      round(el[1] - el[0], 2), round(el[5] - el[1], 2), round(el[20] - el[5], 2))
```

The first extra day adds a lot of log-odds, later days much less per day: the effect is curved, and a **log transform** straightens it.

## 2. Options for non-linear effects

| Approach | Example | Trade-off |
|---|---|---|
| Transform | `np.log1p(los_days)` | simple, one parameter; must guess the shape |
| Categories (bins) | `C(pd.cut(los_days, ...))` | flexible, easy to explain; loses information, steps at cut-offs |
| Polynomial | `los_days + I(los_days**2)` | smooth; wild at the extremes |
| **Spline** | `bs(charlson, knots=[2], degree=2)` | smooth and flexible; harder to explain |

```python
forms = {
    "linear": "los_days",
    "log": "np.log1p(los_days)",
    "binned": "C(pd.cut(los_days, [-1, 0, 1, 3, 6, 13, 200]))",
    "spline": "bs(los_days, knots=[2, 5, 10], degree=3, lower_bound=0, upper_bound=121)",
}
aic = {k: smf.logit(f"readmit_30d ~ {f} + charlson + emergency", data=cohort).fit(disp=0).aic for k, f in forms.items()}
print(pd.Series(aic).sub(min(aic.values())).round(1).rename("AIC minus best"))
```

> **Production pitfall:** `bs(x, df=3)` places knots at quantiles of *whatever data you pass in*. Scoring one patient would then use different knots. Always fix knots and bounds explicitly in a model you will deploy (the project's champion does).

## 3. Residuals

For 0/1 data raw residuals are always either $-p$ or $1-p$, so residual plots look like two bands. Useful versions:

* **Pearson residual** $(y - p)/\sqrt{p(1-p)}$
* **Deviance residual**: signed square root of each observation's deviance contribution
* **Binned residuals**: average residual within bins of predicted risk or of a predictor (a calibration check, lesson 08)

```python
m = smf.logit("readmit_30d ~ los_days + charlson + emergency", data=cohort).fit(disp=0)
binned = cohort.assign(resid=m.resid_response).groupby(cohort["los_days"].clip(upper=15))["resid"].mean()
print("mean residual by length of stay (linear-LOS model):")
print(binned.round(3).to_string())
```

A pattern (negative for short stays, positive for medium, negative again for long) shows the linear term is mis-shaped.

## 4. Influence

A few extreme patients can pull the coefficients. Cook's distance and DFBETAs carry over from linear regression via the IRLS weights.

```python
small = cohort.sample(3000, random_state=1)
mm = sm.GLM(small["readmit_30d"], sm.add_constant(small[["los_days", "charlson"]]), family=sm.families.Binomial()).fit()
infl = mm.get_influence()
cooks = pd.Series(infl.cooks_distance[0], index=small.index)
print(small.loc[cooks.nlargest(3).index, ["los_days", "charlson", "readmit_30d"]].assign(cooks=cooks.nlargest(3).round(4)))
```

The most influential patients have very long stays: a raw linear term gives them enormous leverage. Log-transforming length of stay removes most of the problem.

## 5. Multicollinearity

Correlated predictors inflate standard errors exactly as in linear regression. Variance inflation factors (VIFs) on the design matrix are a quick check; above about 5-10 is a warning.

## 6. Separation: when the maximum likelihood estimate does not exist

If some combination of predictors perfectly predicts the outcome (every patient with a rare flag was readmitted), the log-likelihood keeps increasing as the coefficient goes to infinity. Symptoms: huge coefficients, enormous standard errors, convergence warnings.

```python
rng = np.random.default_rng(5)
n = 300
flag = np.zeros(n)
flag[:6] = 1
y = rng.binomial(1, 0.15, n).astype(float)
y[:6] = 1
X = np.column_stack([np.ones(n), flag])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for it in [5, 20, 40]:
        print(f"after {it} Newton steps: coefficient {fit_logit(X, y, max_iter=it, tol=0).params[1]:.1f}")
print(f"ridge penalty (lambda = 1): coefficient {fit_logit(X, y, l2=1.0).params[1]:.2f}")
```

Remedies: a penalty (ridge, or **Firth's** bias-reduced logistic regression, standard in medical statistics), merging rare categories, or exact logistic regression.

## Practice

**Q1.** Why can't you check linearity by plotting y against x as you would in linear regression?

<details><summary>Answer</summary>

y is only 0 or 1, so a scatter shows two horizontal bands. You need to aggregate (empirical logits or binned rates) and look on the log-odds scale, which is where linearity is assumed.
</details>

**Q2.** A model has a coefficient of 21 with a standard error of 8,000. What is going on?

<details><summary>Answer</summary>

Almost certainly separation: a predictor (or combination) perfectly predicts the outcome in the data, so the MLE is at infinity and the algorithm stopped somewhere large. Use a penalised or Firth model, or collapse the sparse category.
</details>

**Q3.** Binned linear terms are popular in banking scorecards. Give one advantage and one drawback.

<details><summary>Answer</summary>

Advantage: easy to explain and audit, each bin gets its own effect, and monotonic constraints are simple. Drawback: information loss within bins, arbitrary cut-offs, and jumps in predicted risk at the boundaries.
</details>

**Q4.** Your spline model scores well in the notebook but gives different predictions when the IT team scores patients one at a time. Likely cause?

<details><summary>Answer</summary>

Data-dependent knots (e.g. `bs(x, df=3)`), which are recomputed from whatever data is passed in. Fix the knots and boundary knots at training time and store them with the model.
</details>
