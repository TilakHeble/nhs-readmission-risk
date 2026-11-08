# 05 · Grouped data, binomial counts and over-dispersion

🟡 Intermediate · [Course index](README.md) · Previous: [04](04_inference.md) · Next: [06 · Functional form and diagnostics](06_form_diagnostics.md)

## The question

NHS England publishes counts, not patients: "in 2024/25, 240,735 of 1,545,450 discharges from the most deprived fifth of areas were readmitted". Can we still fit a logistic regression? Yes, and it is exactly the same model.

## Binomial logistic regression

If cell *c* has $n_c$ discharges and $y_c$ readmissions,

$$y_c \sim \text{Binomial}(n_c, p_c), \qquad \text{logit}(p_c) = x_c^\top\beta .$$

When everyone in a cell shares the same covariates, the cell totals are **sufficient statistics**: fitting to 330 cells gives *identical* estimates to fitting to the 64.7 million patient rows they summarise.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.models import grouped as G
from nhsread.models.logit_scratch import fit_logit

cells = deprivation_cells(read_compendium())
print(cells[["year", "imd", "age", "sex", "num", "den", "rate"]].head())
print(f"{len(cells)} cells, {cells['den'].sum():,} discharges")
```

## Proof by experiment: cells vs individual rows

```python
toy = pd.DataFrame({"x": [0, 0, 1, 1], "num": [30, 50, 70, 60], "den": [200, 250, 260, 180]})
X = np.column_stack([np.ones(4), toy["x"]])
grouped = fit_logit(X, toy["num"], n=toy["den"])

rows = toy.loc[toy.index.repeat(toy["den"])].copy()
rows["y"] = np.concatenate([[1] * a + [0] * (b - a) for a, b in zip(toy["num"], toy["den"])])
individual = sm.Logit(rows["y"], sm.add_constant(rows["x"])).fit(disp=0)
print("grouped:   ", grouped.params.round(6), "\nindividual:", individual.params.to_numpy().round(6))
```

## Fitting the real cells

In statsmodels, pass the response as two columns (successes, failures):

```python
d = G.prepare(cells)
res = G.fit(d, "C(imd_c) + age + sex + C(year)")
print(f"deviance {res.deviance:,.0f} on {res.df_resid:.0f} df;  Pearson chi2 / df = {G.dispersion(res):.1f}")
```

## Over-dispersion

For grouped binomial data the deviance and Pearson chi-square *are* goodness-of-fit statistics: if the model is right, each is roughly equal to its degrees of freedom, so their ratio (the **dispersion**, $\hat\phi$) is about 1.

Here $\hat\phi$ is far bigger than 1. Real cells vary more than binomial sampling allows because each cell pools many different hospitals, places and case mixes. This is **over-dispersion**, and it is the norm in real health and insurance data.

**Consequence:** the binomial standard errors are too small by a factor of about $\sqrt{\hat\phi}$.

**Fixes:**

| Method | How |
|---|---|
| Quasi-binomial | keep the estimates, multiply the variance by $\hat\phi$ |
| Beta-binomial | let each cell's $p$ vary around the model's prediction |
| Random effects (GLMM) | add a random intercept for the source of extra variation (e.g. hospital) |
| Better model | add the missing structure (interactions, more covariates) |

```python
q = G.fit(d, "C(imd_c) + age + sex + C(year)", scale="X2")
print(pd.DataFrame({"binomial SE": res.bse, "quasi SE": q.bse}).head(5).round(5))
print(f"ratio = sqrt(phi) = {np.sqrt(G.dispersion(res)):.2f}")
```

> **Pitfall:** with a two-column response, statsmodels' own `fit(scale="X2")` computes the dispersion on the proportion scale and *shrinks* the errors. `nhsread.models.grouped.fit` computes Pearson chi-square / df itself. Always check that quasi errors come out **bigger**.

Nested quasi-binomial models are compared with an **F-test** (deviance change / df, divided by $\hat\phi$) rather than the chi-square LR test:

```python
small = G.fit(d, "age + sex + C(year)")
big = G.fit(d, "C(imd_c) + age + sex + C(year)")
f, df1, df2, p = G.quasi_f_test(small, big)
print(f"adding deprivation: F({df1}, {df2}) = {f:.1f}, p = {p:.2g}")
```

## Over-dispersion between hospitals

The same idea drives NHS funnel plots ([notebook 03](../notebooks/03_funnel_benchmarking.ipynb)): with no allowance for over-dispersion, most trusts look "significantly" different from England. Spiegelhalter's method estimates the extra between-trust variance and widens the control limits.

## Practice

**Q1.** Your grouped model has deviance 15,100 on 302 residual df. What is the rough dispersion and how much should you inflate the standard errors?

<details><summary>Answer</summary>

φ ≈ 15,100 / 302 ≈ 50 (using deviance; Pearson is usually preferred, but similar). Standard errors grow by √50 ≈ 7.
</details>

**Q2.** Why is the residual deviance a valid goodness-of-fit test for grouped data but not for 0/1 data?

<details><summary>Answer</summary>

Its chi-square approximation needs each observation's expected counts to be reasonably large. With 0/1 data each "cell" has one trial, so the deviance depends only on the fitted values and has no useful reference distribution.
</details>

**Q3.** Over-dispersion does not change the coefficients in a quasi-binomial model. So why does it matter?

<details><summary>Answer</summary>

Inference. Without correction, p-values are far too small and confidence intervals far too narrow, so you will declare effects, or hospital outliers, that are not supported by the data.
</details>

**Q4.** Give one *substantive* cause of over-dispersion in NHS readmission cells.

<details><summary>Answer</summary>

Each cell mixes patients from many hospitals, local authorities and specialties with different underlying readmission risks (different community services, case mix within age bands, coding practices). Variation in the true p across those groups shows up as extra-binomial variation.
</details>
