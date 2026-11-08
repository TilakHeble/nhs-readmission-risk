# 01 · From yes/no outcomes to logistic regression

🟢 Beginner · [Course index](README.md) · Next: [02 · Fitting by maximum likelihood](02_maximum_likelihood.md)

## The question

A patient is about to leave hospital. Will they be back as an emergency within 30 days? The outcome is **yes or no** (1 or 0). We want a model that turns what we know about the patient into a **probability** between 0 and 1.

## Load the data

The course uses this project's data. Real NHS England readmission rates are used for the grouped lessons, and a synthetic cohort anchored to them for patient-level lessons (see [notebook 04](../notebooks/04_synthetic_cohort.ipynb) for why).

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.synth.simulate import simulate_cohort

cells = deprivation_cells(read_compendium())   # real NHS cells
cohort = simulate_cohort(cells)                # 240,000 synthetic discharges
print(cohort[["age", "los_days", "charlson", "prior_adm_12m", "readmit_30d"]].head())
print(f"readmission rate: {cohort['readmit_30d'].mean():.1%}")
```

## Why not ordinary linear regression?

You *can* regress a 0/1 outcome on predictors with OLS. This is the **linear probability model** (LPM), and it is easy to read: each coefficient is a change in probability.

```python
lpm = smf.ols("readmit_30d ~ charlson + prior_adm_12m + los_days", data=cohort).fit()
p_lpm = lpm.predict(cohort)
print(lpm.params.round(4))
print(f"predicted probabilities range from {p_lpm.min():.3f} to {p_lpm.max():.3f}")
```

Three problems:

1. **Impossible predictions.** A straight line is unbounded. Push the predictors far enough (a long stay, high comorbidity, many prior admissions) and the "probability" goes above 1; with negative slopes it can go below 0. Even in this cohort the highest LPM prediction is just above 1.
2. **The wrong shape.** Real risks bend: going from 2% to 4% is a doubling, but from 90% to 92% is barely a change. A constant slope cannot capture that.
3. **Unequal variance.** A 0/1 outcome has variance $p(1-p)$, which changes with $p$, so OLS standard errors are wrong unless corrected.

The LPM is still useful as a quick approximation when probabilities stay between about 0.2 and 0.8. For risk scores, we need something that respects the 0 to 1 range.

## Odds and log-odds

Probabilities live in $[0, 1]$. Two transformations stretch them onto the whole number line:

| Quantity | Formula | Range | Example: p = 0.15 |
|---|---|---|---|
| probability | $p$ | 0 to 1 | 0.15 |
| odds | $p / (1-p)$ | 0 to ∞ | 0.176 ("about 1 to 5.7") |
| log-odds (logit) | $\log\frac{p}{1-p}$ | −∞ to ∞ | −1.73 |

Because the log-odds can be any number, we can safely model it as a straight line.

```python
for p in [0.01, 0.15, 0.5, 0.85, 0.99]:
    odds = p / (1 - p)
    print(f"p = {p:.2f}   odds = {odds:7.3f}   log-odds = {np.log(odds):+.2f}")
```

## The logistic regression model

$$\log\frac{p_i}{1-p_i} = \beta_0 + \beta_1 x_{i1} + \dots + \beta_k x_{ik}
\quad\Longleftrightarrow\quad
p_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{i1} + \dots)}}$$

The right-hand function is the **sigmoid** (or logistic) function. It turns any number into a probability.

```python
from nhsread.models.logit_scratch import sigmoid

z = np.array([-4, -2, 0, 2, 4])
print(dict(zip(z.tolist(), sigmoid(z).round(3).tolist())))
```

## A first model

```python
m = smf.logit("readmit_30d ~ charlson", data=cohort).fit(disp=0)
print(m.params.round(3))
b0, b1 = m.params
for score in [0, 2, 4, 6]:
    print(f"Charlson {score}: predicted risk {sigmoid(b0 + b1 * score):.1%}   "
          f"observed {cohort.loc[cohort['charlson'] == score, 'readmit_30d'].mean():.1%}")
```

Each extra Charlson point adds the same amount (β₁) to the **log-odds**. On the probability scale the steps get bigger as risk rises, which is the S-shape. Lesson 03 shows how to explain these numbers to a clinician.

## Where it is used for real

- [Notebook 02](../notebooks/02_grouped_logistic.ipynb): logistic regression on real NHS cells
- [Notebook 05](../notebooks/05_patient_model.ipynb): the patient-level risk model

## Practice

**Q1.** A patient's predicted readmission probability is 0.20. What are the odds and the log-odds?

<details><summary>Answer</summary>

Odds = 0.20 / 0.80 = 0.25 (1 to 4). Log-odds = log(0.25) ≈ −1.39.
</details>

**Q2.** Why is the linear probability model often acceptable for an A/B test with conversion rates around 40-60%, but not for a readmission risk score?

<details><summary>Answer</summary>

Near the middle of the range the logistic curve is almost straight, so the LPM approximates it well and its coefficients are easy to read as percentage-point effects. A risk score must produce valid probabilities across the whole range, including very low- and very high-risk patients, where the LPM bends the wrong way or leaves [0, 1].
</details>

**Q3.** The log-odds rises by β₁ = 0.42 per Charlson point. Does the probability rise by the same amount for every point?

<details><summary>Answer</summary>

No. A constant change in log-odds gives a bigger change in probability near p = 0.5 and smaller changes near 0 or 1. The probability change for a small step is approximately β₁·p(1−p), which is largest at p = 0.5.
</details>

**Q4.** What is the probability when the log-odds is 0?

<details><summary>Answer</summary>

0.5, because sigmoid(0) = 1 / (1 + 1) = 0.5. The odds are 1 (evens).
</details>
