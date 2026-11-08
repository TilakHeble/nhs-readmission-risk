# 09 · Validation, thresholds and fairness

🔴 Expert · [Course index](README.md) · Previous: [08](08_evaluation.md) · Next: [10 · Logistic regression in industry](10_industry.md)

## The question

How do you know a model will work on *next year's* patients, and how should a hospital turn a probability into an action?

## Setup

```python
import numpy as np
import pandas as pd

from nhsread.config import MODELS_DIR
from nhsread.evaluation import metrics as M
from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.models import patient as P
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
train, test = P.split(cohort)
champion = P.SavedModel.load(MODELS_DIR / "champion.json")
y, p = test["readmit_30d"].to_numpy(), champion.predict(test)
```

## Levels of validation

| Level | What it tests | Here |
|---|---|---|
| Apparent | performance on the training data (optimistic) | 2022/23-2023/24 |
| Internal (bootstrap / cross-validation) | how much the *modelling process* overfits | Harrell's bootstrap, [notebook 06](../notebooks/06_validation_decisions.ipynb) |
| **Temporal (out-of-time)** | drift over time: new year, new practice | 2024/25 held out |
| External / geographic | a different hospital or population | needs real data from elsewhere |

A random train/test split of the same years mostly tests overfitting. An out-of-time split also tests **drift**, which is what breaks deployed models.

## Harrell's bootstrap optimism, in miniature

```python
rng = np.random.default_rng(0)
dev = train.sample(8_000, random_state=1)      # a small development set, where optimism is visible
formula = P.BASELINE_FORMULA
apparent = M.auc(dev["readmit_30d"], P.fit(dev, formula).predict(dev))
opt = []
for _ in range(20):
    bs = dev.iloc[rng.integers(0, len(dev), len(dev))]
    r = P.fit(bs, formula)
    opt.append(M.auc(bs["readmit_30d"], r.predict(bs)) - M.auc(dev["readmit_30d"], r.predict(dev)))
print(f"apparent AUC {apparent:.3f}, optimism {np.mean(opt):.3f}, corrected {apparent - np.mean(opt):.3f}")
print(f"actual on 2024/25: {M.auc(y, P.fit(dev, formula).predict(test)):.3f}")
```

Each bootstrap refits the **whole** modelling process (including any variable selection) on a resample and measures how much better it looks on its own data than on the original. That average gap is the optimism.

## From probability to decision

A follow-up programme costs £c per patient, prevents a fraction e of readmissions, and a readmission costs £R. Following up a patient with risk p saves $p\,e\,R - c$ on average, so the rule is

$$\text{follow up if } p > t^* = \frac{c}{e\,R}.$$

```python
c, R, e = 150, 2500, 0.20
t_star = c / (e * R)
t_best, curve = M.cost_optimal_threshold(y, p, c, R, e)
print(f"theory threshold {t_star:.0%}, empirical cost-minimising threshold {t_best:.0%}")
print(pd.DataFrame([M.confusion_at(y, p, t) for t in [0.1, 0.2, t_star]])[
    ["threshold", "flagged_pct", "sensitivity", "specificity", "ppv"]].round(3))
```

The theoretical and empirical thresholds agree **only because the model is calibrated**. This is the practical payoff of lesson 08. Note that 0.5 is not special: the right threshold comes from costs and capacity, not from the default `predict()` cut-off.

## Decision curves

Net benefit $= \frac{TP}{n} - \frac{FP}{n}\cdot\frac{t}{1-t}$ puts benefits and harms on one scale at threshold t. A model is useful wherever its net benefit beats both "treat everyone" and "treat no one".

```python
nb = M.net_benefit(y, p, [0.1, 0.2, 0.3])
print(nb.round(4))
```

## Fairness checks

For clinical risk scores the key property is **calibration within groups**: a predicted 30% should mean 30% for every group. Then differences in flag rates reflect differences in need.

```python
rep = M.group_report(y, p, test["imd"].astype(int), t_star)
print(rep[["group", "n", "obs_rate", "mean_pred", "o_to_e", "auc", "flagged_pct"]].round(3).to_string(index=False))
```

Fairness criteria conflict. When base rates differ between groups, a model cannot be calibrated in every group *and* have equal false-positive and false-negative rates in every group (Kleinberg et al., 2016; Chouldechova, 2017). Choose deliberately, and document the choice.

**Dropping a protected variable does not guarantee fairness.** [Notebook 06](../notebooks/06_validation_decisions.ipynb) shows that removing deprivation leaves AUC unchanged but miscalibrates one deprivation group.

## Practice

**Q1.** Why is an out-of-time test more convincing than a random 80/20 split for a model to be deployed next year?

<details><summary>Answer</summary>

A random split draws train and test from the same period and population, so it mostly measures overfitting. An out-of-time split also tests whether relationships and base rates hold as practice, case mix and coding change, which is what happens after deployment.
</details>

**Q2.** Costs: follow-up £100, readmission £3,000, effectiveness 25%. What is the threshold?

<details><summary>Answer</summary>

t* = 100 / (0.25 × 3,000) = 0.133, i.e. follow up patients with predicted risk above about 13%.
</details>

**Q3.** The service only has capacity to follow up 5% of discharges. How do you set the threshold?

<details><summary>Answer</summary>

Use the 95th percentile of predicted risk on recent discharges, so the top 5% are flagged (and monitor it, because the distribution drifts). Capacity constraints override the cost threshold if the cost threshold would flag more than you can serve.
</details>

**Q4.** In your bootstrap you only refit coefficients, but the real process also chose variables with stepwise selection. What goes wrong?

<details><summary>Answer</summary>

Optimism is underestimated. Variable selection is a major source of overfitting, so each bootstrap must repeat the whole procedure, including selection, to capture it.
</details>
