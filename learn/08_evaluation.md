# 08 · Evaluating predictions: discrimination and calibration

🟠 Advanced · [Course index](README.md) · Previous: [07](07_regularisation_imbalance.md) · Next: [09 · Validation and decisions](09_validation_decisions.md)

## The question

A risk model makes two promises. It **ranks** patients correctly (the riskier ones get higher scores), and its **probabilities are right** (of patients given 30%, about 30% are readmitted). These are different properties, and you need to check both.

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
train, test = P.split(cohort)                       # out-of-time: test = 2024/25
champion = P.SavedModel.load(MODELS_DIR / "champion.json")
y, p = test["readmit_30d"].to_numpy(), champion.predict(test)
print(f"{len(test):,} test discharges, observed rate {y.mean():.1%}, mean prediction {p.mean():.1%}")
```

## Discrimination: AUC (the c-statistic)

**AUC = the probability that a randomly chosen readmitted patient gets a higher score than a randomly chosen non-readmitted patient.** 0.5 is a coin toss, 1.0 is perfect. It equals the area under the ROC curve (sensitivity vs 1 − specificity across all thresholds) and is the Mann-Whitney U statistic rescaled.

```python
a, lo, hi = M.auc_ci_delong(y, p)
print(f"AUC {a:.3f} (95% CI {lo:.3f}-{hi:.3f}),  Gini = 2*AUC - 1 = {M.gini(y, p):.3f}")
print(f"best achievable here (true probabilities): {M.auc(y, test['p_true']):.3f}")
```

Banks quote **Gini** (= 2 AUC − 1) for credit scorecards; clinicians quote the **c-statistic** (= AUC). Same information.

AUC is blind to calibration: multiply every prediction by 2 and the AUC does not change.

```python
print(f"AUC of p: {M.auc(y, p):.4f};  AUC of 2p: {M.auc(y, 2 * p):.4f};  AUC of p^3: {M.auc(y, p ** 3):.4f}")
```

## Overall accuracy of probabilities: Brier score and log-loss

* **Brier score** $= \frac{1}{n}\sum (p_i - y_i)^2$: mean squared error of the probabilities (lower is better).
* **Log-loss** $= -\frac{1}{n}\sum [y\log p + (1-y)\log(1-p)]$: the quantity maximum likelihood minimises; punishes confident mistakes hard.

Both are **proper scoring rules**: they are optimised only by the true probabilities, so they reward honesty. Compare with a "know-nothing" model that gives everyone the base rate:

```python
base = np.full_like(p, y.mean())
print(f"Brier: model {M.brier(y, p):.4f} vs base rate {M.brier(y, base):.4f}  "
      f"-> scaled Brier {1 - M.brier(y, p) / M.brier(y, base):.3f}")
```

## Calibration

```python
citl, slope = M.calibration_intercept_slope(y, p)
print(f"calibration-in-the-large {citl:+.3f} (ideal 0);  calibration slope {slope:.3f} (ideal 1)")
print(M.reliability_table(y, p)[["pred", "obs", "n"]].round(3).to_string(index=False))
```

* **Calibration-in-the-large**: are predictions right on average? Fails when the base rate shifts (a new hospital, a new year).
* **Calibration slope**: refit $y$ on logit(p). A slope **below 1** means predictions are too extreme, the classic sign of **overfitting**; above 1 means too timid.
* **Reliability diagram**: observed vs predicted by decile. The most informative single picture.

The **Hosmer-Lemeshow** test compares observed and expected counts by decile. With very large samples it rejects clinically trivial miscalibration; with small samples it misses real problems. Use it alongside the reliability diagram, not instead of it.

## Why calibration matters

A model with an excellent AUC but poor calibration can still rank patients well, but any decision that uses the **probability itself** (a cost threshold, a resource plan, a conversation with a patient) will be wrong. A miscalibrated model can be **recalibrated** without changing its ranking:

```python
from nhsread.models.logit_scratch import logit, sigmoid

p_bad = sigmoid(2.0 * logit(p) + 1.5)          # deliberately over-confident: risks pushed to the extremes
print(f"bad model: AUC {M.auc(y, p_bad):.3f}, mean risk {p_bad.mean():.3f}, slope {M.calibration_intercept_slope(y, p_bad)[1]:.2f}")
cut = len(y) // 2
import statsmodels.api as sm
fit = sm.GLM(y[:cut], sm.add_constant(logit(p_bad[:cut])), family=sm.families.Binomial()).fit()
p_recal = sigmoid(fit.params[0] + fit.params[1] * logit(p_bad[cut:]))
print(f"after logistic recalibration: AUC {M.auc(y[cut:], p_recal):.3f}, slope "
      f"{M.calibration_intercept_slope(y[cut:], p_recal)[1]:.2f}, mean risk {p_recal.mean():.3f}")
```

Recalibration is fitted on the first half and checked on the second half (which is why the AUC differs slightly from the full-sample figure). The ranking is untouched; the slope goes from about 0.5 back to about 1.

This is **Platt scaling**: logistic regression on the model's own logit. It is also how a model is updated for a new hospital using a small local sample.

## Practice

**Q1.** Model A has AUC 0.72 and calibration slope 0.6. Model B has AUC 0.70 and slope 1.0. Which would you deploy for deciding who gets a follow-up call based on a cost threshold?

<details><summary>Answer</summary>

Probably B. A's slope of 0.6 means its predictions are far too extreme (overfitted), so a probability threshold would flag the wrong number of patients. A could be recalibrated, but out of the box B's probabilities can be used directly, at a small cost in ranking.
</details>

**Q2.** Why doesn't the AUC change if you square every prediction?

<details><summary>Answer</summary>

AUC depends only on the ordering of the scores, and squaring (any strictly increasing transform) keeps the order.
</details>

**Q3.** The model's calibration-in-the-large is +0.3 in a new hospital, but the slope is 1. What does that mean and what is the fix?

<details><summary>Answer</summary>

The ranking and spread are fine, but the new hospital's readmission rate is higher than the model expects across the board. Re-estimate the intercept (keep the slopes) using local data.
</details>

**Q4.** What is a "proper scoring rule" and why does it matter for model selection?

<details><summary>Answer</summary>

A score whose expected value is optimised by reporting the true probabilities (log-loss, Brier). Choosing models by a proper score rewards well-calibrated probabilities. Accuracy at a fixed threshold is not proper and can favour worse probability models.
</details>
