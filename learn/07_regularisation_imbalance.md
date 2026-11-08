# 07 · Regularisation and class imbalance

🟠 Advanced · [Course index](README.md) · Previous: [06](06_form_diagnostics.md) · Next: [08 · Evaluating predictions](08_evaluation.md)

## The question

Two common situations: many candidate features (risk of overfitting), and a rare outcome (only ~15% readmitted). What do penalties and re-weighting actually do to a logistic model?

## Setup

```python
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from nhsread.evaluation import metrics as M
from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.models.logit_scratch import sigmoid
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
small = cohort.sample(5_000, random_state=2)   # a single small hospital's worth of data
test = cohort.drop(small.index).sample(50_000, random_state=3)
```

## Penalised likelihood

$$\hat\beta = \arg\max_\beta\ \ell(\beta) - \lambda\Big(\alpha\sum_j|\beta_j| + \tfrac{1-\alpha}{2}\sum_j\beta_j^2\Big)$$

| Penalty | α | Effect |
|---|---|---|
| Ridge (L2) | 0 | shrinks every coefficient towards 0; keeps all; stable with correlated features |
| Lasso (L1) | 1 | sets some coefficients exactly to 0: variable selection |
| Elastic net | between | selection plus stability for groups of correlated features |

In scikit-learn the strength is `C = 1/λ` (smaller C, stronger penalty). **Standardise** features first, otherwise the penalty depends on units. The intercept is not penalised.

## Overfitting in a small sample

```python
feats = ["np.log1p(los_days)", "emergency", "charlson", "prior_ed_6m", "np.log1p(prior_adm_12m)", "care_home",
         "polypharmacy", "medical_specialty", "weekend_discharge", "ward_occupancy", "discharge_hour", "age"]
formula = "readmit_30d ~ (" + " + ".join(feats) + ") ** 2"   # every pairwise interaction: ~78 terms
_, Xs = dmatrices(formula, small, return_type="dataframe")
_, Xt = dmatrices(formula, test, return_type="dataframe")
sc = StandardScaler().fit(Xs.iloc[:, 1:])
Zs, Zt = sc.transform(Xs.iloc[:, 1:]), sc.transform(Xt.iloc[:, 1:])
ys, yt = small["readmit_30d"].to_numpy(), test["readmit_30d"].to_numpy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, model in {
        "no penalty": LogisticRegression(C=1e6, max_iter=5000),
        "ridge C=0.05": LogisticRegression(l1_ratio=0, C=0.05, max_iter=5000),
        "lasso C=0.05": LogisticRegression(l1_ratio=1, C=0.05, solver="liblinear"),
    }.items():
        model.fit(Zs, ys)
        p_tr, p_te = model.predict_proba(Zs)[:, 1], model.predict_proba(Zt)[:, 1]
        print(f"{name:13s} train AUC {M.auc(ys, p_tr):.3f}  test AUC {M.auc(yt, p_te):.3f}  "
              f"test log-loss {M.log_loss(yt, p_te):.4f}  non-zero {np.sum(np.abs(model.coef_) > 1e-8)}")
```

With 5,000 patients and ~78 terms, the unpenalised model fits its training data noticeably better than new data. Penalties trade a little training fit for better performance on new patients.

## Choosing λ by cross-validation

```python
cv = StratifiedKFold(5, shuffle=True, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scores = {C: -cross_val_score(LogisticRegression(l1_ratio=1, C=C, solver="liblinear"), Zs, ys, cv=cv,
                                  scoring="neg_log_loss").mean() for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1]}
print(pd.Series(scores, name="CV log-loss").round(4))
print("best C:", min(scores, key=scores.get))
```

Use **log-loss** (or Brier), not accuracy, to tune a probability model. `LogisticRegressionCV` automates this loop.

## Class imbalance: do not "fix" what is not broken

Readmission is the minority class. A common reflex is to re-weight (`class_weight="balanced"`) or re-sample to 50/50. For logistic regression this only shifts the intercept by about log(weight ratio). **Ranking (AUC) is unchanged; the probabilities become wrong.**

```python
X1 = sc.transform(Xs.iloc[:, 1:])[:, :12]   # main effects only, for clarity
plain = LogisticRegression(max_iter=2000).fit(X1, ys)
bal = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X1, ys)
p_plain, p_bal = plain.predict_proba(X1)[:, 1], bal.predict_proba(X1)[:, 1]
w_ratio = (1 - ys.mean()) / ys.mean()
p_fixed = sigmoid(np.log(p_bal / (1 - p_bal)) - np.log(w_ratio))   # King & Zeng prior correction
print(pd.DataFrame({"plain": [M.auc(ys, p_plain), p_plain.mean()], "balanced": [M.auc(ys, p_bal), p_bal.mean()],
                    "balanced + correction": [M.auc(ys, p_fixed), p_fixed.mean()]},
                   index=["AUC", "mean predicted risk"]).round(3))
print(f"observed rate: {ys.mean():.3f}")
```

When does re-weighting or under-sampling make sense? When the data are so large you must sub-sample (keep all events, sample non-events, then correct the intercept), or with algorithms that genuinely struggle with rare classes. For a calibrated risk score, handle imbalance at the **decision threshold** (lesson 09), not in the fit.

## Practice

**Q1.** Why must features be standardised before ridge or lasso but not before ordinary logistic regression?

<details><summary>Answer</summary>

The penalty sums coefficient sizes, and a coefficient's size depends on its feature's units. Without standardising, features measured in small units (large coefficients) are penalised more. Unpenalised MLE is equivariant to rescaling: the fitted probabilities don't change.
</details>

**Q2.** You under-sample non-readmitted patients to 50/50 to speed up training. What must you do before using the predicted probabilities?

<details><summary>Answer</summary>

Correct the intercept: subtract log(sampling rate of events / sampling rate of non-events), i.e. log of the over-representation factor of events. Otherwise every probability is inflated.
</details>

**Q3.** Lasso dropped a variable your clinicians are sure matters. What could explain it?

<details><summary>Answer</summary>

It may be correlated with variables the lasso kept (which then carry its signal); the lasso picks among correlated features somewhat arbitrarily. Its true effect may also be small relative to the noise at that penalty. Consider elastic net, or forcing it in (unpenalised) for clinical face validity.
</details>

**Q4.** Your colleague reports 85% accuracy for a readmission model. Why is that unimpressive?

<details><summary>Answer</summary>

Predicting "no readmission" for everyone already gives about 85% accuracy when 15% are readmitted. Accuracy at a 0.5 threshold ignores ranking and calibration; use AUC, log-loss/Brier and calibration instead.
</details>
