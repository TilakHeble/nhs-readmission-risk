# Learn logistic regression: from first principles to NHS risk models

A ten-lesson course that teaches logistic regression **on this project's data**: real NHS England readmission counts and a synthetic patient cohort anchored to them. Each lesson explains the idea, shows the maths only where it helps, runs real code, and ends with practice questions (answers are hidden under each question).

You do not need to have read the rest of the repository. Each lesson links to the notebook where the same technique is used for real.

## How to use it

```bash
pip install -e ".[dev]"      # from the repository root
python -m nhsread.ingest.build
```

Then read the lessons in order, running the code blocks in a Python session or notebook. Code blocks within a lesson build on each other. CI executes all of them (`tests/test_learn.py`), so they always run.

## Roadmap

| Level | Lesson | You will learn | Used for real in |
|---|---|---|---|
| 🟢 Beginner | [01 · From yes/no outcomes to logistic regression](01_why_logistic.md) | Why not linear regression, odds and log-odds, the sigmoid, a first model | [Notebook 02](../notebooks/02_grouped_logistic.ipynb) |
| 🟢 Beginner | [02 · Fitting by maximum likelihood](02_maximum_likelihood.md) | Likelihood and log-loss, Newton-Raphson / IRLS from scratch, gradient descent | [Notebook 05](../notebooks/05_patient_model.ipynb) |
| 🟢 Beginner | [03 · Interpreting coefficients](03_interpretation.md) | Odds ratios vs risk ratios, marginal effects, dummies, interactions, confounding | [Notebooks 02, 05](../notebooks/02_grouped_logistic.ipynb) |
| 🟡 Intermediate | [04 · Inference](04_inference.md) | Wald, likelihood-ratio and score tests, intervals, deviance, AIC/BIC, false positives | [Notebook 05](../notebooks/05_patient_model.ipynb) |
| 🟡 Intermediate | [05 · Grouped data and over-dispersion](05_grouped_overdispersion.md) | Binomial counts, sufficient statistics, quasi-binomial errors, quasi F-tests | [Notebooks 02, 03](../notebooks/02_grouped_logistic.ipynb) |
| 🟡 Intermediate | [06 · Functional form and diagnostics](06_form_diagnostics.md) | Empirical logits, transforms, splines, residuals, influence, separation | [Notebook 05](../notebooks/05_patient_model.ipynb) |
| 🟠 Advanced | [07 · Regularisation and class imbalance](07_regularisation_imbalance.md) | Ridge, lasso, elastic net, choosing λ, why re-weighting breaks calibration | [Notebook 05](../notebooks/05_patient_model.ipynb) |
| 🟠 Advanced | [08 · Evaluating predictions](08_evaluation.md) | AUC/Gini, Brier, log-loss, calibration slope and intercept, recalibration | [Notebook 06](../notebooks/06_validation_decisions.ipynb) |
| 🔴 Expert | [09 · Validation, thresholds and fairness](09_validation_decisions.md) | Bootstrap optimism, out-of-time tests, cost-based thresholds, decision curves, fairness | [Notebook 06](../notebooks/06_validation_decisions.ipynb) |
| 🔴 Expert | [10 · Logistic regression in industry](10_industry.md) | NHS indicators, credit scorecards (WoE/IV), propensity scores, monitoring (PSI), communication | [docs/](../docs/) |

**Extras**
- [Cheat sheet](cheat_sheet.md): the formulas and tests on two pages
- [Interview questions](interview_questions.md): 40 questions from basic to senior level, with model answers
- [Glossary](glossary.md)

## The running example

**Which patients are at risk of an emergency readmission within 30 days of leaving hospital, and which hospitals have more readmissions than their patients would predict?**

- **Outcome:** readmitted as an emergency within 30 days (1) or not (0)
- **Real data:** NHS England Compendium I02040: readmissions and discharges by deprivation, age, sex, year and trust, 2014/15 to 2024/25
- **Synthetic patients:** 240,000 discharges whose demographic mix and readmission rates match the real data, with clinical features (length of stay, comorbidity, prior A&E use) and known true effects
