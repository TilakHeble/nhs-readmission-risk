# 10 · Logistic regression in industry

🔴 Expert · [Course index](README.md) · Previous: [09](09_validation_decisions.md)

## Why logistic regression is still everywhere

Gradient-boosted trees and neural networks often rank slightly better. Logistic regression stays the default in regulated and high-stakes settings because its coefficients can be read, challenged and signed off; it is naturally well calibrated; it is stable when data drift; and it is cheap to implement anywhere (a formula and a list of coefficients, as in `models/champion.json`).

## Healthcare (NHS)

| Use | How logistic regression appears |
|---|---|
| **Hospital indicators** (SHMI, readmissions) | a national logistic model gives each patient an expected risk; summed by trust it gives E in O/E; funnel plots with over-dispersion flag outliers ([notebook 03](../notebooks/03_funnel_benchmarking.ipynb)) |
| **Risk prediction** | readmission scores (LACE, PARR, QAdmissions), QRISK-style tools; validated with TRIPOD+AI reporting |
| **Research** | adjusted odds ratios in observational studies; case-control studies (where only odds ratios are estimable) |
| **Population health** | modelling inequalities with deprivation, ethnicity and age; SII/RII ([notebook 02](../notebooks/02_grouped_logistic.ipynb)) |

Software that informs care in the NHS falls under clinical safety standards (DCB0129 for manufacturers, DCB0160 for deploying organisations), and may count as a medical device. A readmission model therefore needs a hazard log, a clinical safety case and post-deployment monitoring, not just a good AUC.

## Banking and credit

Credit scorecards are logistic regressions. The **probability of default** (PD) feeds lending decisions, Basel capital and IFRS 9 provisions. The craft is in feature engineering:

* **Binning + Weight of Evidence (WoE).** Each variable is binned and each bin replaced by $\text{WoE} = \log\frac{\%\text{goods}}{\%\text{bads}}$, which makes the logistic relationship monotone and linear by construction.
* **Information Value** $IV = \sum (\%\text{goods} - \%\text{bads}) \times \text{WoE}$ ranks variables.
* **Points scaling.** Log-odds are mapped to a points scale (for example, +20 points doubles the odds).

```python
import numpy as np
import pandas as pd

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
bins = pd.cut(cohort["los_days"], [-1, 0, 1, 3, 6, 13, 200], labels=["0", "1", "2-3", "4-6", "7-13", "14+"])
t = cohort.groupby(bins, observed=True)["readmit_30d"].agg(events="sum", n="size")
t["non_events"] = t["n"] - t["events"]
pe, pn = t["events"] / t["events"].sum(), t["non_events"] / t["non_events"].sum()
t["WoE"] = np.log(pn / pe)            # banking convention: goods over bads; negative = riskier
t["IV part"] = (pn - pe) * t["WoE"]
print(t[["n", "events", "WoE", "IV part"]].round(3))
print(f"Information Value of length of stay: {t['IV part'].sum():.3f}")
```

Rule-of-thumb IV bands used in credit: below 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, above 0.3 strong. Model risk governance (in the UK, the PRA's SS1/23) requires documentation, independent validation and monitoring for these models, as in [Project 1](https://github.com/TilakHeble/uk-credit-stress-engine).

## Tech, government, research and ecology

* **Tech:** click-through and conversion prediction (logistic regression with hashed features was the industry workhorse for ad CTR), churn models, and analysis of A/B tests with covariate adjustment.
* **Government and policy evaluation:** **propensity scores** (the probability of receiving a programme given characteristics) for matching and weighting in observational evaluations.
* **Epidemiology and research:** adjusted odds ratios, conditional logistic regression for matched case-control designs.
* **Ecology:** species presence/absence models (species distribution models) and occupancy models are logistic regressions on habitat covariates.

## Monitoring a deployed model

Models decay. Two standard checks, run every month or quarter:

1. **Population stability index (PSI)** on the score and key inputs, to detect drift in who is being scored.
2. **Calibration-in-the-large and slope** on recent outcomes, once the 30-day follow-up has passed.

```python
def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))  # unique: discrete data repeat quantiles
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


old = cohort.loc[cohort["year"] == "2022/23", "los_days"]
new = cohort.loc[cohort["year"] == "2024/25", "los_days"]
shifted = new + 2  # pretend a new discharge policy lengthened every stay by 2 days
print(f"PSI length of stay, 2022/23 vs 2024/25: {psi(old, new):.3f};  with a 2-day shift: {psi(old, shifted):.3f}")
```

Common reading: PSI below 0.1 stable, 0.1-0.25 investigate, above 0.25 significant shift.

## Communicating a logistic model

* Lead with **absolute risks** ("about 1 in 3 patients like this are readmitted") and percentage-point differences, not odds ratios.
* Show a **calibration plot** before an AUC: managers trust what they can see matching reality.
* State what the model **cannot** see (social support, what happens after discharge), so nobody mistakes a score for a verdict.
* For a scorecard or a ward tool, a points table or a factor chart (as in the dashboard) beats a coefficient table.

## Practice

**Q1.** Why do banks transform variables to Weight of Evidence before fitting a logistic scorecard?

<details><summary>Answer</summary>

WoE-binned variables have a log-odds relationship with the outcome that is linear and monotone by construction, so each variable needs one coefficient, handles missing values and outliers as their own bins, and produces an explainable points table. It also makes coefficients comparable across variables.
</details>

**Q2.** A readmission model had calibration slope 1.0 at launch. A year later the slope is 0.8 and calibration-in-the-large is −0.2. What happened, and what would you do?

<details><summary>Answer</summary>

Predictions have become too extreme and too high on average: case mix or practice has drifted (for example, more same-day emergency care changing who counts as a discharge). First recalibrate (intercept and slope) on recent data; if discrimination has also fallen, refit or redevelop, and update the documentation and safety case.
</details>

**Q3.** Name one reason a hospital might prefer a logistic model with AUC 0.70 over a gradient-boosted model with AUC 0.72.

<details><summary>Answer</summary>

Any of: transparency for clinical safety review and clinicians' trust; easier implementation in the electronic record (a formula); better-behaved calibration and stability under drift; easier fairness auditing; and a 0.02 AUC gain may not change any decision.
</details>

**Q4.** What is a propensity score and why is it usually estimated with logistic regression?

<details><summary>Answer</summary>

The probability that a unit receives a treatment or programme given its characteristics. Matching or weighting on it balances observed characteristics between treated and untreated groups. Logistic regression is the default because the treatment is binary and the probabilities need to be well calibrated for weighting.
</details>
