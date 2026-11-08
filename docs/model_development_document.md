# Model development document

Structured after the TRIPOD+AI reporting items for prediction models, with additions for the population-level (grouped) models.

## 1. Purpose and scope

| Model | Purpose | Data | Status |
|---|---|---|---|
| **Grouped logistic (M5/M6)** | quantify readmission differences by deprivation, age, sex and year; test the trend in inequality | real NHS cells | analytical |
| **Trust benchmarking** | indirectly standardised O/E ratios with over-dispersion-adjusted funnel limits | real NHS trust counts | analytical / screening |
| **Patient risk model (champion)** | 30-day emergency readmission probability at discharge, to target follow-up | synthetic cohort | demonstration only; not for clinical use |

Intended user of the patient model: a discharge-planning team deciding who receives post-discharge follow-up. Out of scope: diagnosis, individual treatment decisions, use on populations unlike English NHS acute discharges.

## 2. Data

### 2.1 Real NHS data
NHS England, Compendium indicator I02040 (2025 release), Open Government Licence v3.0. 155,573 rows; financial years 2014/15 to 2024/25.

* **Outcome:** emergency readmission to any English hospital within 30 days of discharge.
* **Denominator:** discharges as defined by the NHS England indicator specification (see the Compendium methodology for exclusions).
* **Disclosure control:** sub-national counts rounded to the nearest 5; small counts suppressed (11.6% of sub-national rows).
* **Data contracts** (`src/nhsread/ingest/contracts.py`): Pandera schemas check all 11 years are present, counts are non-negative, readmissions ≤ discharges, and the 330-cell deprivation grid is complete. CI rebuilds processed data from the raw file on every push.

### 2.2 Synthetic patient cohort
240,000 discharges, 2022/23 to 2024/25 (`src/nhsread/synth/simulate.py`, seeded):

1. Demographic cells (year × IMD quintile × age band × sex) sampled in proportion to real NHS denominators.
2. Clinical features drawn conditional on age and deprivation.
3. Outcome from a known logistic model with a hospital random intercept (sd 0.15).
4. Per-cell intercepts solved by root-finding so every cell's mean risk equals the real NHS rate (max error 0.0004).

True effects (log-odds): log(1+LOS) 0.35; emergency 0.60; Charlson 0.30/point to 4 then 0.10; Charlson × under-75 0.05; A&E visits 0.17/visit (cap 6); log(1+prior admissions) 0.50; care home 0.35; polypharmacy 0.25; medical specialty 0.20; weekend 0.08; ward occupancy and discharge hour 0 (noise).

## 3. Part 1 methods: grouped logistic regression

Binomial GLM with logit link on 330 cells (`src/nhsread/models/grouped.py`); cell totals are sufficient statistics, so estimates equal those of a patient-level model with the same covariates. Our NumPy IRLS implementation matches statsmodels to 1e-15.

**Model ladder** (deviance, residual df, Pearson dispersion):

| Model | Terms | Deviance | df | φ |
|---|---|---|---|---|
| M1 | age + sex | 112,102 | 326 | 340 |
| M2 | + IMD | 54,337 | 322 | 168 |
| M3 | + year | 22,060 | 312 | 70 |
| M4 | + IMD × age | 16,968 | 304 | 56 |
| **M5** | + age × sex | 15,111 | 302 | 50 |
| M6 | + year × age | 4,513 | 282 | 16 |

Every step is significant by a quasi-likelihood F-test. M5 is the interpretation model; M6 is a robustness check.

**Over-dispersion.** φ ≫ 1 in every model, so inference is **quasi-binomial** (SEs × √φ ≈ 7 for M5). Implementation note: statsmodels' `fit(scale="X2")` with a two-column response returns a dispersion of about 0.0005 (proportion scale), shrinking SEs; we compute Pearson χ²/df explicitly and test this.

**Inequality.** SII and RII per year from weighted linear and logistic fits on the population ridit of deprivation. Trend test: linear deprivation score × time, with year fixed effects, quasi-binomial errors; repeated with year × age effects.

**Current effects.** Average marginal effects for 2024/25 come from M5's structure refitted on 2024/25 cells only, because the pooled model's deprivation effect (3.1 pp) overstates the current gap (2.4 pp).

## 4. Part 1 methods: trust benchmarking

* **Expected counts:** indirect standardisation. Rebuilt from six age × sex strata; correlation 0.996 with published expected counts (`src/nhsread/funnel.py`).
* **Intervals and banding:** Byar's approximation to exact Poisson intervals for O; banding by comparing 95%/99.8% intervals with 1. Agreement with published bands: 98.7% (2024/25).
* **Over-dispersion:** Spiegelhalter (2005) on the square-root scale: z = 2(√O − √E), 10% winsorisation, φ = mean z², additive τ² = (nφ − (n−1)) / (Σw − Σw²/Σw), w = 4E. Limits (1 ± z√(1/(4E) + τ²))².
* **Shrinkage and ranks:** empirical-Bayes shrinkage on the sqrt scale; 4,000 posterior draws for rank intervals.

2024/25: φ = 62.6, τ = 0.050.

## 5. Part 3 methods: patient risk model

* **Split:** develop on 2022/23-2023/24 (157,229 discharges); validate out-of-time on 2024/25 (82,771).
* **Candidate predictors:** available at discharge only (no post-discharge information, avoiding leakage).
* **Functional form:** guided by empirical logits, chosen by AIC and LR tests: log(1+LOS), A&E visits capped at 6, log(1+prior admissions), quadratic B-spline in Charlson with a **fixed** interior knot at 2 and bounds [0, 9] (data-dependent knots would change at scoring time), Charlson × under-75.
* **Selection:** noise variables dropped (joint LR p = 0.27; BIC lower without them). Lasso, ridge and elastic net on ~80 expanded features were compared by cross-validated log-loss; none beat the 19-term champion.
* **Imbalance:** no re-weighting (15% prevalence). Class-balanced weights were shown to inflate mean predicted risk from 15% to 47% with identical AUC.
* **Final model** (`models/champion.json`, 20 coefficients): stored as formula + coefficients in JSON; scoring reproduces statsmodels to 1e-10 and works for single patients (tested).

**Coefficient recovery (training data):** all nine directly comparable true effects lie inside their 95% intervals (e.g. emergency 0.602 vs 0.60; log LOS 0.345 vs 0.35).

**Comparators:** naive all-linear baseline (`baseline.json`); LACE index recalibrated as a one-variable logistic model (`lace.json`).

## 6. Assumptions and limitations

| Area | Assumption / limitation | Impact |
|---|---|---|
| Part 1 case mix | age and sex only | trust over-dispersion partly reflects unadjusted specialty and diagnosis mix |
| Part 1 design | ecological | associations between areas, not individuals |
| Rounding | counts rounded to 5 | ~1% banding disagreement at limits |
| Part 3 data | synthetic clinical effects | performance figures demonstrate method, not real accuracy |
| Hospital effect | unseen by the patient model | caps achievable AUC; motivates local recalibration |
| Costs | illustrative | threshold and savings are scenario-dependent |

## 7. Reproducibility

```bash
pip install -e ".[dev,dashboard]"
python -m nhsread.ingest.build          # processed data + contracts
make notebooks                          # executes notebooks 01-06
pytest -q                               # 35 tests incl. every lesson code block and the dashboard
```
