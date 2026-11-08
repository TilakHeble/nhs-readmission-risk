# Validation report

Independent-style review of the models described in the [model development document](model_development_document.md). Patient-model results are on synthetic data and validate the **method**; they do not establish real-world clinical performance.

## 1. Summary

| Area | Result | Rating |
|---|---|---|
| Data integrity (real) | contracts pass; totals reconcile with England within 2.5% (records with no IMD lookup excluded) | 🟢 |
| Replication of NHS method | expected counts r = 0.996; banding agreement 98.7% | 🟢 |
| Grouped model fit | strong over-dispersion (φ ≈ 50 for M5); handled with quasi-binomial inference | 🟡 (limited case-mix data) |
| Trust benchmarking | Poisson limits unfit for purpose (73% flagged); random-effects limits recommended | 🟢 with adjustment |
| Patient model discrimination | AUC 0.672 vs ceiling 0.676 and LACE 0.650 | 🟢 |
| Patient model calibration | slope 0.98, CITL +0.011, HL p = 0.63 | 🟢 |
| Stability (bootstrap optimism) | AUC optimism 0.0001 | 🟢 |
| Fairness (calibration by group) | O/E 0.99-1.04 across IMD, age, sex | 🟢 |
| Real-world readiness | no real patient-level validation | 🔴 not for clinical use |

## 2. Implementation verification

* Logistic IRLS (NumPy) vs statsmodels: coefficients and SEs agree to < 1e-13 on individual and grouped data; deviance and Pearson χ² agree.
* Gradient descent converges to the MLE (tested).
* AUC, Brier and log-loss match scikit-learn to 1e-12; DeLong interval brackets the AUC.
* Quasi-binomial SEs are larger than binomial SEs (guards against the statsmodels scale pitfall).
* JSON-stored models score identically to the fitted models, including single-row scoring with the spline.
* Dashboard runs end to end in CI (Streamlit AppTest) and responds to inputs.

## 3. Part 1 checks

* **Sufficiency:** grouped and expanded-row fits give identical coefficients.
* **Robustness of the inequality trend:** deprivation × time coefficient −0.0028/year (quasi SE 0.0006); with year × age effects −0.0038 (SE 0.0003). Sign and significance stable.
* **Residuals:** M5 residuals cluster by age within year; addressed by M6. Deprivation conclusions unchanged.
* **Benchmarking persistence:** year-to-year rank correlation of trust O/E 0.77-0.86 across all ten consecutive pairs, supporting a random-effects (systematic variation) interpretation.
* **Flagged trusts:** include specialist and non-acute providers whose case mix is poorly described by age and sex; flags are screening signals.

## 4. Patient model, out-of-time (2024/25, n = 82,771, rate 14.8%)

| Model | AUC (95% CI) | Brier | Log-loss | CITL | Slope |
|---|---|---|---|---|---|
| **Champion** | 0.672 (0.667-0.678) | 0.1186 | 0.3932 | +0.011 | 0.978 |
| Baseline (all linear) | 0.669 (0.664-0.674) | 0.1191 | 0.3945 | +0.012 | 0.978 |
| LACE (recalibrated) | 0.650 (0.645-0.655) | 0.1206 | 0.3996 | +0.009 | 0.984 |
| Oracle (true risk) | 0.676 (0.671-0.682) | 0.1183 | 0.3919 | +0.005 | 0.992 |

* **Internal validation:** Harrell bootstrap (30 resamples): apparent AUC 0.6756, optimism 0.0001. The small further drop on 2024/25 reflects temporal drift rather than over-fitting.
* **Reliability diagram:** observed and predicted rates agree within sampling error in every decile (5% to 35%).
* **Hospital effect:** case-mix-adjusted O/E for 60 synthetic hospitals correlates 0.89 with their true effects; over-dispersion-adjusted funnel flags 6 at 95%.

## 5. Decision analysis

Assumptions: follow-up £150, readmission £2,500, 20% of readmissions prevented among those followed up.

* Cost-optimal threshold 31% (theory 30%): about 6% of discharges flagged; sensitivity 15%, PPV 38%.
* Cost per discharge: no programme £370; targeted £367; follow up everyone £446.
* Decision curve: champion net benefit exceeds LACE and both reference strategies at every threshold from 5% to 50% (below 5%, following up everyone is as good).

**Finding:** the model reliably identifies higher-risk patients, but the financial case is sensitive to intervention effectiveness. Recommend establishing effectiveness locally before investment.

## 6. Fairness

At the cost-based threshold: O/E 0.99-1.04 for every IMD quintile, age band and sex. Flag rates are higher for older and more deprived patients, consistent with higher observed risk. AUC is lowest for children (0.63); a paediatric service would need different predictors.

Removing IMD as a predictor: AUC 0.6722 vs 0.6724, but the least deprived quintile's O/E becomes 1.065. Recommendation: keep IMD, monitor group calibration.

## 7. Conditions for any real-world use

1. Redevelop and validate on real patient-level data (HES or local EHR), including external validation at another trust.
2. Clinical safety case and hazard log (DCB0129/DCB0160); assess medical-device status.
3. Monitoring: monthly PSI on inputs and score; quarterly calibration once 30-day outcomes mature; triggers for recalibration (|CITL| > 0.1 or slope outside 0.85-1.15).
4. Re-estimate costs and effectiveness locally; revisit the threshold annually.
