# Executive summary: NHS 30-day emergency readmissions

**Questions**
1. Which NHS trusts have more emergency readmissions than their patients would predict, and how confident can we be?
2. Who is readmitted, and is the gap between deprived and affluent areas closing?
3. How would a hospital build, validate and use a patient-level readmission risk score?

**Data.** Parts 1 and 2 use the real NHS England *Compendium: Emergency readmissions to hospital within 30 days of discharge* (I02040, 2025 release): 64.7 million discharges, 2014/15 to 2024/25, by trust, deprivation, age, sex and specialty. Patient-level hospital records (HES) are not public, so Part 3 uses a 240,000-patient **synthetic** cohort whose demographic mix and readmission rates match the real data exactly, with assumed clinical effects.

## Key findings

**1. Published trust banding flags almost three-quarters of trusts; a fairer method flags six.**
In 2024/25, 113 of 154 NHS trusts (73%) sit outside the 99.8% limits of NHS England's standard banding. We rebuilt that method from the published counts: expected counts correlate 0.996 with NHS England's, and banding agrees for 98.7% of trusts. The cause of the mass flagging is **over-dispersion**: trusts differ far more than chance allows (dispersion φ ≈ 63), so Poisson limits are far too narrow for trusts with tens of thousands of discharges. With SHMI-style random-effects limits (Spiegelhalter, 2005), **6 trusts** are flagged at 99.8% and 25 at 95%. The extra variation is real, not noise: trusts' ratios have year-to-year rank correlations of 0.77 to 0.86.

**2. Readmissions are rising, and the deprivation gap has narrowed.**
The England rate rose from 12.6% (2014/15) to 14.7% (2024/25), with a 15.4% spike in 2020/21 when planned care stopped. In 2024/25, coming from the most rather than the least deprived fifth of areas adds about **2.4 percentage points** to a discharged patient's risk after adjusting for age and sex (roughly 156,000 readmissions a year). The gradient is steep for adults and nearly flat for children. The Relative Index of Inequality fell from **1.38 to 1.19** over the decade, mainly because rates rose faster in the least deprived areas. The narrowing is statistically robust under quasi-binomial inference and alternative time-trend specifications. As aggregated data, these results show where to look, not why.

**3. A patient-level model reaches the realistic ceiling and is well calibrated.**
On a held-out year (2024/25), the champion logistic model scores AUC **0.672** (95% CI 0.667-0.678), against **0.650** for the LACE index used in practice and **0.676** for the true probabilities (the best any model could do on these features). Calibration slope is 0.98. Estimated effects recover the known truth: every 95% interval covers the true value, and pure-noise variables are rejected.

**4. A risk score is only worth what it changes.**
For a follow-up programme costing £150 per patient, preventing 20% of readmissions that cost £2,500 each, the cost-optimal rule is to follow up patients above **30% risk**. Under these assumptions, targeting saves about £2.50 per discharge over no programme and £79 over following up everyone. The business case rests on the programme's effectiveness, which is the figure to establish before deployment.

**5. Fairness needs measuring, not assuming.**
The model is calibrated within every deprivation, age and sex group (O/E 0.99 to 1.04). Removing deprivation as a predictor leaves AUC unchanged but leaves the least deprived fifth under-predicted by about 6%.

## What a decision-maker should take away

* Do not read the published trust bands as a league table. Use over-dispersion-adjusted limits, and treat a flag as a reason to look, not a verdict.
* Readmission inequality is narrowing, but not because outcomes in deprived areas improved.
* A readmission score can be built to a realistic standard with logistic regression. Its value depends on calibration and on the effectiveness of what is done for flagged patients.

## Limitations

* Part 1 is ecological (aggregated) and adjusts only for age and sex; specialty and diagnosis mix would explain much of the trust variation. Counts are rounded to 5.
* Part 3 uses synthetic patients: it demonstrates method, not real-world performance. Before clinical use, a model must be developed and externally validated on real records, with a clinical safety case (DCB0129/DCB0160).
* Cost figures are illustrative assumptions, not NHS reference costs.

See the [model development document](model_development_document.md) and [validation report](validation_report.md) for detail.
