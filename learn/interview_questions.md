# Interview questions: logistic regression

[Course index](README.md) · 40 questions, from graduate analyst to senior data scientist. Try each before opening the answer.

## Foundations

**1. Why not use linear regression for a binary outcome?**
<details><summary>Answer</summary>Predictions can fall outside [0, 1], the effect on probability cannot be constant across the range, and the error variance p(1−p) is not constant. The linear probability model can still be a reasonable approximation for rates between about 0.2 and 0.8.</details>

**2. Define odds, log-odds and the logistic function.**
<details><summary>Answer</summary>Odds = p/(1−p); log-odds = log(p/(1−p)); the logistic (sigmoid) function p = 1/(1+e^(−z)) maps any real z to (0, 1) and is the inverse of the logit.</details>

**3. How do you interpret a coefficient of 0.4?**
<details><summary>Answer</summary>A one-unit increase in the predictor adds 0.4 to the log-odds, i.e. multiplies the odds by e^0.4 ≈ 1.49, holding other variables fixed. The probability change depends on the baseline risk.</details>

**4. What's the "divide by 4" rule?**
<details><summary>Answer</summary>The maximum slope of the logistic curve is β·p(1−p) ≤ β/4 at p = 0.5, so β/4 is an upper bound on the percentage-point change per unit of x.</details>

**5. What does logistic regression minimise?**
<details><summary>Answer</summary>The negative log-likelihood of the Bernoulli/binomial model, equivalently the log-loss (cross-entropy).</details>

**6. Is there a closed-form solution?**
<details><summary>Answer</summary>No. It is solved iteratively, usually by Newton-Raphson (IRLS) or a quasi-Newton / gradient method. The log-likelihood is concave, so the optimum is unique when it exists.</details>

**7. What is IRLS?**
<details><summary>Answer</summary>Iteratively reweighted least squares: each Newton step equals a weighted least-squares regression of a "working response" with weights p(1−p). For the canonical logit link it is identical to Newton-Raphson.</details>

**8. Odds ratio vs relative risk: when do they diverge?**
<details><summary>Answer</summary>When the outcome is common. For rare outcomes (under about 10%) OR ≈ RR; for common outcomes the OR is further from 1 than the RR. RR = OR / (1 − p0 + p0·OR).</details>

**9. How would you explain a logistic model's result to a non-technical manager?**
<details><summary>Answer</summary>In absolute risks and percentage points ("patients like this are readmitted about 30% of the time vs 15% on average"), with a calibration plot showing predictions match reality, and with clear limits on what the model cannot see.</details>

**10. What's the difference between a probability model and a classifier?**
<details><summary>Answer</summary>Logistic regression outputs probabilities. A classifier adds a decision threshold. The threshold should come from costs, capacity and objectives, not a default of 0.5.</details>

## Inference and model building

**11. Compare Wald, likelihood-ratio and score tests.**
<details><summary>Answer</summary>Wald uses the estimate and its SE from the full model; LR compares the log-likelihoods of nested models; score uses the gradient at the restricted model. They agree asymptotically; LR is generally the most reliable in finite samples and is invariant to reparameterisation.</details>

**12. When can't you use an LR test?**
<details><summary>Answer</summary>For non-nested models (use AIC/BIC or cross-validated log-loss), and with over-dispersed grouped data use a quasi-likelihood F-test instead of a chi-square.</details>

**13. What is deviance, and is residual deviance a goodness-of-fit test?**
<details><summary>Answer</summary>−2 × log-likelihood relative to the saturated model. For grouped data with reasonable counts, residual deviance ≈ χ² with residual df under a correct model. For individual 0/1 data it is not a valid fit test.</details>

**14. What is over-dispersion and how do you handle it?**
<details><summary>Answer</summary>Grouped counts varying more than the binomial allows (Pearson χ²/df ≫ 1), typically from unmodelled heterogeneity or clustering. Handle with quasi-binomial SEs, beta-binomial models, random effects, or a better-specified model.</details>

**15. How would you check whether a continuous predictor is linear on the logit scale?**
<details><summary>Answer</summary>Empirical logit plots by bins, binned residual plots, comparing linear vs spline/fractional polynomial fits by LR/AIC.</details>

**16. Splines vs binning for non-linear effects?**
<details><summary>Answer</summary>Splines are smooth and use data efficiently; bins are easy to explain and audit but lose information and create jumps. Banks often prefer bins (with WoE); clinical models often use restricted cubic splines.</details>

**17. What is separation and how do you fix it?**
<details><summary>Answer</summary>A predictor (or combination) perfectly predicts the outcome, so the MLE goes to infinity (huge coefficients and SEs). Fix with Firth's penalised likelihood, ridge, collapsing categories, or exact methods.</details>

**18. How does multicollinearity affect logistic regression?**
<details><summary>Answer</summary>As in linear regression: unstable coefficients and inflated SEs, while predictions may be fine. Check VIFs; combine, drop or penalise correlated predictors.</details>

**19. A coefficient changes sign when you add a variable. What does that tell you?**
<details><summary>Answer</summary>Confounding or mediation: the coefficients answer different questions (total vs conditional association). Example: deprivation's effect shrinks or reverses once comorbidity, which lies on the pathway, is included.</details>

**20. How do you handle a categorical variable with 200 levels (e.g. hospital)?**
<details><summary>Answer</summary>Random intercepts (mixed model) or a penalised fixed effect, target/WoE encoding with proper cross-fitting, or grouping rare levels. Plain dummies over-fit small levels and risk separation.</details>

## Regularisation and imbalance

**21. Ridge vs lasso in logistic regression?**
<details><summary>Answer</summary>Ridge shrinks all coefficients and handles correlated predictors smoothly; lasso sets some to zero (selection) but picks somewhat arbitrarily among correlated predictors; elastic net combines them.</details>

**22. Why standardise before penalising?**
<details><summary>Answer</summary>Penalties depend on coefficient size, which depends on units; standardising makes the penalty treat features equally. The intercept is not penalised.</details>

**23. How do you choose the penalty strength?**
<details><summary>Answer</summary>Cross-validation on a proper scoring rule (log-loss or Brier), or information criteria; for time-dependent data, use time-ordered folds.</details>

**24. The outcome is 2% prevalent. Should you balance the classes?**
<details><summary>Answer</summary>Usually not for logistic regression: it shifts only the intercept and destroys calibration without improving ranking. Handle the rarity at the threshold, use proper scoring metrics, and if you subsample non-events for size, correct the intercept by log of the sampling ratio.</details>

**25. What's the King & Zeng prior correction?**
<details><summary>Answer</summary>After sampling events and non-events at different rates, subtract log[(fraction of events sampled)/(fraction of non-events sampled)] from the intercept to recover population probabilities.</details>

## Evaluation and validation

**26. What does AUC measure, and what doesn't it?**
<details><summary>Answer</summary>Ranking: the probability a random event outranks a random non-event. It ignores calibration, prevalence and the costs of decisions; monotone transformations of the scores don't change it.</details>

**27. AUC 0.68 for a readmission model: good or bad?**
<details><summary>Answer</summary>Typical. Published readmission models usually report 0.6-0.75 because much of the risk is determined after discharge. Compare with the existing tool (e.g. LACE) and with the achievable ceiling; an AUC above 0.9 should prompt a leakage check.</details>

**28. Define calibration-in-the-large and calibration slope.**
<details><summary>Answer</summary>CITL: intercept of a logistic model of y with logit(p) as an offset (ideal 0) – average over/under-prediction. Slope: coefficient of logit(p) (ideal 1); below 1 means predictions too extreme (over-fitting).</details>

**29. Why is accuracy a poor metric here?**
<details><summary>Answer</summary>With 15% prevalence, predicting "no" for everyone gives 85% accuracy. Accuracy depends on an arbitrary threshold and isn't a proper scoring rule.</details>

**30. What is a proper scoring rule?**
<details><summary>Answer</summary>A score optimised in expectation only by the true probabilities (log-loss, Brier). Using it for model selection rewards honest, calibrated probabilities.</details>

**31. Explain Harrell's bootstrap optimism correction.**
<details><summary>Answer</summary>Repeat the full modelling process on bootstrap samples; for each, compute performance on the bootstrap sample minus performance on the original data; average this optimism and subtract it from the apparent performance.</details>

**32. Random split vs out-of-time validation?**
<details><summary>Answer</summary>A random split mostly measures over-fitting. Out-of-time also tests stability against drift in case mix, practice and base rates, which is what deployment faces.</details>

**33. How do you recalibrate a model for a new hospital?**
<details><summary>Answer</summary>Fit a logistic model of local outcomes on the model's logit: update the intercept only (if the slope is fine) or intercept and slope (logistic recalibration / Platt scaling). Refit fully only if discrimination is poor.</details>

**34. How do you choose a decision threshold?**
<details><summary>Answer</summary>From costs and benefits (treat if p > cost / (effectiveness × cost of event)), from capacity (top k%), or from decision-curve analysis. It must be revisited as costs, capacity or risk distribution change.</details>

**35. What is decision-curve analysis?**
<details><summary>Answer</summary>Plotting net benefit = TP/n − FP/n × t/(1−t) across thresholds t, against treat-all and treat-none, to show where a model improves decisions.</details>

## Industry and governance

**36. How do NHS hospital indicators like SHMI use logistic regression?**
<details><summary>Answer</summary>A national logistic model gives each patient an expected probability from case mix; summing by trust gives expected deaths/readmissions; O/E ratios are compared on funnel plots with limits widened for over-dispersion (random effects on the sqrt or log scale).</details>

**37. Why might 70% of trusts fall outside 99.8% funnel limits, and what do you do?**
<details><summary>Answer</summary>Huge denominators make Poisson limits very narrow, and real between-trust variation (case mix not captured, coding, local services) is far larger than chance. Estimate over-dispersion (Spiegelhalter's method) and use random-effects limits; also improve the case-mix model.</details>

**38. What is Weight of Evidence and Information Value?**
<details><summary>Answer</summary>WoE for a bin = log(% goods / % bads); replacing a variable by its WoE makes its logit relationship linear and monotone. IV = Σ(% goods − % bads) × WoE summarises predictive power (rule of thumb: >0.3 strong).</details>

**39. How would you monitor a deployed readmission model?**
<details><summary>Answer</summary>Track input and score drift (PSI), outcome-based calibration (CITL, slope) and discrimination once outcomes mature, subgroup calibration (fairness), and data-quality checks; define triggers for recalibration or redevelopment and record them in the safety case.</details>

**40. Should deprivation be a predictor in a clinical risk model?**
<details><summary>Answer</summary>It depends on the use. Including it can improve calibration for deprived patients, which can direct help to them; excluding it ("fairness through unawareness") can leave a group systematically under- or over-predicted. The right approach is to check calibration and error rates by group with and without it and document the decision, involving clinicians and patients.</details>
