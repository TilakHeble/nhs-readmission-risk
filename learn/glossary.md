# Glossary

[Course index](README.md)

| Term | Meaning |
|---|---|
| **AUC / c-statistic** | Probability that a randomly chosen event (readmitted patient) scores higher than a randomly chosen non-event |
| **Average marginal effect (AME)** | Average change in predicted probability across real observations when a predictor changes |
| **Binomial** | Distribution of the number of successes in n independent yes/no trials with the same probability |
| **Brier score** | Mean squared difference between predicted probabilities and 0/1 outcomes |
| **Calibration** | Agreement between predicted probabilities and observed rates |
| **Calibration slope** | Slope from regressing outcomes on the model's logit; below 1 means over-fitting |
| **Charlson index** | Comorbidity score summing weighted long-term conditions |
| **Confounding** | A third variable related to both predictor and outcome that distorts their apparent association |
| **Decision curve** | Net benefit of using a model across threshold probabilities |
| **Deviance** | −2 × log-likelihood relative to a saturated model; differences between nested models are LR statistics |
| **Empirical logit** | log(events / non-events) within a group or bin; used to check linearity |
| **Expected count (E)** | Number of events predicted for a unit from a reference model; denominator of O/E |
| **Fisher information** | Curvature of the log-likelihood; its inverse estimates the covariance of the estimates |
| **Funnel plot** | O/E against expected count with control limits that narrow as units get bigger |
| **Gini** | 2 × AUC − 1; used in credit scoring |
| **HES** | Hospital Episode Statistics: patient-level NHS hospital records (restricted access) |
| **Hosmer-Lemeshow test** | Chi-square comparison of observed and expected events by decile of risk |
| **IMD** | Index of Multiple Deprivation; quintile 1 = most deprived fifth of areas in England |
| **Indirect standardisation** | Comparing observed events with those expected if the unit had reference (national) rates for its case mix |
| **IRLS** | Iteratively reweighted least squares; equivalent to Newton-Raphson for logistic regression |
| **LACE index** | Readmission score from Length of stay, Acuity, Comorbidity and ED visits (0-19) |
| **Lasso / ridge / elastic net** | Penalised regressions using L1, L2 or mixed penalties on coefficient size |
| **Leakage** | Using information in training that would not be available at prediction time |
| **Likelihood ratio test** | Compares nested models via twice the difference in log-likelihoods |
| **Log-loss** | Negative mean log-likelihood; the loss minimised by logistic regression |
| **Logit** | log(p / (1 − p)), the log-odds |
| **Odds ratio** | Ratio of odds between two groups; $e^\beta$ for a logistic coefficient |
| **Optimism** | How much better a model looks on its training data than on new data |
| **Out-of-time validation** | Testing on a later period than the model was developed on |
| **Over-dispersion** | More variation in grouped counts than the binomial (or Poisson) model allows |
| **PSI** | Population stability index; measures drift in a variable's distribution |
| **Propensity score** | Probability of receiving a treatment given characteristics, used to balance comparisons |
| **Quasi-binomial** | Binomial model whose variance is multiplied by an estimated dispersion φ |
| **RII / SII** | Relative / Slope Index of Inequality: modelled gap between the most and least deprived ends of the population |
| **Separation** | A predictor perfectly predicts the outcome, so the MLE is infinite |
| **SHMI** | Summary Hospital-level Mortality Indicator: NHS O/E mortality indicator using funnel plots with over-dispersion |
| **Shrinkage (empirical Bayes)** | Pulling noisy unit estimates towards the overall mean in proportion to their unreliability |
| **Sigmoid** | $1/(1+e^{-z})$; maps the real line to (0, 1) |
| **Spline** | Piecewise polynomial joined smoothly at knots; a flexible non-linear term |
| **TRIPOD** | Reporting guideline for clinical prediction models (TRIPOD+AI, 2024) |
| **Weight of Evidence (WoE)** | log(% non-events / % events) per bin; used to build credit scorecards |
