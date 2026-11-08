# Logistic regression cheat sheet

[Course index](README.md)

## Model

| | |
|---|---|
| Model | $\text{logit}(p) = \log\frac{p}{1-p} = x^\top\beta$, $\;p = \sigma(x^\top\beta) = 1/(1+e^{-x^\top\beta})$ |
| Odds | $p/(1-p)$; odds ratio for a 1-unit change in $x_j$: $e^{\beta_j}$ |
| OR → RR at baseline risk $p_0$ | $RR = OR / (1 - p_0 + p_0\,OR)$ |
| Marginal effect (continuous) | $\partial p/\partial x_j = \beta_j\,p(1-p)$, largest at $p = 0.5$ ($\beta_j/4$: the "divide by 4" rule) |
| Grouped data | $y_c \sim \text{Bin}(n_c, p_c)$; same estimates as individual rows |

## Estimation

| | |
|---|---|
| Log-likelihood | $\ell = \sum y\log p + (1-y)\log(1-p)$ |
| Log-loss | $-\ell/n$ |
| Gradient | $X^\top(y - p)$ |
| Fisher information | $X^\top W X$, $W = \text{diag}\{n\,p(1-p)\}$ |
| Newton / IRLS step | $\beta \leftarrow \beta + (X^\top WX)^{-1}X^\top(y-p)$ |
| Covariance | $(X^\top W X)^{-1}$ (× $\hat\phi$ for quasi-binomial) |
| Ridge / lasso | maximise $\ell - \lambda\sum\beta_j^2/2$ / $\ell - \lambda\sum|\beta_j|$ (standardise first) |

## Inference

| Test | Statistic | Use |
|---|---|---|
| Wald | $z = \hat\beta/SE$ | single coefficients |
| Likelihood ratio | $2(\ell_1 - \ell_0) \sim \chi^2_{df}$ | nested models (preferred) |
| Quasi F | $\frac{(D_0 - D_1)/df}{\hat\phi} \sim F_{df,\,n-k}$ | nested, over-dispersed grouped data |
| AIC / BIC | $-2\ell + 2k$ / $-2\ell + k\log n$ | non-nested comparison |
| Dispersion | $\hat\phi = X^2_{Pearson}/(n-k)$ | grouped data: ≈ 1 if binomial is right |

## Evaluation

| Measure | Formula / meaning | Ideal |
|---|---|---|
| AUC (c-statistic) | P(score of random event > random non-event) | 1 (0.5 = chance) |
| Gini | $2\,AUC - 1$ | 1 |
| Brier | $\frac1n\sum(p - y)^2$ | 0 |
| Calibration-in-the-large | intercept of $y \sim \text{offset}(\text{logit}\,p)$ | 0 |
| Calibration slope | slope of $y \sim \text{logit}\,p$ | 1 (<1 = overfitted) |
| Net benefit at $t$ | $\frac{TP}{n} - \frac{FP}{n}\frac{t}{1-t}$ | above treat-all and treat-none |
| Cost threshold | intervene if $p > c/(e\,R)$ | needs calibration |

## Benchmarking (NHS style)

| | |
|---|---|
| Expected | $E = \sum_i \hat p_i$ (or $\sum_s n_s p_{ref,s}$) |
| Standardised ratio | $SR = O/E$; indicator value $= SR \times$ national rate |
| Poisson funnel (sqrt scale) | limits $\big(1 \pm z/(2\sqrt E)\big)^2$ |
| Over-dispersed funnel | $\big(1 \pm z\sqrt{1/(4E) + \tau^2}\big)^2$, $\tau^2$ from winsorised z-scores |
| Shrinkage | $\sqrt{SR}^{\,shr} = 1 + \frac{\tau^2}{\tau^2 + 1/(4E)}(\sqrt{SR} - 1)$ |
| RII / SII | logistic / linear fit of rate on deprivation ridit |

## Red flags

- Huge coefficient with a huge SE: **separation**
- Quasi SEs *smaller* than binomial ones: dispersion computed on the wrong scale
- Balanced class weights: AUC unchanged, probabilities inflated
- AUC > 0.9 for readmission: look for **leakage** (information from after discharge)
- `bs(x, df=k)` in a deployed model: knots move with the data
