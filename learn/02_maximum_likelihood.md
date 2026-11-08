# 02 · Fitting by maximum likelihood

🟢 Beginner · [Course index](README.md) · Previous: [01](01_why_logistic.md) · Next: [03 · Interpreting coefficients](03_interpretation.md)

## The question

Linear regression picks the line that minimises squared errors. What does logistic regression minimise, and how does the computer find the answer?

## The likelihood

Each patient's outcome is a coin flip with their own probability $p_i$. The probability of seeing exactly the outcomes we saw is the **likelihood**:

$$L(\beta) = \prod_i p_i^{\,y_i} (1-p_i)^{1-y_i}$$

Products of many small numbers underflow, so we work with the **log-likelihood**:

$$\ell(\beta) = \sum_i \big[\,y_i \log p_i + (1-y_i)\log(1-p_i)\,\big]$$

**Maximum likelihood** picks the β that makes the observed data most probable. Minus the average log-likelihood is the **log-loss** (cross-entropy), the same quantity machine-learning libraries minimise.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.models.logit_scratch import fit_logit, gradient_descent, sigmoid
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
X = np.column_stack([np.ones(len(cohort)), cohort["charlson"], np.log1p(cohort["los_days"])])
y = cohort["readmit_30d"].to_numpy()


def log_lik(beta):
    p = sigmoid(X @ beta)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


print(f"all zeros (p = 0.5 for everyone): {log_lik(np.zeros(3)):,.0f}")
print(f"intercept only at the base rate:  {log_lik(np.array([np.log(y.mean() / (1 - y.mean())), 0, 0])):,.0f}")
```

Higher (less negative) is better. Now let the algorithm search.

## Newton-Raphson (also known as IRLS)

There is no closed-form answer like OLS's normal equation, so we iterate. The slope (gradient) and curvature (Hessian) of the log-likelihood are

$$\nabla\ell = X^\top (y - p), \qquad -\nabla^2\ell = X^\top W X,\quad W = \text{diag}\{p_i(1-p_i)\}$$

and each Newton step is $\beta \leftarrow \beta + (X^\top W X)^{-1} X^\top (y - p)$. Each step is a *weighted least-squares* fit, which is why it is also called **iteratively reweighted least squares**.

```python
res = fit_logit(X, y, names=["const", "charlson", "log1p_los"])
print(f"converged in {res.n_iter} iterations, log-likelihood {res.loglik:,.1f}")
print(pd.DataFrame({"coef": res.params, "se": res.bse}, index=res.names).round(4))

ref = sm.Logit(y, X).fit(disp=0)
print("matches statsmodels:", np.allclose(res.params, ref.params, atol=1e-8))
```

Newton converges in a handful of steps because it uses curvature. The inverse of $X^\top W X$ (the **Fisher information**) also gives the standard errors for free.

## Gradient descent

Gradient descent takes small steps downhill on the log-loss using only the gradient. It is slower, but it scales to huge data and is how logistic layers in neural networks are trained.

```python
from sklearn.preprocessing import StandardScaler

sub = cohort.sample(20_000, random_state=0)
Xs = np.column_stack([np.ones(len(sub)), StandardScaler().fit_transform(sub[["charlson", "los_days"]])])
ys = sub["readmit_30d"].to_numpy()
beta_gd, loss = gradient_descent(Xs, ys, lr=0.5, n_iter=2000)
beta_nr = fit_logit(Xs, ys).params
print("gradient descent:", beta_gd.round(3), " Newton:", beta_nr.round(3))
print(f"log-loss went from {loss[0]:.4f} to {loss[-1]:.4f}")
```

**Feature scaling matters for gradient descent** (unscaled features with very different ranges make the loss surface a long narrow valley). Newton's method does not care, because it rescales by the curvature.

## Is the optimum unique?

The log-likelihood of logistic regression is **concave**, so any maximum is the global maximum. There is one important exception: if a predictor perfectly separates the 0s from the 1s, the maximum is at infinity (lesson 06).

## Practice

**Q1.** Write the log-loss for a single patient who was readmitted (y = 1) and given p = 0.8. What if they were given p = 0.1?

<details><summary>Answer</summary>

Log-loss = −log(p) when y = 1. For p = 0.8: −log 0.8 ≈ 0.22. For p = 0.1: −log 0.1 ≈ 2.30. Confident wrong predictions are punished heavily.
</details>

**Q2.** Why does Newton-Raphson need far fewer iterations than gradient descent?

<details><summary>Answer</summary>

It uses second-derivative (curvature) information to choose both the direction and the size of each step. Near the optimum it converges quadratically: the number of correct digits roughly doubles each step. Gradient descent uses only the slope and a fixed step size.
</details>

**Q3.** What does the matrix W = diag{p(1−p)} tell you about which patients carry the most information about β?

<details><summary>Answer</summary>

Patients whose predicted probability is near 0.5 have the largest weight p(1−p) = 0.25: their outcome is genuinely uncertain, so it is informative. Patients predicted near 0 or 1 contribute little.
</details>

**Q4.** Logistic regression is "linear". In what sense?

<details><summary>Answer</summary>

It is linear in the parameters on the log-odds scale: logit(p) = Xβ. It is a generalised linear model with a logit link. The relationship between predictors and the probability itself is non-linear (S-shaped).
</details>
