import numpy as np
import statsmodels.api as sm

from nhsread.models.logit_scratch import fit_logit, gradient_descent, likelihood_ratio_test, sigmoid


def _sim(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.binomial(1, 0.4, n)])
    beta = np.array([-1.5, 0.8, 0.5])
    y = rng.binomial(1, sigmoid(X @ beta))
    return X, y


def test_matches_statsmodels_individual():
    X, y = _sim()
    mine = fit_logit(X, y)
    ref = sm.Logit(y, X).fit(disp=0)
    assert mine.converged
    np.testing.assert_allclose(mine.params, ref.params, atol=1e-8)
    np.testing.assert_allclose(mine.bse, ref.bse, rtol=1e-6)
    assert abs(mine.loglik - ref.llf) < 1e-6


def test_matches_statsmodels_grouped(cells):
    d = cells[cells["year"] == "2024/25"]
    X = np.column_stack([np.ones(len(d)), (d["age"] == "75+").astype(float), (d["imd"] == 1).astype(float)])
    mine = fit_logit(X, d["num"].to_numpy(), n=d["den"].to_numpy())
    ref = sm.GLM(np.column_stack([d["num"], d["den"] - d["num"]]), X, family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(mine.params, ref.params, atol=1e-8)
    np.testing.assert_allclose(mine.deviance, ref.deviance, rtol=1e-8)
    np.testing.assert_allclose(mine.pearson_chi2, ref.pearson_chi2, rtol=1e-6)
    assert abs(mine.loglik - ref.llf) < 1e-4


def test_ridge_shrinks_but_not_intercept():
    X, y = _sim()
    a, b = fit_logit(X, y), fit_logit(X, y, l2=500.0)
    assert np.all(np.abs(b.params[1:]) < np.abs(a.params[1:]))


def test_gradient_descent_approaches_mle():
    X, y = _sim(2000)
    beta, hist = gradient_descent(X, y, lr=0.5, n_iter=4000)
    assert hist[-1] < hist[0]
    np.testing.assert_allclose(beta, fit_logit(X, y).params, atol=0.02)


def test_lr_test_detects_real_effect():
    X, y = _sim()
    lr, df, p = likelihood_ratio_test(fit_logit(X[:, :2], y), fit_logit(X, y))
    assert df == 1 and lr > 10 and p < 1e-3
