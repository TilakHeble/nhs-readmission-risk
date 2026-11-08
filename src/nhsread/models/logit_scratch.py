"""Logistic regression from first principles (NumPy only), used to check statsmodels and to teach.

Supports individual (0/1) data and grouped binomial data (y successes out of n trials), optional
observation weights and an optional ridge (L2) penalty. The fitting algorithm is Newton-Raphson,
which for the canonical logit link is identical to iteratively reweighted least squares (IRLS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable inverse logit."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


@dataclass
class LogitResult:
    params: np.ndarray
    cov: np.ndarray
    loglik: float
    deviance: float
    pearson_chi2: float
    df_resid: int
    n_iter: int
    converged: bool
    names: list[str]

    @property
    def bse(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def z(self) -> np.ndarray:
        return self.params / self.bse

    @property
    def pvalues(self) -> np.ndarray:
        return 2 * stats.norm.sf(np.abs(self.z))

    @property
    def odds_ratios(self) -> np.ndarray:
        return np.exp(self.params)

    @property
    def dispersion(self) -> float:
        """Pearson chi-square / residual df. About 1 for a correctly specified binomial model."""
        return self.pearson_chi2 / self.df_resid

    def aic(self) -> float:
        return -2 * self.loglik + 2 * len(self.params)

    def bic(self, n_obs: float) -> float:
        return -2 * self.loglik + np.log(n_obs) * len(self.params)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        q = stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([self.params - q * self.bse, self.params + q * self.bse])


def _loglik(y, n, p, w):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    # binomial log-likelihood without the constant log C(n, y): matches statsmodels' llf only for
    # n = 1; for grouped data we add the combinatorial term so the numbers line up exactly.
    from scipy.special import gammaln

    const = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    return float(np.sum(w * (const + y * np.log(p) + (n - y) * np.log1p(-p))))


def _deviance(y, n, p, w):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / (n * p)), 0.0)
        t2 = np.where(n - y > 0, (n - y) * np.log((n - y) / (n * (1 - p))), 0.0)
    return float(2 * np.sum(w * (t1 + t2)))


def fit_logit(
    X: np.ndarray,
    y: np.ndarray,
    n: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    l2: float = 0.0,
    penalise_intercept: bool = False,
    names: list[str] | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> LogitResult:
    """Maximum-likelihood logistic regression by Newton-Raphson / IRLS.

    X : (N, k) design matrix including an intercept column if wanted.
    y : successes (0/1 for individual data, counts for grouped data).
    n : trials per row (default 1).
    l2: ridge penalty lambda; objective is loglik - l2/2 * ||beta||^2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = np.ones_like(y) if n is None else np.asarray(n, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    k = X.shape[1]
    pen = np.full(k, l2)
    if not penalise_intercept and l2 > 0:
        const_cols = np.all(X == X[0], axis=0)
        pen[const_cols] = 0.0

    # Start from the overall log-odds in the intercept: fast and safe.
    beta = np.zeros(k)
    p0 = np.clip(np.sum(w * y) / np.sum(w * n), 1e-6, 1 - 1e-6)
    const = np.where(np.all(X == 1, axis=0))[0]
    if len(const):
        beta[const[0]] = np.log(p0 / (1 - p0))

    converged = False
    for it in range(1, max_iter + 1):  # noqa: B007 (it is reported as n_iter)
        p = sigmoid(X @ beta)
        W = w * n * p * (1 - p)  # IRLS weights
        grad = X.T @ (w * (y - n * p)) - pen * beta
        H = (X * W[:, None]).T @ X + np.diag(pen)  # negative Hessian = Fisher information
        step = np.linalg.solve(H, grad)
        # step-halving keeps Newton stable near separation
        old = _loglik(y, n, p, w) - 0.5 * np.sum(pen * beta**2)
        t = 1.0
        while True:
            cand = beta + t * step
            new = _loglik(y, n, sigmoid(X @ cand), w) - 0.5 * np.sum(pen * cand**2)
            if new >= old - 1e-12 or t < 1e-8:
                break
            t /= 2
        beta = cand
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break

    p = sigmoid(X @ beta)
    W = w * n * p * (1 - p)
    H = (X * W[:, None]).T @ X + np.diag(pen)
    cov = np.linalg.inv(H)
    pearson = float(np.sum(w * (y - n * p) ** 2 / (n * p * (1 - p))))
    return LogitResult(
        params=beta,
        cov=cov,
        loglik=_loglik(y, n, p, w),
        deviance=_deviance(y, n, p, w),
        pearson_chi2=pearson,
        df_resid=int(len(y) - k),
        n_iter=it,
        converged=converged,
        names=names or [f"x{i}" for i in range(k)],
    )


def gradient_descent(X, y, lr: float = 0.1, n_iter: int = 5000, l2: float = 0.0):
    """Plain batch gradient descent on the mean log-loss (teaching only: slow, but transparent).

    Returns (beta, loss_history).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    hist = []
    m = len(y)
    for _ in range(n_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) / m + l2 * beta / m
        beta -= lr * grad
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        hist.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
    return beta, np.array(hist)


def likelihood_ratio_test(restricted: LogitResult, full: LogitResult) -> tuple[float, int, float]:
    """LR = 2 (llf_full - llf_restricted) ~ chi2(df) under H0."""
    lr = 2 * (full.loglik - restricted.loglik)
    df = len(full.params) - len(restricted.params)
    return lr, df, float(stats.chi2.sf(lr, df))
