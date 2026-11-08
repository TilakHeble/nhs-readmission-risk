"""Indirect standardisation, confidence intervals, banding and funnel plots (NHS-style benchmarking).

The NHS readmission indicator is an indirectly standardised ratio:

    SR = O / E,   E = sum over case-mix strata s of  n_unit,s * p_England,s

where p_England,s comes from a (logistic) risk model or simply the national rate in each stratum.
The published "indicator value" is SR x the England crude rate.

Over-dispersion follows Spiegelhalter (2005), "Funnel plots for comparing institutional
performance", as used for SHMI: z-scores on the square-root scale, 10% winsorisation, and an
additive random-effects term tau^2 when the dispersion phi exceeds what chance allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Z95 = stats.norm.ppf(0.975)
Z998 = stats.norm.ppf(0.999)


# ----------------------------------------------------------------------------- standardisation
def expected_counts(unit_strata: pd.DataFrame, reference_rates: pd.Series, unit: str, stratum: str,
                    n_col: str = "den") -> pd.Series:
    """Indirect standardisation: E_unit = sum_s n_unit,s * p_ref,s."""
    d = unit_strata.join(reference_rates.rename("p_ref"), on=stratum)
    return (d[n_col] * d["p_ref"]).groupby(d[unit]).sum()


def direct_standardised_rate(rates: pd.Series, standard_population: pd.Series) -> float:
    """Direct standardisation: apply group-specific rates to a standard population."""
    w = standard_population / standard_population.sum()
    return float((rates.reindex(w.index) * w).sum())


# ----------------------------------------------------------------------------- intervals
def byar_ci(observed: np.ndarray, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """Byar's approximation to the exact Poisson interval for an observed count."""
    o = np.asarray(observed, dtype=float)
    lo = np.where(o > 0, o * (1 - 1 / (9 * np.maximum(o, 1e-9)) - z / (3 * np.sqrt(np.maximum(o, 1e-9)))) ** 3, 0.0)
    hi = (o + 1) * (1 - 1 / (9 * (o + 1)) + z / (3 * np.sqrt(o + 1))) ** 3
    return lo, hi


def exact_poisson_ci(observed: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(observed, dtype=float)
    lo = np.where(o > 0, stats.chi2.ppf(alpha / 2, 2 * o) / 2, 0.0)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * o + 2) / 2
    return lo, hi


def ratio_ci(observed, expected, z: float = Z95):
    lo, hi = byar_ci(observed, z)
    e = np.asarray(expected, dtype=float)
    return lo / e, hi / e


def band(observed, expected) -> np.ndarray:
    """NHS banding: compare the unit's 95% / 99.8% interval for SR with 1 (= the England rate)."""
    lo95, hi95 = ratio_ci(observed, expected, Z95)
    lo998, hi998 = ratio_ci(observed, expected, Z998)
    out = np.full(len(lo95), "W", dtype=object)
    out[hi95 < 1] = "B5"
    out[hi998 < 1] = "B1"
    out[lo95 > 1] = "A5"
    out[lo998 > 1] = "A1"
    return out


# ----------------------------------------------------------------------------- funnel limits
def poisson_funnel(expected_grid: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Control limits for SR = O/E under H0: O ~ Poisson(E) (normal approx on the sqrt scale)."""
    e = np.asarray(expected_grid, dtype=float)
    lo = np.maximum(1 - z / (2 * np.sqrt(e)), 0) ** 2
    hi = (1 + z / (2 * np.sqrt(e))) ** 2
    return lo, hi


@dataclass
class Overdispersion:
    phi: float          # dispersion of the winsorised z-scores (1 = Poisson variation only)
    tau2: float         # additive between-unit variance on the sqrt(SR) scale
    n_units: int
    winsor: float

    @property
    def overdispersed(self) -> bool:
        return self.phi > 1 + np.sqrt(2 / self.n_units) * Z95  # rough chi-square test bound


def sqrt_z(observed, expected) -> np.ndarray:
    """z-score on the square-root scale: 2 (sqrt(O) - sqrt(E)); approx N(0,1) under Poisson H0."""
    return 2 * (np.sqrt(np.asarray(observed, float)) - np.sqrt(np.asarray(expected, float)))


def estimate_overdispersion(observed, expected, winsor: float = 0.10) -> Overdispersion:
    """Spiegelhalter (2005) additive random-effects estimate on the sqrt scale (as SHMI)."""
    o = np.asarray(observed, float)
    e = np.asarray(expected, float)
    z = sqrt_z(o, e)
    lo, hi = np.quantile(z, [winsor, 1 - winsor])
    zw = np.clip(z, lo, hi)
    n = len(z)
    phi = float(np.mean(zw**2))
    s2 = 1 / (4 * e)  # variance of sqrt(SR) under H0
    w = 1 / s2
    tau2 = max(0.0, (n * phi - (n - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    return Overdispersion(phi=phi, tau2=float(tau2), n_units=n, winsor=winsor)


def overdispersed_funnel(expected_grid, z: float, tau2: float):
    """Funnel limits widened by an additive random effect: (1 +/- z sqrt(1/(4E) + tau2))^2."""
    e = np.asarray(expected_grid, float)
    s = np.sqrt(1 / (4 * e) + tau2)
    return np.maximum(1 - z * s, 0) ** 2, (1 + z * s) ** 2


def overdispersed_band(observed, expected, tau2: float) -> np.ndarray:
    """Banding after allowing for unexplained between-unit variation (fewer false alarms)."""
    o = np.asarray(observed, float)
    e = np.asarray(expected, float)
    zs = (np.sqrt(o / e) - 1) / np.sqrt(1 / (4 * e) + tau2)
    out = np.full(len(o), "W", dtype=object)
    out[zs < -Z95] = "B5"
    out[zs < -Z998] = "B1"
    out[zs > Z95] = "A5"
    out[zs > Z998] = "A1"
    return out


def funnel_frame(observed, expected, tau2: float | None = None, n_points: int = 200) -> pd.DataFrame:
    """Grid of funnel limits for plotting (Poisson and, optionally, over-dispersed)."""
    e = np.asarray(expected, float)
    grid = np.geomspace(max(e.min() * 0.8, 1), e.max() * 1.2, n_points)
    out = {"expected": grid}
    for name, z in [("95", Z95), ("998", Z998)]:
        out[f"lo{name}"], out[f"hi{name}"] = poisson_funnel(grid, z)
        if tau2 is not None:
            out[f"od_lo{name}"], out[f"od_hi{name}"] = overdispersed_funnel(grid, z, tau2)
    return pd.DataFrame(out)
