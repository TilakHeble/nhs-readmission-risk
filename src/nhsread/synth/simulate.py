"""Synthetic patient-level discharge cohort, anchored to the real NHS readmission rates.

Why synthetic?  Patient-level Hospital Episode Statistics are not public. Rather than use a popular
US dataset, we simulate discharges whose *demographic structure* is taken from the real NHS file:

1. Discharges are sampled in proportion to the real NHS denominators for every
   IMD quintile x age band x sex x year cell (2022/23 to 2024/25).
2. Clinical features (length of stay, admission type, comorbidity, prior ED use, ...) are drawn
   from plausible distributions that depend on age and deprivation.
3. The 30-day readmission outcome comes from a known logistic model. A per-cell intercept is
   solved numerically so that the simulated readmission rate in every cell equals the real NHS rate.

The clinical effect sizes are ASSUMPTIONS (loosely informed by the LACE literature, van Walraven et
al. 2010, CMAJ) and are recorded in TRUE_EFFECTS. Because the truth is known we can check whether
the modelling recovers it - something real data never allows. Performance numbers on this cohort
demonstrate method, not real-world clinical accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from nhsread.config import SEED
from nhsread.models.logit_scratch import sigmoid

COHORT_YEARS = ["2022/23", "2023/24", "2024/25"]

# True log-odds effects used to generate the outcome (documented assumptions).
TRUE_EFFECTS = {
    "log_los": 0.35,              # per unit of log(1 + length of stay in days)
    "emergency": 0.60,            # emergency (vs elective) index admission
    "charlson_le4": 0.30,         # per Charlson point up to 4 ...
    "charlson_gt4": 0.10,         # ... flatter above 4 (non-linear)
    "charlson_x_under75": 0.05,   # comorbidity matters a bit more below 75 (interaction)
    "prior_ed_6m": 0.17,          # per ED attendance in prior 6 months (capped at 6)
    "log_prior_adm_12m": 0.50,    # per unit of log(1 + emergency admissions in prior 12 months)
    "care_home": 0.35,            # discharged to a care home
    "polypharmacy": 0.25,         # 10+ regular medicines
    "medical_specialty": 0.20,    # medical (vs surgical / other)
    "weekend_discharge": 0.08,
    "ward_occupancy": 0.0,        # pure noise: should get ~0 / be dropped by the lasso
    "discharge_hour": 0.0,        # pure noise
}
PROVIDER_SD = 0.15                # between-hospital variation (random intercept, log-odds)
N_PROVIDERS = 60


def _age_years(band: np.ndarray, rng) -> np.ndarray:
    a = np.empty(len(band))
    m = band == "<16"
    a[m] = np.minimum(rng.exponential(4.0, m.sum()), 15.99)
    m = band == "16-74"
    a[m] = 16 + 59 * rng.beta(1.6, 1.2, m.sum())
    m = band == "75+"
    a[m] = 75 + np.minimum(rng.gamma(2.0, 4.5, m.sum()), 29)
    return np.floor(a).astype(int)


def simulate_cohort(cells: pd.DataFrame, n: int = 240_000, seed: int = SEED) -> pd.DataFrame:
    """Return one row per synthetic discharge, with features, the true probability and the outcome."""
    rng = np.random.default_rng(seed)
    c = cells[cells["year"].isin(COHORT_YEARS)].reset_index(drop=True)
    idx = rng.choice(len(c), size=n, p=(c["den"] / c["den"].sum()).to_numpy())
    df = c.loc[idx, ["year", "imd", "age", "sex"]].rename(columns={"age": "age_band"}).reset_index(drop=True)
    df["cell"] = idx
    band = df["age_band"].to_numpy()
    df["age"] = _age_years(band, rng)
    old = (df["age"] >= 75).to_numpy()
    kid = (df["age"] < 16).to_numpy()
    dep = (6 - df["imd"]).to_numpy()  # 1 least ... 5 most deprived

    # --- clinical features (depend on age and deprivation) ---
    age_s = (df["age"].to_numpy() - 50) / 20
    lam_ch = np.exp(-0.9 + 0.55 * age_s + 0.08 * dep - 1.0 * kid)
    charlson = rng.poisson(lam_ch)
    los = np.round(rng.lognormal(0.6 + 0.35 * np.clip(age_s, -1, 2) + 0.05 * charlson, 0.9)).clip(0, 120).astype(int)
    emergency = rng.random(n) < sigmoid(0.2 + 0.35 * age_s + 0.12 * dep + 0.3 * kid)
    prior_ed = rng.negative_binomial(1.2, 1 / (1 + np.exp(-1.0 + 0.15 * dep + 0.2 * charlson) / 1.2))
    prior_adm = rng.poisson(np.exp(-1.6 + 0.3 * charlson + 0.1 * dep + 0.25 * age_s))
    care_home = rng.random(n) < sigmoid(-6.0 + 2.2 * np.clip(age_s, 0, None) + 0.15 * charlson) * old
    polypharm = rng.random(n) < sigmoid(-3.2 + 1.1 * age_s + 0.35 * charlson)
    medical = rng.random(n) < sigmoid(-0.2 + 0.4 * age_s + 0.4 * emergency)
    weekend = rng.random(n) < 2 / 7 * 0.8
    occupancy = np.clip(rng.normal(0.92, 0.05, n), 0.6, 1.05)
    hour = np.clip(rng.normal(15, 3, n), 7, 23).round().astype(int)
    provider = rng.integers(0, N_PROVIDERS, n)
    prov_effect = rng.normal(0, PROVIDER_SD, N_PROVIDERS)

    E = TRUE_EFFECTS
    eta = (
        E["log_los"] * np.log1p(los)
        + E["emergency"] * emergency
        + E["charlson_le4"] * np.minimum(charlson, 4)
        + E["charlson_gt4"] * np.maximum(charlson - 4, 0)
        + E["charlson_x_under75"] * charlson * (~old)
        + E["prior_ed_6m"] * np.minimum(prior_ed, 6)
        + E["log_prior_adm_12m"] * np.log1p(prior_adm)
        + E["care_home"] * care_home
        + E["polypharmacy"] * polypharm
        + E["medical_specialty"] * medical
        + E["weekend_discharge"] * weekend
        + prov_effect[provider]
    )

    # --- per-cell intercept so the simulated rate equals the real NHS rate in every cell ---
    alpha = np.zeros(n)
    rates = c["rate"].to_numpy()
    cell_ids = df["cell"].to_numpy()
    order = np.argsort(cell_ids, kind="stable")
    bounds = np.searchsorted(cell_ids[order], np.arange(len(c) + 1))
    for k in range(len(c)):
        ix = order[bounds[k]:bounds[k + 1]]
        if len(ix) == 0:
            continue
        e = eta[ix]
        a = brentq(lambda a, e=e, r=rates[k]: sigmoid(a + e).mean() - r, -15, 10)
        alpha[ix] = a
    p_true = sigmoid(alpha + eta)

    df = df.assign(
        los_days=los,
        emergency=emergency.astype(int),
        charlson=charlson,
        prior_ed_6m=prior_ed,
        prior_adm_12m=prior_adm,
        care_home=care_home.astype(int),
        polypharmacy=polypharm.astype(int),
        medical_specialty=medical.astype(int),
        weekend_discharge=weekend.astype(int),
        ward_occupancy=occupancy.round(3),
        discharge_hour=hour,
        provider=[f"P{p:02d}" for p in provider],
        p_true=p_true,
        readmit_30d=(rng.random(n) < p_true).astype(int),
    )
    df["patient_id"] = [f"S{i:07d}" for i in range(n)]
    cols = ["patient_id", "year", "provider", "age", "age_band", "sex", "imd"] + [
        c for c in df.columns if c not in {"patient_id", "year", "provider", "age", "age_band", "sex", "imd", "cell"}
    ]
    return df[cols]


def calibration_check(cohort: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Simulated vs real readmission rate by year, IMD quintile and age band."""
    sim = cohort.groupby(["year", "imd", "age_band"]).agg(sim_rate=("readmit_30d", "mean"),
                                                           true_p=("p_true", "mean"), n=("readmit_30d", "size"))
    real = (cells[cells["year"].isin(COHORT_YEARS)].groupby(["year", "imd", "age"])[["num", "den"]].sum()
            .rename_axis(["year", "imd", "age_band"]))
    sim["real_rate"] = real["num"] / real["den"]
    return sim.reset_index()


if __name__ == "__main__":
    from nhsread.config import CELLS_FILE, COHORT_FILE

    cells = pd.read_parquet(CELLS_FILE)
    cohort = simulate_cohort(cells)
    cohort.to_parquet(COHORT_FILE, index=False)
    print(f"{len(cohort):,} synthetic discharges, readmission rate {cohort['readmit_30d'].mean():.3%} -> {COHORT_FILE.name}")
