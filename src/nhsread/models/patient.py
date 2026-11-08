"""Patient-level readmission risk model: feature definitions, fitting, saving and scoring.

Shared by the notebooks, the tests and the dashboard so they can never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from nhsread.config import MODELS_DIR

TRAIN_YEARS = ["2022/23", "2023/24"]
TEST_YEAR = "2024/25"
TARGET = "readmit_30d"

# Baseline: main effects only, linear in everything (what most quick projects fit).
BASELINE_FORMULA = (
    "los_days + emergency + charlson + prior_ed_6m + prior_adm_12m + care_home + polypharmacy"
    " + medical_specialty + weekend_discharge + C(age_band) + sex + C(imd) + ward_occupancy + discharge_hour"
)

# Champion: log transforms for skewed counts, a spline in Charlson, a clinically motivated
# interaction, and the two noise variables removed after the lasso and LR tests drop them.
CHAMPION_FORMULA = (
    "np.log1p(los_days) + emergency + bs(charlson_c, knots=[2], degree=2, lower_bound=0, upper_bound=9)"
    " + charlson_c:under75 + np.minimum(prior_ed_6m, 6) + np.log1p(prior_adm_12m) + care_home"
    " + polypharmacy + medical_specialty + weekend_discharge + C(age_band) + sex + C(imd)"
)

NUMERIC_FEATURES = [
    "age", "los_days", "emergency", "charlson", "prior_ed_6m", "prior_adm_12m", "care_home",
    "polypharmacy", "medical_specialty", "weekend_discharge", "ward_occupancy", "discharge_hour",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["charlson_c"] = d["charlson"].clip(upper=9)
    d["under75"] = (d["age"] < 75).astype(int)
    d["age_band"] = pd.Categorical(d["age_band"], categories=["16-74", "<16", "75+"])
    d["sex"] = pd.Categorical(d["sex"], categories=["Female", "Male"])
    d["imd"] = pd.Categorical(d["imd"].astype(int), categories=[1, 2, 3, 4, 5])
    return d


def split(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split: develop on 2022/23-2023/24, validate on 2024/25."""
    d = add_features(cohort)
    return d[d["year"].isin(TRAIN_YEARS)].reset_index(drop=True), d[d["year"] == TEST_YEAR].reset_index(drop=True)


def fit(train: pd.DataFrame, formula: str = CHAMPION_FORMULA, **kw):
    return smf.logit(f"{TARGET} ~ {formula}", data=train).fit(disp=0, **kw)


def fit_weighted(train: pd.DataFrame, formula: str, weights: np.ndarray):
    """Weighted logistic regression (used for class re-weighting experiments)."""
    import statsmodels.api as sm

    return smf.glm(f"{TARGET} ~ {formula}", data=train, family=sm.families.Binomial(),
                   freq_weights=weights).fit()


@dataclass
class SavedModel:
    """A fitted model stored as plain JSON (formula + coefficients) - no pickles."""

    name: str
    formula: str
    params: dict[str, float]
    train_years: list[str] = field(default_factory=lambda: list(TRAIN_YEARS))
    n_train: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        from patsy import dmatrix

        X = dmatrix(self.formula, add_features(df), return_type="dataframe")
        return X[list(self.params)]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        from nhsread.models.logit_scratch import sigmoid

        X = self.design(df)
        return sigmoid(X.to_numpy() @ np.array(list(self.params.values())))

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> SavedModel:
        return cls(**json.loads(Path(path).read_text()))


def to_saved(res, name: str, formula: str, n_train: int, metrics: dict | None = None) -> SavedModel:
    return SavedModel(name=name, formula=formula, params={k: float(v) for k, v in res.params.items()},
                      n_train=n_train, metrics=metrics or {})
