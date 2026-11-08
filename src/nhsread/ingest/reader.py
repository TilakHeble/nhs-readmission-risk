"""Read the NHS England Compendium I02040 readmissions CSV into a tidy frame.

Notes on the source file
- Counts are published with thousands separators ("336,208.0") and sub-national counts are
  rounded to the nearest 5. Small counts are suppressed as "*".
- `method == "Snapshot"`: all ages, indirectly standardised against England in the same year.
- `method == "Trend"`: by age band, standardised to a fixed reference so years can be compared.
- `value` is the standardised rate per 100 discharges: (observed / expected) x England rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhsread.config import COLUMNS, COMPENDIUM_FILE, IMD_LABELS

NUMERIC = ["value", "lci95", "uci95", "lci998", "uci998", "num", "den", "expected"]


def _to_number(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned.where(cleaned != "*"), errors="coerce")


def read_compendium(path: Path | str = COMPENDIUM_FILE) -> pd.DataFrame:
    """Return the compendium with clean column names, numeric counts and helper columns."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, found {raw.shape[1]}: {list(raw.columns)}")
    raw.columns = COLUMNS
    df = raw.drop(columns=["diag"])  # always "All" in the main file

    df["suppressed"] = df["num"].astype("string").str.contains(r"\*", na=False)
    for c in NUMERIC:
        df[c] = _to_number(df[c])
    df["band"] = df["band"].where(df["band"] != "Err!")
    df["caution"] = df["warning"].eq("Caution")
    df["year_start"] = df["year"].str[:4].astype(int)
    df["crude_rate"] = 100 * df["num"] / df["den"]
    df["sr"] = df["num"] / df["expected"]  # standardised ratio, 1 = as expected
    df["imd"] = df["level_desc"].map(IMD_LABELS).where(df["breakdown"].eq("Deprivation quintile"))
    df["imd"] = df["imd"].astype("Int64")
    for c in ["breakdown", "age", "sex", "method"]:
        df[c] = df[c].astype("category")
    return df.drop(columns=["warning"])


def deprivation_cells(tidy: pd.DataFrame) -> pd.DataFrame:
    """Grouped-binomial cells: IMD quintile x age band x sex x year with readmissions and discharges.

    These are the mutually exclusive cells used to fit the grouped logistic regression.
    """
    m = (
        tidy["breakdown"].eq("Deprivation quintile")
        & tidy["imd"].notna()
        & tidy["age"].isin(["<16", "16-74", "75+"])
        & tidy["sex"].isin(["Male", "Female"])
        & tidy["method"].eq("Trend")
    )
    cells = tidy.loc[m, ["year", "year_start", "imd", "age", "sex", "num", "den"]].copy()
    cells["age"] = cells["age"].astype(str)
    cells["sex"] = cells["sex"].astype(str)
    cells["imd"] = cells["imd"].astype(int)
    cells["num"] = cells["num"].round().astype(int)
    cells["den"] = cells["den"].round().astype(int)
    cells["rate"] = cells["num"] / cells["den"]
    cells["logit"] = np.log(cells["rate"] / (1 - cells["rate"]))
    return cells.sort_values(["year", "imd", "age", "sex"]).reset_index(drop=True)


def providers(tidy: pd.DataFrame, year: str | None = None, age: str = "All", sex: str = "Persons",
              nhs_trusts_only: bool = True) -> pd.DataFrame:
    """Provider-level snapshot rows (observed, expected, discharges, NHS banding)."""
    method = "Snapshot" if age == "All" else "Trend"
    m = (
        tidy["breakdown"].eq("Provider")
        & tidy["age"].eq(age)
        & tidy["sex"].eq(sex)
        & tidy["method"].eq(method)
        & tidy["num"].notna()
        & tidy["expected"].gt(0)
    )
    if year is not None:
        m &= tidy["year"].eq(year)
    if nhs_trusts_only:
        m &= tidy["level"].str.startswith("R")  # NHS trust codes start with R; N* = independent sector
    cols = ["year", "year_start", "level", "level_desc", "num", "den", "expected", "value", "sr",
            "lci95", "uci95", "lci998", "uci998", "band", "caution"]
    return tidy.loc[m, cols].reset_index(drop=True)
