"""Paths and constants shared across the project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

COMPENDIUM_FILE = DATA_RAW / "I02040_readmissions_main_2025.csv"
TIDY_FILE = DATA_PROCESSED / "compendium_tidy.parquet"
CELLS_FILE = DATA_PROCESSED / "deprivation_cells.parquet"
COHORT_FILE = DATA_PROCESSED / "synthetic_cohort.parquet"

# Renamed CSV columns, in file order.
COLUMNS = [
    "year", "period", "breakdown", "level", "level_desc", "age", "sex", "diag", "method",
    "value", "lci95", "uci95", "lci998", "uci998", "num", "den", "expected", "band", "warning",
]

# Mutually exclusive age bands used for modelling ("All" and "16+" are aggregates).
AGE_BANDS = ["<16", "16-74", "75+"]
SEXES = ["Male", "Female"]
YEARS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2014, 2025)]
LATEST_YEAR = YEARS[-1]
COVID_YEAR = "2020/21"

IMD_LABELS = {
    "IMD Group 1 (Most Deprived)": 1,
    "IMD Group 2": 2,
    "IMD Group 3": 3,
    "IMD Group 4": 4,
    "IMD Group 5 (Least Deprived)": 5,
}

# NHS Digital banding codes (control limits on the funnel plot).
BANDS = {
    "A1": "Above 99.8% limit",
    "A5": "Above 95% limit",
    "W": "Within limits",
    "B5": "Below 95% limit",
    "B1": "Below 99.8% limit",
}

SEED = 20251108
