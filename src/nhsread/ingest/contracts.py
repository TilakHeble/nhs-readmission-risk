"""Data contracts: fail fast if NHS England changes the file layout or the numbers stop making sense."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from nhsread.config import YEARS

TIDY_SCHEMA = DataFrameSchema(
    {
        "year": Column(str, Check.isin(YEARS)),
        "breakdown": Column("category"),
        "level": Column(str),
        "age": Column("category", Check.isin(["All", "<16", "16+", "16-74", "75+"])),
        "sex": Column("category", Check.isin(["Persons", "Male", "Female"])),
        "method": Column("category", Check.isin(["Snapshot", "Trend"])),
        "num": Column(float, Check.ge(0), nullable=True),
        "den": Column(float, Check.gt(0), nullable=True),
        "expected": Column(float, Check.ge(0), nullable=True),
        "value": Column(float, Check.ge(0), nullable=True),  # standardised; can exceed 100 when expected is tiny
        "band": Column(str, Check.isin(["A1", "A5", "W", "B5", "B1"]), nullable=True),
    },
    checks=[
        Check(lambda d: (d["num"] <= d["den"]) | d["num"].isna() | d["den"].isna(),
              error="readmissions cannot exceed discharges"),
        Check(lambda d: d["year"].nunique() == len(YEARS), error="every financial year must be present"),
    ],
    strict=False,
    coerce=False,
)

CELLS_SCHEMA = DataFrameSchema(
    {
        "year": Column(str, Check.isin(YEARS)),
        "imd": Column(int, Check.isin([1, 2, 3, 4, 5])),
        "age": Column(str, Check.isin(["<16", "16-74", "75+"])),
        "sex": Column(str, Check.isin(["Male", "Female"])),
        "num": Column(int, Check.ge(0)),
        "den": Column(int, Check.gt(0)),
        "rate": Column(float, Check.in_range(0.03, 0.35)),
    },
    checks=[
        Check(lambda d: len(d) == len(YEARS) * 5 * 3 * 2, error="expected a complete 11x5x3x2 grid"),
        Check(lambda d: not d.duplicated(["year", "imd", "age", "sex"]).any(), error="duplicate cells"),
    ],
    strict=False,
)


def validate_tidy(df):
    return TIDY_SCHEMA.validate(df, lazy=True)


def validate_cells(df):
    return CELLS_SCHEMA.validate(df, lazy=True)


__all__ = ["TIDY_SCHEMA", "CELLS_SCHEMA", "validate_tidy", "validate_cells", "pa"]
