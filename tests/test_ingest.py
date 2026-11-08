import numpy as np

from nhsread.config import YEARS
from nhsread.ingest.contracts import validate_cells, validate_tidy
from nhsread.ingest.reader import providers


def test_reader_shape_and_years(tidy):
    assert len(tidy) > 150_000
    assert sorted(tidy["year"].unique()) == YEARS
    validate_tidy(tidy)


def test_suppressed_counts_become_missing(tidy):
    s = tidy[tidy["suppressed"]]
    assert len(s) > 0 and s["num"].isna().all()


def test_england_rate_2024_25(tidy):
    e = tidy[(tidy["breakdown"] == "England") & (tidy["method"] == "Snapshot") & (tidy["sex"] == "Persons")
             & (tidy["year"] == "2024/25")].iloc[0]
    assert e["num"] == 948_836 and e["den"] == 6_467_596
    assert abs(100 * e["num"] / e["den"] - e["value"]) < 0.05


def test_cells_complete_and_reconcile_with_england(tidy, cells):
    validate_cells(cells)
    eng = tidy[(tidy["breakdown"] == "England") & (tidy["method"] == "Snapshot") & (tidy["sex"] == "Persons")]
    eng = eng.set_index("year")["den"]
    by_year = cells.groupby("year")["den"].sum()
    # cells exclude records with no IMD lookup and are rounded to 5, so allow 2.5%
    assert np.all(np.abs(by_year / eng - 1) < 0.025)


def test_provider_snapshot_values_are_o_over_e_times_england(tidy):
    p = providers(tidy, "2024/25")
    p = p[p["num"] >= 100]
    implied = p["num"] / p["expected"] * 14.67
    assert np.median(np.abs(implied - p["value"])) < 0.1
