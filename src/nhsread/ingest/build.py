"""Build processed datasets from the committed raw CSV.  Run: python -m nhsread.ingest.build"""

from __future__ import annotations

from nhsread.config import CELLS_FILE, DATA_PROCESSED, TIDY_FILE
from nhsread.ingest.contracts import validate_cells, validate_tidy
from nhsread.ingest.reader import deprivation_cells, read_compendium


def build() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tidy = validate_tidy(read_compendium())
    cells = validate_cells(deprivation_cells(tidy))
    tidy.to_parquet(TIDY_FILE, index=False)
    cells.to_parquet(CELLS_FILE, index=False)
    print(f"tidy: {len(tidy):,} rows -> {TIDY_FILE.name}")
    print(f"cells: {len(cells):,} IMD x age x sex x year cells, {cells['den'].sum():,} discharges -> {CELLS_FILE.name}")


if __name__ == "__main__":
    build()
