from __future__ import annotations

import pandas as pd
import pytest

from nhsread.config import COMPENDIUM_FILE
from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.synth.simulate import simulate_cohort


@pytest.fixture(scope="session")
def tidy() -> pd.DataFrame:
    return read_compendium(COMPENDIUM_FILE)


@pytest.fixture(scope="session")
def cells(tidy) -> pd.DataFrame:
    return deprivation_cells(tidy)


@pytest.fixture(scope="session")
def cohort(cells) -> pd.DataFrame:
    return simulate_cohort(cells, n=60_000, seed=7)
