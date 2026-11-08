import numpy as np
import pandas as pd

from nhsread import funnel as F
from nhsread.ingest.reader import providers
from nhsread.models import grouped as G


def test_byar_close_to_exact_poisson():
    o = np.array([5, 20, 100, 1000])
    b = F.byar_ci(o)
    e = F.exact_poisson_ci(o)
    np.testing.assert_allclose(b[0], e[0], rtol=0.01)
    np.testing.assert_allclose(b[1], e[1], rtol=0.01)


def test_replicates_published_nhs_banding(tidy):
    p = providers(tidy, "2024/25")
    p = p[p["band"].notna()]
    agree = (F.band(p["num"], p["expected"]) == p["band"]).mean()
    assert agree > 0.95  # residual disagreement comes from counts rounded to the nearest 5


def test_overdispersion_detected_and_reduces_alarms(tidy):
    p = providers(tidy, "2024/25")
    od = F.estimate_overdispersion(p["num"], p["expected"])
    assert od.overdispersed and od.tau2 > 0
    plain = np.isin(F.band(p["num"], p["expected"]), ["A1", "B1"]).sum()
    adj = np.isin(F.overdispersed_band(p["num"], p["expected"], od.tau2), ["A1", "B1"]).sum()
    assert adj < plain / 3


def test_funnel_limits_narrow_with_size():
    lo, hi = F.poisson_funnel(np.array([10, 100, 10_000]), F.Z95)
    assert np.all(np.diff(hi) < 0) and np.all(np.diff(lo) > 0) and np.all(lo < 1) and np.all(hi > 1)


def test_indirect_standardisation_toy():
    strata = pd.DataFrame({"unit": ["a", "a", "b", "b"], "s": ["young", "old"] * 2, "den": [100, 0, 0, 100]})
    ref = pd.Series({"young": 0.1, "old": 0.2})
    e = F.expected_counts(strata, ref, "unit", "s")
    assert e["a"] == 10 and e["b"] == 20


def test_grouped_model_gradient_and_dispersion(cells):
    res = G.fit(cells, "C(imd_c) + age + sex + C(year)")
    # least-deprived quintile (5) has lower odds than most-deprived (1)
    assert res.params["C(imd_c)[T.5]"] < 0
    assert G.dispersion(res) > 5  # real cells are over-dispersed
    q = G.fit(cells, "C(imd_c) + age + sex + C(year)", scale="X2")
    assert np.all(q.bse > res.bse)


def test_inequality_indices_narrowing(cells):
    a = G.inequality_indices(cells[cells["year"] == "2014/15"])
    b = G.inequality_indices(cells[cells["year"] == "2024/25"])
    assert a["rii_or"] > 1 and b["rii_or"] > 1 and b["rii_or"] < a["rii_or"]
