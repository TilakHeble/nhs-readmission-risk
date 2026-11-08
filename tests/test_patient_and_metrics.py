import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nhsread.evaluation import metrics as M
from nhsread.models import patient as P
from nhsread.synth.simulate import COHORT_YEARS, TRUE_EFFECTS, calibration_check, simulate_cohort


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 4000)
    y = rng.binomial(1, p)
    assert abs(M.auc(y, p) - roc_auc_score(y, p)) < 1e-12
    assert abs(M.brier(y, p) - brier_score_loss(y, p)) < 1e-12
    assert abs(M.log_loss(y, p) - log_loss(y, p)) < 1e-12
    a, lo, hi = M.auc_ci_delong(y, p)
    assert lo < a < hi


def test_perfectly_calibrated_predictions():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.5, 50_000)
    y = rng.binomial(1, p)
    citl, slope = M.calibration_intercept_slope(y, p)
    assert abs(citl) < 0.05 and abs(slope - 1) < 0.05


def test_lace_range():
    s = M.lace_score([0, 1, 30], [0, 1, 1], [0, 2, 8], [0, 1, 9])
    assert list(s) == [0, 1 + 3 + 2 + 1, 7 + 3 + 5 + 4]


def test_cost_threshold_sensible():
    rng = np.random.default_rng(3)
    p = rng.beta(2, 10, 20_000)
    y = rng.binomial(1, p)
    cheap, _ = M.cost_optimal_threshold(y, p, 50, 3000, 0.2)
    dear, _ = M.cost_optimal_threshold(y, p, 400, 3000, 0.2)
    assert cheap < dear


def test_cohort_matches_real_rates(cohort, cells):
    k = calibration_check(cohort, cells)
    assert set(cohort["year"]) == set(COHORT_YEARS)
    assert np.max(np.abs(k["true_p"] - k["real_rate"])) < 0.01


def test_cohort_reproducible(cells):
    a = simulate_cohort(cells, n=2000, seed=11)
    b = simulate_cohort(cells, n=2000, seed=11)
    assert a.equals(b)


def test_model_recovers_true_effects_and_scores_from_json(cohort, tmp_path):
    train, test = P.split(cohort)
    res = P.fit(train)
    assert abs(res.params["emergency"] - TRUE_EFFECTS["emergency"]) < 0.15
    assert abs(res.params["np.log1p(los_days)"] - TRUE_EFFECTS["log_los"]) < 0.1
    saved = P.to_saved(res, "t", P.CHAMPION_FORMULA, len(train))
    loaded = P.SavedModel.load(saved.save(tmp_path / "m.json"))
    np.testing.assert_allclose(loaded.predict(test), res.predict(test), atol=1e-10)
    np.testing.assert_allclose(loaded.predict(test.iloc[[0]]), res.predict(test).iloc[[0]], atol=1e-10)
    assert M.auc(test["readmit_30d"], loaded.predict(test)) > 0.63
