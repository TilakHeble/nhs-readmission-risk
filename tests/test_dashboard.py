import pytest

from nhsread.config import MODELS_DIR, ROOT

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "champion.json").exists(), reason="models absent")


def test_dashboard_runs_and_responds():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboards" / "app.py"), default_timeout=300).run()
    assert not at.exception
    flagged_adjusted = at.metric[1].value
    at.toggle[0].set_value(False).run()  # switch off the over-dispersion adjustment
    assert not at.exception
    assert at.metric[1].value != flagged_adjusted
    risk = next(m.value for m in at.metric if m.label == "30-day readmission risk")
    at.slider[0].set_value(30).run()  # change age
    assert not at.exception
    assert next(m.value for m in at.metric if m.label == "30-day readmission risk") != risk
