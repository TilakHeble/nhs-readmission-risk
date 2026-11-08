"""NHS 30-day readmissions: benchmarking and risk dashboard.

Run locally:  streamlit run dashboards/app.py
Part 1 tabs use the real NHS England Compendium file committed in data/raw.
The risk calculator uses the champion model trained on the synthetic cohort (see notebook 04).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from nhsread import funnel as F  # noqa: E402
from nhsread.config import BANDS, MODELS_DIR, YEARS  # noqa: E402
from nhsread.evaluation import metrics as M  # noqa: E402
from nhsread.ingest.reader import deprivation_cells, providers, read_compendium  # noqa: E402
from nhsread.models import grouped as G  # noqa: E402
from nhsread.models import patient as P  # noqa: E402
from nhsread.synth.simulate import simulate_cohort  # noqa: E402

BAND_COLOURS = {"A1": "#c2410c", "A5": "#eb6834", "W": "#a8a7a2", "B5": "#1baf7a", "B1": "#0f7a55"}
IMD_COLOURS = ["#7a2e0e", "#c2410c", "#eda100", "#5aa0e6", "#1d4f91"]

st.set_page_config(page_title="NHS Readmissions: Risk & Benchmarking", page_icon="🏥", layout="wide")


@st.cache_data(show_spinner="Reading the NHS England file…")
def load_nhs():
    tidy = read_compendium()
    return tidy, deprivation_cells(tidy)


@st.cache_resource(show_spinner=False)
def load_models():
    return {n: P.SavedModel.load(MODELS_DIR / f"{n}.json") for n in ["champion", "lace"]}


@st.cache_data(show_spinner="Scoring the 2024/25 validation cohort…")
def validation_scores():
    _, cells = load_nhs()
    cohort = simulate_cohort(cells)
    _, test = P.split(cohort)
    m = load_models()
    return test["readmit_30d"].to_numpy(), m["champion"].predict(test), m["lace"].predict(test)


tidy, cells = load_nhs()
summary = json.loads((MODELS_DIR / "validation_summary.json").read_text())

st.title("🏥 NHS 30-day emergency readmissions")
st.caption(
    "Real NHS England data (Compendium I02040, 2014/15 to 2024/25, Open Government Licence) for hospital "
    "benchmarking and inequality; a model trained on a **synthetic** patient cohort for the risk calculator. "
    "Source code and notebooks: github.com/TilakHeble/nhs-readmission-risk"
)
tab1, tab2, tab3 = st.tabs(["📊 Hospital benchmarking", "⚖️ Deprivation & inequality", "🧮 Patient risk calculator"])

# ------------------------------------------------------------------------------------------------ tab 1
with tab1:
    c1, c2 = st.columns([1, 3])
    with c1:
        year = st.selectbox("Financial year", YEARS[::-1], index=0)
        adjust = st.toggle("Allow for over-dispersion (SHMI-style)", value=True,
                           help="Widen the limits for genuine between-trust variation (Spiegelhalter 2005).")
        trusts_only = st.toggle("NHS trusts only", value=True, help="Exclude independent-sector providers.")
    p = providers(tidy, year, nhs_trusts_only=trusts_only)
    p = p[p["expected"] >= 5].copy()
    od = F.estimate_overdispersion(p["num"], p["expected"])
    p["our band"] = F.overdispersed_band(p["num"], p["expected"], od.tau2) if adjust else F.band(p["num"], p["expected"])
    flagged = int(p["our band"].isin(["A1", "B1"]).sum())
    with c1:
        st.metric("Trusts", len(p))
        st.metric("Flagged at 99.8%", f"{flagged} ({flagged / len(p):.0%})")
        st.metric("Dispersion φ", f"{od.phi:.0f}", help="1 = only chance variation between trusts")
    fr = F.funnel_frame(p["num"], p["expected"], tau2=od.tau2)
    pre = "od_" if adjust else ""
    lines = pd.concat([
        fr[["expected", f"{pre}{k}"]].rename(columns={f"{pre}{k}": "sr"}).assign(limit=k) for k in ["lo95", "hi95", "lo998", "hi998"]
    ])
    lines["level"] = np.where(lines["limit"].str.contains("998"), "99.8%", "95%")
    lines = lines[lines["sr"].between(0.2, 1.8)]  # draw limits only inside the plotted range
    base = alt.Chart(lines).mark_line(color="#52514e", strokeWidth=1).encode(
        x=alt.X("expected:Q", scale=alt.Scale(type="log"), title="expected readmissions (log scale)"),
        y=alt.Y("sr:Q", scale=alt.Scale(domain=[0.2, 1.8], clamp=True), title="observed / expected"),
        detail="limit:N", strokeDash=alt.StrokeDash("level:N", scale=alt.Scale(domain=["95%", "99.8%"], range=[[4, 3], [1, 0]]),
                                                     legend=alt.Legend(title="limit")))
    pts = alt.Chart(p.assign(sr_plot=p["sr"].clip(0.2, 1.8))).mark_circle(size=55, opacity=0.9).encode(
        x="expected:Q", y="sr_plot:Q",
        color=alt.Color("our band:N", scale=alt.Scale(domain=list(BAND_COLOURS), range=list(BAND_COLOURS.values())),
                        legend=alt.Legend(title="band")),
        tooltip=[alt.Tooltip("level_desc:N", title="trust"), alt.Tooltip("den:Q", title="discharges", format=","),
                 alt.Tooltip("num:Q", title="readmissions", format=","), alt.Tooltip("expected:Q", format=",.0f"),
                 alt.Tooltip("sr:Q", title="O/E", format=".3f"), alt.Tooltip("band:N", title="published NHS band"),
                 alt.Tooltip("our band:N")])
    rule = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color="#52514e").encode(y="y:Q")
    with c2:
        st.altair_chart((base + rule + pts).properties(height=430), use_container_width=True)
        st.caption("Hover for details. Bands: " + "; ".join(f"{k} = {v}" for k, v in BANDS.items()))
    st.subheader("Trusts outside the limits")
    out = p[p["our band"] != "W"].sort_values("sr", ascending=False)
    st.dataframe(out[["level_desc", "den", "num", "expected", "sr", "band", "our band"]].rename(columns={
        "level_desc": "trust", "den": "discharges", "num": "readmissions", "sr": "O/E", "band": "published band"}),
        hide_index=True, use_container_width=True)

    st.subheader("One trust over time")
    names = sorted(providers(tidy, None)["level_desc"].unique())
    pick = st.selectbox("Trust", names, index=names.index("LEEDS TEACHING HOSPITALS NHS TRUST") if "LEEDS TEACHING HOSPITALS NHS TRUST" in names else 0)
    hist = providers(tidy, None)
    hist = hist[hist["level_desc"] == pick].sort_values("year")
    band_ch = alt.Chart(hist).mark_area(opacity=0.2, color="#2a78d6").encode(x=alt.X("year:N", title=None), y=alt.Y("lci95:Q", title="standardised rate (%)", scale=alt.Scale(zero=False)), y2="uci95:Q")
    line_ch = alt.Chart(hist).mark_line(point=True, color="#2a78d6").encode(x="year:N", y="value:Q",
                                                                             tooltip=["year", "value", "band"])
    eng = tidy[(tidy["breakdown"] == "England") & (tidy["method"] == "Snapshot") & (tidy["sex"] == "Persons")]
    eng_ch = alt.Chart(eng).mark_line(strokeDash=[4, 3], color="#52514e").encode(x="year:N", y="value:Q")
    st.altair_chart((band_ch + line_ch + eng_ch).properties(height=260), use_container_width=True)
    st.caption("Blue: trust's indirectly standardised rate with 95% interval. Dashed: England.")

# ------------------------------------------------------------------------------------------------ tab 2
with tab2:
    ineq = pd.DataFrame({y: G.inequality_indices(cells[cells["year"] == y]) for y in YEARS}).T.reset_index(names="year")
    a, b, c = st.columns(3)
    a.metric("Most vs least deprived, 2024/25", f"{ineq['rate_most'].iloc[-1]:.1f}% vs {ineq['rate_least'].iloc[-1]:.1f}%")
    b.metric("Relative Index of Inequality", f"{ineq['rii_or'].iloc[-1]:.2f}", f"{ineq['rii_or'].iloc[-1] - ineq['rii_or'].iloc[0]:+.2f} since 2014/15",
             delta_color="inverse")
    c.metric("Slope Index of Inequality", f"{ineq['sii_pp'].iloc[-1]:.1f} pts", f"{ineq['sii_pp'].iloc[-1] - ineq['sii_pp'].iloc[0]:+.1f} since 2014/15",
             delta_color="inverse")
    age = st.radio("Age band", ["All ages", "<16", "16-74", "75+"], horizontal=True)
    sel = cells if age == "All ages" else cells[cells["age"] == age]
    r = sel.groupby(["year", "imd"])[["num", "den"]].sum().reset_index()
    r["rate"] = 100 * r["num"] / r["den"]
    r["quintile"] = r["imd"].map({1: "1 most deprived", 2: "2", 3: "3", 4: "4", 5: "5 least deprived"})
    st.altair_chart(alt.Chart(r).mark_line(point=True).encode(
        x=alt.X("year:N", title=None), y=alt.Y("rate:Q", title="readmission rate (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("quintile:N", scale=alt.Scale(range=IMD_COLOURS), title="IMD quintile"),
        tooltip=["year", "quintile", alt.Tooltip("rate:Q", format=".2f")]).properties(height=360), use_container_width=True)
    st.caption("Crude rates from real NHS cells. The gap has narrowed mainly because rates rose faster in the least deprived "
               "areas. A grouped logistic model with quasi-binomial errors (notebook 02) confirms the narrowing.")

# ------------------------------------------------------------------------------------------------ tab 3
with tab3:
    st.info("This calculator uses a model trained on a **synthetic** cohort whose age, sex, deprivation and year mix "
            "matches real NHS data, but whose clinical effects are assumptions. It demonstrates the method; it is "
            "**not** a clinical tool.", icon="⚠️")
    m = load_models()
    left, right = st.columns([1, 1.3])
    with left:
        age_y = st.slider("Age", 0, 100, 78)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        imd = st.select_slider("Deprivation quintile of home area", options=[1, 2, 3, 4, 5], value=2,
                               format_func=lambda q: {1: "1 (most)", 5: "5 (least)"}.get(q, str(q)))
        los = st.slider("Length of stay (days)", 0, 60, 6)
        emergency = st.toggle("Emergency admission", value=True)
        charlson = st.slider("Charlson comorbidity score", 0, 9, 3)
        ed = st.slider("A&E visits in previous 6 months", 0, 10, 2)
        adm = st.slider("Emergency admissions in previous 12 months", 0, 8, 1)
        care_home = st.toggle("Discharged to a care home")
        poly = st.toggle("10+ regular medicines", value=True)
        medical = st.toggle("Medical specialty", value=True)
        weekend = st.toggle("Weekend discharge")
    band = "<16" if age_y < 16 else ("75+" if age_y >= 75 else "16-74")
    patient = pd.DataFrame([{
        "age": age_y, "age_band": band, "sex": sex, "imd": imd, "los_days": los, "emergency": int(emergency),
        "charlson": charlson, "prior_ed_6m": ed, "prior_adm_12m": adm, "care_home": int(care_home),
        "polypharmacy": int(poly), "medical_specialty": int(medical), "weekend_discharge": int(weekend),
        "ward_occupancy": 0.92, "discharge_hour": 15, "year": P.TEST_YEAR,
    }])
    risk = float(m["champion"].predict(patient)[0])
    lace_pts = int(M.lace_score(los, int(emergency), charlson, ed))
    lace_risk = float(m["lace"].predict(patient)[0])
    with right:
        k1, k2, k3 = st.columns(3)
        k1.metric("30-day readmission risk", f"{risk:.0%}", f"{(risk - summary['event_rate']) * 100:+.0f} pts vs average",
                  delta_color="inverse")
        k2.metric("LACE index", f"{lace_pts} / 19", f"≈ {lace_risk:.0%} risk", delta_color="off")
        k3.metric("Average discharge", f"{summary['event_rate']:.0%}")

        X = m["champion"].design(patient).iloc[0]
        contrib = pd.Series(X.to_numpy() * np.array(list(m["champion"].params.values())), index=X.index)
        groups = {"np.log1p(los_days)": "length of stay", "emergency": "emergency admission", "bs(": "comorbidity",
                  "charlson_c:under75": "comorbidity", "np.minimum(prior_ed_6m": "prior A&E visits",
                  "np.log1p(prior_adm_12m)": "prior admissions", "care_home": "care home", "polypharmacy": "polypharmacy",
                  "medical_specialty": "medical specialty", "weekend_discharge": "weekend discharge", "C(age_band)": "age band",
                  "sex": "sex", "C(imd)": "deprivation"}
        lab = [next((v for k, v in groups.items() if i.startswith(k)), None) for i in contrib.index]
        g = contrib.groupby(lab).sum()
        g = g[g.abs() > 1e-9].sort_values()
        dfc = pd.DataFrame({"factor": g.index, "log_odds": g.values, "odds multiplier": np.exp(g.values)})
        st.markdown("**What drives this patient's risk** (multiplies the odds, relative to a baseline patient)")
        st.altair_chart(alt.Chart(dfc).mark_bar().encode(
            x=alt.X("log_odds:Q", title="contribution to log-odds"),
            y=alt.Y("factor:N", sort=list(dfc["factor"]), title=None),
            color=alt.condition("datum.log_odds > 0", alt.value("#eb6834"), alt.value("#1baf7a")),
            tooltip=["factor", alt.Tooltip("odds multiplier:Q", format=".2f")]).properties(height=300), use_container_width=True)

        st.caption("Each bar is that factor's effect *given all the others*. For older patients, much of the extra "
                   "risk is carried by comorbidity, length of stay and prior admissions, so the age term itself can be small "
                   "or negative.")

        st.markdown("**Should this patient get post-discharge follow-up?**")
        cc1, cc2, cc3 = st.columns(3)
        c_int = cc1.number_input("Follow-up cost (£)", 10, 2000, summary["cost_assumptions"]["intervention"], step=10)
        c_read = cc2.number_input("Readmission cost (£)", 500, 20000, summary["cost_assumptions"]["readmission"], step=100)
        eff = cc3.slider("Readmissions prevented", 0.0, 0.6, summary["cost_assumptions"]["effectiveness"], 0.01, format="%.2f")
        thr = c_int / max(eff * c_read, 1e-9)
        decision = "✅ Offer follow-up" if risk >= thr else "➖ Standard discharge"
        st.write(f"Cost-based threshold: follow up when risk > **{min(thr, 1):.0%}** "
                 f"(cost ÷ (effectiveness × readmission cost)). This patient: **{decision}**.")

        y_val, p_ch, _ = validation_scores()
        _, curve = M.cost_optimal_threshold(y_val, p_ch, c_int, c_read, eff, grid=np.linspace(0.02, 0.8, 79))
        none_cost = c_read * y_val.mean()
        curve["vs no programme (£ per discharge)"] = none_cost - curve["cost_per_discharge"]
        st.altair_chart(alt.Chart(curve).mark_line(color="#2a78d6").encode(
            x=alt.X("threshold:Q", axis=alt.Axis(format="%"), title="threshold"),
            y=alt.Y("vs no programme (£ per discharge):Q", title="saving vs no programme (£ per discharge)"),
            tooltip=[alt.Tooltip("threshold:Q", format=".0%"), alt.Tooltip("vs no programme (£ per discharge):Q", format=",.2f")])
            + alt.Chart(pd.DataFrame({"t": [min(thr, 0.8)]})).mark_rule(strokeDash=[4, 3], color="#52514e").encode(x="t:Q"),
            use_container_width=True)
        st.caption("Across the 2024/25 synthetic validation cohort. The peak sits at the cost-based threshold "
                   "because the model is well calibrated.")

st.divider()
st.caption(f"Champion model, out-of-time validation on {summary['validation_year']}: AUC {summary['auc']['champion']:.3f} "
           f"(LACE {summary['auc']['lace']:.3f}; best achievable {summary['auc']['oracle (true risk)']:.3f}), "
           f"calibration slope {summary['calibration_slope']['champion']:.2f}.")
