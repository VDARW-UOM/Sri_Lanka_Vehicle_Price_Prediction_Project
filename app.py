"""Streamlit MSc assignment demo: vehicle price estimates with global and local XAI.

Reads the files saved by Notebook 02: the validation-selected price model, the glass-box EBM,
test metrics for all candidates, and the XAI outputs. EBM contributions explain ONLY EBM, never XGBoost.
"""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
OUT = BASE / "vehicle_ml_outputs"
DATA = OUT / "cleaned_vehicle_data.csv"
XGB_MODEL = OUT / "trained_vehicle_price_model.pkl"
EBM_MODEL = OUT / "ebm_vehicle_price_model.pkl"
PERM_CSV = OUT / "feature_importance.csv"
EBM_IMP = OUT / "ebm_global_importance.csv"
METRICS = OUT / "test_metrics.csv"
EBM_SHAPES = OUT / "ebm_shape_functions.png"
META = OUT / "model_metadata.json"

# Original-feature schema; must match Notebook 02 (Section 1).
CAT = ["Brand", "Model", "Edition", "Transmission", "FuelType", "BodyType", "Negotiable"]
NUM = ["ManufactureYear", "EngineCC", "MileageKM"]
FEATURES = CAT + NUM
EBM_PRICE_DIVISOR = 1_000_000  # EBM is trained on PriceLKR / 1,000,000 (Notebook 02, Section 3)


def prepare_ebm_input(table: pd.DataFrame) -> pd.DataFrame:
    """Retain original feature names; let EBM handle numeric NaNs natively."""
    x = table[FEATURES].copy()
    for col in CAT:
        x[col] = x[col].fillna("Unknown").astype(str)
    for col in NUM:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
    return x


def normalize_importance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Support both original Notebook 02 importance CSV column spellings."""
    value_col = next((v for v in ("MAE_Increase_LKR", "IncreaseInMAE_LKR", "MAE_increase_when_shuffled_LKR")
                      if v in frame.columns), None)
    if value_col is None or "Feature" not in frame.columns:
        raise ValueError("Unrecognized feature importance CSV columns")
    out = frame[["Feature", value_col]].rename(columns={value_col: "MAE_increase_LKR"}).copy()
    out["MAE_increase_LKR"] = pd.to_numeric(out["MAE_increase_LKR"], errors="coerce")
    return out.dropna().sort_values("MAE_increase_LKR", ascending=False)


def ebm_local_contributions(model, one_record: pd.DataFrame) -> dict:
    """EBM explanation in LKR. Does NOT explain the separate XGBoost estimate.

    The EBM is fitted to PriceLKR / 1,000,000 with an identity link, so its
    intercept and individual main-effect term scores add up exactly to its
    prediction in millions of rupees. Do not use with a log-target EBM.
    """
    x = prepare_ebm_input(one_record)
    if len(x) != 1:
        raise ValueError("Local EBM explanations require exactly one record")
    term_values = np.asarray(model.eval_terms(x), dtype=float).reshape(1, -1)[0]
    names = list(model.term_names_)
    if len(names) != len(term_values):
        raise ValueError("Term names do not match EBM scores")
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    terms = [{"Feature": name, "Contribution_LKR": float(value * EBM_PRICE_DIVISOR)}
             for name, value in zip(names, term_values)]
    reconstructed_lkr = (intercept + float(np.sum(term_values))) * EBM_PRICE_DIVISOR
    predicted_lkr = float(np.asarray(model.predict(x)).ravel()[0] * EBM_PRICE_DIVISOR)
    if not np.isclose(reconstructed_lkr, predicted_lkr, atol=1.0):
        raise ValueError("EBM additive explanation does not reconstruct its prediction")
    return {
        "intercept_lkr": intercept * EBM_PRICE_DIVISOR,
        "sum_lkr": predicted_lkr,
        "terms": sorted(terms, key=lambda r: abs(r["Contribution_LKR"]), reverse=True),
    }


st.set_page_config(page_title="Vehicle Price Prediction & XAI · MSc", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1160px; padding-top:1.9rem;}
.hero {background:linear-gradient(110deg,#edf5fc,#eaf7f4);border:1px solid #d6e7ed;
  border-radius:18px;padding:1.6rem 1.8rem;margin-bottom:1rem;}
.hero h1 {font-size:2.15rem;line-height:1.14;letter-spacing:-.035em;color:#163453;margin:.2rem 0 .5rem;}
.hero p {margin:0;color:#3c5b70;font-size:.98rem;}
.eyebrow {color:#297e86;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:750;}
.prediction {background:#f2f9fc;border:1px solid #cfe4ed;border-radius:14px;padding:1.3rem;}
.prediction .number {font-size:2.1rem;color:#143b54;font-weight:800;}
.prediction .hint {color:#476579;font-size:.86rem;}
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class='hero'><div class='eyebrow'>MSc in Artificial Intelligence · Machine Learning Assignment</div>
<h1>Used Vehicle Price Prediction & Explainable AI</h1>
<p>Sri Lankan advertised vehicle prices · XGBoost comparator · Explainable Boosting Machine (EBM)</p></div>""",
unsafe_allow_html=True)

if not DATA.exists() or not XGB_MODEL.exists():
    st.error("Missing cleaned dataset or XGBoost model. Run Notebooks 01 and 02 first.")
    st.stop()


@st.cache_resource(show_spinner="Loading trained models…")
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def read_csv(path: str):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


records = read_csv(str(DATA))
assert set(FEATURES + ["PriceLKR"]).issubset(records.columns), "Unexpected cleaned data schema"
for col in CAT:
    records[col] = records[col].fillna("Unknown").astype(str)
for col in NUM:
    records[col] = pd.to_numeric(records[col], errors="coerce")
xgb = load_model(str(XGB_MODEL))
meta = read_json(str(META)) if META.exists() else {}
selected_name = meta.get("model", "XGBoost")
# Older output folders may lack the EBM; the app then runs without its sections.
ebm = load_model(str(EBM_MODEL)) if EBM_MODEL.exists() else None


def lkr(value: float) -> str:
    return "Rs. {:,.0f}".format(float(value))


def choices(col: str, subset: pd.DataFrame | None = None) -> list[str]:
    table = records if subset is None or subset.empty else subset
    options = sorted({str(v) for v in table[col].dropna() if str(v).strip()})
    return options or ["Unknown"]


def option_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


prediction_tab, xai_tab, metrics_tab, about_tab = st.tabs([
    "🚗 Estimate price", "🔎 Explainable AI (XAI)",
    "📊 Model comparison", "📖 Method & limitations",
])

with prediction_tab:
    st.subheader("Enter vehicle specifications")
    st.caption("This research predicts a seller's advertised asking price, not a confirmed sale price.")
    with st.container():
        left, right = st.columns(2, gap="large")
        with left:
            brands = choices("Brand")
            brand = st.selectbox("Brand", brands, index=option_index(brands, "Toyota"))
            brand_subset = records[records.Brand == brand]
            models = choices("Model", brand_subset)
            vehicle_model = st.selectbox("Model", models)
            subset = brand_subset[brand_subset.Model == vehicle_model]
            edition = st.selectbox("Edition", choices("Edition", subset))
            year = st.number_input("Manufacture year", min_value=1950, max_value=date.today().year + 1,
                                   value=min(2018, date.today().year), step=1)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=800000,
                                      value=75000, step=1000)
        with right:
            fuel = st.selectbox("Fuel type", choices("FuelType", subset))
            transmission = st.selectbox("Transmission", choices("Transmission", subset))
            body = st.selectbox("Body type", choices("BodyType", subset))
            negotiable = st.selectbox("Negotiable", choices("Negotiable"))
            median_cc = subset.EngineCC.median()
            common_cc = int(median_cc) if np.isfinite(median_cc) else 1500
            engine_cc = st.number_input("Engine capacity (cc)", min_value=0, max_value=8000,
                                        value=max(0, min(common_cc, 8000)), step=100,
                                        help="Enter 0 for electric or unknown engine capacity.")
        submitted = st.button("Predict asking price", type="primary", width="stretch")

    if submitted:
        st.session_state["vehicle_input"] = pd.DataFrame([{
            "Brand": brand, "Model": vehicle_model, "Edition": edition,
            "ManufactureYear": year, "Transmission": transmission,
            "FuelType": fuel, "BodyType": body,
            "EngineCC": np.nan if engine_cc == 0 else float(engine_cc),
            "MileageKM": float(mileage), "Negotiable": negotiable,
        }])[FEATURES]

    row = st.session_state.get("vehicle_input")
    if row is not None:
        try:
            estimate = max(0.0, float(xgb.predict(row)[0]))
            st.markdown(
                f"<div class='prediction'><div>Validation-selected model ({escape(selected_name)}) · advertised price estimate</div>"
                f"<div class='number'>{lkr(estimate)}</div>"
                "<div class='hint'>Indicative model prediction. Not a verified market valuation.</div></div>",
                unsafe_allow_html=True,
            )
            if ebm is None:
                st.info("EBM model not found. Run `02_Vehicle_Price_Regression_Submission.ipynb` "
                        "to enable the EBM estimate and local explanation.")
            else:
                ebm_prediction = float(ebm.predict(prepare_ebm_input(row))[0]) * EBM_PRICE_DIVISOR
                st.metric("Glass-box EBM estimate", lkr(max(0.0, ebm_prediction)))
                with st.expander("Explain THIS EBM prediction (local XAI)", expanded=True):
                    try:
                        local = ebm_local_contributions(ebm, row)
                        st.caption("The baseline and contributions sum to the EBM prediction. "
                                   "Contributions are in LKR and can be positive or negative. "
                                   "They explain the EBM estimate only, NOT the selected model's estimate above.")
                        st.write("**EBM baseline (intercept):**", lkr(local["intercept_lkr"]))
                        contrib = pd.DataFrame(local["terms"])
                        st.bar_chart(contrib.set_index("Feature")["Contribution_LKR"])
                        st.dataframe(contrib.style.format({"Contribution_LKR": "{:+,.0f}"}),
                                     hide_index=True, width="stretch")
                        st.write("**EBM result reconstructed:**", lkr(local["sum_lkr"]))
                    except Exception as exc:
                        st.warning("Local explanation could not be generated: " + str(exc))
        except Exception as exc:
            st.error("Prediction failed: " + str(exc))
    else:
        st.caption("Enter specifications and select **Predict asking price** to view an estimate.")

with xai_tab:
    st.subheader("Which features matter overall? — permutation importance")
    st.markdown(f"The same model-agnostic test is applied to **{selected_name}** (the selected "
                "black-box model) and to **EBM**: one feature is shuffled at a time and the increase in "
                "validation MAE is measured. Larger increases mean the model relies more on that feature.")
    if PERM_CSV.exists():
        try:
            raw_importance = read_csv(str(PERM_CSV))
            imp = normalize_importance_table(raw_importance).set_index("Feature")
            imp = imp.rename(columns={"MAE_increase_LKR": selected_name})
            if "EBM_MAE_Increase_LKR" in raw_importance:
                imp["EBM"] = raw_importance.set_index("Feature")["EBM_MAE_Increase_LKR"]
            st.bar_chart(imp, stack=False, horizontal=True)
            st.dataframe(imp.style.format("{:,.0f}"), width="stretch")
            st.caption("Global explanation across validation records, not the explanation "
                       "of any one vehicle. Correlated features can share importance; "
                       "permutation importance does not prove causation.")
        except Exception as exc:
            st.warning("Couldn't display permutation importance: " + str(exc))
    else:
        st.info("Execute Notebook 02 to generate `feature_importance.csv`.")

    if ebm is not None:
        st.divider()
        st.subheader("EBM built-in explanations")
        if EBM_IMP.exists():
            eimp = read_csv(str(EBM_IMP))
            if {"Feature", "MeanAbsoluteContribution_MillionLKR"}.issubset(eimp):
                st.bar_chart(eimp.set_index("Feature")["MeanAbsoluteContribution_MillionLKR"],
                             horizontal=True)
                st.caption("Term importance: mean absolute size of each EBM term, in million LKR.")
        if EBM_SHAPES.exists():
            st.image(str(EBM_SHAPES), width="stretch")
            st.caption("Shape functions: the amount each numeric feature adds to or subtracts from "
                       "EBM's baseline at every value. Because EBM is additive, these curves are the "
                       "model itself. They describe EBM only, not the selected model.")

with metrics_tab:
    st.subheader("Independent held-out test comparison")
    if METRICS.exists():
        results = read_csv(str(METRICS)).copy()
        display_cols = [c for c in ("Model", "MAE_LKR", "RMSE_LKR", "R2", "Median_APE_pct")
                        if c in results.columns]
        st.dataframe(results[display_cols].style.format({
            "MAE_LKR": "{:,.0f}", "RMSE_LKR": "{:,.0f}",
            "R2": "{:.3f}", "Median_APE_pct": "{:.1f}",
        }), hide_index=True, width="stretch")
        st.bar_chart(results.set_index("Model")["MAE_LKR"])
        st.caption("Lower MAE and RMSE are better; higher R² is better. R² does not mean "
                   "'percentage of exact correct price predictions'.")
        selected = results[results.Model == selected_name]
        if not selected.empty:
            result = selected.iloc[0]
            a, b, c = st.columns(3)
            a.metric(f"Selected {selected_name} MAE", lkr(result.MAE_LKR))
            b.metric(f"Selected {selected_name} RMSE", lkr(result.RMSE_LKR))
            c.metric(f"Selected {selected_name} R²", f"{result.R2:.3f}")
    else:
        st.warning("Please execute Notebook 02 to generate test metrics.")
    st.info("**Model-selection rule:** all candidates, EBM included, competed on validation MAE "
            "before held-out test results were reviewed. EBM is kept alongside the selected model "
            "for its built-in explanations, whatever its accuracy rank.")

with about_tab:
    st.subheader("Implementation and research context")
    st.markdown("""
    **Problem and dataset.** Predict seller-advertised asking prices from publicly
    accessible Sri Lankan vehicle listings scraped from ikman.lk.
    This is not a verified transaction-price database.

    **Data cleaning.** Standardized column names and categorical text; parsed
    price/mileage/engine numerics; removed exact duplicate records; treated invalid
    or implausible measurements as missing; recorded a cleaning audit.

    **Validation.** Group-disjoint train/validation/test splits based on all original
    input features; preprocessing fitted on training folds; held-out test used for
    final reporting rather than model selection.

    **Candidate models.** A median baseline, a Decision Tree, three black-box tree
    ensembles (Random Forest, XGBoost, LightGBM; log-price target, sklearn
    preprocessing pipeline) and a glass-box Explainable Boosting Machine (EBM;
    native categorical terms, additive effects in **millions of LKR**), all trained
    and evaluated under the same protocol.

    **Selection.** Lowest validation MAE among configured and cross-validation-tuned
    candidates; the default-settings XGBoost won and serves the price estimate.
    EBM is clearly less accurate on held-out data (see *Model comparison*); it
    trades accuracy for built-in transparency.

    **XAI.** Permutation importance for both models (global); EBM term importance
    and shape functions (global); and an exact decomposition of each EBM estimate
    into a baseline plus feature contributions (local). EBM explanations describe
    EBM only, not XGBoost. None of these establishes causality.

    **Limitations and ethics.** No advert timestamp or unique ID, possible reposts
    and deposit advertisements, uneven brand representation, limited coverage
    of luxury vehicles, model uncertainty, and no verified consent assessment
    for redistributing the original advertisements. The prototype should not
    be used for loans, insurance decisions, or formal valuations without
    professional inspection and independent market evidence.
    """)
    st.caption("Academic prototype · Results reflect the collected listings, not real-time prices.")

st.divider()
st.caption("MSc Machine Learning assignment · Interpretability and responsible use · Sri Lanka")
