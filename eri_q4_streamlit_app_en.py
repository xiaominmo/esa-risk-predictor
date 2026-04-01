import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "outputs" / "eri_q4_paper" / "artifacts"
META = json.loads((ARTIFACT_DIR / "artifact_metadata.json").read_text(encoding="utf-8"))
FORMAL_MODEL = joblib.load(ARTIFACT_DIR / "best_formal_model.joblib")
COMPACT_MODEL = joblib.load(ARTIFACT_DIR / "compact_model.joblib")

FEATURE_DISPLAY = META["feature_display"]
FORMAL_FEATURES = META["formal_features"]
COMPACT_FEATURES = META["compact_features"]
SEX_TO_MODEL = {"Male": "男", "Female": "女"}

DISPLAY_LABEL_OVERRIDES = {
    "current_hyporesponse": "Current-quarter ESA hyporesponse",
    "esa_dose_per_kg_index": "Current-quarter ESA dose per kg",
    "eri_index": "Current-quarter ERI",
    "dialysis_frequency_index": "Dialysis frequency (sessions/week)",
    "hypotension_any_index": "Intradialytic hypotension",
}


def risk_label(probability, cutoffs):
    if probability < cutoffs["low_high_cut"]:
        return "Low risk"
    if probability < cutoffs["mid_high_cut"]:
        return "Intermediate risk"
    return "High risk"


def make_empty_row():
    return {feature: np.nan for feature in FORMAL_FEATURES}


def predict_probability(model, payload):
    frame = pd.DataFrame([payload])
    return float(model.predict_proba(frame)[0, 1])


def metric_card(title, value, help_text=""):
    st.metric(title, value)
    if help_text:
        st.caption(help_text)


def english_feature_label(feature):
    return DISPLAY_LABEL_OVERRIDES.get(feature, FEATURE_DISPLAY.get(feature, feature))


def select_mapped_option(label, options, key):
    display_options = list(options.keys())
    selected = st.selectbox(label, options=display_options, key=key)
    return options[selected], selected


def compact_inputs():
    model_payload = make_empty_row()
    display_payload = {}
    left, right = st.columns(2)

    with left:
        value = st.number_input("Dry weight (kg)", min_value=25.0, max_value=120.0, value=58.0, step=0.1)
        model_payload["dry_weight_index"] = value
        display_payload["dry_weight_index"] = value

        value = st.number_input("Hemoglobin (g/L)", min_value=50.0, max_value=180.0, value=110.0, step=1.0)
        model_payload["hb_index"] = value
        display_payload["hb_index"] = value

        value = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
        model_payload["age"] = value
        display_payload["age"] = value

        model_value, display_value = select_mapped_option("Sex", SEX_TO_MODEL, key="compact_sex")
        model_payload["sex"] = model_value
        display_payload["sex"] = display_value

    with right:
        value = st.number_input("spKt/V", min_value=0.2, max_value=3.0, value=1.40, step=0.01)
        model_payload["spktv_index"] = value
        display_payload["spktv_index"] = value

        value = st.number_input("URR (%)", min_value=10.0, max_value=95.0, value=70.0, step=0.1)
        model_payload["urr_index"] = value
        display_payload["urr_index"] = value

        value = st.number_input("Albumin (g/L)", min_value=20.0, max_value=55.0, value=39.0, step=0.1)
        model_payload["albumin_index"] = value
        display_payload["albumin_index"] = value

        value = st.number_input("Dialysis vintage (months)", min_value=0, max_value=300, value=48, step=1)
        model_payload["dialysis_vintage_months"] = value
        display_payload["dialysis_vintage_months"] = value

    return model_payload, display_payload


def formal_inputs():
    model_payload = make_empty_row()
    display_payload = {}
    st.caption(
        "The full predictor permits missing optional items. Missing values will be handled according to "
        "the preprocessing strategy used during model training."
    )

    group1, group2, group3 = st.columns(3)
    with group1:
        value = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1, key="f_age")
        model_payload["age"] = value
        display_payload["age"] = value

        model_value, display_value = select_mapped_option("Sex", SEX_TO_MODEL, key="f_sex")
        model_payload["sex"] = model_value
        display_payload["sex"] = display_value

        value = st.number_input(
            "Dialysis vintage (months)",
            min_value=0,
            max_value=300,
            value=48,
            step=1,
            key="f_vintage",
        )
        model_payload["dialysis_vintage_months"] = value
        display_payload["dialysis_vintage_months"] = value

        value = st.selectbox("Vascular access", options=["AVF", "Catheter", "Graft", "Unknown"], index=0, key="f_access")
        model_payload["access_type"] = value
        display_payload["access_type"] = value

        value = st.selectbox(
            "Dialysis frequency (sessions/week)",
            options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5],
            index=4,
            key="f_freq",
        )
        model_payload["dialysis_frequency_index"] = value
        display_payload["dialysis_frequency_index"] = value

        model_value, display_value = select_mapped_option("HDF", {"Yes": 1.0, "No": 0.0}, key="f_hdf")
        model_payload["hdf_index"] = model_value
        display_payload["hdf_index"] = display_value

    with group2:
        value = st.number_input("Dry weight (kg)", min_value=25.0, max_value=120.0, value=58.0, step=0.1, key="f_weight")
        model_payload["dry_weight_index"] = value
        display_payload["dry_weight_index"] = value

        value = st.number_input("Hemoglobin (g/L)", min_value=50.0, max_value=180.0, value=110.0, step=1.0, key="f_hb")
        model_payload["hb_index"] = value
        display_payload["hb_index"] = value

        value = st.number_input("Albumin (g/L)", min_value=20.0, max_value=55.0, value=39.0, step=0.1, key="f_albumin")
        model_payload["albumin_index"] = value
        display_payload["albumin_index"] = value

        value = st.number_input("CRP (mg/L)", min_value=0.0, max_value=200.0, value=4.0, step=0.1, key="f_crp")
        model_payload["crp_index"] = value
        display_payload["crp_index"] = value

        value = st.number_input("TSAT (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, key="f_tsat")
        model_payload["tsat_index"] = value
        display_payload["tsat_index"] = value

        value = st.number_input("Ferritin (ng/mL)", min_value=0.0, max_value=3000.0, value=150.0, step=1.0, key="f_ferritin")
        model_payload["ferritin_index"] = value
        display_payload["ferritin_index"] = value

    with group3:
        value = st.number_input("spKt/V", min_value=0.2, max_value=3.0, value=1.40, step=0.01, key="f_spktv")
        model_payload["spktv_index"] = value
        display_payload["spktv_index"] = value

        value = st.number_input("URR (%)", min_value=10.0, max_value=95.0, value=70.0, step=0.1, key="f_urr")
        model_payload["urr_index"] = value
        display_payload["urr_index"] = value

        value = st.number_input(
            "Current-quarter ESA dose (IU/week)",
            min_value=0.0,
            max_value=60000.0,
            value=10000.0,
            step=500.0,
            key="f_esa",
        )
        model_payload["esa_dose_index"] = value
        display_payload["esa_dose_index"] = value

        value = st.number_input(
            "Current-quarter ESA dose per kg",
            min_value=0.0,
            max_value=1000.0,
            value=180.0,
            step=1.0,
            key="f_esakg",
        )
        model_payload["esa_dose_per_kg_index"] = value
        display_payload["esa_dose_per_kg_index"] = value

        value = st.number_input("Current-quarter ERI", min_value=0.0, max_value=80.0, value=16.0, step=0.1, key="f_eri")
        model_payload["eri_index"] = value
        display_payload["eri_index"] = value

        value = st.selectbox("Current-quarter ESA hyporesponse", options=["No", "Yes"], index=0, key="f_hypo")
        model_payload["current_hyporesponse"] = value
        display_payload["current_hyporesponse"] = value

    with st.expander("Optional supplementary variables"):
        a, b, c = st.columns(3)
        with a:
            value = st.number_input("Platelet count", min_value=0.0, max_value=1000.0, value=250.0, step=1.0, key="f_plt")
            model_payload["platelet_index"] = value
            display_payload["platelet_index"] = value

            value = st.number_input("Phosphorus (mmol/L)", min_value=0.0, max_value=5.0, value=2.0, step=0.01, key="f_phos")
            model_payload["phosphorus_index"] = value
            display_payload["phosphorus_index"] = value

            value = st.number_input("PTH", min_value=0.0, max_value=3000.0, value=250.0, step=1.0, key="f_pth")
            model_payload["pth_index"] = value
            display_payload["pth_index"] = value

        with b:
            value = st.number_input("SBP (mmHg)", min_value=60.0, max_value=240.0, value=145.0, step=1.0, key="f_sbp")
            model_payload["sbp_mean_index"] = value
            display_payload["sbp_mean_index"] = value

            value = st.number_input("DBP (mmHg)", min_value=30.0, max_value=140.0, value=78.0, step=1.0, key="f_dbp")
            model_payload["dbp_mean_index"] = value
            display_payload["dbp_mean_index"] = value

            value = st.selectbox("Iron deficiency", options=["Yes", "No"], index=0, key="f_iron_def")
            model_payload["iron_deficiency_index"] = value
            display_payload["iron_deficiency_index"] = value

        with c:
            value = st.selectbox("Low albumin", options=["No", "Yes"], index=0, key="f_lowalb")
            model_payload["low_albumin_index"] = value
            display_payload["low_albumin_index"] = value

            value = st.selectbox("Inflammation", options=["No", "Yes"], index=0, key="f_infl")
            model_payload["inflammation_index"] = value
            display_payload["inflammation_index"] = value

            model_value, display_value = select_mapped_option(
                "Intradialytic hypotension",
                {"No": 0.0, "Yes": 1.0},
                key="f_hypotension",
            )
            model_payload["hypotension_any_index"] = model_value
            display_payload["hypotension_any_index"] = display_value

    return model_payload, display_payload


def render_result(probability, cutoffs, threshold, display_payload, title):
    band = risk_label(probability, cutoffs)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Predicted risk", f"{probability:.1%}")
    with c2:
        metric_card("Risk category", band)
    with c3:
        metric_card("High-risk threshold", f"{threshold:.3f}", "Training-set ROC optimal cutoff")

    st.progress(min(max(probability, 0.0), 1.0))
    st.markdown(
        f"**{title}**: The model estimates a **{probability:.1%}** probability that this patient will "
        "enter the `ERI Top 25%` ESA resistance phenotype in the next quarter."
    )

    summary_rows = []
    for key, value in display_payload.items():
        if pd.isna(value):
            continue
        summary_rows.append({"Variable": english_feature_label(key), "Input value": str(value)})
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="Next-Quarter ESA Resistance Risk Predictor", page_icon="🩺", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1.2rem; max-width: 1200px;}
        .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
        .stTabs [data-baseweb="tab"] {height: 44px; padding-left: 1rem; padding-right: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Next-Quarter ESA Resistance Risk Predictor")
    st.caption(
        f"The primary outcome was defined as entry into the top quartile of ERI in the subsequent quarter "
        f"(training-set Q75 = {META['eri_q75_train']:.4f}). The full calculator is based on the primary "
        "random forest model, whereas the compact calculator is based on a logistic regression model using "
        "8 routinely available variables."
    )

    tab1, tab2 = st.tabs(["Compact Calculator", "Full Calculator"])

    with tab1:
        st.subheader("Compact Calculator")
        st.write(
            "This streamlined interface uses 8 routinely available variables and is suitable for rapid "
            "demonstration in outpatient, clinical, or manuscript settings."
        )
        compact_model_payload, compact_display_payload = compact_inputs()
        if st.button("Calculate compact-model risk", type="primary"):
            payload = {feature: compact_model_payload.get(feature, np.nan) for feature in COMPACT_FEATURES}
            probability = predict_probability(COMPACT_MODEL, payload)
            render_result(
                probability,
                META["compact_risk_bands"],
                META["compact_threshold"],
                compact_display_payload,
                "Compact model result",
            )

    with tab2:
        st.subheader("Full Calculator")
        st.write(
            "This interface uses the primary formal model. Optional fields can be left unfilled and will "
            "be handled according to the original model preprocessing pipeline."
        )
        formal_model_payload, formal_display_payload = formal_inputs()
        if st.button("Calculate full-model risk", type="primary"):
            probability = predict_probability(FORMAL_MODEL, formal_model_payload)
            render_result(
                probability,
                META["risk_bands"],
                META["best_threshold"],
                formal_display_payload,
                "Full model result",
            )


if __name__ == "__main__":
    main()
