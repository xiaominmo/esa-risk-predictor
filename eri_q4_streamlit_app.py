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


def risk_label(probability, cutoffs):
    if probability < cutoffs["low_high_cut"]:
        return "低风险"
    if probability < cutoffs["mid_high_cut"]:
        return "中风险"
    return "高风险"


def make_empty_row():
    return {feature: np.nan for feature in FORMAL_FEATURES}


def predict_probability(model, payload):
    frame = pd.DataFrame([payload])
    return float(model.predict_proba(frame)[0, 1])


def metric_card(title, value, help_text=""):
    st.metric(title, value)
    if help_text:
        st.caption(help_text)


def select_binary(label, key, yes="Yes", no="No", default="No"):
    return st.selectbox(label, options=[no, yes], index=0 if default == no else 1, key=key)


def compact_inputs():
    payload = make_empty_row()
    left, right = st.columns(2)
    with left:
        payload["dry_weight_index"] = st.number_input("干体重 (kg)", min_value=25.0, max_value=120.0, value=58.0, step=0.1)
        payload["hb_index"] = st.number_input("血红蛋白 Hb (g/L)", min_value=50.0, max_value=180.0, value=110.0, step=1.0)
        payload["age"] = st.number_input("年龄 (岁)", min_value=18, max_value=100, value=60, step=1)
        payload["sex"] = st.selectbox("性别", options=["男", "女"], index=0)
    with right:
        payload["spktv_index"] = st.number_input("spKt/V", min_value=0.2, max_value=3.0, value=1.40, step=0.01)
        payload["urr_index"] = st.number_input("URR (%)", min_value=10.0, max_value=95.0, value=70.0, step=0.1)
        payload["albumin_index"] = st.number_input("白蛋白 (g/L)", min_value=20.0, max_value=55.0, value=39.0, step=0.1)
        payload["dialysis_vintage_months"] = st.number_input("透析龄 (月)", min_value=0, max_value=300, value=48, step=1)
    return payload


def formal_inputs():
    payload = make_empty_row()
    st.caption("完整版预测器允许缺失项留空，模型会按训练集规则自动填补。")
    group1, group2, group3 = st.columns(3)
    with group1:
        payload["age"] = st.number_input("年龄 (岁)", min_value=18, max_value=100, value=60, step=1, key="f_age")
        payload["sex"] = st.selectbox("性别", options=["男", "女"], index=0, key="f_sex")
        payload["dialysis_vintage_months"] = st.number_input("透析龄 (月)", min_value=0, max_value=300, value=48, step=1, key="f_vintage")
        payload["access_type"] = st.selectbox("血管通路", options=["AVF", "Catheter", "Graft", "Unknown"], index=0, key="f_access")
        payload["dialysis_frequency_index"] = st.selectbox("透析频次 (次/周)", options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5], index=4, key="f_freq")
        payload["hdf_index"] = st.selectbox("是否 HDF", options=[1.0, 0.0], index=0, key="f_hdf")
    with group2:
        payload["dry_weight_index"] = st.number_input("干体重 (kg)", min_value=25.0, max_value=120.0, value=58.0, step=0.1, key="f_weight")
        payload["hb_index"] = st.number_input("Hb (g/L)", min_value=50.0, max_value=180.0, value=110.0, step=1.0, key="f_hb")
        payload["albumin_index"] = st.number_input("白蛋白 (g/L)", min_value=20.0, max_value=55.0, value=39.0, step=0.1, key="f_albumin")
        payload["crp_index"] = st.number_input("CRP (mg/L)", min_value=0.0, max_value=200.0, value=4.0, step=0.1, key="f_crp")
        payload["tsat_index"] = st.number_input("TSAT (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, key="f_tsat")
        payload["ferritin_index"] = st.number_input("铁蛋白 (ng/mL)", min_value=0.0, max_value=3000.0, value=150.0, step=1.0, key="f_ferritin")
    with group3:
        payload["spktv_index"] = st.number_input("spKt/V", min_value=0.2, max_value=3.0, value=1.40, step=0.01, key="f_spktv")
        payload["urr_index"] = st.number_input("URR (%)", min_value=10.0, max_value=95.0, value=70.0, step=0.1, key="f_urr")
        payload["esa_dose_index"] = st.number_input("当季 ESA 剂量 (IU/周)", min_value=0.0, max_value=60000.0, value=10000.0, step=500.0, key="f_esa")
        payload["esa_dose_per_kg_index"] = st.number_input("当季 ESA 剂量/kg", min_value=0.0, max_value=1000.0, value=180.0, step=1.0, key="f_esakg")
        payload["eri_index"] = st.number_input("当季 ERI", min_value=0.0, max_value=80.0, value=16.0, step=0.1, key="f_eri")
        payload["current_hyporesponse"] = select_binary("当季 ESA 低反应", key="f_hypo")

    with st.expander("可选补充变量"):
        a, b, c = st.columns(3)
        with a:
            payload["platelet_index"] = st.number_input("血小板", min_value=0.0, max_value=1000.0, value=250.0, step=1.0, key="f_plt")
            payload["phosphorus_index"] = st.number_input("血磷 (mmol/L)", min_value=0.0, max_value=5.0, value=2.0, step=0.01, key="f_phos")
            payload["pth_index"] = st.number_input("PTH", min_value=0.0, max_value=3000.0, value=250.0, step=1.0, key="f_pth")
        with b:
            payload["sbp_mean_index"] = st.number_input("收缩压 (mmHg)", min_value=60.0, max_value=240.0, value=145.0, step=1.0, key="f_sbp")
            payload["dbp_mean_index"] = st.number_input("舒张压 (mmHg)", min_value=30.0, max_value=140.0, value=78.0, step=1.0, key="f_dbp")
            payload["iron_deficiency_index"] = select_binary("铁缺乏", key="f_iron_def", default="Yes")
        with c:
            payload["low_albumin_index"] = select_binary("低白蛋白", key="f_lowalb")
            payload["inflammation_index"] = select_binary("炎症", key="f_infl")
            payload["hypotension_any_index"] = st.selectbox("透析中低血压", options=[0.0, 1.0], index=0, key="f_hypotension")
    return payload


def render_result(probability, cutoffs, threshold, payload, title):
    band = risk_label(probability, cutoffs)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("预测风险", f"{probability:.1%}")
    with c2:
        metric_card("风险分层", band)
    with c3:
        metric_card("高风险阈值", f"{threshold:.3f}", "训练集 ROC 最优阈值")

    st.progress(min(max(probability, 0.0), 1.0))
    st.markdown(f"**{title}**：模型预测该患者下一季度进入 `ERI Top 25%` ESA 抵抗表型的概率为 **{probability:.1%}**。")

    summary_rows = []
    for key, value in payload.items():
        if pd.isna(value):
            continue
        summary_rows.append({"变量": FEATURE_DISPLAY.get(key, key), "输入值": str(value)})
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="ESA 抵抗风险预测器", page_icon="🩺", layout="wide")
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

    st.title("下一季度 ESA 抵抗风险预测器")
    st.caption(
        f"正式结局定义为下一季度进入 ERI 最高四分位（训练集 Q75={META['eri_q75_train']:.4f}）。"
        f" 正式模型为 {META['best_model_name']}，同时提供简化版计算器。"
    )

    tab1, tab2 = st.tabs(["简化版计算器", "完整版预测器"])

    with tab1:
        st.subheader("简化版计算器")
        st.write("只需输入 8 个常规变量，适合门诊或研究展示时快速估算。")
        compact_payload = compact_inputs()
        if st.button("计算简化版风险", type="primary"):
            payload = {feature: compact_payload.get(feature, np.nan) for feature in COMPACT_FEATURES}
            probability = predict_probability(COMPACT_MODEL, payload)
            render_result(probability, META["compact_risk_bands"], META["compact_threshold"], payload, "简化版模型结果")

    with tab2:
        st.subheader("完整版预测器")
        st.write("调用正式最优模型；未填写项目会自动按训练集规则补全。")
        formal_payload = formal_inputs()
        if st.button("计算正式模型风险", type="primary"):
            probability = predict_probability(FORMAL_MODEL, formal_payload)
            render_result(probability, META["risk_bands"], META["best_threshold"], formal_payload, "正式模型结果")


if __name__ == "__main__":
    main()
