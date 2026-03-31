import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ID_COL = "\u4e3bID"
CENTER_COL = "\u7cfb\u7edf_\u521b\u5efa\u8005"
STATUS_COL = "\u900f\u6790\u72b6\u6001_\u60a3\u8005\u72b6\u6001"
AGE_COL = "\u57fa\u672c\u60c5\u51b5_\u5e74\u9f84"
SEX_COL = "\u57fa\u672c\u60c5\u51b5_\u6027\u522b"
VINTAGE_COL = "\u57fa\u672c\u60c5\u51b5_\u900f\u6790\u9f84"
PRIMARY_COL = "\u539f\u53d1\u75c5\u8bca\u65ad_\u4e3b\u8981\u539f\u53d1\u75c5\u75be\u75c5"
ACCESS_COL = "\u901a\u8def_\u8840\u7ba1\u901a\u8def\u7c7b\u578b_HY2"

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
TRANSITIONS = [("Q1", "Q2", "Q3"), ("Q2", "Q3", "Q4"), ("Q3", "Q4", None)]
QUARTER_MONTHS = {
    "Q1": ["M1", "M2", "M3"],
    "Q2": ["M4", "M5", "M6"],
    "Q3": ["M7", "M8", "M9"],
    "Q4": ["M10", "M11", "M12"],
}

MISSING_TEXTS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "--",
    "\u672a\u67e5",
    "\u4e0d\u8be6",
}

NUMERIC_FEATURES = [
    "age",
    "dialysis_vintage_months",
    "hb_index",
    "platelet_index",
    "dry_weight_index",
    "esa_dose_index",
    "esa_dose_per_kg_index",
    "eri_index",
    "ferritin_index",
    "tsat_index",
    "albumin_index",
    "crp_index",
    "phosphorus_index",
    "pth_index",
    "spktv_index",
    "urr_index",
    "sbp_mean_index",
    "dbp_mean_index",
]

CATEGORICAL_FEATURES = [
    "sex",
    "primary_disease_group",
    "access_type",
    "esa_use_index",
    "iron_use_index",
    "hif_use_index",
    "dialysis_frequency_index",
    "hdf_index",
    "hp_index",
    "current_hyporesponse",
    "iron_deficiency_index",
    "low_albumin_index",
    "inflammation_index",
    "hypotension_any_index",
]

OR_FEATURES = [
    "age",
    "male",
    "dialysis_vintage_months",
    "diabetic_kidney_disease",
    "catheter_access",
    "hb_index",
    "esa_dose_per_kg_index",
    "iron_deficiency_index_num",
    "iron_use_index_num",
    "albumin_index",
    "crp_index",
    "spktv_index",
    "hypotension_any_index_num",
    "phosphorus_index",
]


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip("'").strip('"')
    if not text:
        return np.nan
    if text.lower() in MISSING_TEXTS:
        return np.nan
    return text


def clean_categorical(series):
    return series.map(normalize_text)


def to_numeric(series):
    cleaned = clean_categorical(series).astype("string")
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def parse_frequency(series):
    cleaned = clean_categorical(series)

    def _parse(text):
        if pd.isna(text):
            return np.nan
        if "\u6b21/2\u5468" in text:
            num = pd.to_numeric(pd.Series([text]).str.extract(r"(\d+\.?\d*)")[0], errors="coerce").iloc[0]
            return num / 2 if pd.notna(num) else np.nan
        if "-" in text and "\u6b21/\u5468" in text:
            left = text.split("\u6b21/\u5468")[0]
            vals = [float(v) for v in left.split("-") if v.strip().isdigit()]
            return sum(vals) / len(vals) if vals else np.nan
        return pd.to_numeric(pd.Series([text]).str.extract(r"(\d+\.?\d*)")[0], errors="coerce").iloc[0]

    return cleaned.map(_parse)


def encode_yes_no(series):
    cleaned = clean_categorical(series)
    return cleaned.map(
        {
            "\u662f": 1,
            "\u5426": 0,
            "\u6709": 1,
            "\u65e0": 0,
            "\u4f7f\u7528": 1,
            "\u672a\u4f7f\u7528": 0,
        }
    )


def map_primary_group(series):
    cleaned = clean_categorical(series)

    def _map(text):
        if pd.isna(text):
            return "Unknown"
        if "\u7cd6\u5c3f\u75c5" in text:
            return "Diabetic kidney disease"
        if "\u9ad8\u8840\u538b" in text:
            return "Hypertensive nephropathy"
        if "\u539f\u53d1\u6027\u80be\u5c0f\u7403" in text:
            return "Primary glomerular disease"
        if "\u611f\u67d3" in text or "\u7ed3\u77f3" in text:
            return "Infection or stone disease"
        if "\u539f\u53d1\u75c5\u4e0d\u660e\u786e" in text:
            return "Unknown primary disease"
        return "Other"

    return cleaned.map(_map)


def map_access_type(series):
    cleaned = clean_categorical(series)

    def _map(text):
        if pd.isna(text):
            return "Unknown"
        if "\u7f6e\u7ba1" in text:
            return "Catheter"
        if "\u5185\u7618" in text:
            return "AVF"
        if "\u79fb\u690d" in text:
            return "Graft"
        return text

    return cleaned.map(_map)


def quarter_col(prefix, quarter):
    return f"{prefix}_{quarter}"


def month_col(prefix, month):
    return f"{prefix}_{month}"


def quarter_bp_mean(df, prefix, quarter):
    cols = [month_col(prefix, m) for m in QUARTER_MONTHS[quarter]]
    matrix = pd.concat([to_numeric(df[c]) for c in cols], axis=1)
    return matrix.mean(axis=1, skipna=True)


def quarter_any(df, prefix, quarter):
    cols = [month_col(prefix, m) for m in QUARTER_MONTHS[quarter]]
    clean = pd.concat([clean_categorical(df[c]) for c in cols], axis=1)
    yes = clean.eq("\u6709")
    out = yes.max(axis=1).astype(float)
    out[clean.isna().all(axis=1)] = np.nan
    return out


def read_year(path, year):
    df = pd.read_csv(path, encoding="gb18030", dtype=str, low_memory=False).fillna("")
    df["year"] = year
    df["patient_id"] = clean_categorical(df[ID_COL])
    df["center"] = clean_categorical(df[CENTER_COL]).fillna("Unknown")
    df["status"] = clean_categorical(df[STATUS_COL]).fillna("Unknown")
    df["age"] = to_numeric(df[AGE_COL])
    df["sex"] = clean_categorical(df[SEX_COL]).fillna("Unknown")
    df["dialysis_vintage_months"] = to_numeric(df[VINTAGE_COL])
    df["primary_disease"] = clean_categorical(df[PRIMARY_COL]).fillna("Unknown")
    df["primary_disease_group"] = map_primary_group(df[PRIMARY_COL])
    df["access_type"] = map_access_type(df[ACCESS_COL])
    return df


def quarter_features(df, quarter, prefix):
    out = pd.DataFrame(index=df.index)
    out[f"esa_use_{prefix}"] = clean_categorical(df[quarter_col("ESA_\u662f\u5426\u4f7f\u7528", quarter)])
    out[f"esa_dose_{prefix}"] = to_numeric(df[quarter_col("ESA_\u5242\u91cf_\u5468", quarter)])
    out[f"esa_unit_{prefix}"] = clean_categorical(df[quarter_col("ESA_\u5242\u91cf\u5355\u4f4d", quarter)])
    out[f"hif_use_{prefix}"] = clean_categorical(df[quarter_col("HIF_\u662f\u5426\u4f7f\u7528", quarter)])
    out[f"iron_use_{prefix}"] = clean_categorical(df[quarter_col("\u94c1\u5242_\u662f\u5426\u4f7f\u7528", quarter)])
    out[f"dialysis_frequency_{prefix}"] = parse_frequency(
        df[quarter_col("\u900f\u6790\u5904\u65b9_\u900f\u6790_\u9891\u6b21", quarter)]
    )
    out[f"hdf_{prefix}"] = encode_yes_no(df[quarter_col("\u900f\u6790\u5904\u65b9_HDF\u6cbb\u7597", quarter)])
    out[f"hp_{prefix}"] = encode_yes_no(df[quarter_col("\u900f\u6790\u5904\u65b9_HP\u6cbb\u7597", quarter)])
    out[f"dry_weight_{prefix}"] = to_numeric(
        df[quarter_col("\u900f\u6790\u5145\u5206\u6027_\u5e72\u4f53\u91cd_\u6570\u503c", quarter)]
    )
    out[f"urr_{prefix}"] = to_numeric(df[quarter_col("\u900f\u6790\u5145\u5206\u6027_URR", quarter)])
    out[f"spktv_{prefix}"] = to_numeric(df[quarter_col("\u900f\u6790\u5145\u5206\u6027_spKtV", quarter)])
    out[f"hb_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u8840\u5e38\u89c4_\u8840\u7ea2\u86cb\u767d", quarter)]
    )
    out[f"platelet_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u8840\u5e38\u89c4_\u8840\u5c0f\u677f", quarter)]
    )
    out[f"tsat_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u94c1\u4ee3\u8c22_\u8f6c\u94c1\u9971\u548c\u5ea6", quarter)]
    )
    out[f"ferritin_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u94c1\u4ee3\u8c22_\u94c1\u86cb\u767d", quarter)]
    )
    out[f"albumin_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u751f\u5316\u68c0\u67e5_\u8840\u767d\u86cb\u767d", quarter)]
    )
    out[f"crp_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u8425\u517b\u4e0e\u708e\u75c7_C\u53cd\u5e94\u86cb\u767d", quarter)]
    )
    out[f"phosphorus_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u9aa8\u77ff\u7269\u8d28\u4ee3\u8c22_\u8840\u78f7", quarter)]
    )
    out[f"pth_{prefix}"] = to_numeric(
        df[quarter_col("\u5b9e\u9a8c\u5ba4\u68c0\u67e5_\u9aa8\u77ff\u7269\u8d28\u4ee3\u8c22_PTH", quarter)]
    )
    out[f"sbp_mean_{prefix}"] = quarter_bp_mean(df, "\u8840\u538b_\u900f\u6790\u524dSBP", quarter)
    out[f"dbp_mean_{prefix}"] = quarter_bp_mean(df, "\u8840\u538b_\u900f\u6790\u524dDBP", quarter)
    out[f"hypotension_any_{prefix}"] = quarter_any(df, "\u8840\u538b_\u900f\u6790\u76f8\u5173\u4f4e\u8840\u538b", quarter)
    return out


def build_transition_data(df):
    parts = []
    base = df[
        [
            "year",
            "patient_id",
            "center",
            "status",
            "age",
            "sex",
            "dialysis_vintage_months",
            "primary_disease",
            "primary_disease_group",
            "access_type",
        ]
    ].copy()
    for index_q, outcome_q, next2_q in TRANSITIONS:
        part = base.copy()
        part["index_quarter"] = index_q
        part["outcome_quarter"] = outcome_q
        part = pd.concat(
            [part, quarter_features(df, index_q, "index"), quarter_features(df, outcome_q, "outcome")],
            axis=1,
        )
        if next2_q is not None:
            part = pd.concat([part, quarter_features(df, next2_q, "next2")], axis=1)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["male"] = out["sex"].eq("\u7537").astype(int)
    out["diabetic_kidney_disease"] = out["primary_disease_group"].eq("Diabetic kidney disease").astype(int)
    out["catheter_access"] = out["access_type"].eq("Catheter").astype(int)
    return out


def engineer_dataset(data):
    df = data.copy()
    df["esa_dose_per_kg_index"] = df["esa_dose_index"] / df["dry_weight_index"]
    df["eri_index"] = df["esa_dose_index"] / df["dry_weight_index"] / (df["hb_index"] / 10.0)
    df["eri_outcome"] = df["esa_dose_outcome"] / df["dry_weight_outcome"] / (df["hb_outcome"] / 10.0)

    df["iron_deficiency_index"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["tsat_index"].notna() | df["ferritin_index"].notna(), "iron_deficiency_index"] = "No"
    df.loc[(df["tsat_index"] < 20) | (df["ferritin_index"] < 200), "iron_deficiency_index"] = "Yes"

    df["low_albumin_index"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["albumin_index"].notna(), "low_albumin_index"] = "No"
    df.loc[df["albumin_index"] < 35, "low_albumin_index"] = "Yes"

    df["inflammation_index"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["crp_index"].notna(), "inflammation_index"] = "No"
    df.loc[df["crp_index"] > 5, "inflammation_index"] = "Yes"

    df["current_hyporesponse"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["esa_dose_index"].notna() & df["hb_index"].notna(), "current_hyporesponse"] = "No"
    df.loc[(df["esa_dose_index"] >= 10000) & (df["hb_index"] < 100), "current_hyporesponse"] = "Yes"

    outcome_base = (
        df["esa_use_outcome"].eq("\u4f7f\u7528")
        & df["esa_dose_outcome"].notna()
        & df["hb_outcome"].notna()
        & df["dry_weight_outcome"].notna()
        & (df["dry_weight_outcome"] > 0)
    )
    outcome_no_hif = outcome_base & ~df["hif_use_outcome"].eq("\u4f7f\u7528")
    main_logic = ((df["esa_dose_outcome"] >= 10000) & (df["hb_outcome"] < 100)).fillna(False)
    df["label_main"] = np.where(outcome_no_hif, main_logic.astype(int), np.nan)
    df["label_main_keep_hif"] = np.where(outcome_base, main_logic.astype(int), np.nan)
    df["label_hb105"] = np.where(
        outcome_no_hif,
        (((df["esa_dose_outcome"] >= 10000) & (df["hb_outcome"] < 105)).fillna(False)).astype(int),
        np.nan,
    )
    df["label_dose12000"] = np.where(
        outcome_no_hif,
        (((df["esa_dose_outcome"] >= 12000) & (df["hb_outcome"] < 100)).fillna(False)).astype(int),
        np.nan,
    )
    df["label_dose15000"] = np.where(
        outcome_no_hif,
        (((df["esa_dose_outcome"] >= 15000) & (df["hb_outcome"] < 100)).fillna(False)).astype(int),
        np.nan,
    )
    df["label_eri12"] = np.where(outcome_no_hif, (df["eri_outcome"] >= 12).fillna(False).astype(int), np.nan)
    df["label_eri15"] = np.where(outcome_no_hif, (df["eri_outcome"] >= 15).fillna(False).astype(int), np.nan)
    df["label_eri15_4"] = np.where(outcome_no_hif, (df["eri_outcome"] >= 15.4).fillna(False).astype(int), np.nan)
    iron_replete_mask = outcome_no_hif & (df["tsat_outcome"] >= 20).fillna(False) & (df["ferritin_outcome"] >= 200).fillna(False)
    df["label_iron_replete"] = np.where(iron_replete_mask, main_logic.astype(int), np.nan)
    if {"esa_use_next2", "esa_dose_next2", "hb_next2", "dry_weight_next2", "hif_use_next2"}.issubset(df.columns):
        next2_base = (
            df["esa_use_next2"].eq("\u4f7f\u7528")
            & df["esa_dose_next2"].notna()
            & df["hb_next2"].notna()
            & df["dry_weight_next2"].notna()
            & (df["dry_weight_next2"] > 0)
            & ~df["hif_use_next2"].eq("\u4f7f\u7528")
        )
        next2_logic = ((df["esa_dose_next2"] >= 10000) & (df["hb_next2"] < 100)).fillna(False)
        df["label_persistent"] = np.where(
            outcome_no_hif & next2_base,
            (main_logic & next2_logic).astype(int),
            np.nan,
        )
    else:
        df["label_persistent"] = np.nan

    df["iron_deficiency_index_num"] = pd.Series(df["iron_deficiency_index"]).map({"Yes": 1, "No": 0})
    df["iron_use_index_num"] = pd.Series(df["iron_use_index"]).map({"\u4f7f\u7528": 1, "\u672a\u4f7f\u7528": 0})
    df["hypotension_any_index_num"] = df["hypotension_any_index"].astype(float)

    df["adult"] = df["age"] >= 18
    df["minimum_feature_set"] = (
        df["age"].notna()
        & df["sex"].notna()
        & df["dialysis_vintage_months"].notna()
        & df["hb_index"].notna()
        & df["dry_weight_index"].notna()
        & df["esa_use_index"].notna()
    )
    df["main_analysis_flag"] = df["adult"] & df["minimum_feature_set"] & df["label_main"].notna()
    df["main_keep_hif_flag"] = df["adult"] & df["minimum_feature_set"] & df["label_main_keep_hif"].notna()
    df["iron_complete_flag"] = df["main_analysis_flag"] & df["tsat_index"].notna() & df["ferritin_index"].notna()
    df["continuous_pair_flag"] = df["main_analysis_flag"] & df["label_persistent"].notna()
    return df.replace([np.inf, -np.inf], np.nan)


def build_pipeline(model_name):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if model_name == "logistic":
        model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif model_name == "hgb":
        model = GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=300,
            random_state=42,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def choose_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    return max(min(threshold, 0.95), 0.05)


def calibration_stats(y_true, y_prob):
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    try:
        intercept_fit = sm.GLM(
            y_true,
            np.ones((len(y_true), 1)),
            family=sm.families.Binomial(),
            offset=logit_p,
        ).fit()
        slope_fit = sm.GLM(
            y_true,
            sm.add_constant(logit_p),
            family=sm.families.Binomial(),
        ).fit()
        return float(intercept_fit.params[0]), float(slope_fit.params[1])
    except Exception:
        return np.nan, np.nan


def metric_frame(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    specificity = ((y_true == 0) & (y_pred == 0)).sum() / max((y_true == 0).sum(), 1)
    npv = ((y_true == 0) & (y_pred == 0)).sum() / max((y_pred == 0).sum(), 1)
    calibration_intercept, calibration_slope = calibration_stats(y_true, y_prob)
    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "calibration_intercept": float(calibration_intercept) if pd.notna(calibration_intercept) else np.nan,
        "calibration_slope": float(calibration_slope) if pd.notna(calibration_slope) else np.nan,
    }


def cross_validated_predictions(train_df, label_col, model_name):
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[label_col].astype(int).values
    groups = train_df["patient_id"].values
    oof = np.zeros(len(train_df))
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pipe = build_pipeline(model_name)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = choose_threshold(y, oof)
    return oof, threshold, metric_frame(y, oof, threshold)


def fit_and_predict(train_df, test_df, label_col, model_name, threshold):
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[label_col].astype(int).values
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[label_col].astype(int).values
    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    return pipe, prob, metric_frame(y_test, prob, threshold)


def build_or_table(train_df, label_col):
    data = train_df.copy()
    data["male"] = data["sex"].eq("\u7537").astype(int)
    data["diabetic_kidney_disease"] = data["primary_disease_group"].eq("Diabetic kidney disease").astype(int)
    data["catheter_access"] = data["access_type"].eq("Catheter").astype(int)
    data["iron_deficiency_index_num"] = data["iron_deficiency_index"].map({"Yes": 1, "No": 0})
    data["iron_use_index_num"] = data["iron_use_index"].map({"\u4f7f\u7528": 1, "\u672a\u4f7f\u7528": 0})
    data["hypotension_any_index_num"] = data["hypotension_any_index"].fillna(0)
    X = data[OR_FEATURES].apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    X = sm.add_constant(X).astype(float)
    y = data[label_col].astype(int).values
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["center"]},
    )
    conf = fit.conf_int()
    return pd.DataFrame(
        {
            "feature": X.columns,
            "coef": fit.params,
            "or": np.exp(fit.params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
            "p_value": fit.pvalues,
        }
    ).reset_index(drop=True)


def logistic_coefficients(pipe):
    preprocessor = pipe.named_steps["preprocessor"]
    num_base = preprocessor.transformers_[0][2]
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    num_names = list(num_base)
    if getattr(num_imputer, "indicator_", None) is not None:
        for idx in num_imputer.indicator_.features_:
            num_names.append(f"{num_base[idx]}_missing")
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
            preprocessor.transformers_[1][2]
        )
    )
    feature_names = num_names + cat_names
    coef = pipe.named_steps["model"].coef_[0]
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": coef, "abs_coefficient": np.abs(coef)}
    ).sort_values("abs_coefficient", ascending=False)


def permutation_importance_frame(pipe, valid_df):
    X = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = valid_df["label_main"].astype(int).values
    result = permutation_importance(
        pipe,
        X,
        y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def calibration_curve_frame(y_true, y_prob, bins=10):
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    return (
        frame.groupby("bin", observed=False)
        .agg(mean_pred=("p", "mean"), mean_obs=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )


def decision_curve(y_true, y_prob, thresholds):
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append(
            {
                "threshold": threshold,
                "model": net_benefit,
                "treat_all": treat_all,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_plots(y_true, logistic_prob, hgb_prob):
    fpr1, tpr1, _ = roc_curve(y_true, logistic_prob)
    fpr2, tpr2, _ = roc_curve(y_true, hgb_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr1, tpr1, label="Logistic")
    plt.plot(fpr2, tpr2, label="GradientBoosting")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Temporal validation ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temporal_validation_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    for label, prob in [("Logistic", logistic_prob), ("GradientBoosting", hgb_prob)]:
        curve = calibration_curve_frame(y_true, prob)
        plt.plot(curve["mean_pred"], curve["mean_obs"], marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed risk")
    plt.title("Temporal validation calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temporal_validation_calibration.png", dpi=180)
    plt.close()

    thresholds = np.linspace(0.05, 0.5, 19)
    dca_logistic = decision_curve(y_true, logistic_prob, thresholds)
    dca_hgb = decision_curve(y_true, hgb_prob, thresholds)
    plt.figure(figsize=(6, 5))
    plt.plot(dca_logistic["threshold"], dca_logistic["model"], label="Logistic")
    plt.plot(dca_hgb["threshold"], dca_hgb["model"], label="GradientBoosting")
    plt.plot(dca_logistic["threshold"], dca_logistic["treat_all"], linestyle="--", label="Treat all")
    plt.plot(dca_logistic["threshold"], dca_logistic["treat_none"], linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temporal_validation_decision_curve.png", dpi=180)
    plt.close()
    dca_logistic.to_csv(OUTPUT_DIR / "decision_curve_logistic.csv", index=False)
    dca_hgb.to_csv(OUTPUT_DIR / "decision_curve_hgb.csv", index=False)


def sensitivity_runs(train_df, valid_df):
    specs = [
        ("label_main", "Main outcome"),
        ("label_hb105", "Hb < 105 g/L"),
        ("label_dose12000", "ESA >= 12000 IU/week"),
        ("label_dose15000", "ESA >= 15000 IU/week"),
        ("label_eri12", "ERI >= 12"),
        ("label_eri15", "ERI >= 15"),
        ("label_eri15_4", "ERI >= 15.4"),
        ("label_iron_replete", "Iron-replete main outcome"),
        ("label_persistent", "Persistent hyporesponse"),
    ]
    rows = []
    for label_col, label_name in specs:
        tr = train_df[train_df[label_col].notna()].copy()
        va = valid_df[valid_df[label_col].notna()].copy()
        if len(tr) < 200 or len(va) < 50 or tr[label_col].sum() < 25 or va[label_col].sum() < 10:
            continue
        _, threshold, _ = cross_validated_predictions(tr, label_col, "logistic")
        _, _, metrics = fit_and_predict(tr, va, label_col, "logistic", threshold)
        rows.append({"analysis": label_name, **metrics})
    return pd.DataFrame(rows)


def summary_table(train_df, valid_df, valid_new_df):
    rows = []
    for name, frame in [
        ("train_2024", train_df),
        ("validation_2025", valid_df),
        ("validation_2025_new_patients", valid_new_df),
    ]:
        rows.append(
            {
                "dataset": name,
                "n_records": int(len(frame)),
                "n_patients": int(frame["patient_id"].nunique()),
                "events_main": int(frame["label_main"].sum()),
                "event_rate_main": float(frame["label_main"].mean()),
                "events_eri15": int(frame["label_eri15"].sum()),
                "event_rate_eri15": float(frame["label_eri15"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary_df, metrics_df, odds_df, sensitivity_df):
    summary_text = summary_df.to_string(index=False)
    metrics_text = metrics_df.to_string(index=False)
    odds_text = odds_df.sort_values("p_value").head(12).to_string(index=False)
    sensitivity_text = (
        sensitivity_df.to_string(index=False)
        if not sensitivity_df.empty
        else "No sensitivity analysis met the minimum sample threshold."
    )
    lines = [
        "# ESA hyporesponsiveness study report",
        "",
        "Main operational outcome: next-quarter ESA use, no HIF use, ESA dose >= 10000 IU/week, and Hb < 100 g/L.",
        "",
        "## Dataset summary",
        summary_text,
        "",
        "## Model performance",
        metrics_text,
        "",
        "## Selected odds ratios",
        odds_text,
        "",
        "## Sensitivity analyses",
        sensitivity_text,
        "",
        "## Literature anchors",
        "- KDIGO 2012: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-Anemia-Guideline-English.pdf",
        "- KDIGO 2026 summary: https://kdigo.org/wp-content/uploads/2026/01/KDIGO-2026-Anemia-in-CKD-Guideline-Executive-Summary.pdf",
        "- Wu 2022 review: https://pmc.ncbi.nlm.nih.gov/articles/PMC9021651/",
        "- Sibbel 2015: https://pubmed.ncbi.nlm.nih.gov/26283069/",
        "- Gilbertson 2013: https://pmc.ncbi.nlm.nih.gov/articles/PMC3586346/",
        "- Okazaki 2014: https://pubmed.ncbi.nlm.nih.gov/24603656/",
    ]
    (OUTPUT_DIR / "study_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    year_files = {2024: ROOT / "2024年.csv", 2025: ROOT / "2025年.csv"}
    data = [engineer_dataset(build_transition_data(read_year(path, year))) for year, path in year_files.items()]
    all_data = pd.concat(data, ignore_index=True)

    ids_2024 = set(all_data.loc[all_data["year"] == 2024, "patient_id"].dropna().unique())
    ids_2025 = set(all_data.loc[all_data["year"] == 2025, "patient_id"].dropna().unique())
    all_data["is_2025_new_patient"] = (all_data["year"] == 2025) & all_data["patient_id"].isin(ids_2025 - ids_2024)

    main_df = all_data[all_data["main_analysis_flag"]].copy()
    train_df = main_df[main_df["year"] == 2024].copy()
    valid_df = main_df[main_df["year"] == 2025].copy()
    valid_new_df = valid_df[valid_df["is_2025_new_patient"]].copy()

    _, logistic_threshold, logistic_cv_metrics = cross_validated_predictions(train_df, "label_main", "logistic")
    logistic_pipe, logistic_valid_prob, logistic_valid_metrics = fit_and_predict(
        train_df, valid_df, "label_main", "logistic", logistic_threshold
    )
    _, logistic_new_prob, logistic_new_metrics = fit_and_predict(
        train_df, valid_new_df, "label_main", "logistic", logistic_threshold
    )

    _, hgb_threshold, hgb_cv_metrics = cross_validated_predictions(train_df, "label_main", "hgb")
    hgb_pipe, hgb_valid_prob, hgb_valid_metrics = fit_and_predict(
        train_df, valid_df, "label_main", "hgb", hgb_threshold
    )
    _, hgb_new_prob, hgb_new_metrics = fit_and_predict(
        train_df, valid_new_df, "label_main", "hgb", hgb_threshold
    )

    metrics_df = pd.DataFrame(
        [
            {"split": "internal_cv_2024", "model": "Logistic", **logistic_cv_metrics},
            {"split": "temporal_validation_2025", "model": "Logistic", **logistic_valid_metrics},
            {"split": "new_patient_validation_2025", "model": "Logistic", **logistic_new_metrics},
            {"split": "internal_cv_2024", "model": "GradientBoosting", **hgb_cv_metrics},
            {"split": "temporal_validation_2025", "model": "GradientBoosting", **hgb_valid_metrics},
            {"split": "new_patient_validation_2025", "model": "GradientBoosting", **hgb_new_metrics},
        ]
    )

    odds_df = build_or_table(train_df, "label_main")
    logistic_importance_df = logistic_coefficients(logistic_pipe)
    hgb_importance_df = permutation_importance_frame(hgb_pipe, valid_df)
    sensitivity_df = sensitivity_runs(train_df, valid_df)
    summary_df = summary_table(train_df, valid_df, valid_new_df)

    y_valid = valid_df["label_main"].astype(int).values
    save_plots(y_valid, logistic_valid_prob, hgb_valid_prob)
    calibration_curve_frame(y_valid, logistic_valid_prob).to_csv(OUTPUT_DIR / "calibration_logistic.csv", index=False)
    calibration_curve_frame(y_valid, hgb_valid_prob).to_csv(OUTPUT_DIR / "calibration_hgb.csv", index=False)

    main_df.to_csv(OUTPUT_DIR / "analysis_dataset_main.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    odds_df.to_csv(OUTPUT_DIR / "logistic_odds_ratios.csv", index=False, encoding="utf-8-sig")
    logistic_importance_df.to_csv(OUTPUT_DIR / "logistic_feature_importance.csv", index=False, encoding="utf-8-sig")
    hgb_importance_df.to_csv(OUTPUT_DIR / "hgb_feature_importance.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_metrics.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_DIR / "esa_study_results.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="dataset_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        odds_df.to_excel(writer, sheet_name="odds_ratios", index=False)
        logistic_importance_df.to_excel(writer, sheet_name="logit_importance", index=False)
        hgb_importance_df.to_excel(writer, sheet_name="hgb_importance", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)

    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(
            {
                "logistic_threshold": logistic_threshold,
                "hgb_threshold": hgb_threshold,
                "dataset_summary": summary_df.to_dict(orient="records"),
                "model_metrics": metrics_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(summary_df, metrics_df, odds_df, sensitivity_df)


if __name__ == "__main__":
    main()
