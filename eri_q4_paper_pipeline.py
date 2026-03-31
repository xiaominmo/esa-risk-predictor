import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency, mannwhitneyu, skew, ttest_ind
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esa_study_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_transition_data, calibration_curve_frame, choose_threshold, decision_curve, engineer_dataset, read_year


OUTPUT_DIR = ROOT / "outputs" / "eri_q4_paper"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
for directory in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "label_eri_q4_traincut"
TARGET_NAME = "ESA hyporesponsiveness (ERI top quartile phenotype)"

MODEL_DISPLAY = {
    "LR": "LR",
    "DT": "DT",
    "RF": "RF",
    "XGBoost": "XGBoost",
    "SVM": "SVM",
    "KNN": "KNN",
    "LightGBM": "LightGBM",
}

FEATURE_DISPLAY = {
    "age": "Age (years)",
    "dialysis_vintage_months": "Dialysis vintage (months)",
    "hb_index": "Hemoglobin (g/L)",
    "platelet_index": "Platelet count",
    "dry_weight_index": "Dry weight (kg)",
    "esa_dose_index": "ESA dose (IU/week)",
    "esa_dose_per_kg_index": "ESA dose per kg",
    "eri_index": "Current-quarter ERI",
    "ferritin_index": "Ferritin (ng/mL)",
    "tsat_index": "TSAT (%)",
    "albumin_index": "Albumin (g/L)",
    "crp_index": "CRP (mg/L)",
    "phosphorus_index": "Phosphorus (mmol/L)",
    "pth_index": "PTH",
    "spktv_index": "spKt/V",
    "urr_index": "URR (%)",
    "sbp_mean_index": "SBP (mmHg)",
    "dbp_mean_index": "DBP (mmHg)",
    "sex": "Sex",
    "primary_disease_group": "Primary disease",
    "access_type": "Vascular access",
    "esa_use_index": "ESA use",
    "iron_use_index": "Iron use",
    "hif_use_index": "HIF use",
    "dialysis_frequency_index": "Dialysis frequency",
    "hdf_index": "HDF",
    "hp_index": "HP",
    "current_hyporesponse": "Current ESA hyporesponse",
    "iron_deficiency_index": "Iron deficiency",
    "low_albumin_index": "Low albumin",
    "inflammation_index": "Inflammation",
    "hypotension_any_index": "Intradialytic hypotension",
}

YES_NO_MAP = {
    "1.0": "Yes",
    "0.0": "No",
    "1": "Yes",
    "0": "No",
    "使用": "Use",
    "未使用": "No use",
    "Yes": "Yes",
    "No": "No",
    "男": "Male",
    "女": "Female",
}

MAIN_TABLE_VARS = [
    ("Age, years", "age", "continuous"),
    ("Sex, n (%)", "sex", "categorical"),
    ("Dialysis vintage, months", "dialysis_vintage_months", "continuous"),
    ("Primary kidney disease, n (%)", "primary_disease_group", "categorical"),
    ("Vascular access, n (%)", "access_type", "categorical"),
    ("Dialysis frequency, times/week", "dialysis_frequency_index", "continuous"),
    ("HDF, n (%)", "hdf_index", "categorical"),
    ("Dry weight, kg", "dry_weight_index", "continuous"),
    ("Hemoglobin, g/L", "hb_index", "continuous"),
    ("Ferritin, ng/mL", "ferritin_index", "continuous"),
    ("TSAT, %", "tsat_index", "continuous"),
    ("Albumin, g/L", "albumin_index", "continuous"),
    ("CRP, mg/L", "crp_index", "continuous"),
    ("Phosphorus, mmol/L", "phosphorus_index", "continuous"),
    ("PTH", "pth_index", "continuous"),
    ("spKt/V", "spktv_index", "continuous"),
    ("URR, %", "urr_index", "continuous"),
    ("SBP, mmHg", "sbp_mean_index", "continuous"),
    ("DBP, mmHg", "dbp_mean_index", "continuous"),
    ("Iron deficiency, n (%)", "iron_deficiency_index", "categorical"),
    ("Low albumin, n (%)", "low_albumin_index", "categorical"),
    ("Inflammation, n (%)", "inflammation_index", "categorical"),
    ("Current ESA hyporesponse, n (%)", "current_hyporesponse", "categorical"),
    (f"{TARGET_NAME}, n (%)", TARGET_COL, "binary_event"),
]

FULL_TABLE_VARS = [
    ("Age, years", "age", "continuous"),
    ("Sex, n (%)", "sex", "categorical"),
    ("Dialysis vintage, months", "dialysis_vintage_months", "continuous"),
    ("Primary kidney disease, n (%)", "primary_disease_group", "categorical"),
    ("Vascular access, n (%)", "access_type", "categorical"),
    ("ESA use, n (%)", "esa_use_index", "categorical"),
    ("Iron use, n (%)", "iron_use_index", "categorical"),
    ("HIF use, n (%)", "hif_use_index", "categorical"),
    ("Dialysis frequency, times/week", "dialysis_frequency_index", "continuous"),
    ("HDF, n (%)", "hdf_index", "categorical"),
    ("HP, n (%)", "hp_index", "categorical"),
    ("Dry weight, kg", "dry_weight_index", "continuous"),
    ("Hemoglobin, g/L", "hb_index", "continuous"),
    ("Platelet count", "platelet_index", "continuous"),
    ("ESA dose, IU/week", "esa_dose_index", "continuous"),
    ("ESA dose per kg", "esa_dose_per_kg_index", "continuous"),
    ("Current-quarter ERI", "eri_index", "continuous"),
    ("Ferritin, ng/mL", "ferritin_index", "continuous"),
    ("TSAT, %", "tsat_index", "continuous"),
    ("Albumin, g/L", "albumin_index", "continuous"),
    ("CRP, mg/L", "crp_index", "continuous"),
    ("Phosphorus, mmol/L", "phosphorus_index", "continuous"),
    ("PTH", "pth_index", "continuous"),
    ("spKt/V", "spktv_index", "continuous"),
    ("URR, %", "urr_index", "continuous"),
    ("SBP, mmHg", "sbp_mean_index", "continuous"),
    ("DBP, mmHg", "dbp_mean_index", "continuous"),
    ("Iron deficiency, n (%)", "iron_deficiency_index", "categorical"),
    ("Low albumin, n (%)", "low_albumin_index", "categorical"),
    ("Inflammation, n (%)", "inflammation_index", "categorical"),
    ("Intradialytic hypotension, n (%)", "hypotension_any_index", "categorical"),
    ("Current ESA hyporesponse, n (%)", "current_hyporesponse", "categorical"),
    (f"{TARGET_NAME}, n (%)", TARGET_COL, "binary_event"),
]


def load_raw_analysis():
    year_files = {2024: ROOT / "2024年.csv", 2025: ROOT / "2025年.csv"}
    yearly = []
    for year, path in year_files.items():
        yearly.append(engineer_dataset(build_transition_data(read_year(path, year))))
    all_data = pd.concat(yearly, ignore_index=True)
    ids_2024 = set(all_data.loc[all_data["year"] == 2024, "patient_id"].dropna().unique())
    ids_2025 = set(all_data.loc[all_data["year"] == 2025, "patient_id"].dropna().unique())
    all_data["is_2025_new_patient"] = (all_data["year"] == 2025) & all_data["patient_id"].isin(ids_2025 - ids_2024)
    return all_data


def apply_eri_q4_definition(all_data):
    df = all_data.copy()
    outcome_base = (
        df["esa_use_outcome"].eq("使用")
        & df["esa_dose_outcome"].notna()
        & df["hb_outcome"].notna()
        & df["dry_weight_outcome"].notna()
        & (df["dry_weight_outcome"] > 0)
    )
    outcome_no_hif = outcome_base & ~df["hif_use_outcome"].eq("使用")
    train_eligible = (df["year"] == 2024) & outcome_no_hif & df["adult"] & df["minimum_feature_set"]
    eri_q75_train = float(df.loc[train_eligible, "eri_outcome"].quantile(0.75))

    target_flag = (df["eri_outcome"] >= eri_q75_train).fillna(False).astype(int)
    df[TARGET_COL] = np.where(outcome_no_hif, target_flag, np.nan)
    df["label_eri_q4_yearcut"] = np.nan
    for year, year_frame in df.loc[outcome_no_hif].groupby("year"):
        year_cut = float(year_frame["eri_outcome"].quantile(0.75))
        year_mask = (df["year"] == year) & outcome_no_hif
        df.loc[year_mask, "label_eri_q4_yearcut"] = (df.loc[year_mask, "eri_outcome"] >= year_cut).fillna(False).astype(int)

    df["eri_q4_analysis_flag"] = df["adult"] & df["minimum_feature_set"] & df[TARGET_COL].notna()
    summary = {
        "eri_q75_train": eri_q75_train,
        "train_n": int(df.loc[(df["year"] == 2024) & df["eri_q4_analysis_flag"]].shape[0]),
        "train_events": int(df.loc[(df["year"] == 2024) & df["eri_q4_analysis_flag"], TARGET_COL].sum()),
        "valid_n": int(df.loc[(df["year"] == 2025) & df["eri_q4_analysis_flag"]].shape[0]),
        "valid_events": int(df.loc[(df["year"] == 2025) & df["eri_q4_analysis_flag"], TARGET_COL].sum()),
    }
    return df, summary


def split_analysis_sets(df):
    main_df = df[df["eri_q4_analysis_flag"]].copy()
    train_df = main_df[main_df["year"] == 2024].copy()
    valid_df = main_df[main_df["year"] == 2025].copy()
    valid_new_df = valid_df[valid_df["is_2025_new_patient"]].copy()
    return main_df, train_df, valid_df, valid_new_df


def build_preprocessor():
    return ColumnTransformer(
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_models(scale_pos_weight):
    return {
        "LR": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE),
        "DT": DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "LightGBM": LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
        ),
    }


def model_pipeline(estimator):
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def metric_frame(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }


def cross_validated_predictions(train_df, estimator):
    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df["patient_id"].values
    oof = np.zeros(len(train_df))
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pipe = model_pipeline(clone(estimator))
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = choose_threshold(y, oof)
    return oof, threshold, metric_frame(y, oof, threshold)


def fit_full_model(train_df, estimator, threshold):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL].astype(int).values
    pipe = model_pipeline(clone(estimator))
    pipe.fit(X_train, y_train)
    train_prob = pipe.predict_proba(X_train)[:, 1]
    return pipe, train_prob, metric_frame(y_train, train_prob, threshold)


def score_validation(pipe, valid_df, threshold):
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET_COL].astype(int).values
    valid_prob = pipe.predict_proba(X_valid)[:, 1]
    return valid_prob, metric_frame(y_valid, valid_prob, threshold)


def pretty_feature_name(name):
    if "_missing" in name:
        base = name.replace("_missing", "")
        return f"{FEATURE_DISPLAY.get(base, base)} (missing)"
    for raw in sorted(FEATURE_DISPLAY, key=len, reverse=True):
        prefix = raw + "_"
        if name.startswith(prefix):
            level = YES_NO_MAP.get(name[len(prefix) :], name[len(prefix) :])
            return f"{FEATURE_DISPLAY.get(raw, raw)}: {level}"
    return FEATURE_DISPLAY.get(name, name)


def get_feature_names(pipe):
    preprocessor = pipe.named_steps["preprocessor"]
    num_base = list(preprocessor.transformers_[0][2])
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    if getattr(num_imputer, "indicator_", None) is not None:
        for idx in num_imputer.indicator_.features_:
            num_base.append(f"{preprocessor.transformers_[0][2][idx]}_missing")
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(preprocessor.transformers_[1][2])
    )
    return [pretty_feature_name(name) for name in num_base + cat_names]


def curve_payload(valid_df, valid_prob):
    y_true = valid_df[TARGET_COL].astype(int).values
    roc = roc_curve(y_true, valid_prob)
    calibration = calibration_curve_frame(y_true, valid_prob, bins=10)
    dca = decision_curve(y_true, valid_prob, thresholds=np.linspace(0.05, 0.75, 29))
    return {"roc": roc, "calibration": calibration, "dca": dca, "auc": float(roc_auc_score(y_true, valid_prob))}


def select_best_model(results_long):
    valid = results_long[results_long["split"] == "validation_2025"].copy()
    valid = valid.sort_values(["roc_auc", "pr_auc", "f1"], ascending=False)
    return valid.iloc[0]["model"]


def format_pvalue(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def summarize_continuous(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA", False
    use_mean = abs(float(skew(s))) < 1
    if use_mean:
        return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}", True
    return f"{s.median():.2f} ({s.quantile(0.25):.2f}, {s.quantile(0.75):.2f})", False


def pvalue_continuous(train_s, valid_s):
    train_s = pd.to_numeric(train_s, errors="coerce").dropna()
    valid_s = pd.to_numeric(valid_s, errors="coerce").dropna()
    if train_s.empty or valid_s.empty:
        return np.nan
    train_normal = abs(float(skew(train_s))) < 1
    valid_normal = abs(float(skew(valid_s))) < 1
    try:
        if train_normal and valid_normal:
            return float(ttest_ind(train_s, valid_s, equal_var=False, nan_policy="omit").pvalue)
        return float(mannwhitneyu(train_s, valid_s, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def display_level(value):
    return YES_NO_MAP.get(str(value), str(value))


def pvalue_categorical(train_s, valid_s):
    levels = sorted(set(train_s.dropna().astype(str).unique()) | set(valid_s.dropna().astype(str).unique()))
    if not levels:
        return np.nan
    train_counts = train_s.astype(str).value_counts().reindex(levels, fill_value=0)
    valid_counts = valid_s.astype(str).value_counts().reindex(levels, fill_value=0)
    try:
        return float(chi2_contingency(np.vstack([train_counts.values, valid_counts.values])).pvalue)
    except Exception:
        return np.nan


def format_binary_event(frame, column):
    values = frame[column].dropna().astype(int)
    if values.empty:
        return "NA"
    count = int(values.sum())
    total = int(values.shape[0])
    return f"{count} ({count / total * 100:.1f}%)"


def build_baseline_table(total_df, train_df, valid_df, specs):
    rows = []
    for label, column, kind in specs:
        if kind == "continuous":
            total_text, _ = summarize_continuous(total_df[column])
            valid_text, _ = summarize_continuous(valid_df[column])
            train_text, _ = summarize_continuous(train_df[column])
            pvalue = pvalue_continuous(train_df[column], valid_df[column])
            rows.append({"Characteristics": label, "Total": total_text, "Validation/Test": valid_text, "Training": train_text, "p value": format_pvalue(pvalue)})
        elif kind == "categorical":
            pvalue = pvalue_categorical(train_df[column], valid_df[column])
            rows.append({"Characteristics": label, "Total": "", "Validation/Test": "", "Training": "", "p value": format_pvalue(pvalue)})
            levels = sorted(set(total_df[column].dropna().astype(str).unique()))
            for level in levels:
                total_count = int(total_df[column].astype(str).eq(level).sum())
                valid_count = int(valid_df[column].astype(str).eq(level).sum())
                train_count = int(train_df[column].astype(str).eq(level).sum())
                total_n = int(total_df[column].notna().sum())
                valid_n = int(valid_df[column].notna().sum())
                train_n = int(train_df[column].notna().sum())
                rows.append(
                    {
                        "Characteristics": f"  {display_level(level)}",
                        "Total": f"{total_count} ({total_count / max(total_n, 1) * 100:.1f}%)",
                        "Validation/Test": f"{valid_count} ({valid_count / max(valid_n, 1) * 100:.1f}%)",
                        "Training": f"{train_count} ({train_count / max(train_n, 1) * 100:.1f}%)",
                        "p value": "",
                    }
                )
        elif kind == "binary_event":
            pvalue = pvalue_categorical(train_df[column], valid_df[column])
            rows.append(
                {
                    "Characteristics": label,
                    "Total": format_binary_event(total_df, column),
                    "Validation/Test": format_binary_event(valid_df, column),
                    "Training": format_binary_event(train_df, column),
                    "p value": format_pvalue(pvalue),
                }
            )
    return pd.DataFrame(rows)


def build_metrics_tables(results_long):
    split_map = {"train_full": "Training", "validation_2025": "Validation"}
    rows = []
    for model in MODEL_DISPLAY:
        row = {"Model": MODEL_DISPLAY[model]}
        for split_key, split_label in split_map.items():
            frame = results_long[(results_long["model"] == model) & (results_long["split"] == split_key)].iloc[0]
            row[f"{split_label} n"] = int(frame["n"])
            row[f"{split_label} events"] = int(frame["events"])
            row[f"{split_label} AUC"] = f"{frame['roc_auc']:.3f}"
            row[f"{split_label} PR AUC"] = f"{frame['pr_auc']:.3f}"
            row[f"{split_label} Sensitivity"] = f"{frame['sensitivity']:.3f}"
            row[f"{split_label} Specificity"] = f"{frame['specificity']:.3f}"
            row[f"{split_label} Accuracy"] = f"{frame['accuracy']:.3f}"
            row[f"{split_label} PPV"] = f"{frame['ppv']:.3f}"
            row[f"{split_label} NPV"] = f"{frame['npv']:.3f}"
            row[f"{split_label} F1"] = f"{frame['f1']:.3f}"
            row[f"{split_label} Brier"] = f"{frame['brier']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_table_bundle(frame, stem):
    frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    frame.to_excel(TABLE_DIR / f"{stem}.xlsx", index=False)
    (TABLE_DIR / f"{stem}.txt").write_text(frame.to_string(index=False), encoding="utf-8")


def plot_combined_curves(curve_store):
    colors = ["#0b6e4f", "#c1121f", "#003049", "#9d4edd", "#ff8c42", "#457b9d", "#6a994e"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for color, model in zip(colors, MODEL_DISPLAY):
        fpr, tpr, _ = curve_store[model]["roc"]
        axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{MODEL_DISPLAY[model]} (AUC={curve_store[model]['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    axes[0].set_xlabel("1 - Specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("ROC Curves")

    for color, model in zip(colors, MODEL_DISPLAY):
        cal = curve_store[model]["calibration"]
        axes[1].plot(cal["mean_pred"], cal["mean_obs"], marker="o", lw=1.8, color=color, label=MODEL_DISPLAY[model])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Observed probability")
    axes[1].set_title("Calibration Curves")

    for color, model in zip(colors, MODEL_DISPLAY):
        dca = curve_store[model]["dca"]
        axes[2].plot(dca["threshold"], dca["model"], lw=2, color=color, label=MODEL_DISPLAY[model])
    example_dca = next(iter(curve_store.values()))["dca"]
    axes[2].plot(example_dca["threshold"], example_dca["treat_all"], linestyle="--", color="black", lw=1.5, label="Treat all")
    axes[2].plot(example_dca["threshold"], example_dca["treat_none"], linestyle=":", color="gray", lw=1.5, label="Treat none")
    axes[2].set_xlabel("Threshold probability")
    axes[2].set_ylabel("Net benefit")
    axes[2].set_title("Decision Curve Analysis")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(FIG_DIR / "figure_models_roc_calibration_dca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_shap(best_model_name, pipe, train_df, valid_df):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names(pipe)
    train_sample = train_df.sample(min(400, len(train_df)), random_state=RANDOM_STATE)
    valid_sample = valid_df.sample(min(400, len(valid_df)), random_state=RANDOM_STATE)
    X_bg = pd.DataFrame(preprocessor.transform(train_sample[FEATURE_COLUMNS]), columns=feature_names)
    X_eval = pd.DataFrame(preprocessor.transform(valid_sample[FEATURE_COLUMNS]), columns=feature_names)

    if best_model_name in {"DT", "RF", "XGBoost", "LightGBM"}:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_eval)
        shap_values = raw[1] if isinstance(raw, list) and len(raw) > 1 else raw
    elif best_model_name == "LR":
        explainer = shap.LinearExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_eval)
    else:
        background = X_bg.sample(min(80, len(X_bg)), random_state=RANDOM_STATE)
        eval_small = X_eval.sample(min(120, len(X_eval)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(eval_small, nsamples=150)
        shap_values = raw[1] if isinstance(raw, list) else raw
        X_eval = eval_small

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    importance = pd.DataFrame({"feature": X_eval.columns, "mean_abs_shap": np.abs(shap_array).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False
    )

    top20 = importance.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, 6))
    plt.barh(top20["feature"], top20["mean_abs_shap"], color="#1d3557")
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"SHAP Bar Plot: {MODEL_DISPLAY[best_model_name]}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_shap_summary_bar.png", dpi=220, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_array, X_eval, show=False, max_display=20, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_shap_summary_dot.png", dpi=220, bbox_inches="tight")
    plt.close()

    numeric_labels = [FEATURE_DISPLAY.get(name, name) for name in NUMERIC_FEATURES]
    dependence_records = []
    numeric_ranked = [feat for feat in importance["feature"] if feat in numeric_labels]
    for idx, feature in enumerate(numeric_ranked[:5], start=1):
        shap.dependence_plot(feature, shap_array, X_eval, show=False, interaction_index=None)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(":", "").replace(" ", "_")
        out_path = FIG_DIR / f"figure_shap_dependence_{idx}_{safe_name}.png"
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        dependence_records.append({"feature": feature, "path": str(out_path)})

    importance.to_csv(TABLE_DIR / "shap_importance.csv", index=False, encoding="utf-8-sig")
    return importance, dependence_records


def collapse_to_raw_feature(display_name):
    for raw, label in FEATURE_DISPLAY.items():
        if display_name == label or display_name.startswith(label + ":") or display_name.startswith(label + " (missing)"):
            return raw
    return display_name


def choose_compact_features(shap_importance):
    preferred = [
        "age",
        "sex",
        "dialysis_vintage_months",
        "access_type",
        "dry_weight_index",
        "hb_index",
        "albumin_index",
        "crp_index",
        "tsat_index",
        "ferritin_index",
        "spktv_index",
        "urr_index",
    ]
    selected = []
    for feature in shap_importance["feature"]:
        raw = collapse_to_raw_feature(feature)
        if raw in preferred and raw not in selected:
            selected.append(raw)
        if len(selected) >= 8:
            break
    for raw in preferred:
        if raw not in selected:
            selected.append(raw)
        if len(selected) >= 8:
            break
    return selected[:8]


def build_compact_model(train_df, valid_df, compact_features):
    compact_numeric = [f for f in compact_features if f in NUMERIC_FEATURES]
    compact_categorical = [f for f in compact_features if f in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), compact_numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
                    ]
                ),
                compact_categorical,
            ),
        ]
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))])
    X_train = train_df[compact_features]
    y_train = train_df[TARGET_COL].astype(int).values
    X_valid = valid_df[compact_features]
    y_valid = valid_df[TARGET_COL].astype(int).values
    pipe.fit(X_train, y_train)
    train_prob = pipe.predict_proba(X_train)[:, 1]
    valid_prob = pipe.predict_proba(X_valid)[:, 1]
    threshold = choose_threshold(y_train, train_prob)
    return pipe, threshold, metric_frame(y_train, train_prob, threshold), metric_frame(y_valid, valid_prob, threshold), train_prob


def risk_cutoffs(train_prob):
    return {"low_high_cut": float(np.quantile(train_prob, 0.50)), "mid_high_cut": float(np.quantile(train_prob, 0.75))}


def run_models(train_df, valid_df, valid_new_df):
    pos = float(train_df[TARGET_COL].sum())
    neg = float(len(train_df) - pos)
    scale_pos_weight = neg / pos if pos else 1.0
    models = make_models(scale_pos_weight=scale_pos_weight)
    rows = []
    fitted = {}
    curve_store = {}
    for name, estimator in models.items():
        _, threshold, cv_metrics = cross_validated_predictions(train_df, estimator)
        pipe, train_prob, train_metrics = fit_full_model(train_df, estimator, threshold)
        valid_prob, valid_metrics = score_validation(pipe, valid_df, threshold)
        _, new_metrics = score_validation(pipe, valid_new_df, threshold)
        fitted[name] = {"pipe": pipe, "threshold": threshold, "train_prob": train_prob}
        curve_store[name] = curve_payload(valid_df, valid_prob)
        rows.append({"split": "internal_cv_2024", "model": name, **cv_metrics})
        rows.append({"split": "train_full", "model": name, **train_metrics})
        rows.append({"split": "validation_2025", "model": name, **valid_metrics})
        rows.append({"split": "new_patient_validation_2025", "model": name, **new_metrics})
    return pd.DataFrame(rows), fitted, curve_store


def write_report(definition_summary, train_df, valid_df, valid_new_df, results_long, best_name, shap_importance, compact_features, compact_metrics):
    best_valid = results_long[(results_long["model"] == best_name) & (results_long["split"] == "validation_2025")].iloc[0]
    lines = [
        "# ERI-Q4 paper-style analysis package",
        "",
        "## Cohort and endpoint",
        f"- Main endpoint: {TARGET_NAME}",
        f"- 2024 training ERI Q75 cutoff: {definition_summary['eri_q75_train']:.4f}",
        f"- Training set: {len(train_df)} records, {int(train_df[TARGET_COL].sum())} events",
        f"- Validation set: {len(valid_df)} records, {int(valid_df[TARGET_COL].sum())} events",
        f"- 2025 new-patient validation: {len(valid_new_df)} records, {int(valid_new_df[TARGET_COL].sum())} events",
        "",
        "## Best-performing model",
        f"- Model: {MODEL_DISPLAY[best_name]}",
        f"- Validation ROC AUC: {best_valid['roc_auc']:.3f}",
        f"- Validation PR AUC: {best_valid['pr_auc']:.3f}",
        f"- Validation F1: {best_valid['f1']:.3f}",
        "",
        "## Top SHAP features",
    ]
    lines.extend([f"- {row.feature}: {row.mean_abs_shap:.4f}" for row in shap_importance.head(10).itertuples(index=False)])
    lines.extend(
        [
            "",
            "## Compact calculator",
            f"- Compact features: {', '.join(FEATURE_DISPLAY.get(f, f) for f in compact_features)}",
            f"- Training ROC AUC: {compact_metrics['train']['roc_auc']:.3f}",
            f"- Validation ROC AUC: {compact_metrics['valid']['roc_auc']:.3f}",
        ]
    )
    (OUTPUT_DIR / "study_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    all_data = load_raw_analysis()
    all_data, definition_summary = apply_eri_q4_definition(all_data)
    main_df, train_df, valid_df, valid_new_df = split_analysis_sets(all_data)

    main_df.to_csv(OUTPUT_DIR / "analysis_dataset_eri_q4_main.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([definition_summary]).to_csv(OUTPUT_DIR / "definition_summary.csv", index=False, encoding="utf-8-sig")

    baseline_main = build_baseline_table(main_df, train_df, valid_df, MAIN_TABLE_VARS)
    baseline_full = build_baseline_table(main_df, train_df, valid_df, FULL_TABLE_VARS)
    save_table_bundle(baseline_main, "table1_baseline_main")
    save_table_bundle(baseline_full, "tableS1_baseline_full")

    results_long, fitted, curve_store = run_models(train_df, valid_df, valid_new_df)
    results_long.to_csv(TABLE_DIR / "model_metrics_long.csv", index=False, encoding="utf-8-sig")
    results_long.to_excel(TABLE_DIR / "model_metrics_long.xlsx", index=False)
    metrics_main = build_metrics_tables(results_long)
    save_table_bundle(metrics_main, "table2_model_metrics_main")

    best_name = select_best_model(results_long)
    plot_combined_curves(curve_store)

    best_pipe = fitted[best_name]["pipe"]
    best_threshold = float(fitted[best_name]["threshold"])
    best_train_prob = fitted[best_name]["train_prob"]
    shap_importance, dependence_records = compute_shap(best_name, best_pipe, train_df, valid_df)

    compact_features = choose_compact_features(shap_importance)
    compact_pipe, compact_threshold, compact_train_metrics, compact_valid_metrics, compact_train_prob = build_compact_model(
        train_df, valid_df, compact_features
    )

    artifact_meta = {
        "target_col": TARGET_COL,
        "target_name": TARGET_NAME,
        "eri_q75_train": definition_summary["eri_q75_train"],
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "compact_threshold": compact_threshold,
        "risk_bands": risk_cutoffs(best_train_prob),
        "compact_risk_bands": risk_cutoffs(compact_train_prob),
        "formal_features": FEATURE_COLUMNS,
        "compact_features": compact_features,
        "feature_display": FEATURE_DISPLAY,
        "compact_metrics": {"train": compact_train_metrics, "valid": compact_valid_metrics},
        "dependence_plots": dependence_records,
    }

    joblib.dump(best_pipe, ARTIFACT_DIR / "best_formal_model.joblib")
    joblib.dump(compact_pipe, ARTIFACT_DIR / "compact_model.joblib")
    (ARTIFACT_DIR / "artifact_metadata.json").write_text(json.dumps(artifact_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(
        definition_summary,
        train_df,
        valid_df,
        valid_new_df,
        results_long,
        best_name,
        shap_importance,
        compact_features,
        {"train": compact_train_metrics, "valid": compact_valid_metrics},
    )

    with pd.ExcelWriter(OUTPUT_DIR / "eri_q4_paper_results.xlsx") as writer:
        baseline_main.to_excel(writer, sheet_name="Table1_main", index=False)
        baseline_full.to_excel(writer, sheet_name="TableS1_full", index=False)
        metrics_main.to_excel(writer, sheet_name="Table2_metrics", index=False)
        results_long.to_excel(writer, sheet_name="Metrics_long", index=False)
        shap_importance.to_excel(writer, sheet_name="SHAP_importance", index=False)
        pd.DataFrame([definition_summary]).to_excel(writer, sheet_name="Definition", index=False)

    summary = {
        "definition": definition_summary,
        "best_model": best_name,
        "best_validation_metrics": results_long[(results_long["model"] == best_name) & (results_long["split"] == "validation_2025")].iloc[0].to_dict(),
        "compact_features": compact_features,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
