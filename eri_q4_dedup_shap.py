import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eri_q4_paper_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_DISPLAY,
    NUMERIC_FEATURES,
    TARGET_COL,
    choose_threshold,
    metric_frame,
    pretty_feature_name,
)


OUTPUT_DIR = ROOT / "outputs" / "eri_q4_paper" / "dedup_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_FEATURES = ["esa_dose_per_kg_index", "esa_dose_index", "current_hyporesponse"]
KEPT_ANEMIA_SIGNAL = "eri_index"
RANDOM_STATE = 42
PAPER_FONT_FAMILY = "Arial"
PAPER_FONT_SIZE = 10.5
PAPER_PNG_DPI = 300

PAPER_LABEL_MAP = {
    "Hemoglobin (g/L)": "Hemoglobin",
    "Dry weight (kg)": "Dry weight",
    "Age (years)": "Age",
    "Albumin (g/L)": "Albumin",
    "URR (%)": "URR",
    "DBP (mmHg)": "DBP",
    "TSAT (%)": "TSAT",
    "Dialysis vintage (months)": "Dialysis vintage (months)",
    "Ferritin (ng/mL)": "Ferritin",
    "SBP (mmHg)": "SBP",
    "Phosphorus (mmol/L)": "Phosphorus",
    "CRP (mg/L)": "CRP",
    "Sex: Male": "Male",
    "Vascular access: AVF": "AVF",
    "Vascular access: Catheter": "Catheter",
    "HDF: Yes": "HDF",
}


matplotlib.rcParams.update(
    {
        "font.family": PAPER_FONT_FAMILY,
        "font.size": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_FONT_SIZE,
        "axes.titlesize": PAPER_FONT_SIZE,
        "xtick.labelsize": PAPER_FONT_SIZE,
        "ytick.labelsize": PAPER_FONT_SIZE,
        "legend.fontsize": PAPER_FONT_SIZE,
        "figure.titlesize": PAPER_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_data():
    df = pd.read_csv(ROOT / "outputs" / "eri_q4_paper" / "analysis_dataset_eri_q4_main.csv", encoding="utf-8-sig")
    train_df = df[df["year"] == 2024].copy()
    valid_df = df[df["year"] == 2025].copy()
    return train_df, valid_df


def paper_feature_name(name):
    return PAPER_LABEL_MAP.get(name, name)


def save_figure_bundle(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PAPER_PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")


def feature_sets():
    numeric = [f for f in NUMERIC_FEATURES if f not in EXCLUDED_FEATURES]
    categorical = [f for f in CATEGORICAL_FEATURES if f not in EXCLUDED_FEATURES]
    return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
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
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_pipeline(numeric_features, categorical_features):
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("model", model)])


def get_feature_names(pipe, numeric_features, categorical_features):
    preprocessor = pipe.named_steps["preprocessor"]
    num_base = list(numeric_features)
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    if getattr(num_imputer, "indicator_", None) is not None:
        for idx in num_imputer.indicator_.features_:
            num_base.append(f"{numeric_features[idx]}_missing")
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
    )
    return [pretty_feature_name(name) for name in num_base + cat_names]


def cross_validated_threshold(train_df, feature_columns, numeric_features, categorical_features):
    X = train_df[feature_columns]
    y = train_df[TARGET_COL].astype(int).values
    groups = train_df["patient_id"].values
    oof = np.zeros(len(train_df))
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pipe = build_pipeline(numeric_features, categorical_features)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = choose_threshold(y, oof)
    return threshold, metric_frame(y, oof, threshold)


def compute_shap_outputs(pipe, train_df, valid_df, feature_columns, numeric_features, categorical_features):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names(pipe, numeric_features, categorical_features)

    train_sample = train_df.sample(min(400, len(train_df)), random_state=RANDOM_STATE)
    valid_sample = valid_df.sample(min(400, len(valid_df)), random_state=RANDOM_STATE)
    X_bg = pd.DataFrame(preprocessor.transform(train_sample[feature_columns]), columns=feature_names)
    X_eval = pd.DataFrame(preprocessor.transform(valid_sample[feature_columns]), columns=feature_names)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_eval)
    shap_values = raw[1] if isinstance(raw, list) and len(raw) > 1 else raw
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    display_names = [paper_feature_name(name) for name in X_eval.columns]
    X_eval_plot = X_eval.copy()
    X_eval_plot.columns = display_names

    importance = pd.DataFrame({"feature": display_names, "mean_abs_shap": np.abs(shap_array).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False
    )

    top20 = importance.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.barh(top20["feature"], top20["mean_abs_shap"], color="#d95f1a", edgecolor="#d95f1a")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.98, bottom=0.14)
    save_figure_bundle(fig, "修改后shap_summary_bar")
    save_figure_bundle(fig, "dedup_shap_summary_bar")
    plt.close(fig)

    shap.summary_plot(shap_array, X_eval_plot, show=False, max_display=20, plot_size=(8.6, 6.0), color_bar_label="Feature value")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    fig.subplots_adjust(left=0.29, right=0.92, top=0.98, bottom=0.13)
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=PAPER_FONT_SIZE)
        cbar_ax.set_ylabel("Feature value", fontsize=PAPER_FONT_SIZE)
    save_figure_bundle(fig, "修改后shap_summary_dot")
    save_figure_bundle(fig, "dedup_shap_summary_dot")
    plt.close(fig)

    numeric_labels = [paper_feature_name(FEATURE_DISPLAY.get(name, name)) for name in numeric_features]
    dependence_records = []
    numeric_ranked = [feat for feat in importance["feature"] if feat in numeric_labels]
    for idx, feature in enumerate(numeric_ranked[:5], start=1):
        shap.dependence_plot(feature, shap_array, X_eval_plot, show=False, interaction_index=None)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(":", "").replace(" ", "_")
        out_path = OUTPUT_DIR / f"dedup_dependence_{idx}_{safe_name}.png"
        plt.savefig(out_path, dpi=PAPER_PNG_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        dependence_records.append({"feature": feature, "path": str(out_path)})

    importance.to_csv(OUTPUT_DIR / "dedup_shap_importance.csv", index=False, encoding="utf-8-sig")
    return importance, dependence_records


def main():
    train_df, valid_df = load_data()
    numeric_features, categorical_features = feature_sets()
    feature_columns = numeric_features + categorical_features

    threshold, cv_metrics = cross_validated_threshold(train_df, feature_columns, numeric_features, categorical_features)

    pipe = build_pipeline(numeric_features, categorical_features)
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL].astype(int).values
    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COL].astype(int).values
    pipe.fit(X_train, y_train)

    train_prob = pipe.predict_proba(X_train)[:, 1]
    valid_prob = pipe.predict_proba(X_valid)[:, 1]
    train_metrics = metric_frame(y_train, train_prob, threshold)
    valid_metrics = metric_frame(y_valid, valid_prob, threshold)

    importance, dependence_records = compute_shap_outputs(
        pipe, train_df, valid_df, feature_columns, numeric_features, categorical_features
    )

    metrics = pd.DataFrame(
        [
            {"split": "internal_cv_2024", **cv_metrics},
            {"split": "train_full", **train_metrics},
            {"split": "validation_2025", **valid_metrics},
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "dedup_model_metrics.csv", index=False, encoding="utf-8-sig")

    report = {
        "approach": "ERI-only de-redundant interpretation model",
        "kept_anemia_signal": KEPT_ANEMIA_SIGNAL,
        "excluded_features": EXCLUDED_FEATURES,
        "feature_count": len(feature_columns),
        "validation_metrics": valid_metrics,
        "top_shap_features": importance.head(10).to_dict(orient="records"),
        "dependence_plots": dependence_records,
    }
    (OUTPUT_DIR / "dedup_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# De-redundant SHAP secondary model",
        "",
        "This interpretation-focused secondary model retains `eri_index` as the single composite anemia-treatment signal,",
        "while removing `esa_dose_per_kg_index`, `esa_dose_index`, and `current_hyporesponse` to reduce overlap.",
        "",
        "## Excluded redundant features",
        f"- {', '.join(EXCLUDED_FEATURES)}",
        "",
        "## Validation performance",
        f"- ROC AUC: {valid_metrics['roc_auc']:.3f}",
        f"- PR AUC: {valid_metrics['pr_auc']:.3f}",
        f"- F1: {valid_metrics['f1']:.3f}",
        f"- Accuracy: {valid_metrics['accuracy']:.3f}",
        "",
        "## Top SHAP features",
    ]
    lines.extend([f"- {row.feature}: {row.mean_abs_shap:.4f}" for row in importance.head(10).itertuples(index=False)])
    (OUTPUT_DIR / "dedup_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
