import json

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from esa_multimodel_shap import (
    MULTI_DIR,
    ROOT,
    calibration_curve_frame,
    choose_threshold,
    decision_curve,
    get_feature_names,
    load_analysis_data,
    pretty_feature_name,
)

import matplotlib.pyplot as plt


SUBDIR = MULTI_DIR / "restricted_clinical_model"
SUBDIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

RESTRICTED_NUMERIC_FEATURES = [
    "age",
    "dialysis_vintage_months",
    "dry_weight_index",
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

RESTRICTED_CATEGORICAL_FEATURES = [
    "sex",
    "primary_disease_group",
    "access_type",
    "dialysis_frequency_index",
    "hdf_index",
    "hp_index",
    "iron_deficiency_index",
    "low_albumin_index",
    "inflammation_index",
    "hypotension_any_index",
]

FEATURE_COLUMNS = RESTRICTED_NUMERIC_FEATURES + RESTRICTED_CATEGORICAL_FEATURES


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
                RESTRICTED_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
                    ]
                ),
                RESTRICTED_CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline():
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def cross_validated_predictions(train_df):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label_main"].astype(int).values
    groups = train_df["patient_id"].values
    oof = np.zeros(len(train_df))
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pipe = build_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = choose_threshold(y, oof)
    return oof, threshold, metric_frame(y, oof, threshold)


def fit_and_score(train_df, test_df, threshold):
    pipe = build_pipeline()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label_main"].astype(int).values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label_main"].astype(int).values
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    return pipe, prob, metric_frame(y_test, prob, threshold)


def shap_explain(pipe, train_df, valid_df):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    feature_names = [pretty_feature_name(x) for x in get_feature_names(pipe)]

    train_sample = train_df.sample(min(300, len(train_df)), random_state=RANDOM_STATE)
    valid_sample = valid_df.sample(min(300, len(valid_df)), random_state=RANDOM_STATE)
    X_bg = preprocessor.transform(train_sample[FEATURE_COLUMNS])
    X_eval = preprocessor.transform(valid_sample[FEATURE_COLUMNS])
    X_eval_df = pd.DataFrame(X_eval, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_eval_df)
    shap_values = raw[1] if isinstance(raw, list) else raw
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    importance_df = pd.DataFrame(
        {
            "feature": X_eval_df.columns,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 6))
    top = importance_df.head(20).sort_values("mean_abs_shap")
    plt.barh(top["feature"], top["mean_abs_shap"], color="#1D7874")
    plt.xlabel("mean(|SHAP value|)")
    plt.title("Restricted clinical model SHAP top 20")
    plt.tight_layout()
    plt.savefig(SUBDIR / "restricted_shap_top20_bar.png", dpi=180)
    plt.close()

    shap.summary_plot(
        shap_array,
        X_eval_df,
        max_display=20,
        show=False,
        plot_size=(9, 6),
    )
    plt.tight_layout()
    plt.savefig(SUBDIR / "restricted_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(SUBDIR / "restricted_shap_importance.csv", index=False, encoding="utf-8-sig")
    return importance_df


def save_curves(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Restricted RF AUC={roc_auc_score(y_true, prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Restricted model ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUBDIR / "restricted_model_roc.png", dpi=180)
    plt.close()

    calibration_curve_frame(y_true, prob).to_csv(SUBDIR / "restricted_model_calibration.csv", index=False)
    thresholds = np.linspace(0.05, 0.5, 19)
    decision_curve(y_true, prob, thresholds).to_csv(SUBDIR / "restricted_model_decision_curve.csv", index=False)


def main():
    _, train_df, valid_df, valid_new_df = load_analysis_data()

    _, threshold, cv_metrics = cross_validated_predictions(train_df)
    pipe, valid_prob, valid_metrics = fit_and_score(train_df, valid_df, threshold)
    _, new_prob, new_metrics = fit_and_score(train_df, valid_new_df, threshold)

    metrics_df = pd.DataFrame(
        [
            {"split": "internal_cv_2024", "model": "RestrictedRandomForest", **cv_metrics},
            {"split": "temporal_validation_2025", "model": "RestrictedRandomForest", **valid_metrics},
            {"split": "new_patient_validation_2025", "model": "RestrictedRandomForest", **new_metrics},
        ]
    )

    shap_df = shap_explain(pipe, train_df, valid_df)
    save_curves(valid_df["label_main"].astype(int).values, valid_prob)

    metrics_df.to_csv(SUBDIR / "restricted_model_metrics.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(SUBDIR / "restricted_model_results.xlsx") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        shap_df.to_excel(writer, sheet_name="shap_importance", index=False)

    report = [
        "# Restricted clinical model SHAP analysis",
        "",
        "Removed variables: Hb-derived and ESA treatment-driven features such as current hyporesponse, ERI, ESA dose, ESA dose per kg, ESA use, HIF use, iron treatment use, and platelet.",
        "",
        "Retained domains: demographics, dialysis parameters, iron metabolism, inflammation, and nutrition-related indicators.",
        "",
        "## Model performance",
        metrics_df.to_string(index=False),
        "",
        "## Top SHAP features",
        shap_df.head(20).to_string(index=False),
    ]
    (SUBDIR / "restricted_model_report.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "retained_feature_domains": [
            "demographics",
            "dialysis parameters",
            "iron metabolism",
            "inflammation",
            "nutrition indicators",
        ],
        "metrics": metrics_df.to_dict(orient="records"),
        "top_shap_features": shap_df.head(20).to_dict(orient="records"),
    }
    (SUBDIR / "restricted_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
