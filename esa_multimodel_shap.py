import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from esa_study_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    ROOT,
    build_transition_data,
    calibration_curve_frame,
    choose_threshold,
    decision_curve,
    engineer_dataset,
    read_year,
)

import matplotlib.pyplot as plt


MULTI_DIR = OUTPUT_DIR / "multimodel"
MULTI_DIR.mkdir(exist_ok=True)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


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


def make_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
        ),
        "GaussianNB": GaussianNB(),
    }


def model_pipeline(estimator):
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(pipe):
    preprocessor = pipe.named_steps["preprocessor"]
    num_base = list(preprocessor.transformers_[0][2])
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    if getattr(num_imputer, "indicator_", None) is not None:
        for idx in num_imputer.indicator_.features_:
            num_base.append(f"{preprocessor.transformers_[0][2][idx]}_missing")
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
            preprocessor.transformers_[1][2]
        )
    )
    return [pretty_feature_name(name) for name in num_base + cat_names]


def pretty_feature_name(name):
    replacements = {
        "sex_男": "sex_male",
        "sex_女": "sex_female",
        "esa_use_index_使用": "esa_use_index_use",
        "esa_use_index_未使用": "esa_use_index_no_use",
        "iron_use_index_使用": "iron_use_index_use",
        "iron_use_index_未使用": "iron_use_index_no_use",
        "hif_use_index_使用": "hif_use_index_use",
        "hif_use_index_未使用": "hif_use_index_no_use",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


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


def cross_validated_predictions(train_df, estimator):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label_main"].astype(int).values
    groups = train_df["patient_id"].values
    oof = np.zeros(len(train_df))
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pipe = model_pipeline(clone(estimator))
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold = choose_threshold(y, oof)
    return oof, threshold, metric_frame(y, oof, threshold)


def fit_and_score(train_df, valid_df, estimator, threshold):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label_main"].astype(int).values
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["label_main"].astype(int).values
    pipe = model_pipeline(clone(estimator))
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_valid)[:, 1]
    return pipe, prob, metric_frame(y_valid, prob, threshold)


def load_analysis_data():
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
    return main_df, train_df, valid_df, valid_new_df


def select_best_model(results_df):
    valid = results_df[results_df["split"] == "temporal_validation_2025"].copy()
    valid = valid.sort_values(["roc_auc", "pr_auc", "f1"], ascending=False)
    return valid.iloc[0]["model"]


def shap_explain(best_name, pipe, train_df, valid_df):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    feature_names = get_feature_names(pipe)

    train_sample = train_df.sample(min(300, len(train_df)), random_state=RANDOM_STATE)
    valid_sample = valid_df.sample(min(300, len(valid_df)), random_state=RANDOM_STATE)
    X_bg = preprocessor.transform(train_sample[FEATURE_COLUMNS])
    X_eval = preprocessor.transform(valid_sample[FEATURE_COLUMNS])
    X_bg_df = pd.DataFrame(X_bg, columns=feature_names)
    X_eval_df = pd.DataFrame(X_eval, columns=feature_names)

    tree_models = {"DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost"}
    if best_name in tree_models:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_eval_df)
        if isinstance(raw, list):
            shap_values = raw[1] if len(raw) > 1 else raw[0]
        else:
            shap_values = raw
    elif best_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_bg_df)
        shap_values = explainer.shap_values(X_eval_df)
    else:
        background = X_bg_df.sample(min(80, len(X_bg_df)), random_state=RANDOM_STATE)
        eval_small = X_eval_df.sample(min(120, len(X_eval_df)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(eval_small, nsamples=150)
        shap_values = raw[1] if isinstance(raw, list) else raw
        X_eval_df = eval_small

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    mean_abs = np.abs(shap_array).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": X_eval_df.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 6))
    top = importance_df.head(20).sort_values("mean_abs_shap")
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2E6F95")
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"Top SHAP features: {best_name}")
    plt.tight_layout()
    plt.savefig(MULTI_DIR / "best_model_shap_top20_bar.png", dpi=180)
    plt.close()

    shap.summary_plot(
        shap_array,
        X_eval_df,
        max_display=20,
        show=False,
        plot_size=(9, 6),
    )
    plt.tight_layout()
    plt.savefig(MULTI_DIR / "best_model_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(MULTI_DIR / "best_model_shap_importance.csv", index=False, encoding="utf-8-sig")
    return importance_df


def plot_model_comparison(results_df):
    valid = results_df[results_df["split"] == "temporal_validation_2025"].sort_values("roc_auc", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(valid["model"], valid["roc_auc"], color="#457B9D")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ROC AUC")
    plt.title("Temporal validation ROC AUC by model")
    plt.tight_layout()
    plt.savefig(MULTI_DIR / "model_auc_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(valid["model"], valid["pr_auc"], color="#E76F51")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("PR AUC")
    plt.title("Temporal validation PR AUC by model")
    plt.tight_layout()
    plt.savefig(MULTI_DIR / "model_prauc_comparison.png", dpi=180)
    plt.close()


def plot_best_model_curves(best_name, valid_df, valid_prob):
    y_true = valid_df["label_main"].astype(int).values
    fpr, tpr, _ = roc_curve(y_true, valid_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} ROC AUC={roc_auc_score(y_true, valid_prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Best model ROC: {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MULTI_DIR / "best_model_roc.png", dpi=180)
    plt.close()

    calibration_curve_frame(y_true, valid_prob).to_csv(MULTI_DIR / "best_model_calibration.csv", index=False)
    thresholds = np.linspace(0.05, 0.5, 19)
    decision_curve(y_true, valid_prob, thresholds).to_csv(MULTI_DIR / "best_model_decision_curve.csv", index=False)


def build_report(reference_summary, results_df, best_name, shap_df):
    lines = [
        "# Multimodel ESA hyporesponsiveness analysis",
        "",
        "## Reference paper alignment",
        reference_summary,
        "",
        "## Model comparison",
        results_df.to_string(index=False),
        "",
        f"## Best model: {best_name}",
        "",
        "The best model was chosen by temporal validation ROC AUC, with PR AUC and F1 used as tie-breakers.",
        "",
        "## Top SHAP features",
        shap_df.head(20).to_string(index=False),
    ]
    (MULTI_DIR / "multimodel_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    reference_summary = (
        "The referenced paper compared C5.0 decision tree, logistic regression, and SVM; "
        "it evaluated models with ROC and confusion-matrix metrics, then selected the best-performing model."
    )

    _, train_df, valid_df, valid_new_df = load_analysis_data()
    models = make_models()

    rows = []
    fitted = {}
    validation_probs = {}
    for name, estimator in models.items():
        _, threshold, cv_metrics = cross_validated_predictions(train_df, estimator)
        pipe, valid_prob, valid_metrics = fit_and_score(train_df, valid_df, estimator, threshold)
        _, new_prob, new_metrics = fit_and_score(train_df, valid_new_df, estimator, threshold)
        fitted[name] = pipe
        validation_probs[name] = valid_prob
        rows.append({"split": "internal_cv_2024", "model": name, **cv_metrics})
        rows.append({"split": "temporal_validation_2025", "model": name, **valid_metrics})
        rows.append({"split": "new_patient_validation_2025", "model": name, **new_metrics})

    results_df = pd.DataFrame(rows)
    best_name = select_best_model(results_df)
    best_pipe = fitted[best_name]
    best_prob = validation_probs[best_name]
    shap_df = shap_explain(best_name, best_pipe, train_df, valid_df)

    results_df.to_csv(MULTI_DIR / "multimodel_metrics.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(MULTI_DIR / "multimodel_results.xlsx") as writer:
        results_df.to_excel(writer, sheet_name="metrics", index=False)
        shap_df.to_excel(writer, sheet_name="shap_importance", index=False)

    plot_model_comparison(results_df)
    plot_best_model_curves(best_name, valid_df, best_prob)
    build_report(reference_summary, results_df, best_name, shap_df)

    summary = {
        "best_model": best_name,
        "temporal_validation": results_df[results_df["split"] == "temporal_validation_2025"].sort_values(
            ["roc_auc", "pr_auc"], ascending=False
        ).to_dict(orient="records"),
    }
    (MULTI_DIR / "multimodel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
