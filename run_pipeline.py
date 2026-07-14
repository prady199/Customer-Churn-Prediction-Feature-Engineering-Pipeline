"""
run_pipeline.py
================
End-to-end orchestration script for the Customer Churn Prediction project.

Pipeline stages:
    1. Load & clean data          (src.preprocessing)
    2. Feature engineering        (src.feature_engineering)
    3. Train/test split
    4. Train baseline models      (src.train_model)
    5. Hyperparameter tuning      (GridSearchCV)
    6. Evaluate best model        (src.evaluation)
    7. Feature importance
    8. Save best model            (joblib)
    9. Business insights + single-customer prediction demo

Run with:  python run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    get_classification_report,
    get_cross_val_scores,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataCleaner
from src.train_model import ModelTrainer, get_baseline_models, tune_model
from src.utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, get_logger, save_object

logger = get_logger(__name__)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. LOAD & CLEAN DATA
    # ------------------------------------------------------------------
    cleaner = DataCleaner()
    df_clean = cleaner.clean(DATA_DIR / "Telco-Customer-Churn.csv")

    # ------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df_clean)

    # Persist the fitted feature engineer for inference-time reuse
    save_object(fe, MODELS_DIR / "feature_engineer.pkl")

    X = df_features.drop(columns=["Churn"])
    y = df_features["Churn"]

    # ------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT (80/20, stratified to preserve churn ratio)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 4. TRAIN BASELINE MODELS
    # ------------------------------------------------------------------
    trainer = ModelTrainer()
    models = get_baseline_models()
    comparison_df = trainer.train_and_evaluate(models, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print("STEP 6-7: MODEL COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df.round(4).to_string())
    comparison_df.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")

    best_baseline_name = comparison_df.index[0]
    print(f"\nBest baseline model (by ROC AUC): {best_baseline_name}")

    # ------------------------------------------------------------------
    # 5. HYPERPARAMETER TUNING (Random Forest, Decision Tree, Gradient Boosting)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)

    tuned_models = {}
    best_params_all = {}

    rf_grid = {
        "n_estimators": [200, 300],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
    }
    tuned_models["Random Forest (Tuned)"], best_params_all["Random Forest"] = tune_model(
        RandomForestClassifier(random_state=42), rf_grid, X_train, y_train
    )

    dt_grid = {
        "max_depth": [4, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    tuned_models["Decision Tree (Tuned)"], best_params_all["Decision Tree"] = tune_model(
        DecisionTreeClassifier(random_state=42), dt_grid, X_train, y_train
    )

    gb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    }
    tuned_models["Gradient Boosting (Tuned)"], best_params_all["Gradient Boosting"] = tune_model(
        GradientBoostingClassifier(random_state=42), gb_grid, X_train, y_train
    )

    with open(REPORTS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    # Evaluate tuned models alongside baselines to pick the overall best.
    # Use a fresh ModelTrainer instance so its results_ dict doesn't
    # accumulate the baseline results already captured in comparison_df.
    tuned_trainer = ModelTrainer()
    tuned_comparison = tuned_trainer.train_and_evaluate(tuned_models, X_train, y_train, X_test, y_test)
    trainer.fitted_models_.update(tuned_trainer.fitted_models_)  # merge fitted models for later lookup
    full_comparison = pd.concat([comparison_df, tuned_comparison]).sort_values("ROC AUC", ascending=False)
    print("\nFull comparison (baseline + tuned):")
    print(full_comparison.round(4).to_string())
    full_comparison.round(4).to_csv(REPORTS_DIR / "full_model_comparison.csv")

    best_model_name = full_comparison.index[0]
    best_model = trainer.fitted_models_[best_model_name]
    print(f"\n*** OVERALL BEST MODEL: {best_model_name} ***")
    print(full_comparison.loc[best_model_name].round(4))

    # ------------------------------------------------------------------
    # 6. MODEL EVALUATION (best model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: MODEL EVALUATION")
    print("=" * 70)

    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best)

    report = get_classification_report(y_test, y_pred_best)
    print(report)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n\n{report}")

    plot_roc_curve(best_model, X_test, y_test)
    plot_precision_recall_curve(best_model, X_test, y_test)
    plot_learning_curve(best_model, X_train, y_train)
    cv_scores = get_cross_val_scores(best_model, X_train, y_train)
    print(f"5-Fold CV ROC AUC scores: {cv_scores.round(4)} | Mean: {cv_scores.mean():.4f}")

    # ------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 10: FEATURE IMPORTANCE")
    print("=" * 70)
    imp_df = plot_feature_importance(best_model, list(X_train.columns), top_n=20)
    if imp_df is None:
        # The overall best model (e.g. Logistic Regression) has no
        # feature_importances_ attribute. Fall back to the tuned Random
        # Forest -- a tree-based model in our lineup -- purely to surface
        # feature importance for business interpretation purposes.
        logger.info("Best model has no feature_importances_; using Random Forest (Tuned) for importance chart.")
        importance_source = tuned_models.get("Random Forest (Tuned)") or trainer.fitted_models_.get("Random Forest")
        imp_df = plot_feature_importance(importance_source, list(X_train.columns), top_n=20)
    if imp_df is not None:
        print(imp_df.to_string(index=False))
        imp_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # ------------------------------------------------------------------
    # 8. SAVE BEST MODEL
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 11: MODEL SAVING")
    print("=" * 70)
    save_object(best_model, MODELS_DIR / "best_model.pkl")
    logger.info(f"Saved best model ({best_model_name}) to models/best_model.pkl")

    # Also save the two explicitly requested named models for the portfolio
    if "Random Forest" in trainer.fitted_models_:
        save_object(trainer.fitted_models_["Random Forest"], MODELS_DIR / "random_forest.pkl")
    if "Logistic Regression" in trainer.fitted_models_:
        save_object(trainer.fitted_models_["Logistic Regression"], MODELS_DIR / "logistic_regression.pkl")

    save_object(list(X_train.columns), MODELS_DIR / "feature_columns.pkl")

    # Demo: load the model back and predict on a sample test customer
    from src.utils import load_object

    reloaded_model = load_object(MODELS_DIR / "best_model.pkl")
    sample_customer = X_test.iloc[[0]]
    pred = reloaded_model.predict(sample_customer)[0]
    proba = reloaded_model.predict_proba(sample_customer)[0, 1]
    print(f"\nDemo prediction on a sample customer: "
          f"Predicted Churn = {'Yes' if pred == 1 else 'No'} (probability={proba:.3f})")

    # Save metadata for README/report generation
    metadata = {
        "best_model_name": best_model_name,
        "best_model_metrics": full_comparison.loc[best_model_name].round(4).to_dict(),
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_mean_roc_auc": round(float(cv_scores.mean()), 4),
    }
    with open(REPORTS_DIR / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nPipeline complete. All models and reports saved.")


if __name__ == "__main__":
    main()
