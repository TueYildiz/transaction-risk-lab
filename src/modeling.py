from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import duckdb
from baseline import evaluate_selection
import pandas as pd
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

NUMERIC_FEATURES = ["TransactionAmt"]
CATEGORICAL_FEATURES = ["ProductCD"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HISTORY_FEATURES = [
    "hist_count",
    "hist_valid_amount_count",
    "hist_mean_amount",
    "hist_std_amount",
    "hist_max_amount",
    "hist_total_amount",
    "time_since_previous",
    "amount_to_hist_mean",
    "amount_minus_hist_mean",
    "is_new_group",
]

def load_model_data():
    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                t.TransactionID,
                t.TransactionDT,
                t.TransactionAmt,
                t.ProductCD,
                t.isFraud,
                s.split,
                h.hist_count,
                h.hist_valid_amount_count,
                h.hist_mean_amount,
                h.hist_std_amount,
                h.hist_max_amount,
                h.hist_total_amount,
                h.time_since_previous,
                h.amount_to_hist_mean,
                h.amount_minus_hist_mean,
                h.is_new_group
            FROM read_parquet(?) AS t
            INNER JOIN read_parquet(?) AS s
                ON t.TransactionID = s.TransactionID
            INNER JOIN read_parquet(?) AS h
                ON t.TransactionID = h.TransactionID
            WHERE s.split IN ('train', 'validation')
            ORDER BY t.TransactionDT, t.TransactionID
            """,
            [
                str(DATA_DIR / "joined_transactions.parquet"),
                str(DATA_DIR / "split_membership.parquet"),
                str(DATA_DIR / "history_features.parquet"),
            ],
        ).df()
    

def build_logistic_pipeline(numeric_features=None):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

def build_lightgbm_pipeline(numeric_features=None):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=100,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                ),
            ),
        ]
    )


if __name__ == "__main__":
    data = load_model_data()

    expected_rows = {
        "train": 324797,
        "validation": 59054,
    }

    actual_rows = data["split"].value_counts().to_dict()

    if actual_rows != expected_rows:
        raise ValueError("Model-data JOIN changed the expected split counts.")

    if not data["TransactionID"].is_unique:
        raise ValueError("Model-data JOIN produced duplicate transaction IDs.")

    print("PASS: Model-data JOIN checks completed.")



    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()

    numeric_columns = NUMERIC_FEATURES + HISTORY_FEATURES

    train[numeric_columns] = train[numeric_columns].astype("float64")
    validation[numeric_columns] = validation[numeric_columns].astype("float64")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["isFraud"]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["isFraud"]

    print("Training features:", X_train.shape)
    print("Validation features:", X_validation.shape)
    print("Feature columns:", FEATURE_COLUMNS)

    model = build_logistic_pipeline()
    model.fit(X_train, y_train)

    validation_scores = model.predict_proba(X_validation)[:, 1]

    print("PASS: Logistic Regression training completed.")
    print("Validation score count:", len(validation_scores))
    print("First five scores:", validation_scores[:5])
    scored_validation = validation.assign(
        risk_score=validation_scores
    )

    capacity = int(len(scored_validation) * 0.01)

    selected = (
        scored_validation.sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
        .head(capacity)
    )

    metrics = evaluate_selection(validation, selected)

    print("\nLogistic Regression at 1% review capacity:")

    for name in [
        "true_positives",
        "false_positives",
        "precision",
        "recall",
        "fdr",
        "review_rate",
    ]:
        value = metrics[name]

        if name in {"true_positives", "false_positives"}:
            print(f"{name}: {value}")
        else:
            formatted = f"{value:.4%}" if value is not None else "N/A"
            print(f"{name}: {formatted}")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "logistic_base",
        "split": "validation",
        "feature_count": len(FEATURE_COLUMNS),
        **metrics,
    }

    pd.DataFrame([result]).to_csv(
        report_dir / "validation_logistic_base.csv",
        index=False,
    )

    print("Base Logistic Regression report saved.")

    history_numeric_features = NUMERIC_FEATURES + HISTORY_FEATURES
    history_feature_columns = (
        history_numeric_features + CATEGORICAL_FEATURES
    )

    history_model = build_logistic_pipeline(
        numeric_features=history_numeric_features
    )

    history_model.fit(
        train[history_feature_columns],
        y_train,
    )

    history_scores = history_model.predict_proba(
        validation[history_feature_columns]
    )[:, 1]

    history_selected = (
        validation.assign(risk_score=history_scores)
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
        .head(capacity)
    )

    history_metrics = evaluate_selection(
        validation,
        history_selected,
    )

    print("\nLogistic Regression with historical features:")

    for name in ["true_positives", "precision", "recall", "fdr"]:
        value = history_metrics[name]

        if name == "true_positives":
            print(f"{name}: {value}")
        else:
            formatted = f"{value:.4%}" if value is not None else "N/A"
            print(f"{name}: {formatted}")

    history_result = {
        "model": "logistic_history",
        "split": "validation",
        "feature_count": len(history_feature_columns),
        **history_metrics,
    }

    pd.DataFrame([history_result]).to_csv(
        report_dir / "validation_logistic_history.csv",
        index=False,
    )

    lightgbm_base = build_lightgbm_pipeline()
    lightgbm_base.fit(X_train, y_train)

    lightgbm_scores = lightgbm_base.predict_proba(X_validation)[:, 1]

    lightgbm_selected = (
        validation.assign(risk_score=lightgbm_scores)
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
        .head(capacity)
    )

    lightgbm_metrics = evaluate_selection(validation, lightgbm_selected)

    print("\nLightGBM base at 1% review capacity:")

    for name in ["true_positives", "precision", "recall", "fdr"]:
        value = lightgbm_metrics[name]

        if name == "true_positives":
            print(f"{name}: {value}")
        else:
            formatted = f"{value:.4%}" if value is not None else "N/A"
            print(f"{name}: {formatted}")

    pd.DataFrame([{
        "model": "lightgbm_base",
        "split": "validation",
        "feature_count": len(FEATURE_COLUMNS),
        **lightgbm_metrics,
    }]).to_csv(
        report_dir / "validation_lightgbm_base.csv",
        index=False,
    )

    lightgbm_history = build_lightgbm_pipeline(
        numeric_features=history_numeric_features
    )

    lightgbm_history.fit(
        train[history_feature_columns],
        y_train,
    )

    lightgbm_history_scores = lightgbm_history.predict_proba(
        validation[history_feature_columns]
    )[:, 1]

    lightgbm_history_selected = (
        validation.assign(risk_score=lightgbm_history_scores)
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
        .head(capacity)
    )

    lightgbm_history_metrics = evaluate_selection(
        validation,
        lightgbm_history_selected,
    )

    print("\nLightGBM with historical features:")

    for name in ["true_positives", "precision", "recall", "fdr"]:
        value = lightgbm_history_metrics[name]

        if name == "true_positives":
            print(f"{name}: {value}")
        else:
            formatted = f"{value:.4%}" if value is not None else "N/A"
            print(f"{name}: {formatted}")

    pd.DataFrame([{
        "model": "lightgbm_history",
        "split": "validation",
        "feature_count": len(history_feature_columns),
        **lightgbm_history_metrics,
    }]).to_csv(
        report_dir / "validation_lightgbm_history.csv",
        index=False,
    )
    
    
    comparison = pd.DataFrame([
        {"model": "logistic_base", **metrics},
        {"model": "logistic_history", **history_metrics},
        {"model": "lightgbm_base", **lightgbm_metrics},
        {"model": "lightgbm_history", **lightgbm_history_metrics},
    ])

    comparison = comparison.sort_values("recall", ascending=False)

    comparison.to_csv(
        report_dir / "validation_model_comparison.csv",
        index=False,
    )

    print("\nValidation model comparison:")
    print(
        comparison[
            ["model", "true_positives", "precision", "recall", "fdr"]
        ].to_string(index=False)
    )

    score_sets = {
        "logistic_base": validation_scores,
        "logistic_history": history_scores,
        "lightgbm_base": lightgbm_scores,
        "lightgbm_history": lightgbm_history_scores,
    }

    ranking_results = []

    for name, scores in score_sets.items():
        ranking_results.append({
            "model": name,
            "average_precision": average_precision_score(
                y_validation, scores
            ),
            "roc_auc": roc_auc_score(y_validation, scores),
        })

    ranking_report = pd.DataFrame(ranking_results)

    ranking_report.to_csv(
        report_dir / "validation_ranking_metrics.csv",
        index=False,
    )

    print("\nValidation ranking metrics:")
    print(ranking_report.to_string(index=False))