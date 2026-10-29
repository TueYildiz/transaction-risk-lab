import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline import evaluate_selection
from modeling import (
    PROJECT_ROOT,
    CATEGORICAL_FEATURES,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    build_lightgbm_pipeline,
    build_logistic_pipeline,
    load_model_data,
)

def build_backtest_windows(train):
    quantiles = [0.40, 0.60, 0.80, 1.00]
    cutoffs = train["TransactionDT"].quantile(quantiles).tolist()

    windows = []

    for fold, (train_end, evaluation_end) in enumerate(
        zip(cutoffs[:-1], cutoffs[1:]),
        start=1,
    ):
        training = train.loc[
            train["TransactionDT"] <= train_end
        ].copy()

        evaluation = train.loc[
            (train["TransactionDT"] > train_end)
            & (train["TransactionDT"] <= evaluation_end)
        ].copy()

        if training.empty or evaluation.empty:
            raise ValueError(f"Fold {fold} contains an empty period.")

        if (
            training["TransactionDT"].max()
            >= evaluation["TransactionDT"].min()
        ):
            raise ValueError(f"Fold {fold} has overlapping time periods.")

        windows.append((fold, training, evaluation))

    return windows

def run_backtests(windows):
    results = []

    candidates = [
        ("logistic_base", build_logistic_pipeline, NUMERIC_FEATURES),
        (
            "logistic_history",
            build_logistic_pipeline,
            NUMERIC_FEATURES + HISTORY_FEATURES,
        ),
        ("lightgbm_base", build_lightgbm_pipeline, NUMERIC_FEATURES),
        (
            "lightgbm_history",
            build_lightgbm_pipeline,
            NUMERIC_FEATURES + HISTORY_FEATURES,
        ),
    ]

    for fold, training, evaluation in windows:
        training = training.copy()
        evaluation = evaluation.copy()

        numeric_columns = NUMERIC_FEATURES + HISTORY_FEATURES
        training[numeric_columns] = training[numeric_columns].astype("float64")
        evaluation[numeric_columns] = evaluation[numeric_columns].astype("float64")

        capacity = int(len(evaluation) * 0.01)

        for name, builder, numeric_features in candidates:
            columns = numeric_features + CATEGORICAL_FEATURES
            model = builder(numeric_features=numeric_features)

            model.fit(training[columns], training["isFraud"])

            scores = model.predict_proba(evaluation[columns])[:, 1]

            selected = (
                evaluation.assign(risk_score=scores)
                .sort_values(
                    ["risk_score", "TransactionID"],
                    ascending=[False, True],
                )
                .head(capacity)
            )

            metrics = evaluate_selection(evaluation, selected)

            results.append({
                "fold": fold,
                "model": name,
                "training_rows": len(training),
                "evaluation_rows": len(evaluation),
                **metrics,
            })

            print(
                f"Fold {fold} | {name} | "
                f"fraud_hits={metrics['true_positives']} | "
                f"recall={metrics['recall']:.2%}"
            )

    return pd.DataFrame(results)



if __name__ == "__main__":
    data = load_model_data()
    train = data.loc[data["split"] == "train"].copy()

    windows = build_backtest_windows(train)
    results = run_backtests(windows)

    if len(results) != 12:
        raise ValueError("Expected exactly 12 backtest results.")

    if results.duplicated(["fold", "model"]).any():
        raise ValueError("Duplicate fold-model results found.")

    report_dir = PROJECT_ROOT / "reports"

    report_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        report_dir / "model_backtests.csv",
        index=False,
    )

    print("\nMean recall across folds:")
    print(
        results.groupby("model")["recall"]
        .mean()
        .sort_values(ascending=False)
    )

    print("Backtest results saved.")
