import duckdb
import joblib
import pandas as pd
import numpy as np
import json
import hashlib

from src.baseline import evaluate_selection

from sklearn.metrics import brier_score_loss, log_loss

from src.calibration import scores_to_log_odds
from src.modeling import PROJECT_ROOT


def load_policy_data():
    data_dir = PROJECT_ROOT / "data" / "processed"

    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                t.TransactionID,
                t.TransactionDT,
                t.TransactionAmt,
                t.ProductCD,
                t.isFraud,
                h.* EXCLUDE (
                    TransactionID,
                    TransactionDT,
                    TransactionAmt
                )
            FROM read_parquet(?) AS t
            INNER JOIN read_parquet(?) AS s
                ON t.TransactionID = s.TransactionID
            INNER JOIN read_parquet(?) AS h
                ON t.TransactionID = h.TransactionID
            WHERE s.split = 'policy'
            ORDER BY t.TransactionDT, t.TransactionID
            """,
            [
                str(data_dir / "joined_transactions.parquet"),
                str(data_dir / "split_membership.parquet"),
                str(data_dir / "history_features.parquet"),
            ],
        ).df()


def select_fdr_threshold(policy_data, scores):
    max_review_rate = 0.01
    fdr_target = 0.50
    capacity = int(len(policy_data) * max_review_rate)

    ranked = (
        policy_data.assign(risk_score=np.asarray(scores))
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranked["selected_count"] = np.arange(1, len(ranked) + 1)
    ranked["fraud_hits"] = ranked["isFraud"].cumsum()
    ranked["observed_fdr"] = (
        1 - ranked["fraud_hits"] / ranked["selected_count"]
    )

    threshold_end = ranked["risk_score"].ne(
        ranked["risk_score"].shift(-1)
    )

    eligible = ranked.loc[
        threshold_end
        & ranked["selected_count"].le(capacity)
        & ranked["observed_fdr"].le(fdr_target)
    ]

    if eligible.empty:
        raise ValueError(
            "No non-empty threshold meets the policy constraints."
        )

    best = eligible.sort_values(
        ["fraud_hits", "selected_count"],
        ascending=[False, True],
    ).iloc[0]

    threshold = float(best["risk_score"])
    selected = ranked.loc[ranked["risk_score"] >= threshold].copy()

    return threshold, evaluate_selection(policy_data, selected)



def apply_review_policy(data, scores, threshold, max_review_rate=0.01):
    score_array = np.asarray(scores, dtype=float)

    if score_array.shape != (len(data),):
        raise ValueError("Scores must contain one value per transaction.")

    if not np.isfinite(score_array).all():
        raise ValueError("Scores contain non-finite values.")

    if not data["TransactionID"].is_unique:
        raise ValueError("Transaction IDs must be unique.")

    capacity = int(len(data) * max_review_rate)

    ranked = (
        data.assign(risk_score=score_array)
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
    )

    return ranked.loc[
        ranked["risk_score"] >= threshold
    ].head(capacity).copy()








if __name__ == "__main__":
    bundle = joblib.load(
        PROJECT_ROOT / "artifacts" / "calibrated_model_bundle.joblib"
    )

    policy_data = load_policy_data()

    if len(policy_data) != 59054:
        raise ValueError("Unexpected policy row count.")

    if not policy_data["TransactionID"].is_unique:
        raise ValueError("Duplicate policy transaction IDs.")

    numeric_features = bundle["numeric_features"]

    policy_data[numeric_features] = (
        policy_data[numeric_features].astype("float64")
    )

    raw_scores = bundle["model"].predict_proba(
        policy_data[bundle["feature_columns"]]
    )[:, 1]

    calibrated_scores = bundle["calibrator"].predict_proba(
        scores_to_log_odds(raw_scores)
    )[:, 1]

    target = policy_data["isFraud"]

    comparison = pd.DataFrame([
        {
            "score_type": name,
            "brier_score": brier_score_loss(target, scores),
            "log_loss": log_loss(target, scores, labels=[0, 1]),
        }
        for name, scores in [
            ("raw", raw_scores),
            ("calibrated", calibrated_scores),
        ]
    ])

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        report_dir / "policy_calibration_comparison.csv",
        index=False,
    )

    print("Policy rows:", len(policy_data))
    print(comparison.to_string(index=False))

    ranked_policy = (
        policy_data.assign(risk_score=raw_scores)
        .sort_values(
            ["risk_score", "TransactionID"],
            ascending=[False, True],
        )
    )

    capacity_results = []

    for rate in [0.001, 0.0025, 0.005, 0.0075, 0.01]:
        capacity = int(len(policy_data) * rate)
        selected = ranked_policy.head(capacity)

        metrics = evaluate_selection(policy_data, selected)

        capacity_results.append({
            "requested_review_rate": rate,
            **metrics,
        })

    capacity_report = pd.DataFrame(capacity_results)

    capacity_report.to_csv(
        report_dir / "policy_capacity_comparison.csv",
        index=False,
    )

    print("\nPolicy capacity comparison:")
    print(
        capacity_report[
            [
                "selected_transactions",
                "true_positives",
                "precision",
                "recall",
                "fdr",
                "review_rate",
            ]
        ].to_string(index=False)
    )

    threshold, threshold_metrics = select_fdr_threshold(
        policy_data,
        raw_scores,
    )

    print("\nSelected policy threshold:", threshold)
    print("Selected transactions:", threshold_metrics["selected_transactions"])
    print(f"Observed precision: {threshold_metrics['precision']:.2%}")
    print(f"Observed recall: {threshold_metrics['recall']:.2%}")
    print(f"Observed FDR: {threshold_metrics['fdr']:.2%}")



    review_input = policy_data[["TransactionID"]]

    selected = apply_review_policy(
        review_input,
        raw_scores,
        threshold,
    )

    selected_with_labels = selected.merge(
        policy_data[["TransactionID", "isFraud"]],
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    applied_metrics = evaluate_selection(
        policy_data,
        selected_with_labels,
    )

    assert len(selected) == threshold_metrics["selected_transactions"]
    assert applied_metrics["true_positives"] == threshold_metrics["true_positives"]

    print("PASS: Target-free policy application matches threshold selection.")

    model_path = (
        PROJECT_ROOT / "artifacts" / "calibrated_model_bundle.joblib"
    )

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    policy_config = {
        "model_name": bundle["model_name"],
        "model_bundle_sha256": model_hash,
        "score_type": "raw",
        "selection_method": "threshold_then_batch_capacity",
        "threshold": threshold,
        "max_review_rate": 0.01,
        "fdr_target": 0.50,
        "tie_breaker": "TransactionID ascending",
        "development_split": "policy",
        "fdr_guaranteed": False,
    }

    policy_path = PROJECT_ROOT / "artifacts" / "review_policy.json"

    if policy_path.exists():
        existing = json.loads(policy_path.read_text(encoding="utf-8"))

        if existing != policy_config:
            raise ValueError(
                "Saved policy differs. Review changes before replacing it."
            )
    else:
        policy_path.write_text(
            json.dumps(policy_config, indent=2),
            encoding="utf-8",
        )

    pd.DataFrame([applied_metrics]).to_csv(
        report_dir / "policy_selected_metrics.csv",
        index=False,
    )

    print("Saved review policy:", policy_path)