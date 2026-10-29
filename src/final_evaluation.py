import hashlib
import json
import duckdb
import numpy as np
import joblib

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.baseline import evaluate_selection
from src.policy import apply_review_policy



from src.modeling import PROJECT_ROOT


def load_frozen_artifacts():
    artifact_dir = PROJECT_ROOT / "artifacts"

    model_path = artifact_dir / "calibrated_model_bundle.joblib"
    policy_path = artifact_dir / "review_policy.json"

    policy = json.loads(
        policy_path.read_text(encoding="utf-8")
    )

    actual_hash = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    if actual_hash != policy["model_bundle_sha256"]:
        raise ValueError(
            "Model bundle changed after policy selection."
        )

    bundle = joblib.load(model_path)

    if bundle["model_name"] != policy["model_name"]:
        raise ValueError("Model and policy names do not match.")

    if policy["score_type"] != "raw":
        raise ValueError("Expected the selected raw-score policy.")

    if not 0 <= policy["threshold"] <= 1:
        raise ValueError("Invalid policy threshold.")

    if not 0 < policy["max_review_rate"] <= 1:
        raise ValueError("Invalid review capacity.")

    return bundle, policy

def load_test_data():
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
            WHERE s.split = 'test'
            ORDER BY t.TransactionDT, t.TransactionID
            """,
            [
                str(data_dir / "joined_transactions.parquet"),
                str(data_dir / "split_membership.parquet"),
                str(data_dir / "history_features.parquet"),
            ],
        ).df()


def evaluate_final_test(bundle, policy):
    test = load_test_data()

    if len(test) != 88581:
        raise ValueError("Unexpected test row count.")

    if not test["TransactionID"].is_unique:
        raise ValueError("Duplicate test transaction IDs.")

    numeric = bundle["numeric_features"]
    test[numeric] = test[numeric].astype("float64")

    scores = bundle["model"].predict_proba(
        test[bundle["feature_columns"]]
    )[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("Test scores contain non-finite values.")

    selected = apply_review_policy(
        test[["TransactionID"]],
        scores,
        threshold=policy["threshold"],
        max_review_rate=policy["max_review_rate"],
    )

    selected = selected.merge(
        test[["TransactionID", "isFraud"]],
        on="TransactionID",
        validate="one_to_one",
    )

    decision_metrics = evaluate_selection(test, selected)
    target = test["isFraud"]

    probability_metrics = {
        "average_precision": float(
            average_precision_score(target, scores)
        ),
        "roc_auc": float(roc_auc_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(
            log_loss(target, scores, labels=[0, 1])
        ),
    }

    observed_fdr = decision_metrics["fdr"]

    return {
        "model_name": bundle["model_name"],
        "policy": policy,
        "test_rows": len(test),
        "probability_metrics": probability_metrics,
        "decision_metrics": decision_metrics,
        "fdr_target_met_on_test": (
            observed_fdr <= policy["fdr_target"]
            if observed_fdr is not None
            else None
        ),
    }

if __name__ == "__main__":
    bundle, policy = load_frozen_artifacts()

    report_path = PROJECT_ROOT / "reports" / "final_test.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        result = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        if result["policy"] != policy:
            raise ValueError(
                "Saved final report belongs to a different policy."
            )

        print("Showing the saved report; test was not reevaluated.")
    else:
        result = evaluate_final_test(bundle, policy)

        report_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        print("Final test evaluated and report saved.")

    print(json.dumps(result, indent=2))
    