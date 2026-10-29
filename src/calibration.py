import joblib
import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression


from src.modeling import (
    PROJECT_ROOT,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    HISTORY_FEATURES,
    build_lightgbm_pipeline,
    load_model_data,
)

def load_calibration_data():
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
            WHERE s.split = 'calibration'
            ORDER BY t.TransactionDT, t.TransactionID
            """,
            [
                str(data_dir / "joined_transactions.parquet"),
                str(data_dir / "split_membership.parquet"),
                str(data_dir / "history_features.parquet"),
            ],
        ).df()
    

def scores_to_log_odds(scores):
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)
    

if __name__ == "__main__":
    development = load_model_data()

    if set(development["split"].unique()) != {"train", "validation"}:
        raise ValueError("Unexpected splits in development training data.")

    if not development["TransactionID"].is_unique:
        raise ValueError("Duplicate transaction IDs in development data.")

    numeric_features = NUMERIC_FEATURES + HISTORY_FEATURES
    feature_columns = numeric_features + CATEGORICAL_FEATURES

    development[numeric_features] = (
        development[numeric_features].astype("float64")
    )

    model = build_lightgbm_pipeline(
        numeric_features=numeric_features
    )

    model.fit(
        development[feature_columns],
        development["isFraud"],
    )

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "selected_model.joblib"

    joblib.dump(
        {
            "model": model,
            "model_name": "lightgbm_history",
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "training_splits": ["train", "validation"],
            "training_rows": len(development),
        },
        model_path,
    )

    print("Development training rows:", len(development))
    print("Saved model:", model_path)
    
    
    
    calibration_data = load_calibration_data()

    if len(calibration_data) != 59054:
        raise ValueError("Unexpected calibration row count.")

    if not calibration_data["TransactionID"].is_unique:
        raise ValueError("Duplicate calibration transaction IDs.")

    if (
        development["TransactionDT"].max()
        >= calibration_data["TransactionDT"].min()
    ):
        raise ValueError("Development and calibration periods overlap.")

    print("PASS: Calibration data checks completed.")
    print("Calibration rows:", len(calibration_data))

    calibration_data[numeric_features] = (
        calibration_data[numeric_features].astype("float64")
    )

    raw_scores = model.predict_proba(
        calibration_data[feature_columns]
    )[:, 1]

    calibration_target = calibration_data["isFraud"]

    if calibration_target.nunique() != 2:
        raise ValueError("Calibration requires both target classes.")

    calibrator = LogisticRegression(
        C=1e6,
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(
        scores_to_log_odds(raw_scores),
        calibration_target,
    )

    calibrated_scores = calibrator.predict_proba(
        scores_to_log_odds(raw_scores)
    )[:, 1]

    if not np.isfinite(calibrated_scores).all():
        raise ValueError("Calibrated scores contain non-finite values.")

    slope = float(calibrator.coef_[0, 0])

    if slope <= 0:
        raise ValueError("Calibration unexpectedly reverses the risk ranking.")

    print("PASS: Sigmoid calibrator fitted.")
    print("Calibration slope:", round(slope, 4))
    print("Calibrated score count:", len(calibrated_scores))

    
    
    
    
    bundle_path = artifact_dir / "calibrated_model_bundle.joblib"

    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "model_name": "lightgbm_history",
            "calibration_method": "sigmoid_on_clipped_log_odds",
            "clip_epsilon": 1e-6,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "training_splits": ["train", "validation"],
            "training_rows": len(development),
            "calibration_split": "calibration",
            "calibration_rows": len(calibration_data),
        },
        bundle_path,
    )

    restored = joblib.load(bundle_path)

    restored_raw_scores = restored["model"].predict_proba(
        calibration_data[restored["feature_columns"]]
    )[:, 1]

    restored_scores = restored["calibrator"].predict_proba(
        scores_to_log_odds(restored_raw_scores)
    )[:, 1]

    np.testing.assert_allclose(
        restored_scores,
        calibrated_scores,
        rtol=1e-12,
        atol=1e-12,
    )

    print("PASS: Model bundle round-trip predictions verified.")
    print("Saved bundle:", bundle_path)