import json
import time

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.features import PROJECT_ROOT
from src.replay import HISTORY_COLUMNS, replay_history


def load_development_features():
    data_dir = PROJECT_ROOT / "data" / "processed"

    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                h.*,
                s.split
            FROM read_parquet(?) AS h
            INNER JOIN read_parquet(?) AS s
                ON h.TransactionID = s.TransactionID
            WHERE s.split != 'test'
            ORDER BY h.TransactionDT, h.TransactionID
            """,
            [
                str(data_dir / "history_features.parquet"),
                str(data_dir / "split_membership.parquet"),
            ],
        ).df()


def assert_feature_parity(expected, actual):
    expected = (
        expected.set_index("TransactionID")[HISTORY_COLUMNS]
        .sort_index()
        .astype("float64")
    )

    actual = (
        actual.set_index("TransactionID")[HISTORY_COLUMNS]
        .sort_index()
        .astype("float64")
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=False,
        rtol=1e-7,
        atol=1e-7,
    )


def build_monitoring_report(data):
    reference = data.loc[data["split"] == "train"]
    current = data.loc[data["split"] == "policy"]
    records = []

    for column in [
        "TransactionAmt",
        "hist_count",
        "time_since_previous",
    ]:
        old = reference[column].astype("float64")
        new = current[column].astype("float64")

        old_values = old.dropna().to_numpy()
        new_values = new.dropna().to_numpy()

        psi = None

        if len(old_values) and len(new_values):
            edges = np.unique(
                np.quantile(old_values, np.linspace(0, 1, 11))
            )

            if len(edges) >= 2:
                edges[0] = -np.inf
                edges[-1] = np.inf

                old_counts = (
                    np.histogram(old_values, bins=edges)[0] + 0.5
                )
                new_counts = (
                    np.histogram(new_values, bins=edges)[0] + 0.5
                )

                p = old_counts / old_counts.sum()
                q = new_counts / new_counts.sum()

                psi = float(np.sum((q - p) * np.log(q / p)))

        records.append({
            "feature": column,
            "reference_split": "train",
            "current_split": "policy",
            "reference_missing_rate": float(old.isna().mean()),
            "current_missing_rate": float(new.isna().mean()),
            "reference_median": old.median(),
            "current_median": new.median(),
            "psi_descriptive": psi,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    started = time.perf_counter()
    data = load_development_features()

    if data.empty or not data["TransactionID"].is_unique:
        raise ValueError("Development input must be non-empty and unique.")

    replay_input = data[
        [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "group_key",
        ]
    ]

    # Keep the entire timestamp group in the first batch.
    cutoff = replay_input["TransactionDT"].iloc[len(data) // 2]

    first_batch = replay_input.loc[
        replay_input["TransactionDT"] <= cutoff
    ]
    second_batch = replay_input.loc[
        replay_input["TransactionDT"] > cutoff
    ]

    report_dir = PROJECT_ROOT / "reports"
    artifact_dir = PROJECT_ROOT / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    first_result, checkpoint = replay_history(first_batch)

    checkpoint_path = artifact_dir / "replay_checkpoint.joblib"
    joblib.dump(checkpoint, checkpoint_path)
    restored = joblib.load(checkpoint_path)

    second_result, checkpoint = replay_history(
        second_batch,
        checkpoint=restored,
    )

    replayed = pd.concat(
        [first_result, second_result],
        ignore_index=True,
    )

    assert_feature_parity(data, replayed)

    retried, retry_checkpoint = replay_history(
        replay_input,
        checkpoint=checkpoint,
    )

    if not retried.empty or retry_checkpoint != checkpoint:
        raise ValueError("Duplicate retry changed replay state.")

    joblib.dump(checkpoint, checkpoint_path)

    monitoring = build_monitoring_report(data)
    monitoring.to_csv(
        report_dir / "development_monitoring.csv",
        index=False,
    )

    report = {
        "scope": "all development splits; test excluded",
        "rows": len(data),
        "features_checked": HISTORY_COLUMNS,
        "sql_replay_parity": True,
        "checkpoint_resume_parity": True,
        "duplicate_retry_noop": True,
        "seconds": round(time.perf_counter() - started, 2),
    }

    (report_dir / "replay_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("PASS: Full development SQL/replay parity verified.")
    print("PASS: Checkpoint/resume and duplicate retry verified.")
    print("Development rows:", len(data))
    print("Elapsed seconds:", report["seconds"])
    print("\nDescriptive monitoring:")
    print(monitoring.to_string(index=False))