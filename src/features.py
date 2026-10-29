from pathlib import Path
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "joined_transactions.parquet"


def inspect_group_fields():
    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                COUNT(*) AS transactions,
                COUNT(*) FILTER (WHERE card1 IS NULL) AS missing_card1,
                COUNT(*) FILTER (WHERE card4 IS NULL) AS missing_card4,
                COUNT(*) FILTER (WHERE addr1 IS NULL) AS missing_addr1,
                COUNT(*) FILTER (
                    WHERE card1 IS NULL
                       OR card4 IS NULL
                       OR addr1 IS NULL
                ) AS missing_any_group_field
            FROM read_parquet(?)
            """,
            [str(INPUT_PATH)],
        ).df()

def preview_group_keys():
    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                TransactionID,
                card1,
                card4,
                addr1,
                CASE
                    WHEN card1 IS NULL
                      OR card4 IS NULL
                      OR addr1 IS NULL
                    THEN NULL
                    ELSE to_json(list_value(
                        CAST(card1 AS VARCHAR),
                        CAST(card4 AS VARCHAR),
                        CAST(addr1 AS VARCHAR)
                    ))
                END AS group_key
            FROM read_parquet(?)
            ORDER BY TransactionID
            LIMIT 10
            """,
            [str(INPUT_PATH)],
        ).df()

def demonstrate_history():
    with duckdb.connect() as con:
        return con.execute("""
            WITH example AS (
                SELECT *
                FROM (
                    VALUES
                        (1, 'group_a', 10, 20.0),
                        (2, 'group_a', 20, 40.0),
                        (3, 'group_a', 20, 60.0),
                        (4, 'group_a', 30, 90.0)
                ) AS t(
                    TransactionID,
                    group_key,
                    TransactionDT,
                    TransactionAmt
                )
            )
            SELECT
                *,
                COUNT(*) OVER history AS hist_count,
                AVG(TransactionAmt) OVER history AS hist_mean_amount
            FROM example
            WINDOW history AS (
                PARTITION BY group_key
                ORDER BY TransactionDT
                RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                EXCLUDE GROUP
            )
            ORDER BY TransactionDT, TransactionID
        """).df()
    
def build_history_features(input_path=INPUT_PATH):
     with duckdb.connect() as con:
        return con.execute(
            """
            WITH keyed AS (
                SELECT
                    TransactionID,
                    TransactionDT,
                    TransactionAmt,
                    CASE
                        WHEN card1 IS NULL
                          OR card4 IS NULL
                          OR addr1 IS NULL
                        THEN NULL
                        ELSE to_json(list_value(
                            CAST(card1 AS VARCHAR),
                            CAST(card4 AS VARCHAR),
                            CAST(addr1 AS VARCHAR)
                        ))
                    END AS group_key
                FROM read_parquet(?)
            ),
            history AS (
                SELECT
                    TransactionID,
                    COUNT(*) OVER past AS hist_count,
                    COUNT(TransactionAmt) OVER past AS hist_valid_amount_count,
                    AVG(TransactionAmt) OVER past AS hist_mean_amount,
                    MAX(TransactionDT) OVER past AS previous_transaction_time,
                    MAX(TransactionAmt) OVER past AS hist_max_amount,
                    SUM(TransactionAmt) OVER past AS hist_total_amount,
                    STDDEV_POP(TransactionAmt) OVER past AS hist_std_amount
                FROM keyed
                WHERE group_key IS NOT NULL
                WINDOW past AS (
                    PARTITION BY group_key
                    ORDER BY TransactionDT
                    RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    EXCLUDE GROUP
                )
            )
            SELECT
                k.*,
                h.hist_count,
                h.hist_valid_amount_count,
                h.hist_mean_amount,
                h.hist_max_amount,
                h.hist_total_amount,
                h.hist_std_amount,
                CASE
                    WHEN k.group_key IS NULL THEN NULL
                    WHEN h.hist_count = 0 THEN 1
                    ELSE 0
                END AS is_new_group,
                k.TransactionAmt
                    / NULLIF(h.hist_mean_amount, 0)
                    AS amount_to_hist_mean,
                k.TransactionAmt - h.hist_mean_amount
                    AS amount_minus_hist_mean,
                k.TransactionDT - h.previous_transaction_time
                    AS time_since_previous
            FROM keyed AS k
            LEFT JOIN history AS h
                ON k.TransactionID = h.TransactionID
            ORDER BY k.TransactionDT, k.TransactionID
            """,
            [str(input_path)],
        ).df()
    





def save_features(features):
    output_path = (
        PROJECT_ROOT / "data" / "processed" / "history_features.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.register("feature_output", features)

        con.execute(
            """
            COPY feature_output
            TO ?
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(output_path)],
        )

        reloaded = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(output_path)],
        ).df()

    expected = features.sort_values("TransactionID").reset_index(drop=True)
    actual = reloaded.sort_values("TransactionID").reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)

    return output_path



def validate_features(features):
    history_columns = [
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

    if features["TransactionID"].isna().any():
        raise ValueError("Feature output contains missing transaction IDs.")

    if not features["TransactionID"].is_unique:
        raise ValueError("Feature output contains duplicate transaction IDs.")

    valid_key = features["group_key"].notna()
    missing_key = ~valid_key

    if features.loc[missing_key, history_columns].notna().any().any():
        raise ValueError("Missing group keys must have missing history features.")

    counts = features.loc[valid_key, "hist_count"]

    if counts.isna().any() or (counts < 0).any():
        raise ValueError("Valid groups must have non-negative historical counts.")

    has_history = valid_key & features["hist_count"].gt(0).fillna(False)
    no_history = valid_key & features["hist_count"].eq(0).fillna(False)

    gaps = features.loc[has_history, "time_since_previous"]

    if gaps.isna().any() or (gaps <= 0).any():
        raise ValueError("Transactions with history must have positive time gaps.")

    if features.loc[
        no_history, "time_since_previous"
    ].notna().any():
        raise ValueError("New groups must have missing time gaps.")

    expected_new = counts.eq(0).astype(int)
    actual_new = features.loc[valid_key, "is_new_group"]

    if actual_new.isna().any() or not actual_new.eq(expected_new).all():
        raise ValueError("New-group indicators do not match historical counts.")

    valid_amount_counts = features.loc[
        valid_key, "hist_valid_amount_count"
    ]

    if not valid_amount_counts.eq(counts).all():
        raise ValueError("Historical amount counts do not match transaction counts.")
    

if __name__ == "__main__":
    features = build_history_features()
    validate_features(features)

    print("PASS: Feature validation completed.")

    output_path = save_features(features)

    print("PASS: Feature Parquet round-trip validation completed.")
    print("Feature shape:", features.shape)
    print("Saved features:", output_path)
