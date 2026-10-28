from pathlib import Path
import pandas as pd
import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"
def load_data():
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "isFraud",
        ],
    )

    identity = pd.read_csv(
        IDENTITY_PATH,
        usecols=[
            "TransactionID",
            "DeviceType",
            "DeviceInfo",
        ],
    )

    return transactions, identity

def validate_ids(transactions, identity):
    for name, frame in [
        ("transactions", transactions),
        ("identity", identity),
    ]:
        if frame["TransactionID"].isna().any():
            raise ValueError(f"{name}: missing TransactionID values.")

        if frame["TransactionID"].duplicated().any():
            raise ValueError(f"{name}: duplicate TransactionID values.")
        
def join_data(transactions, identity):
    with duckdb.connect() as con:
        con.register("transactions_input", transactions)
        con.register("identity_input", identity)

        joined = con.execute("""
            SELECT
                t.*,
                i.DeviceType,
                i.DeviceInfo,
                i.TransactionID IS NOT NULL AS has_identity
            FROM transactions_input AS t
            LEFT JOIN identity_input AS i
                ON t.TransactionID = i.TransactionID
        """).df()

    return joined
def validate_join(transactions, joined):
    if len(joined) != len(transactions):
        raise ValueError("JOIN changed the transaction row count.")

    if joined["TransactionID"].isna().any():
        raise ValueError("JOIN output contains missing TransactionID values.")

    if not joined["TransactionID"].is_unique:
        raise ValueError("JOIN output contains duplicate TransactionID values.")


def validate_target(joined):
    if not joined["isFraud"].isin([0, 1]).all():
        raise ValueError("Target contains missing or unexpected values.")



def validate_time(joined):
    if not np.isfinite(joined["TransactionDT"]).all():
        raise ValueError("TransactionDT contains missing or infinite values.") 


def validate_amount(joined):
    amounts = joined["TransactionAmt"]

    if not np.isfinite(amounts).all():
        raise ValueError("TransactionAmt contains missing or infinite values.")

    if (amounts < 0).any():
        raise ValueError("TransactionAmt contains negative values.")  

def save_data(joined):
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "joined_transactions.parquet"

    with duckdb.connect() as con:
        con.register("joined_output", joined)
        con.execute(
            """
            COPY joined_output
            TO ?
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(output_path)],
        )

    return output_path

def validate_saved_data(joined, output_path):
    with duckdb.connect() as con:
        reloaded = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(output_path)],
        ).df()

    expected = joined.sort_values("TransactionID").reset_index(drop=True)
    actual = reloaded.sort_values("TransactionID").reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)

if __name__ == "__main__":
    transactions, identity = load_data()
    validate_ids(transactions, identity)

    joined = join_data(transactions, identity)

    validate_join(transactions, joined)
    validate_target(joined)
    validate_time(joined)
    validate_amount(joined)

    print("PASS: All validations completed.")

    output_path = save_data(joined)
    validate_saved_data(joined, output_path)
    print("PASS: Parquet round-trip validation completed.")

    print("Rows before JOIN:", len(transactions))
    print("Rows after JOIN:", len(joined))
    print("Identity matches:", joined["has_identity"].sum())
    print("Saved file:", output_path)

