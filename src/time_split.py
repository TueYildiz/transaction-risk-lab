from pathlib import Path
import pandas as pd
import duckdb
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "joined_transactions.parquet"

SPLIT_NAMES = ["train", "validation", "calibration", "policy", "test"]
SPLIT_QUANTILES = [0.55, 0.65, 0.75, 0.85]


def load_joined_data():
    with duckdb.connect() as con:
        return con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(INPUT_PATH)],
        ).df()

def assign_splits(joined, time_cutoffs):
    result = joined.copy()

    bins = [
        float("-inf"),
        *time_cutoffs.tolist(),
        float("inf"),
    ]

    result["split"] = pd.cut(
        result["TransactionDT"],
        bins=bins,
        labels=SPLIT_NAMES,
        right=True,
    )

    return result

def validate_splits(joined, split_data):
    if len(split_data) != len(joined):
        raise ValueError("Split assignment changed the row count.")

    if split_data["split"].isna().any():
        raise ValueError("Some transactions have no split.")

    summary = (
        split_data.groupby("split", observed=False)
        .agg(
            rows=("TransactionID", "size"),
            time_min=("TransactionDT", "min"),
            time_max=("TransactionDT", "max"),
        )
        .reindex(SPLIT_NAMES)
    )

    if (summary["rows"] == 0).any():
        raise ValueError("At least one split is empty.")

    for previous, following in zip(SPLIT_NAMES[:-1], SPLIT_NAMES[1:]):
        if (
            summary.loc[previous, "time_max"]
            >= summary.loc[following, "time_min"]
        ):
            raise ValueError(
                f"Time overlap between {previous} and {following}."
            )

    return summary

def save_split_protocol(time_cutoffs):
    protocol_dir = PROJECT_ROOT / "artifacts"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = protocol_dir / "split_protocol.json"

    protocol = {
        "time_column": "TransactionDT",
        "split_names": SPLIT_NAMES,
        "quantiles": SPLIT_QUANTILES,
        "cutoffs": time_cutoffs.tolist(),
        "right_inclusive": True,
    }

    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))

        if existing != protocol:
            raise ValueError(
                "Split protocol changed. Review the existing protocol."
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2),
            encoding="utf-8",
        )

    return protocol_path

def save_split_membership(split_data):
    output_path = (
        PROJECT_ROOT / "data" / "processed" / "split_membership.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    membership = split_data[["TransactionID", "split"]].copy()
    membership["split"] = membership["split"].astype(str)

    with duckdb.connect() as con:
        con.register("membership_output", membership)

        con.execute(
            """
            COPY membership_output
            TO ?
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(output_path)],
        )

    return output_path


def validate_saved_membership(split_data, membership_path):
    with duckdb.connect() as con:
        actual = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(membership_path)],
        ).df()

    expected = split_data[["TransactionID", "split"]].copy()

    expected["split"] = expected["split"].astype("string")
    actual["split"] = actual["split"].astype("string")

    expected = expected.sort_values("TransactionID").reset_index(drop=True)
    actual = actual.sort_values("TransactionID").reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)



if __name__ == "__main__":
    joined = load_joined_data()
    time_cutoffs = joined["TransactionDT"].quantile(SPLIT_QUANTILES)

    split_data = assign_splits(joined, time_cutoffs)
    summary = validate_splits(joined, split_data)

    protocol_path = save_split_protocol(time_cutoffs)
    membership_path = save_split_membership(split_data)
    validate_saved_membership(split_data, membership_path)
    print("PASS: Split membership round-trip validation completed.")
    print("Split membership:", membership_path)
    print("Split protocol:", protocol_path)

    print("PASS: Split validation completed.")
    print(summary)
