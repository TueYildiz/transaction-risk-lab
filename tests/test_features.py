from pathlib import Path
import tempfile

import duckdb
import pandas as pd

from src.features import build_history_features


def test_same_time_history():
    fixture = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0, 90.0],
        "card1": [100, 100, 100, 100],
        "card4": ["visa", "visa", "visa", "visa"],
        "addr1": [200.0, 200.0, 200.0, 200.0],
    })

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fixture.parquet"

        with duckdb.connect() as con:
            con.register("fixture_input", fixture)
            con.execute(
                "COPY fixture_input TO ? (FORMAT PARQUET)",
                [str(path)],
            )

        actual = build_history_features(path).set_index("TransactionID")

    assert actual.loc[[1, 2, 3, 4], "hist_count"].tolist() == [
        0, 1, 1, 3
    ], "Same-time transactions leaked into historical counts."

    assert actual.loc[[2, 3, 4], "hist_mean_amount"].tolist() == [
        20.0, 20.0, 40.0
    ], "Unexpected historical means."

    assert actual.loc[[2, 3, 4], "time_since_previous"].tolist() == [
        10, 10, 10
    ], "Unexpected historical time gaps."

    print("PASS: Production feature function excludes same-time transactions.")

def test_future_invariance():
    fixture = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0, 90.0],
        "card1": [100, 100, 100, 100],
        "card4": ["visa", "visa", "visa", "visa"],
        "addr1": [200.0, 200.0, 200.0, 200.0],
    })

    changed = fixture.copy()
    changed.loc[changed["TransactionID"] == 4, "TransactionAmt"] = 9999.0

    results = []

    with tempfile.TemporaryDirectory() as folder:
        for name, frame in [("original", fixture), ("changed", changed)]:
            path = Path(folder) / f"{name}.parquet"

            with duckdb.connect() as con:
                con.register("fixture_input", frame)
                con.execute(
                    "COPY fixture_input TO ? (FORMAT PARQUET)",
                    [str(path)],
                )

            result = build_history_features(path)
            earlier = result.loc[
                result["TransactionDT"] < 30
            ].set_index("TransactionID").sort_index()

            results.append(earlier)

    pd.testing.assert_frame_equal(results[0], results[1])

    print("PASS: Future amount changes do not affect earlier features.")

def test_row_order_invariance():
    fixture = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0, 90.0],
        "card1": [100, 100, 100, 100],
        "card4": ["visa", "visa", "visa", "visa"],
        "addr1": [200.0, 200.0, 200.0, 200.0],
    })

    shuffled = fixture.sample(frac=1, random_state=42)
    results = []

    with tempfile.TemporaryDirectory() as folder:
        for name, frame in [
            ("original", fixture),
            ("shuffled", shuffled),
        ]:
            path = Path(folder) / f"{name}.parquet"

            with duckdb.connect() as con:
                con.register("fixture_input", frame)
                con.execute(
                    "COPY fixture_input TO ? (FORMAT PARQUET)",
                    [str(path)],
                )

            result = (
                build_history_features(path)
                .set_index("TransactionID")
                .sort_index()
            )

            results.append(result)

    pd.testing.assert_frame_equal(results[0], results[1])

    print("PASS: Input row order does not affect feature values.")

