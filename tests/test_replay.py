import duckdb
import pandas as pd
import joblib
import pytest

from src.replay import HISTORY_COLUMNS
from copy import deepcopy

from src.features import build_history_features
from src.replay import replay_history

def test_missing_group_key_has_no_history():
    transactions = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [10, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0],
        "group_key": [None, "group_a", None],
    })

    result, _ = replay_history(transactions)
    result = result.set_index("TransactionID")

    assert result.loc[[1, 3]].isna().all().all(), (
        "Missing group keys must have missing historical features."
    )

def test_replay_matches_sql(tmp_path):
    fixture = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0, 90.0],
        "card1": [100, 100, 100, 100],
        "card4": ["visa", "visa", "visa", "visa"],
        "addr1": [200.0, 200.0, 200.0, 200.0],
    })

    path = tmp_path / "fixture.parquet"

    with duckdb.connect() as con:
        con.register("fixture_input", fixture)
        con.execute(
            "COPY fixture_input TO ? (FORMAT PARQUET)",
            [str(path)],
        )

    sql_result = build_history_features(path)

    replay_input = sql_result[
        [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "group_key",
        ]
    ]

    replay_result, _ = replay_history(replay_input)

    columns = [
        "hist_count",
        "hist_mean_amount",
        "time_since_previous",
        "amount_to_hist_mean",
        "amount_minus_hist_mean",
        "hist_total_amount",
        "is_new_group",
        "hist_max_amount",
        "hist_valid_amount_count",
        "hist_std_amount",
    ]

    expected = (
        sql_result.set_index("TransactionID")[columns]
        .sort_index()
        .astype("float64")
    )

    actual = (
        replay_result.set_index("TransactionID")[columns]
        .sort_index()
        .astype("float64")
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )


def test_checkpoint_resume_matches_full_replay(tmp_path):
    transactions = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 20, 30],
        "TransactionAmt": [20.0, 40.0, 60.0, 90.0],
        "group_key": ["group_a"] * 4,
    })

    full_result, full_checkpoint = replay_history(transactions)

    first_batch = transactions.loc[
        transactions["TransactionDT"] <= 20
    ]
    second_batch = transactions.loc[
        transactions["TransactionDT"] > 20
    ]

    first_result, checkpoint = replay_history(first_batch)

    checkpoint_path = tmp_path / "checkpoint.joblib"
    joblib.dump(checkpoint, checkpoint_path)
    restored_checkpoint = joblib.load(checkpoint_path)

    second_result, resumed_checkpoint = replay_history(
        second_batch,
        checkpoint=restored_checkpoint,
    )

    resumed_result = pd.concat(
        [first_result, second_result],
        ignore_index=True,
    )

    expected = (
        full_result.set_index("TransactionID")[HISTORY_COLUMNS]
        .sort_index()
        .astype("float64")
    )

    actual = (
        resumed_result.set_index("TransactionID")[HISTORY_COLUMNS]
        .sort_index()
        .astype("float64")
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )

    assert resumed_checkpoint == full_checkpoint

    assert checkpoint["last_time"] == 20
    assert checkpoint["state"]["group_a"]["count"] == 3

def test_resume_rejects_completed_timestamp():
    first_batch = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [10, 20],
        "TransactionAmt": [20.0, 40.0],
        "group_key": ["group_a", "group_a"],
    })

    late_transaction = pd.DataFrame({
        "TransactionID": [3],
        "TransactionDT": [20],
        "TransactionAmt": [60.0],
        "group_key": ["group_a"],
    })

    _, checkpoint = replay_history(first_batch)

    with pytest.raises(
        ValueError,
        match="New batches must start after",
    ):
        replay_history(
            late_transaction,
            checkpoint=checkpoint,
        )

def test_duplicate_retry_is_noop():
    transactions = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [10, 20],
        "TransactionAmt": [20.0, 40.0],
        "group_key": ["group_a", "group_a"],
    })

    _, checkpoint = replay_history(transactions)
    before = deepcopy(checkpoint)

    retried, after = replay_history(
        transactions,
        checkpoint=checkpoint,
    )

    assert retried.empty
    assert after == before
    assert checkpoint == before


def test_conflicting_duplicate_is_rejected():
    transactions = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [10],
        "TransactionAmt": [20.0],
        "group_key": ["group_a"],
    })

    _, checkpoint = replay_history(transactions)

    changed = transactions.copy()
    changed["TransactionAmt"] = 999.0

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        replay_history(changed, checkpoint=checkpoint)


def test_resume_rejects_completed_timestamp():
    first_batch = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [20],
        "TransactionAmt": [40.0],
        "group_key": ["group_a"],
    })

    late_transaction = pd.DataFrame({
        "TransactionID": [2],
        "TransactionDT": [20],
        "TransactionAmt": [60.0],
        "group_key": ["group_a"],
    })

    _, checkpoint = replay_history(first_batch)

    with pytest.raises(
        ValueError,
        match="New batches must start after",
    ):
        replay_history(late_transaction, checkpoint=checkpoint)
    
