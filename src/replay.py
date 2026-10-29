import math
from copy import deepcopy

import numpy as np
import pandas as pd


HISTORY_COLUMNS = [
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


def empty_state():
    return {
        "count": 0,
        "total": 0.0,
        "last_time": None,
        "maximum": None,
        "mean": 0.0,
        "m2": 0.0,
    }


def record_signature(row):
    key = row["group_key"]

    return (
        float(row["TransactionDT"]),
        float(row["TransactionAmt"]),
        None if key is None or pd.isna(key) else str(key),
    )


def replay_history(transactions, checkpoint=None):
    for column in ["TransactionAmt", "TransactionDT"]:
        values = transactions[column].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values.")

    if transactions["TransactionID"].isna().any():
        raise ValueError("Transaction IDs must not be missing.")

    checkpoint = (
        deepcopy(checkpoint)
        if checkpoint is not None
        else {"state": {}, "last_time": None, "seen": {}}
    )

    state = checkpoint["state"]
    seen = checkpoint.setdefault("seen", {})
    output = []

    ordered = transactions.sort_values(
        ["TransactionDT", "TransactionID"]
    )

    for timestamp, same_time in ordered.groupby("TransactionDT"):
        pending = {}

        for row in same_time.to_dict("records"):
            identifier = int(row["TransactionID"])
            signature = record_signature(row)

            if identifier in seen:
                if seen[identifier] != signature:
                    raise ValueError("Conflicting duplicate transaction.")
                continue

            if identifier in pending:
                if pending[identifier][1] != signature:
                    raise ValueError("Conflicting duplicate transaction.")
                continue

            pending[identifier] = (row, signature)

        if not pending:
            continue

        if (
            checkpoint["last_time"] is not None
            and timestamp <= checkpoint["last_time"]
        ):
            raise ValueError(
                "New batches must start after the last completed timestamp."
            )

        # Generate features before updating any same-time state.
        for identifier, (row, signature) in pending.items():
            key = signature[2]
            amount = float(row["TransactionAmt"])

            history = dict.fromkeys(HISTORY_COLUMNS)

            if key is not None:
                previous = state.get(key, empty_state())
                count = previous["count"]
                mean = previous["total"] / count if count else None

                history = {
                    "hist_count": count,
                    "hist_valid_amount_count": count,
                    "hist_mean_amount": mean,
                    "hist_std_amount": (
                        math.sqrt(max(previous["m2"] / count, 0.0))
                        if count else None
                    ),
                    "hist_max_amount": previous["maximum"],
                    "hist_total_amount": (
                        previous["total"] if count else None
                    ),
                    "time_since_previous": (
                        timestamp - previous["last_time"]
                        if previous["last_time"] is not None
                        else None
                    ),
                    "amount_to_hist_mean": (
                        amount / mean
                        if mean is not None and mean != 0
                        else None
                    ),
                    "amount_minus_hist_mean": (
                        amount - mean if mean is not None else None
                    ),
                    "is_new_group": int(count == 0),
                }

            output.append({
                "TransactionID": identifier,
                **history,
            })

        # Update state only after the entire timestamp is scored.
        for identifier, (row, signature) in pending.items():
            key = signature[2]

            if key is not None:
                previous = state.setdefault(key, empty_state())
                amount = float(row["TransactionAmt"])

                previous["count"] += 1

                delta = amount - previous["mean"]
                previous["mean"] += delta / previous["count"]
                previous["m2"] += (
                    delta * (amount - previous["mean"])
                )

                previous["total"] += amount
                previous["last_time"] = float(timestamp)
                previous["maximum"] = (
                    amount
                    if previous["maximum"] is None
                    else max(previous["maximum"], amount)
                )

            seen[identifier] = signature

        checkpoint["last_time"] = float(timestamp)

    return (
        pd.DataFrame(
            output,
            columns=["TransactionID"] + HISTORY_COLUMNS,
        ),
        checkpoint,
    )
