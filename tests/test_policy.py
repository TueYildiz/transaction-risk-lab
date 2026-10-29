import pandas as pd

from src.policy import apply_review_policy


def test_threshold_filtering():
    data = pd.DataFrame({"TransactionID": [1, 2, 3, 4]})
    scores = [0.90, 0.60, 0.40, 0.10]

    selected = apply_review_policy(
        data,
        scores,
        threshold=0.60,
        max_review_rate=1.0,
    )

    assert selected["TransactionID"].tolist() == [1, 2]


def test_capacity_and_tie_breaking():
    data = pd.DataFrame({"TransactionID": [4, 3, 2, 1]})
    scores = [0.80, 0.80, 0.80, 0.90]

    selected = apply_review_policy(
        data,
        scores,
        threshold=0.50,
        max_review_rate=0.50,
    )

    assert len(selected) == 2
    assert selected["TransactionID"].tolist() == [1, 2]


def test_no_scores_above_threshold():
    data = pd.DataFrame({"TransactionID": [1, 2]})

    selected = apply_review_policy(
        data,
        [0.10, 0.20],
        threshold=0.50,
        max_review_rate=1.0,
    )

    assert selected.empty