from pathlib import Path
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_development_data():
    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                t.TransactionID,
                t.TransactionDT,
                t.TransactionAmt,
                t.isFraud,
                s.split
            FROM read_parquet(?) AS t
            INNER JOIN read_parquet(?) AS s
                ON t.TransactionID = s.TransactionID
            WHERE s.split IN ('train', 'validation')
            """,
            [
                str(DATA_DIR / "joined_transactions.parquet"),
                str(DATA_DIR / "split_membership.parquet"),
            ],
        ).df()

def select_amount_baseline(validation, review_rate=0.01):
    review_capacity = int(len(validation) * review_rate)

    ranked = validation.sort_values(
        ["TransactionAmt", "TransactionID"],
        ascending=[False, True],
    )

    selected = ranked.head(review_capacity).copy()

    return selected, review_capacity

def evaluate_selection(validation, selected):
    total_fraud = int(validation["isFraud"].sum())
    true_positives = int(selected["isFraud"].sum())
    false_positives = len(selected) - true_positives
    false_negatives = total_fraud - true_positives

    return {
        "selected_transactions": len(selected),
        "total_fraud": total_fraud,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / len(selected) if len(selected) else None
        ),
        "recall": (
            true_positives / total_fraud if total_fraud else None
        ),
        "fdr": (
            false_positives / len(selected) if len(selected) else None
        ),
        "review_rate": len(selected) / len(validation),
    }

def select_random_baseline(validation, review_rate=0.01, seed=42):
    review_capacity = int(len(validation) * review_rate)

    ordered = validation.sort_values("TransactionID")

    return ordered.sample(
        n=review_capacity,
        random_state=seed,
    ).copy()

if __name__ == "__main__":
    data = load_development_data()

    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    
    selected, capacity = select_amount_baseline(validation)

    print("Review capacity:", capacity)
    print("Selected transactions:", len(selected))
    print(f"Actual review rate: {len(selected) / len(validation):.2%}")

    metrics = evaluate_selection(validation, selected)

    for name, value in metrics.items():
        if name in {"precision", "recall", "fdr", "review_rate"}:
            formatted = f"{value:.4%}" if value is not None else "N/A"
            print(f"{name}: {formatted}")
        else:
            print(f"{name}: {value}")


    random_selected = select_random_baseline(validation)
    random_metrics = evaluate_selection(validation, random_selected)

    print("\nRandom baseline:")

    for name in ["true_positives", "precision", "recall", "fdr"]:
        value = random_metrics[name]

        if name == "true_positives":
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4%}")
        random_results = []
    

    for seed in range(100):
        random_selected = select_random_baseline(validation, seed=seed)
        result = evaluate_selection(validation, random_selected)
        result["seed"] = seed
        random_results.append(result)

    random_summary = pd.DataFrame(random_results)

    print("\nRandom baseline across 100 seeds:")
    print(
        random_summary[
            ["true_positives", "precision", "recall"]
        ].agg(["mean", "std", "min", "max"]).round(4)
    )

    expected_random_hits = (
        capacity * validation["isFraud"].mean()
    )

    print(f"\nExpected random fraud hits: {expected_random_hits:.2f}")
    
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    amount_report = pd.DataFrame([
        {"method": "amount_top_1pct", **metrics}
    ])

    amount_report.to_csv(
        report_dir / "validation_amount_baseline.csv",
        index=False,
    )

    random_summary.to_csv(
        report_dir / "validation_random_baseline_runs.csv",
        index=False,
    )

    random_summary[
        ["true_positives", "precision", "recall", "fdr", "review_rate"]
    ].agg(["mean", "std", "min", "max"]).to_csv(
        report_dir / "validation_random_baseline_summary.csv"
    )

    print("Baseline reports saved.")