from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_training_data():
    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT t.*
            FROM read_parquet(?) AS t
            INNER JOIN read_parquet(?) AS s
                ON t.TransactionID = s.TransactionID
            WHERE s.split = 'train'
            """,
            [
                str(DATA_DIR / "joined_transactions.parquet"),
                str(DATA_DIR / "split_membership.parquet"),
            ],
        ).df()
    
def save_product_summary(train):
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        train.groupby("ProductCD")
        .agg(
            transactions=("TransactionID", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
        .sort_values("fraud_rate", ascending=False)
    )

    output_path = report_dir / "train_product_summary.csv"
    summary.to_csv(output_path)

    return output_path

def save_identity_summary(train):
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        train.groupby("has_identity")
        .agg(
            transactions=("TransactionID", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
    )

    output_path = report_dir / "train_identity_summary.csv"
    summary.to_csv(output_path)

    return output_path

def save_weekly_summary(train):
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    seconds_per_week = 7 * 24 * 60 * 60
    train_start = train["TransactionDT"].min()

    data = train.assign(
        relative_week=(
            (train["TransactionDT"] - train_start) // seconds_per_week
        ).astype(int)
    )

    summary = (
        data.groupby("relative_week")
        .agg(
            transactions=("TransactionID", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            time_min=("TransactionDT", "min"),
            time_max=("TransactionDT", "max"),
        )
    )

    output_path = report_dir / "train_weekly_summary.csv"
    summary.to_csv(output_path)

    return output_path

def save_weekly_plot(weekly_report_path):
    summary = pd.read_csv(weekly_report_path)

    # Exclude the final partial week in the current training period.
    full_weeks = summary.loc[summary["relative_week"] < 13]

    figure_dir = PROJECT_ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].bar(
        full_weeks["relative_week"],
        full_weeks["transactions"],
    )
    axes[0].set_ylabel("Transaction count")
    axes[0].set_title("Training-period weekly transaction volume")

    axes[1].plot(
        full_weeks["relative_week"],
        full_weeks["fraud_rate"] * 100,
        marker="o",
    )

    overall_rate = (
        summary["fraud_count"].sum() / summary["transactions"].sum()
    )

    axes[1].axhline(
        overall_rate * 100,
        color="red",
        linestyle="--",
        label="Overall training fraud rate",
    )
    axes[1].set_xlabel("Week relative to training start")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].legend()

    fig.tight_layout()

    output_path = figure_dir / "train_weekly_fraud.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

if __name__ == "__main__":
    train = load_training_data()

    print("Training rows:", len(train))
    print(f"Training fraud rate: {train['isFraud'].mean():.2%}")

    product_report_path = save_product_summary(train)
    print("Product summary saved:", product_report_path)

    identity_report_path = save_identity_summary(train)
    print("Identity summary saved:", identity_report_path)

    weekly_report_path = save_weekly_summary(train)
    print("Weekly summary saved:", weekly_report_path)

    figure_path = save_weekly_plot(weekly_report_path)
    print("Weekly figure saved:", figure_path)