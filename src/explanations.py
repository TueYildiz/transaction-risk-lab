import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_evaluation import load_frozen_artifacts
from src.policy import load_policy_data
from src.modeling import PROJECT_ROOT


if __name__ == "__main__":
    bundle, policy = load_frozen_artifacts()
    data = load_policy_data()

    # Use a reproducible sample from the policy period.
    sample = data.sample(
        n=min(1000, len(data)),
        random_state=42,
    ).copy()

    numeric_features = bundle["numeric_features"]

    sample[numeric_features] = (
        sample[numeric_features].astype("float64")
    )

    pipeline = bundle["model"]
    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessing.transform(
        sample[bundle["feature_columns"]]
    )

    feature_names = preprocessing.get_feature_names_out()

    contributions = classifier.booster_.predict(
        transformed,
        pred_contrib=True,
    )

    if hasattr(contributions, "toarray"):
        contributions = contributions.toarray()

    contributions = np.asarray(contributions)

    raw_margins = classifier.booster_.predict(
        transformed,
        raw_score=True,
    )

    np.testing.assert_allclose(
        contributions.sum(axis=1),
        raw_margins,
        rtol=1e-6,
        atol=1e-6,
    )

    # The final contribution column is the expected-value baseline.
    feature_contributions = contributions[:, :-1]

    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_contribution": (
            np.abs(feature_contributions).mean(axis=0)
        ),
        "mean_signed_contribution": (
            feature_contributions.mean(axis=0)
        ),
    }).sort_values(
        "mean_absolute_contribution",
        ascending=False,
    )

    report_dir = PROJECT_ROOT / "reports"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        report_dir / "policy_feature_contributions.csv",
        index=False,
    )

    top_features = summary.head(15).sort_values(
        "mean_absolute_contribution"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_features["feature"],
        top_features["mean_absolute_contribution"],
    )

    ax.set_xlabel("Mean absolute TreeSHAP contribution (raw log-odds)")
    ax.set_title("Policy-period sample: model contributions")

    fig.tight_layout()
    fig.savefig(
        figure_dir / "policy_feature_contributions.png",
        dpi=150,
    )
    plt.close(fig)

    print("PASS: Contributions sum to raw model margins.")
    print("Policy sample rows:", len(sample))
    print(summary.head(15).to_string(index=False))