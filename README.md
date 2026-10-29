
| TransactionID | unique transaction identifier | Used to join the transaction and identity tables.|
| TransactionDT | Elapsed time from a reference point, rather than a calendar date| |
| TransactionAmt | transaction payment amound| input feature|
| isFraud | 1, fraud. 0, legitimate transaction| target we want to predict |
| ProductCD | product code | a input feature, |


# Transaction Risk Lab

## Project Objective

Build a fraud risk scoring pipeline for online transactions to help a risk
operations team prioritize manual reviews.

Under an assumed review capacity of 1% of transactions, the project aims to
maximize fraud recall while tracking precision. Performance will be compared
against simple rule-based baselines using time-based validation.

The project will cover data quality checks, historical feature engineering,
model evaluation, probability calibration, and reproducible workflows.

The review policy will aim to maximize fraud recall subject to a maximum
review rate of 1% and a false discovery rate (FDR) target. The FDR target
will be selected during policy development and evaluated on a held-out
time period.


## Initial Data Checks

- Loaded 590,540 transactions and 144,233 identity records.
- Confirmed unique TransactionID values in both tables.
- Applied a one-to-one LEFT JOIN without changing the transaction count.
- Identity coverage is approximately 24.4%.


### Product-level findings

- Product C has the highest training fraud rate: 10.71%.
- Product W has the largest fraud count: 4,610, despite a lower fraud rate of 2.03%.
- Review prioritization should consider both risk and transaction volume.
- These findings describe the training period and require validation on later periods.


### Identity coverage and product mix

- Identity coverage varies substantially by ProductCD in the training period.
- No Product W transactions have matching identity records.
- Within Product C, fraud rates are 6.07% without identity records and
  11.31% with identity records.
- Missing-identity groups for H, R, and S are too small for firm conclusions.
- These associations do not establish causation.


### Temporal findings

- Weekly fraud rates vary within the training period.
- Between relative weeks 3 and 4, fraud rates increase within products
  C, H, R, and S, while the rate for W decreases slightly.
- Both product mix and within-product fraud rates vary over time.
- These observations support time-based evaluation but do not establish
  the causes of the changes.
- The final partial week is excluded from full-week comparisons.


## M3 — Historical Feature Engineering

Implemented 10 historical features using DuckDB window functions.

- Proxy groups use card1, card4, and addr1.
- Only strictly earlier transactions contribute to history.
- Same-time transactions are excluded.
- Missing group keys produce missing historical features.
- Historical features do not use fraud labels.

Validation includes same-time exclusion, future-amount invariance,
input row-order invariance, and Parquet round-trip equality.

Run feature generation:

    python src/features.py

Run tests:

    python -m pytest tests/test_features.py -v

Output: data/processed/history_features.parquet