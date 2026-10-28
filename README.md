## İlk veri sözlüğü

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

