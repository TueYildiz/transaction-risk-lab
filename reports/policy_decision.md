# M5 — Calibration and Review Policy

## Model

LightGBM with historical features was refitted on train and validation:
383,851 transactions.

A sigmoid calibrator was fitted on the separate calibration period:
59,054 transactions.

## Calibration Evaluation

On the later policy period:

| Scores | Brier Score | Log Loss |
|---|---:|---:|
| Raw | 0.031362 | 0.131514 |
| Calibrated | 0.031422 | 0.131573 |

Calibration did not improve these metrics.
Raw scores were selected for policy development.

## Review Policy

- Offline batch selection.
- Maximum review rate: 1%.
- Development FDR target: 50%.
- Review transactions at or above the saved score threshold.
- If capacity is exceeded, prioritize higher scores.
- Break score ties by TransactionID ascending.

## Policy-Period Results

- Selected transactions: 71.
- Fraud hits: 37.
- Precision: 52.11%.
- Recall: 1.83%.
- Observed FDR: 47.89%.

## Limitations

The threshold was selected using policy-period labels.
These are development results, not independent test results.

The 50% FDR target is a scenario assumption, not a statistical guarantee.
It may not hold in future periods.

The policy is linked to its model bundle through a SHA-256 hash.