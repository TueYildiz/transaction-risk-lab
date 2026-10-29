# M4 — Model Selection

## Evaluation Protocol

Models were trained on the training split and evaluated on the later
validation split. Calibration, policy, and test periods were not used.

Review capacity was fixed at 590 transactions out of 59,054 validation
transactions. Selection used offline batch ranking.

## Validation Results

| Model | Fraud Hits | Precision | Recall | Average Precision | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression base | 75 | 12.71% | 3.53% | 0.0960 | 0.7406 |
| Logistic Regression history | 61 | 10.34% | 2.87% | 0.0898 | 0.7201 |
| LightGBM base | 142 | 24.07% | 6.68% | 0.1337 | 0.7709 |
| LightGBM history | 182 | 30.85% | 8.56% | 0.1519 | 0.7824 |

The amount-based baseline detected 22 fraudulent transactions.
Random selection averaged 21.24 fraud hits across 100 seeds.

## Backtesting

Three expanding-window folds were constructed within the training period.

Mean recall:
- Logistic Regression base: 3.09%.
- Logistic Regression history: 2.63%.
- LightGBM base: 5.80%.
- LightGBM history: 6.01%.

Historical features improved LightGBM recall in two of three folds.
Their contribution was modest on average and varied across periods.

## Decision

Advance LightGBM with historical features to calibration and policy
development using the current hyperparameters.

## Limitations

- Validation results are development results, not final test results.
- Calibration has not been assessed.
- No FDR constraint has been enforced yet.
- Proxy groups do not represent verified customer identities.
- Review selection currently assumes offline batch ranking.
