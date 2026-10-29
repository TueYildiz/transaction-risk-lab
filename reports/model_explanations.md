# Model Explanations

TreeSHAP contributions were computed for a reproducible sample of
1,000 policy-period transactions using the frozen LightGBM model.

Contributions were verified to sum to the raw model margin,
including the expected-value baseline.

TransactionAmt had the largest mean absolute contribution.
ProductCD_C and time_since_previous followed.

Historical features contributed to model scores, but their incremental
predictive value must be assessed through feature comparisons.

hist_valid_amount_count had zero contribution in this sample.
It duplicates hist_count under the current non-missing-amount assumption.

Contributions explain raw log-odds, not calibrated probabilities.
They do not establish causation.