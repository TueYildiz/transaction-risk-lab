# Historical Feature Dictionary

## Group Definition

A proxy group is defined by card1, card4, and addr1.
It is not a verified customer or card identifier.

If any group field is missing, the group key and historical features
remain missing. Transactions are preserved.

## Time Rule

Only transactions with strictly earlier TransactionDT values contribute
to history. Same-time transactions are excluded.

Historical features do not use isFraud.

## Features

| Feature | Definition |
|---|---|
| hist_count | Number of earlier transactions in the group. |
| hist_valid_amount_count | Number of earlier transactions with non-missing amounts. |
| hist_mean_amount | Mean amount of earlier transactions. |
| hist_std_amount | Population standard deviation of earlier amounts. |
| hist_max_amount | Maximum earlier amount. |
| hist_total_amount | Sum of earlier amounts. |
| time_since_previous | Seconds since the latest strictly earlier transaction. |
| amount_to_hist_mean | Current amount divided by historical mean; missing if the mean is zero or unavailable. |
| amount_minus_hist_mean | Current amount minus historical mean. |
| is_new_group | 1 when a valid group has no earlier transactions; otherwise 0. Missing for unknown groups. |

## Checks

- Known-example count and mean verification.
- Missing-key handling.
- Positive time gaps for transactions with history.
- Same-time exclusion using the feature-generation function.
- Future-amount invariance on a small fixture.
- Input row-order invariance on a small fixture.
- Parquet round-trip equality.
