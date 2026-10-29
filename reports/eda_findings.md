# Training EDA Findings

## Scope

All exploratory findings use the training split only:
324,797 transactions, including 11,001 fraudulent transactions.

## Class Imbalance

The training fraud rate is 3.39%.
Predicting every transaction as legitimate would achieve approximately
96.61% accuracy while detecting no fraud.

## Transaction Amount

Fraudulent transactions have a slightly higher median amount:
77.00 versus 68.95 for legitimate transactions.

However, large amounts also occur among legitimate transactions.
Amount alone is not sufficient to identify fraud.

## Product Differences

Product C has the highest fraud rate: 10.71%.
Product W has the largest fraud count: 4,610, despite a 2.03% fraud rate.

## Identity Coverage

Fraud rates are 6.59% with identity records and 2.10% without them.
Identity coverage varies by product, so the aggregate comparison
is affected by product mix.

These associations do not establish causation.

## Temporal Variation

Fraud rates range from 2.07% to 4.42% across full weekly windows.
Both product mix and within-product fraud rates vary over time.

The final partial week is excluded from the weekly chart.

## Implications

- Use time-based evaluation.
- Evaluate precision, recall, and FDR under review capacity constraints.
- Measure whether product and identity features improve later-period results.
- Avoid treating missing identity information as proof of fraud.