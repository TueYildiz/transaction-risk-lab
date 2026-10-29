## Frozen Evaluation

The selected LightGBM history model and raw-score review policy
were evaluated on 88,581 held-out transactions.

The model and threshold were fixed before test evaluation.

## Results

- Reviewed transactions: 535.
- Fraudulent transactions detected: 40 out of 3,083.
- Precision: 7.48%.
- Recall: 1.30%.
- False discovery rate: 92.52%.
- Review rate: 0.60%.
- Average precision: 0.0965.
- ROC AUC: 0.7636.

## Interpretation

The maximum review capacity of 1% was respected.
The development FDR target of 50% was not met on the final test.

The policy-period precision did not generalize to the later test period.
Ranking performance alone did not ensure a reliable review threshold.

Temporal distribution changes and threshold-selection uncertainty
are possible explanations, but their contributions have not been
established.

## Evaluation Boundary

The final test has now been observed.
Any subsequent model or threshold changes are exploratory with respect
to this period.

A new independent evaluation requires previously unseen data.