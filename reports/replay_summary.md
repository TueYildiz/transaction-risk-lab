## Verification

- Checked all 501,959 development transactions; test excluded.
- Verified SQL/Python parity for all 10 historical features.
- Verified checkpoint serialization and resume.
- Confirmed identical duplicate retries do not update state.
- Rejected conflicting duplicates and new late-arriving transactions.
- All 12 automated tests passed.

## Runtime

The validation workflow took 97.88 seconds on the local machine.
This includes data loading, replay, comparison, and reporting.
It is not a production latency benchmark.

## Monitoring

Training and policy feature distributions were compared using
missing rates, medians, and descriptive PSI.

Historical counts increase as group history accumulates.
Distribution changes alone do not establish model degradation.

## Limitations

- Replay assumes finite, non-missing amounts and timestamps.
- Same-time transactions must remain in the same batch.
- State and duplicate signatures are stored in memory.
- Replay parity validates feature generation, not end-to-end
  model scoring or review-policy execution.