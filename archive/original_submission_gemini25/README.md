# Original-submission analysis outputs (Gemini 2.5 Flash)

Every file here was computed for the original submission, against
`gemini-2.5-flash` and the pre-correction Condition C/D prompt. They are
superseded by the tables in `results/`, but they are kept for two reasons.

1. **They are the only surviving record of the Gemini 2.5 Flash results.**
   Google retired `gemini-2.5-flash` (the API returns 404 for keys issued after
   the cutoff), and the per-instance JSONL files for that model were never
   committed. `S3_error_rates.csv` holds its full confusion matrices for all
   four conditions; `S1_paired_tests_supplement.csv` holds its paired-test
   counts. Those numbers cannot be regenerated.

2. They document the analysis policy used in the original submission, before
   the jointly-valid restriction and the exact-binomial McNemar rule.

Do not delete. The revised manuscript reports `gemini-3.6-flash`, the successor
Google's own error message directs callers to, re-run from scratch.
