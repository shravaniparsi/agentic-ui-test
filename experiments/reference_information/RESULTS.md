# Matched reference experiment results

All 294 reference-generation jobs and all 1,470 verification jobs are complete across 147 canonical tasks and five verifiers. Results are from a new experiment; they do not reconstruct the original missing reference inputs.

## Direct paired accuracy comparison

| Verifier | Joint valid n | With criteria | Task-only | Difference pp | Paired 95% CI pp | Raw p | Bonferroni p | Holm p |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| claude-sonnet-4 | 133 | 82.0% | 78.2% | +3.8 | [-0.8, 8.3] | 0.1797 | 0.8984 | 0.8984 |
| gemini-3.6-flash | 147 | 75.5% | 75.5% | +0.0 | [-5.4, 5.4] | 1.0000 | 1.0000 | 1.0000 |
| gpt-4.1 | 147 | 74.1% | 74.8% | -0.7 | [-6.8, 5.4] | 1.0000 | 1.0000 | 1.0000 |
| gpt-4.1-mini | 147 | 68.7% | 66.7% | +2.0 | [-5.4, 9.5] | 0.7103 | 1.0000 | 1.0000 |
| gpt-4.1-nano | 143 | 70.6% | 69.9% | +0.7 | [-5.6, 7.0] | 1.0000 | 1.0000 | 1.0000 |

Comparisons significant after Bonferroni: none. Comparisons significant before correction: none.

Differences are criteria-enriched minus task-only accuracy on the same jointly valid tasks per verifier. Intervals are paired bootstrap percentile intervals from 10,000 resamples (seed 20260927); they are not simultaneous intervals. McNemar uses the two-sided exact binomial test for fewer than 25 discordant pairs and continuity-corrected asymptotic testing otherwise. Bonferroni and Holm cover all five planned comparisons. The contingency cells and matched IDs accompany the CSV.

## Coverage and operational accuracy

| Verifier | Criteria valid /147 | Task-only valid /147 | Criteria all-task accuracy | Task-only all-task accuracy | Criteria success F1 | Task-only success F1 |
|---|---:|---:|---:|---:|---:|---:|
| claude-sonnet-4 | 138 | 138 | 76.9% | 72.1% | 0.576 | 0.484 |
| gemini-3.6-flash | 147 | 147 | 75.5% | 75.5% | 0.500 | 0.500 |
| gpt-4.1 | 147 | 147 | 74.1% | 74.8% | 0.500 | 0.448 |
| gpt-4.1-mini | 147 | 147 | 68.7% | 66.7% | 0.477 | 0.449 |
| gpt-4.1-nano | 146 | 144 | 70.7% | 68.0% | 0.344 | 0.241 |

All-task accuracy uses 147 as denominator and counts invalid verdicts as unsuccessful verification. Success F1 uses valid responses within each arm and can therefore have differing samples. Invalid verdicts were retained and never retried to obtain a preferred answer. Explicit API failures were logged separately and retried as described in PROTOCOL.md.

## Interpretation

This is the direct same-generator control requested by Reviewer 2 Concern 2. The only generation-input difference is inclusion of benchmark evaluation criteria; the wrapper, generator configuration, task, screenshot, verifier prompt and within-provider settings are held fixed. Exact prompts and returned generator/verifier model IDs are retained.

Nonsignificance does not establish equivalence or show that task-only references are ineffective. The older causal statement that evaluation criteria are the primary cause of the text-reference benefit is not justified by this completed control. No fresh no-reference arm was included, so the control does not estimate a new A-to-B benefit. Keep historical and new results separate, and describe criteria-enriched references as supplying target information that may be absent from the user instruction. Deployment recommendations depend on the availability of trustworthy criteria.

## Integrity and provenance

All output-to-input integrity checks passed. Records preserve raw responses, exact textual requests, request and returned model identifiers, timestamps, token usage, reference hashes and screenshot hashes. All 147 agent final images were recovered from the user-supplied public trajectory archive using the historical final-embedded-image rule. Of these, 146 are 1280 by 2048 and vwa_shopping_310 is 844 by 393; the same unchanged image is used in both arms.

Verification attempts: 1484, including 14 failed API attempts. Successful verification responses: 1,470. Generation calls: 294. Conservative cumulative accounting estimate: $7.25; this includes a $0.10 reserve for each failed call and is not an actual provider invoice.

The original four missing-reference IDs remain unrecovered. Historical code inserts an empty reference instead of dropping such tasks. A separate exhaustive exclusion sensitivity analysis shows that removing any up to four records cannot eliminate the four retained historical GPT/Claude A-to-B significances after 30-test Bonferroni correction. That exclusion bound does not recover the original inputs or predict outcomes under replacement references.

Files: paired_results.csv, analysis.json, integrity_report.json, generation_requests.jsonl, generation_results.jsonl, verification_schedule.jsonl, verification_results.jsonl, tasks.jsonl, screenshot_recovery.json, field_examples.json, PROTOCOL.md. Credentials are stored outside this experiment directory and must never be included in a research release.
