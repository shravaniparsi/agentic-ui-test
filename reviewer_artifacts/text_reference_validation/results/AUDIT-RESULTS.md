# Two-author text-reference audit results

Package hash: `252ae159397102d937bd9c4d785eeaac230967af16b64443d826c2e5bc068856`

Both authors completed 294 blinded records independently.
Strict dual-approved references: **213/294**.
Task pairs with both variants strictly dual-approved: **78/147**.
Records with at least one rating disagreement: **80/294**.

## Agreement

| Question | Raw agreement | Cohen kappa |
|---|---:|---:|
| task_faithful | 0.990 | 0.745 |
| criteria_consistent | 0.997 | 0.665 |
| unsupported_details_absent | 0.952 | 0.279 |
| observable_final_state | 1.000 | not estimable |
| sufficiently_specific | 0.765 | -0.017 |
| overall_usable | 0.762 | 0.141 |

## Overall usability by arm

| Author | Criteria-enriched approved | Task-only approved |
|---|---:|---:|
| A | 140/147 | 140/147 |
| B | 134/147 | 87/147 |

## Strict-subset verifier sensitivity

The rule retains a task only when both authors answered Yes to all six questions for both reference variants.

| Verifier | Jointly valid n | Accuracy difference (pp) | Raw p | Bonferroni p | Holm p |
|---|---:|---:|---:|---:|---:|
| claude-sonnet-4 | 68 | +2.94 | 0.6250 | 1.0000 | 1.0000 |
| gemini-3.6-flash | 78 | -1.28 | 1.0000 | 1.0000 | 1.0000 |
| gpt-4.1 | 78 | -1.28 | 1.0000 | 1.0000 | 1.0000 |
| gpt-4.1-mini | 78 | +2.56 | 0.7744 | 1.0000 | 1.0000 |
| gpt-4.1-nano | 76 | -3.95 | 0.5488 | 1.0000 | 1.0000 |

Specificity and overall-usability agreement is low and is retained transparently rather than resolved through post-hoc relabelling.
The strict subset is a conservative sensitivity set. It does not replace adjudication of disagreements or establish equivalence.
The audit applies to the fresh 294-reference experiment and does not reconstruct historical primary inputs.
