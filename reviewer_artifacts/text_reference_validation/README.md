# Text-reference validation package

This package creates two offline, blinded review interfaces for the 294 fresh
references in `experiments/reference_information/generation_results.jsonl`.
No model calls or network access are required.

Generate or verify the package:

```bash
python3 reviewer_artifacts/text_reference_validation/prepare_audit.py
```

The command writes:

- `author_a_review.html`
- `author_b_review.html`
- `blind_map.json`
- `audit_manifest.json`

Each author opens only their assigned HTML file, completes all 294 records
without consultation, checks the independence confirmation, and selects
**Export final JSON**. Partial backup exports are allowed, but the analysis
rejects them as final inputs.

For a documentation pass, import the author's own JSON and use **Next flagged
without note** to visit each No or Uncertain record that still lacks a brief
explanation. Do not change a rating merely to increase agreement.

After both authors finish, analyze their exports:

```bash
python3 reviewer_artifacts/text_reference_validation/analyze_audit.py \
  --author-a /path/to/text_reference_audit_author_A.json \
  --author-b /path/to/text_reference_audit_author_B.json
```

The analysis writes a Markdown report, merged ratings, disagreement queue,
strict dual-approved records and task pairs, and a verifier sensitivity table.
Read `PROTOCOL.md` before rating.
