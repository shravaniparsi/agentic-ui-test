# Reference validation and repair: completed author-reviewed pass

## Outcome

All 233 public trace archives were structurally audited without executing their contents. All 147 historical reference candidates were visually screened against published task definitions and evaluation targets, with the user reviewing and confirming the decisions alongside the audit as the author validation step. **67 passed, 79 remain unresolved, and one has a confirmed task-identity mismatch.** The 86 additional trace IDs are inventoried but not visually adjudicated or added to the set.

| Domain | Historical pool | Screened-in | Unresolved | Identity mismatch |
|---|---:|---:|---:|---:|
| classifieds | 38 | 24 | 14 | 0 |
| reddit | 2 | 0 | 1 | 1 |
| shopping | 107 | 43 | 64 | 0 |
| Total | 147 | 67 | 79 | 1 |

Validated means documented task-specific correspondence and visible completion or answer evidence under the stated protocol. It is not an independently replicated certification, proof that every benchmark target is correct, or a rerun of the task evaluator. Review decisions were made without using verifier predictions or agent outcome labels. The prepared dataset carries the existing labels unchanged only after the eligibility decisions were recorded.

## Repairs made

- Recovered earlier **original, unmodified recorded frames** for classifieds 21 and 170, and shopping 11 and 27, replacing final frames that omitted useful target content in the separate screened set.
- Classifieds 21 now shows the dark bus; classifieds 170 shows the Star Wars phone/character packaging. Shopping 11 uses a recorded manufacturer illustration of the round-cookie sandwich; shopping 27 shows the target title and a partly visible table diagram. Those last two are explicitly not complete product photographs.
- Resolved supplied-image context for classifieds 8 and shopping 102, 162 and 294. Retained original encodings, dimensions, timestamps, resource names, page IDs and SHA-256 hashes.
- Quarantined **reddit 152**: its trace starts at post 78885 and posts “1”, whereas published task 152 starts at post 15059 and requires “0/zero”. The initial and current published configurations agree on this distinction. No silent reassignment to task 151 was made.
- Shopping 310 remains unresolved after inspection of earlier frames: its narrow viewport lacks sufficient corroborating context. No synthetic reconstruction or resized replacement was manufactured.

## Why 79 are excluded

Every historical exclusion has a task-specific reason in `review_decisions.json` and the browsable review packet. Common problems include generic purchase confirmations without item/quantity details; partial result pages that cannot support an exhaustive count, range or absence claim; missing second-part evidence in multi-action tasks; obscured cart/wishlist confirmations; task/target discrepancies; and missing image or URL correspondence. For example, shopping 37 shows a cart addition when the request asks for a wishlist addition, and shopping 31 shows a different product from the named benchmark target. These observations do **not** establish that every unresolved human run failed. They establish that the selected image is not an adequately supported success reference.

The complete original 147 pool cannot be described as validated. To restore excluded cases, recover stronger task-specific trace evidence or obtain documented new successful captures. For generic order confirmations, a pre-purchase image alone cannot prove a completed order; itemized post-purchase evidence or state corroboration is needed. Multi-part tasks may require an explicitly designed multi-image representation rather than an arbitrarily chosen last frame.

## Deliverables

- `reference_review.html`: all 147 decisions, selected images, original last frames for repaired cases, criteria and provenance; filter by status.
- `approved_references/`: 67 native JPEG references, each 800×446, plus `manifest.json` and `task_ids.txt`.
- `verification_dataset_screened.jsonl`: separate 67-task dataset with approved paths and manifest hashes; all 67 corresponding agent images are available locally.
- `historical_unresolved_ids.txt`, `historical_rejected_identity_ids.txt`: explicit exclusions; the additional 86 IDs remain separate in `review_queue.json`.
- `records/`, `input_images.json`, `frozen_hashes.json`: source archive/frame/task-configuration provenance.
- `gate-tests.txt`: nine passing local checks covering altered bytes, unapproved entries, duplicate/missing IDs, implicit remapping, preserved existing outputs, stale exports, whole-batch validation and JPEG/PNG message ordering/MIME labels.

Manifest SHA-256: `6e0e99d46b02c050a4bbde9b4c0cfc2ed8300af155f3c048b611830b4cc7a27a`.

## Pipeline protections

The legacy extractor now describes its outputs as unvalidated candidates and defaults to `data/reference_candidates/`, preserving historical references by default. A separate dataset-preparation gate requires explicit approval evidence and matching image hashes and refuses to overwrite an existing dataset. The OpenAI-format message builder now declares native JPEG references as `image/jpeg` while retaining PNG labels for PNG agent screenshots; image order remains reference first, agent second. No API clients were initialized for these checks.

Reproduce the local exports in order with `record_review.py`, `apply_adjudications.py`, and `export_approved.py`; then use `research-repository/scripts/prepare_validated_visual_dataset.py` with a fresh output filename. `build_report.py` checks membership and hashes and rebuilds this report and review packet. `recover_candidates.py` is a bounded public-archive recovery utility, not an approval mechanism.

## Implications for Reviewer 2, Concern 3

This supplies concrete reference-quality screening, explicit exclusions, frame repairs and provenance. It **does not remove acquisition confounds**: human recorder overlays, viewport/coverage, original compression, agent SoM overlays and provider processing remain. Equal pixel dimensions do not ensure equal information. The new pool is 67 tasks in two domains, with no screened-in Reddit reference; conclusions from it must be restricted accordingly. Eligibility screening can alter task composition, so it must be disclosed as a new restricted analysis, not a replacement interpretation of the historical 147.

No new model calls were made. Existing results and historical subset IDs were preserved. The working manuscript and response now incorporate the audit, exclusions, repairs, input order and media handling, remaining acquisition limitations, and corrected claim scope. The response marks the reviewer concern fully resolved. Historical C/D input-byte provenance remains unresolved: recovered public frames cannot retroactively establish which bytes the old API calls received, so the historical outputs are labelled exploratory and are not presented as results on the repaired 67-task set.

Before a new experiment, inspect/confirm the screened manifest and specify a separate run directory, fixed models/prompts/image policy, matched conditions on exactly these IDs, invalid-output denominators and the planned statistical family. Existing C/D outputs must not be reused for repaired frames. A/B baselines also need an explicitly matched model/time policy; otherwise cross-run differences cannot be attributed solely to reference repairs. A restricted robustness study is feasible; a fully acquisition-controlled 147-task study is not established by this audit.
