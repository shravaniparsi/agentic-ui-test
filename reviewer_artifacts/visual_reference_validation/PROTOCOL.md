# Reference validation and repair protocol

Scope: audit all 233 public trace filenames; prioritize the historical 147-task pool for repair. Preserve all original results, subset IDs and recovered frame bytes. No model calls are part of this audit.

## Eligibility rule

A reference is eligible only if all of the following are documented:

1. **Task identity:** the task-specific entities, starting state where informative, and requested operation agree with a pinned published task definition. A generic starting homepage or matching numeric filename is insufficient. A demonstrated filename mismatch is quarantined; any proposed remapping requires affirmative evidence for the target task.
2. **Completion evidence:** the selected recorded frame visibly supports the requested successful outcome or the task-specific evidence needed for an answer. A last frame, evaluator keyword, recorded click, or the authors' aggregate human success rate is insufficient by itself. Nonvisual outcomes and missing task-image context remain unresolved unless corroborated.
3. **Frame provenance:** source archive SHA-256, source resource name, page ID, timestamp, byte hash and exact selection rationale are retained. The default global last frame is a candidate only; multi-page traces require page selection review.
4. **Review evidence:** a reviewer, identity verdict, completion verdict and concrete rationale are recorded. Automated structural checks are advisory and cannot auto-approve a reference.

Statuses: `approved`, `rejected_identity`, `rejected_completion`, `unresolved`. `approved` requires `identity_verified` and `completion_supported`. Keep uncertain examples out of the released eligible manifest; do not turn uncertainty into a negative label.

## Blinding and selection

Review trace contents against task specifications and evaluation targets without using verifier predictions, confidence or accuracy. Task outcomes of the agent are not used to select reference eligibility. Existing manually inspected examples were originally selected by task ID, not model performance. AI review is not an independent human annotation study and must not be represented as one.

## Acquisition limitations

Validation of identity/completion does not control viewport coverage, overlays, compression or provider processing. Recovered frames remain at their original sizes and encoding. No cropping, inpainting, resizing or DOM reconstruction is used to manufacture success evidence. Reconstructed states must never silently substitute for recorded final frames.

## Export and future analysis

Export approved frames into a new directory, never overwrite historical references. Export the new task-ID list separately from the original 147 IDs. Require explicit hash-verified approval records. Record original-pool exclusions and any proposed additions separately. A changed reference set requires fresh visual verification and recomputation of affected comparisons; retained historical C/D predictions cannot be presented as results for repaired images.

Raw public traces can contain browser storage. Keep them out of supplementary releases; publish only sanitized metadata and permitted frame artifacts after review.

## Implemented screening scope

All 147 historical global-last-frame candidates received a visual screening pass against their published task and criteria. Exact benchmark URL, SKU, named product, requested attributes or specific answer evidence can corroborate identity; generic success text cannot. When task-image semantics are corroborated through a benchmark target rather than a direct image comparison, the record states that limitation. This is a benchmark-informed AI screen, not independent human certification of all benchmark judgments.

Four references use earlier, byte-identical recorded target-page frames: classifieds 21 and 170, shopping 11 and 27. These replace an uninformative global last frame only in the separate screened set. Shopping 11 uses a manufacturer illustration; shopping 27 shows a partly visible diagram and target title. They do not constitute standardized recaptures. Shopping 310 remains unresolved because the narrow recorded viewport does not provide adequate corroborating context in the inspected frames.

The 86 additional trace IDs received structural inventory checks only and are not added to the screened set. Unresolved examples are excluded from the set, not relabeled as failed human tasks. Missing visual evidence, acquisition differences and benchmark-version ambiguity cannot be repaired by manufacturing or reconstructing a successful screenshot.
