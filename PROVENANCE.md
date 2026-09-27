# Data and model provenance

This file separates historical evidence from work performed during the IEEE Access revision.

## Historical study

The original and revision-era verification runs used the authors' personal API accounts.
The committed result rows preserve task IDs, predictions, confidence values and aggregate
analysis inputs. They do not preserve provider-returned model snapshots or a complete ledger
of exact request times. The historical label `claude-sonnet-4` is therefore a study label;
Git history does not independently establish whether every historical request was served by
the same provider snapshot. This limitation should remain visible in any manuscript claim.

The tag `v1.1-ieee-access-r1` preserves the repository as it existed for the first revision.
It is useful for historical reconstruction, but it does not contain every later reviewer
audit or the exact original text references for four affected records. The code's historical
empty-reference fallback is documented; the four IDs cannot be recovered from available Git
history.

## New matched reference-information experiment

`experiments/reference_information/` is a new, predeclared 147-task paired experiment run on
2026-09-27. It generated criteria-enriched and task-only references with the same requested
GPT-4.1 Nano model and evaluated both arms with five verifiers. Raw records retain requested
and returned model identifiers, request IDs, UTC timestamps, exact text inputs, settings,
usage, raw responses and screenshot/reference hashes. The Claude endpoint requested for this
experiment was `claude-sonnet-4-6`; raw analysis rows retain the legacy configuration key
`claude-sonnet-4`, while manuscript display labels identify Claude Sonnet 4.6. This experiment tests the criteria-field contrast and does not reconstruct the
missing historical references or provide a new no-reference baseline.

## Visual-reference audit

`reviewer_artifacts/visual_reference_validation/` records the audit of the public Playwright
trace inventory. All 233 archives received structural checks and all 147 historical
candidates received visual screening. Sixty-seven candidates were approved, 79 remained
unresolved and one was rejected for identity. The two authors reviewed this work during the
revision, independently of their separate annotation study, but the individual screening
records identify the process accurately as an AI-assisted visual review rather than an
independent human annotation study. Four approved references use an earlier recorded frame;
the original bytes and provenance are retained.

The audit does not control viewport coverage, overlays, compression, provider image handling
or domain composition. No new C/D model outputs are claimed for the repaired 67-task set.

## Blinded text-reference audit

`reviewer_artifacts/text_reference_validation/` contains the completed two-author audit of
all 294 references in the fresh matched experiment. The two manuscript authors rated every
reference independently in different randomized orders. The interfaces hid the generation
arm, paired reference, verifier outputs, verifier correctness and agent outcome label. The
release preserves the protocol, package hash, deterministic interfaces, derived agreement
tables, disagreement queue, strict approved-ID lists and five-verifier sensitivity results.
Private author export files remain excluded.

The audit yields 213 strictly dual-approved individual references and 78 complete task
pairs. Agreement is high for task faithfulness and criteria consistency but low for
specificity and overall usability; these disagreements are retained rather than reconciled
after observing downstream results. The audit applies only to the fresh 294-reference
experiment and does not reconstruct or validate the unavailable historical primary inputs.

## Failure-taxonomy reliability study

`iaa_labeling/` and `results/iaa_kappa_3cat_human.txt` contain the separate two-author
reliability analysis for the original obvious, deceptive and partial labels. The authors
worked independently without discussion, but blinding to the original labels has not been
established. Cohen's kappa is 0.000, so category-level results are released only as a
descriptive record and are not used to support a deployment recommendation.

## Release policy

The internal historical result-file key `claude-sonnet-4` is a legacy label. The direct
Anthropic client was configured to request `claude-sonnet-4-6`; historical rows do not retain
provider-returned model identifiers, while the fresh matched experiment records
`claude-sonnet-4-6` as returned. Display labels in the manuscript use Claude Sonnet 4.6.

The final reviewer-evidence tag is `v1.3-ieee-access-final`. Its immutable archive is
[10.5281/zenodo.23002266](https://doi.org/10.5281/zenodo.23002266), within the stable concept DOI
[10.5281/zenodo.22990530](https://doi.org/10.5281/zenodo.22990530). The earlier version-specific
DOI 10.5281/zenodo.22990531 identifies the immutable v1.2 archive.
A SHA-256 manifest at the repository root covers the final evidence package. API credentials,
raw browser traces, working manuscripts, peer-review correspondence, temporary reports and
the 537 MB intermediate action-state export are excluded. The obsolete files were removed
from all public history. A private local rollback bundle was created before the authorized
history rewrite and is not part of the release.
