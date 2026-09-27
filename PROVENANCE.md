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
experiment was `claude-sonnet-4-6`; the public study label remains `claude-sonnet-4` for table
continuity. This experiment tests the criteria-field contrast and does not reconstruct the
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

## Release policy

The reviewer-evidence tag is `v1.2-ieee-access-reviewer-evidence`. A SHA-256 manifest at the
repository root covers the evidence package. API credentials, raw browser traces, working
manuscripts, peer-review correspondence, temporary reports and the 537 MB intermediate action
state export are excluded. The obsolete files were removed from all public history. A private local rollback bundle was
created before the authorized history rewrite and is not part of the release.
