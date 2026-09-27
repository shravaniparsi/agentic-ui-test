# Visual-reference validation package

This directory contains the disclosed audit of the public human Playwright traces used to
construct the historical visual-reference pool. Read `PROTOCOL.md` for the eligibility rule
and `REFERENCE-VALIDATION-REPORT.md` for results and limitations.

The release includes 67 approved native JPEG frames with a hash-bearing manifest, all audit
decisions and compact source records. It excludes raw traces, browser storage, generated HTML,
temporary downloads, duplicate task images and the full frame-review cache. Those materials
are not required to inspect the decisions or reproduce the released screened dataset.

The screening was AI-assisted and reviewed by the authors during revision. It is not an
independent human annotation study. The original 147-task C/D predictions are not results on
this repaired 67-task set; fresh model calls would be required for that claim.
