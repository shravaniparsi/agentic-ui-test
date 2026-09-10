# Original cross-reference results

These are the cross-reference runs from the originally submitted paper, produced
against `data/verification_dataset_xref_claude.jsonl` and
`data/verification_dataset_nocriteria.jsonl` as they existed at submission time.

Those two input datasets were never committed and are lost, and the result
records store only `text_reference_generator`, not the reference text itself, so
these runs cannot be reproduced or extended. They are kept here as the record of
the original experiment.

The files in `results/cross_refs/` supersede them: they were regenerated with the
same generators and prompts, on datasets that are now committed, so every model
(including the replacement Gemini verifier) sees an identical reference set.
