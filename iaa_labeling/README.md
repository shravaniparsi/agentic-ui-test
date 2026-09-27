# Failure-taxonomy inter-annotator study

This directory contains the two-author reliability study for the manuscript's
original obvious, deceptive and partial failure taxonomy.

`data/iaa_sample.jsonl` is a deterministic random sample of 100 ground-truth
FAILURE instances from `data/verification_dataset.jsonl`, selected with seed 42.
The Flask interface in `app.py` presents the task text, full-resolution final
screenshot and the three category definitions. The two manuscript authors used
the interface independently and did not discuss labels before completion.
They were not external annotators, and blinding to the original labels has not
been established.

The completed labels are:

- `data/iaa_labels_3cat_human1.csv`
- `data/iaa_labels_3cat_human2.csv`

Recompute the agreement analysis from the repository root:

```bash
python3 scripts/recompute_iaa.py \
  --a iaa_labeling/data/iaa_labels_3cat_human1.csv \
  --b iaa_labeling/data/iaa_labels_3cat_human2.csv
```

The reported Cohen's kappa is 0.000, with raw and chance-expected agreement of
0.330. The paired 10,000-resample bootstrap interval is approximately -0.131 to
+0.132. This result does not validate the taxonomy; category-level manuscript
results are retained only as descriptions of the original single-annotator
labels.
