# Reviewer evidence

Generated offline by scripts/reviewer_evidence.py. No new model calls. Model labels follow filenames; provider identity still requires provenance evidence.

Thresholds 1 through 10 are swept on the observed data. Coverage and escalation use all assigned tasks, including invalid outputs. Invalid outputs always escalate. Risk is the error fraction among automatically accepted verdicts. Success precision conditions only on accepted SUCCESS predictions. Empty cells indicate an undefined rate, never zero risk. Wilson intervals are pointwise 95% binomial intervals, not simultaneous bounds across thresholds. No held-out validation or deployment guarantee is claimed.

Calibration bins are fixed at 1-3, 4-6, 7-10 and contain only SUCCESS predictions. Uncertainty uses 2,000 unstratified task bootstrap resamples, implemented with multinomial confusion-count resampling, seed 20260927. Bootstrap intervals are marginal percentile 95% intervals over valid outputs; they do not account for model rerun variation, domain clustering, or missing-output bias.
