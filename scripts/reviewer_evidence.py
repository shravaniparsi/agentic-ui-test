"""Offline reviewer evidence from preserved outputs; no model API calls.

Reports all-task coverage (invalid responses escalate), success precision with
Wilson 95% intervals, success-conditional calibration, and bootstrap uncertainty.
Threshold sweeps are descriptive, in-sample analyses, not deployment guarantees.
Model names are file labels, not independently verified provider identities.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_evidence"
MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "claude-sonnet-4", "gemini-3.6-flash"]
VALID = {"SUCCESS", "FAILURE"}


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["instance_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs: {path}")
    return rows


def wilson(k, n):
    if not n:
        return "", ""
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, center-radius), min(1, center+radius)


def ratio(k, n):
    return k/n if n else ""


def dump(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    subset = set((ROOT / "data/visual_subset_ids.txt").read_text().split())
    thresholds, calibration, uncertainty, invalids = [], [], [], []
    rng = np.random.default_rng(20260927)
    for model in MODELS:
        for condition in "ABCD":
            rows = load(ROOT / f"{model}_{condition}.jsonl")
            expected = 147 if condition in "CD" else 909
            assert len(rows) == expected
            if condition in "CD":
                assert {r["instance_id"] for r in rows} == subset
            valid = [r for r in rows if r["verdict"] in VALID]
            for r in valid:
                assert r["ground_truth"] in VALID
                assert isinstance(r.get("confidence"), (int, float)) and 1 <= r["confidence"] <= 10
            correct = sum(r["verdict"] == r["ground_truth"] for r in valid)
            context = {"model_file_label": model, "condition": condition}
            invalids.append(dict(context, n_total=len(rows), n_valid=len(valid),
                api_errors=sum(r["verdict"] == "API_ERROR" for r in rows),
                parse_errors=sum(r["verdict"] == "PARSE_ERROR" for r in rows),
                other_invalid=sum(r["verdict"] not in VALID | {"API_ERROR", "PARSE_ERROR"} for r in rows),
                valid_accuracy=correct/len(valid), all_task_accuracy=correct/len(rows),
                invalid_escalation_rate=1-len(valid)/len(rows)))
            for threshold in range(1, 11):
                auto = [r for r in valid if r["confidence"] >= threshold]
                successes = [r for r in auto if r["verdict"] == "SUCCESS"]
                auto_correct = sum(r["verdict"] == r["ground_truth"] for r in auto)
                tp = sum(r["ground_truth"] == "SUCCESS" for r in successes)
                lo, hi = wilson(tp, len(successes))
                thresholds.append(dict(context, threshold=threshold, n_total=len(rows),
                    n_valid=len(valid), auto_n=len(auto), auto_correct=auto_correct,
                    coverage=len(auto)/len(rows), escalation_rate=1-len(auto)/len(rows),
                    auto_accuracy=ratio(auto_correct,len(auto)),
                    auto_risk=ratio(len(auto)-auto_correct,len(auto)),
                    success_n=len(successes), success_true_positive=tp,
                    success_precision=ratio(tp,len(successes)),
                    success_precision_wilson_low=lo, success_precision_wilson_high=hi,
                    false_acceptance_per_task=(len(successes)-tp)/len(rows)))
            successes = [r for r in valid if r["verdict"] == "SUCCESS"]
            for low, high in [(1,3),(4,6),(7,10)]:
                bucket = [r for r in successes if low <= r["confidence"] <= high]
                k = sum(r["ground_truth"] == "SUCCESS" for r in bucket)
                lo, hi = wilson(k, len(bucket))
                calibration.append(dict(context, confidence_bin=f"{low}-{high}",
                    success_n=len(bucket), mean_confidence=ratio(sum(r["confidence"]/10 for r in bucket),len(bucket)),
                    success_precision=ratio(k,len(bucket)), precision_wilson_low=lo, precision_wilson_high=hi))
            # Multinomial resampling is equivalent to resampling binary outcome
            # records for these confusion-matrix statistics. Valid responses only.
            counts = np.array([
                sum(r["ground_truth"] == g and r["verdict"] == v for r in valid)
                for g,v in [("SUCCESS","SUCCESS"),("FAILURE","SUCCESS"),
                            ("FAILURE","FAILURE"),("SUCCESS","FAILURE")]])
            draws = rng.multinomial(len(valid), counts/len(valid), size=2000)
            def statistics(a):
                tp, fp, tn, fn = a.T
                def div(num, den):
                    return np.divide(num, den, out=np.zeros_like(num,dtype=float), where=den!=0)
                return {"accuracy":(tp+tn)/len(valid), "success_f1":div(2*tp,2*tp+fp+fn),
                    "balanced_accuracy":(div(tp,tp+fn)+div(tn,tn+fp))/2}
            point = statistics(counts.reshape(1,4))
            for name, values in statistics(draws).items():
                lo, hi = np.quantile(values,[.025,.975])
                uncertainty.append(dict(context, n_valid=len(valid), metric=name,
                    estimate=float(point[name][0]), bootstrap_low=float(lo), bootstrap_high=float(hi)))
    dump("thresholds_all_tasks.csv", thresholds)
    dump("success_conditional_calibration.csv", calibration)
    dump("uncertainty.csv", uncertainty)
    dump("invalid_response_policies.csv", invalids)
    (OUT / "README.md").write_text(
        "# Reviewer evidence\n\nGenerated offline by scripts/reviewer_evidence.py. "
        "No new model calls. Model labels follow filenames; provider identity still requires provenance evidence.\n\n"
        "Thresholds 1 through 10 are swept on the observed data. Coverage and escalation use all assigned tasks, "
        "including invalid outputs. Invalid outputs always escalate. Risk is the error fraction among automatically "
        "accepted verdicts. Success precision conditions only on accepted SUCCESS predictions. Empty cells indicate "
        "an undefined rate, never zero risk. Wilson intervals are pointwise 95% binomial intervals, not simultaneous "
        "bounds across thresholds. No held-out validation or deployment guarantee is claimed.\n\n"
        "Calibration bins are fixed at 1-3, 4-6, 7-10 and contain only SUCCESS predictions. "
        "Uncertainty uses 2,000 unstratified task bootstrap resamples, implemented with multinomial confusion-count "
        "resampling, seed 20260927. Bootstrap intervals are marginal percentile 95% intervals over valid outputs; "
        "they do not account for model rerun variation, domain clustering, or missing-output bias.\n")
    print(f"Wrote {len(thresholds)} threshold rows, {len(calibration)} calibration rows, "
          f"{len(uncertainty)} uncertainty rows and {len(invalids)} invalid-policy rows to {OUT}")


if __name__ == "__main__":
    main()
