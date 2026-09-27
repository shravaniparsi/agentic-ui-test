#!/usr/bin/env python3
"""Domain-stratified paired analysis for the primary Condition A vs B contrast.

This is an offline robustness analysis over the retained task-level outputs. It
does not make model calls. Each model/domain comparison uses only task IDs with
a valid SUCCESS or FAILURE verdict in both conditions. McNemar testing follows
the manuscript policy: exact binomial when b+c < 25, otherwise asymptotic
chi-square with continuity correction. Holm and Bonferroni adjustments treat
the 5 models x 3 domains as one 15-test family.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "reviewer_evidence" / "domain_stratified_ab.csv"
MODELS = [
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "claude-sonnet-4",
    "gemini-3.6-flash",
]
DOMAINS = ["classifieds", "shopping", "reddit"]
VALID = {"SUCCESS", "FAILURE"}


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["instance_id"]] = row
    return rows


def metrics(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    tp = sum(truth == "SUCCESS" and verdict == "SUCCESS" for truth, verdict in pairs)
    fp = sum(truth == "FAILURE" and verdict == "SUCCESS" for truth, verdict in pairs)
    tn = sum(truth == "FAILURE" and verdict == "FAILURE" for truth, verdict in pairs)
    fn = sum(truth == "SUCCESS" and verdict == "FAILURE" for truth, verdict in pairs)
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n
    f1_denominator = 2 * tp + fp + fn
    success_f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    return accuracy, success_f1


def mcnemar(b: int, c: int) -> tuple[str, float | None, float]:
    discordant = b + c
    if discordant == 0:
        return "none", None, 1.0
    if discordant < 25:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
        return "exact_binomial", None, min(1.0, 2 * tail)
    statistic = (abs(b - c) - 1) ** 2 / discordant
    return "asymptotic_cc", statistic, math.erfc(math.sqrt(statistic / 2))


def adjust(rows: list[dict]) -> None:
    family_size = len(rows)
    for row in rows:
        row["bonferroni_p"] = min(1.0, family_size * row["raw_p"])

    running_max = 0.0
    for rank, index in enumerate(sorted(range(family_size), key=lambda i: rows[i]["raw_p"])):
        adjusted = min(1.0, (family_size - rank) * rows[index]["raw_p"])
        running_max = max(running_max, adjusted)
        rows[index]["holm_p"] = running_max

    for row in rows:
        row["holm_significant_0_05"] = row["holm_p"] < 0.05
        row["bonferroni_significant_0_05"] = row["bonferroni_p"] < 0.05


def main() -> None:
    rows: list[dict] = []
    for model in MODELS:
        condition_a = load(ROOT / f"{model}_A.jsonl")
        condition_b = load(ROOT / f"{model}_B.jsonl")
        for domain in DOMAINS:
            ids = sorted(
                instance_id
                for instance_id in condition_a.keys() & condition_b.keys()
                if condition_a[instance_id]["domain"] == domain
                and condition_b[instance_id]["domain"] == domain
                and condition_a[instance_id]["verdict"] in VALID
                and condition_b[instance_id]["verdict"] in VALID
            )
            a_pairs = [(condition_a[i]["ground_truth"], condition_a[i]["verdict"]) for i in ids]
            b_pairs = [(condition_b[i]["ground_truth"], condition_b[i]["verdict"]) for i in ids]
            a_correct = [truth == verdict for truth, verdict in a_pairs]
            b_correct = [truth == verdict for truth, verdict in b_pairs]
            lost = sum(a and not b for a, b in zip(a_correct, b_correct))
            gained = sum(not a and b for a, b in zip(a_correct, b_correct))
            test, statistic, raw_p = mcnemar(lost, gained)
            accuracy_a, f1_a = metrics(a_pairs)
            accuracy_b, f1_b = metrics(b_pairs)
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "jointly_valid_n": len(ids),
                    "accuracy_a": accuracy_a,
                    "accuracy_b": accuracy_b,
                    "accuracy_delta_pp": 100 * (accuracy_b - accuracy_a),
                    "success_f1_a": f1_a,
                    "success_f1_b": f1_b,
                    "success_f1_delta": f1_b - f1_a,
                    "b_a_correct_b_wrong": lost,
                    "c_a_wrong_b_correct": gained,
                    "test": test,
                    "statistic": statistic,
                    "raw_p": raw_p,
                }
            )

    adjust(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    positive_accuracy = sum(row["accuracy_delta_pp"] > 0 for row in rows)
    positive_f1 = sum(row["success_f1_delta"] > 0 for row in rows)
    holm = sum(row["holm_significant_0_05"] for row in rows)
    bonferroni = sum(row["bonferroni_significant_0_05"] for row in rows)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(rows)} comparisons)")
    print(f"Positive A-to-B accuracy changes: {positive_accuracy}/{len(rows)}")
    print(f"Positive A-to-B success-F1 changes: {positive_f1}/{len(rows)}")
    print(f"Significant after Holm correction: {holm}/{len(rows)}")
    print(f"Significant after Bonferroni correction: {bonferroni}/{len(rows)}")


if __name__ == "__main__":
    main()
