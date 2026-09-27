#!/usr/bin/env python3
"""Validate two independent author exports and analyze text-reference quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments" / "reference_information"
QUESTIONS = [
    "task_faithful",
    "criteria_consistent",
    "unsupported_details_absent",
    "observable_final_state",
    "sufficiently_specific",
    "overall_usable",
]
CATEGORIES = ["yes", "no", "uncertain"]
VALID = {"SUCCESS", "FAILURE"}


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kappa(a: list[str], b: list[str]) -> tuple[float, float, float]:
    assert len(a) == len(b) and a
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[c] / len(a)) * (cb[c] / len(b)) for c in CATEGORIES)
    value = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return observed, expected, value


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def mcnemar(b: int, c: int) -> tuple[str, float | None, float]:
    n = b + c
    if n == 0:
        return "none", None, 1.0
    if n < 25:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
        return "exact_binomial", None, min(1.0, 2 * tail)
    statistic = max(0, abs(b - c) - 1) ** 2 / n
    return "asymptotic_cc", statistic, math.erfc(math.sqrt(statistic / 2))


def f1(rows: list[tuple[str, str]]) -> float:
    tp = sum(t == p == "SUCCESS" for t, p in rows)
    fp = sum(t == "FAILURE" and p == "SUCCESS" for t, p in rows)
    fn = sum(t == "SUCCESS" and p == "FAILURE" for t, p in rows)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def validate_export(payload: dict, author: str, package_hash: str, blind_ids: set[str]) -> dict:
    assert payload["protocol_version"] == "text-reference-audit-v1"
    assert payload["package_hash"] == package_hash
    assert payload["author_code"] == author
    assert payload["complete"] is True and payload["independence_confirmed"] is True
    assert payload["record_count"] == payload["completed_count"] == len(blind_ids) == 294
    assert set(payload["order"]) == blind_ids and len(payload["order"]) == 294
    assert set(payload["labels"]) == blind_ids
    for blind_id, label in payload["labels"].items():
        for question in QUESTIONS:
            assert label.get(question) in CATEGORIES, (blind_id, question, label.get(question))
    return payload["labels"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-a", required=True, type=Path)
    parser.add_argument("--author-b", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blind = read_json(HERE / "blind_map.json")
    mapping = blind["records"]
    blind_ids = set(mapping)
    labels_a = validate_export(read_json(args.author_a), "A", blind["package_hash"], blind_ids)
    labels_b = validate_export(read_json(args.author_b), "B", blind["package_hash"], blind_ids)

    merged = []
    disagreement = []
    for blind_id in sorted(blind_ids):
        base = {"blind_id": blind_id, **mapping[blind_id]}
        row = base.copy()
        any_disagreement = False
        for question in QUESTIONS:
            row[f"author_a_{question}"] = labels_a[blind_id][question]
            row[f"author_b_{question}"] = labels_b[blind_id][question]
            agree = labels_a[blind_id][question] == labels_b[blind_id][question]
            row[f"agree_{question}"] = agree
            any_disagreement |= not agree
        row["author_a_notes"] = labels_a[blind_id].get("notes", "")
        row["author_b_notes"] = labels_b[blind_id].get("notes", "")
        strict = all(labels_a[blind_id][q] == labels_b[blind_id][q] == "yes" for q in QUESTIONS)
        row["strict_dual_approved"] = strict
        merged.append(row)
        if any_disagreement:
            disagreement.append(row)

    def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        if fieldnames is None:
            fieldnames = list(rows[0]) if rows else list(merged[0])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "merged_author_ratings.csv", merged)
    write_csv(args.output_dir / "disagreement_queue.csv", disagreement)

    strict_records = [r for r in merged if r["strict_dual_approved"]]
    (args.output_dir / "strict_dual_approved_blind_ids.txt").write_text("\n".join(r["blind_id"] for r in strict_records) + ("\n" if strict_records else ""))
    approved_by_instance = defaultdict(set)
    for row in strict_records:
        approved_by_instance[row["instance_id"]].add(row["variant"])
    strict_tasks = sorted(i for i, variants in approved_by_instance.items() if variants == {"with_criteria", "task_only"})
    (args.output_dir / "strict_dual_approved_task_pairs.txt").write_text("\n".join(strict_tasks) + ("\n" if strict_tasks else ""))

    agreement_rows = []
    for question in QUESTIONS:
        a = [labels_a[i][question] for i in sorted(blind_ids)]
        b = [labels_b[i][question] for i in sorted(blind_ids)]
        observed, expected, value = kappa(a, b)
        agreement_rows.append({
            "question": question,
            "raw_agreement": observed,
            "chance_expected_agreement": expected,
            "cohen_kappa_3cat": value,
            **{f"author_a_{c}": Counter(a)[c] for c in CATEGORIES},
            **{f"author_b_{c}": Counter(b)[c] for c in CATEGORIES},
        })
    write_csv(args.output_dir / "agreement_summary.csv", agreement_rows)

    approval_rows = []
    for author, labels in [("A", labels_a), ("B", labels_b)]:
        for variant in ["with_criteria", "task_only"]:
            for domain in ["all", "classifieds", "shopping", "reddit"]:
                ids = [i for i, m in mapping.items() if m["variant"] == variant and (domain == "all" or m["domain"] == domain)]
                yes = sum(labels[i]["overall_usable"] == "yes" for i in ids)
                no = sum(labels[i]["overall_usable"] == "no" for i in ids)
                uncertain = len(ids) - yes - no
                lo, hi = wilson(yes, len(ids))
                approval_rows.append({"author": author, "variant": variant, "domain": domain, "n": len(ids), "yes": yes, "no": no, "uncertain": uncertain, "approval_rate": yes / len(ids), "wilson95_low": lo, "wilson95_high": hi})
    write_csv(args.output_dir / "overall_usable_by_arm_domain.csv", approval_rows)

    # Recompute the fresh criteria-enriched versus task-only comparison on the
    # conservative task pairs approved on every question by both authors.
    all_verifications = [r for r in read_jsonl(EXP / "verification_results.jsonl") if r.get("status") == "ok"]
    successful = {}
    for row in all_verifications:
        successful[row["job_id"]] = row
    assert len(successful) == 1470
    sensitivity = []
    models = sorted({r["model_label"] for r in successful.values()})
    for model in models:
        arms = {
            variant: {r["instance_id"]: r for r in successful.values() if r["model_label"] == model and r["variant"] == variant}
            for variant in ["with_criteria", "task_only"]
        }
        ids = [i for i in strict_tasks if arms["with_criteria"][i].get("verdict") in VALID and arms["task_only"][i].get("verdict") in VALID]
        both = criteria_only = task_only = neither = 0
        c_pairs, t_pairs = [], []
        for i in ids:
            c, t = arms["with_criteria"][i], arms["task_only"][i]
            assert c["ground_truth"] == t["ground_truth"]
            cc = c["verdict"] == c["ground_truth"]
            tc = t["verdict"] == t["ground_truth"]
            if cc and tc: both += 1
            elif cc: criteria_only += 1
            elif tc: task_only += 1
            else: neither += 1
            c_pairs.append((c["ground_truth"], c["verdict"]))
            t_pairs.append((t["ground_truth"], t["verdict"]))
        test, statistic, p = mcnemar(criteria_only, task_only)
        n = len(ids)
        sensitivity.append({
            "model": model,
            "strict_approved_task_pairs": len(strict_tasks),
            "jointly_valid_n": n,
            "both_correct": both,
            "criteria_only_correct": criteria_only,
            "task_only_correct": task_only,
            "neither_correct": neither,
            "criteria_accuracy": (both + criteria_only) / n if n else "",
            "task_only_accuracy": (both + task_only) / n if n else "",
            "difference_pp": 100 * (criteria_only - task_only) / n if n else "",
            "criteria_success_f1": f1(c_pairs) if n else "",
            "task_only_success_f1": f1(t_pairs) if n else "",
            "test": test,
            "statistic": statistic if statistic is not None else "",
            "raw_p": p,
        })
    if sensitivity:
        ordered = sorted(range(len(sensitivity)), key=lambda i: sensitivity[i]["raw_p"])
        running = 0.0
        for rank, index in enumerate(ordered):
            row = sensitivity[index]
            row["bonferroni_p"] = min(1.0, 5 * row["raw_p"])
            running = max(running, min(1.0, (5 - rank) * row["raw_p"]))
            row["holm_p"] = running
        write_csv(args.output_dir / "strict_subset_verifier_sensitivity.csv", sensitivity)

    report = [
        "# Two-author text-reference audit results",
        "",
        f"Package hash: `{blind['package_hash']}`",
        "",
        f"Both authors completed {len(blind_ids)} blinded records independently.",
        f"Strict dual-approved references: **{len(strict_records)}/{len(blind_ids)}**.",
        f"Task pairs with both variants strictly dual-approved: **{len(strict_tasks)}/147**.",
        f"Records with at least one rating disagreement: **{len(disagreement)}/{len(blind_ids)}**.",
        "",
        "## Agreement",
        "",
        "| Question | Raw agreement | Cohen kappa |",
        "|---|---:|---:|",
    ]
    for row in agreement_rows:
        report.append(f"| {row['question']} | {row['raw_agreement']:.3f} | {row['cohen_kappa_3cat']:.3f} |")
    report += [
        "",
        "The strict subset is a conservative sensitivity set. It does not replace adjudication of disagreements.",
        "The audit applies to the fresh 294-reference experiment and does not reconstruct historical primary inputs.",
        "",
    ]
    (args.output_dir / "AUDIT-RESULTS.md").write_text("\n".join(report))
    print("Validated two complete independent exports.")
    print(f"Strict dual-approved references: {len(strict_records)}/294")
    print(f"Strict dual-approved task pairs: {len(strict_tasks)}/147")
    print(f"Disagreement records: {len(disagreement)}/294")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
