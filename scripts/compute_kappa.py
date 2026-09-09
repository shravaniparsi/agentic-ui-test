#!/usr/bin/env python3
"""
Compute Cohen's kappa from two annotation files.
Used for inter-annotator agreement (IAA) study.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_labels(path):
    """Load labels from CSV or JSONL."""
    labels = {}
    path = Path(path)

    if path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("label"):
                    labels[row["instance_id"]] = row["label"]
    else:
        with open(path) as f:
            for line in f:
                r = json.loads(line.strip())
                if r.get("label"):
                    labels[r["instance_id"]] = r["label"]
    return labels


def cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa for two annotators."""
    common = sorted(set(labels_a.keys()) & set(labels_b.keys()))
    if not common:
        return 0.0, {"error": "no common instances"}

    la = [labels_a[i] for i in common]
    lb = [labels_b[i] for i in common]
    n = len(common)

    classes = sorted(set(la) | set(lb))

    # Confusion matrix
    confusion = {c: Counter() for c in classes}
    for a, b in zip(la, lb):
        confusion[a][b] += 1

    # Observed agreement
    po = sum(confusion[c][c] for c in classes) / n

    # Expected agreement
    cnt_a = Counter(la)
    cnt_b = Counter(lb)
    pe = sum((cnt_a[c] / n) * (cnt_b[c] / n) for c in classes)

    # Kappa
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    # Per-class agreement
    by_class = {c: confusion[c][c] / max(1, cnt_a[c]) for c in classes}

    # Interpretation (Landis & Koch 1977)
    if kappa >= 0.81:
        interpretation = "almost perfect"
    elif kappa >= 0.61:
        interpretation = "substantial"
    elif kappa >= 0.41:
        interpretation = "moderate"
    elif kappa >= 0.21:
        interpretation = "fair"
    elif kappa >= 0.0:
        interpretation = "slight"
    else:
        interpretation = "worse than chance"

    return kappa, {
        "n": n,
        "po": round(po, 4),
        "pe": round(pe, 4),
        "by_class": {c: round(v, 3) for c, v in by_class.items()},
        "confusion": {ca: dict(confusion[ca]) for ca in classes},
        "interpretation": interpretation,
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa")
    parser.add_argument("--a", required=True, help="Annotator A labels (CSV or JSONL)")
    parser.add_argument("--b", required=True, help="Annotator B labels (CSV or JSONL)")
    args = parser.parse_args()

    labels_a = load_labels(args.a)
    labels_b = load_labels(args.b)

    print(f"Annotator A: {len(labels_a)} labels")
    print(f"Annotator B: {len(labels_b)} labels")

    kappa, info = cohens_kappa(labels_a, labels_b)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print(f"\n{'='*50}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"Interpretation: {info['interpretation']}")
    print(f"{'='*50}")
    print(f"  n (common instances): {info['n']}")
    print(f"  p_observed: {info['po']}")
    print(f"  p_expected: {info['pe']}")
    print(f"\nPer-class agreement:")
    for cls, agr in info["by_class"].items():
        print(f"  {cls}: {agr:.1%}")
    print(f"\nConfusion matrix (rows=A, cols=B):")
    print(f"  {'':12s}", end="")
    for cls in info["classes"]:
        print(f"{cls:>12s}", end="")
    print()
    for ca in info["classes"]:
        print(f"  {ca:12s}", end="")
        for cb in info["classes"]:
            print(f"{info['confusion'][ca].get(cb, 0):>12d}", end="")
        print()

    print(f"\nManuscript text:")
    print(f'  "Two annotators independently labeled n={info["n"]} failure instances. '
          f'Cohen\'s kappa = {kappa:.2f} ({info["interpretation"]} agreement)."')


if __name__ == "__main__":
    main()
