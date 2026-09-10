#!/usr/bin/env python3
"""
recompute_iaa.py - Recompute IAA with proper methodology.

This addresses Phase 4 of the audit: Fix failure taxonomy/IAA.
"""

import argparse
import csv
from collections import Counter
from sklearn.metrics import cohen_kappa_score

CATEGORIES = ("obvious", "deceptive", "partial")


def load_labels(path):
    """Load labels from CSV file."""
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['instance_id']] = row['label']
    return labels


def compute_iaa(labels_a, labels_b, taxonomy='2cat'):
    """Compute IAA between two labelers."""
    # Find common instances
    common = sorted(set(labels_a.keys()) & set(labels_b.keys()))
    
    if taxonomy == '2cat':
        # Map to 2 categories: failed or partial
        a_labels = ['failed' if labels_a[iid] == 'failed' else 'partial' for iid in common]
        b_labels = ['failed' if labels_b[iid] == 'failed' else 'partial' for iid in common]
    else:
        # Keep original 3 categories
        a_labels = [labels_a[iid] for iid in common]
        b_labels = [labels_b[iid] for iid in common]
    
    # Compute kappa
    kappa = cohen_kappa_score(a_labels, b_labels)
    
    # Compute agreement
    agreement = sum(1 for a, b in zip(a_labels, b_labels) if a == b) / len(common)
    
    # Count labels
    a_counts = Counter(a_labels)
    b_counts = Counter(b_labels)
    
    return {
        'n': len(common),
        'kappa': kappa,
        'agreement': agreement,
        'a_counts': dict(a_counts),
        'b_counts': dict(b_counts)
    }


def confusion_matrix(labels_a, labels_b, common, categories=CATEGORIES):
    """Return a printable confusion matrix of annotator A (rows) vs B (columns)."""
    present = [c for c in categories
               if any(labels_a[i] == c or labels_b[i] == c for i in common)]
    counts = Counter((labels_a[i], labels_b[i]) for i in common)
    width = max(10, max((len(c) for c in present), default=10) + 2)
    lines = ["  " + "".join(["A\\B".ljust(width)] + [c.ljust(width) for c in present])]
    for ra in present:
        cells = [str(counts.get((ra, cb), 0)).ljust(width) for cb in present]
        lines.append("  " + "".join([ra.ljust(width)] + cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Recompute IAA on the three-way taxonomy")
    ap.add_argument("--a", default="iaa_labeling/data/iaa_labels_3cat_human1.csv",
                    help="CSV of annotator 1")
    ap.add_argument("--b", default="iaa_labeling/data/iaa_labels_3cat_human2.csv",
                    help="CSV of annotator 2")
    args = ap.parse_args()

    labels_a = load_labels(args.a)
    labels_b = load_labels(args.b)
    common = sorted(set(labels_a) & set(labels_b))

    print("=== IAA ANALYSIS (three-way taxonomy) ===\n")
    print(f"Annotator 1: {args.a} ({len(labels_a)} labels)")
    print(f"Annotator 2: {args.b} ({len(labels_b)} labels)")
    print(f"Instances labeled by both: {len(common)}\n")

    if not common:
        print("error: the two annotators share no labeled instances")
        raise SystemExit(1)

    unexpected = {l for i in common for l in (labels_a[i], labels_b[i])
                  if l not in CATEGORIES}
    if unexpected:
        print(f"warning: labels outside the taxonomy: {sorted(unexpected)}\n")

    result = compute_iaa(labels_a, labels_b, taxonomy="3cat")
    print("3-category taxonomy (obvious/deceptive/partial):")
    print(f"  N: {result['n']}")
    print(f"  Cohen's kappa: {result['kappa']:.3f}")
    print(f"  Raw agreement: {result['agreement']:.3f}")
    print(f"  Annotator 1 distribution: {result['a_counts']}")
    print(f"  Annotator 2 distribution: {result['b_counts']}")

    print("\nConfusion matrix (rows = annotator 1, columns = annotator 2):")
    print(confusion_matrix(labels_a, labels_b, common))

    print("\n=== INTERPRETATION ===")
    k = result["kappa"]
    if k < 0.4:
        print(f"kappa = {k:.3f} (< 0.40): poor agreement; report as exploratory.")
    elif k < 0.6:
        print(f"kappa = {k:.3f} (0.40-0.60): moderate; report Section 5.3 as exploratory.")
    else:
        print(f"kappa = {k:.3f} (>= 0.60): substantial agreement.")


if __name__ == '__main__':
    main()
