#!/usr/bin/env python3
"""
recompute_iaa.py - Recompute IAA with proper methodology.

This addresses Phase 4 of the audit: Fix failure taxonomy/IAA.
"""

import csv
import json
from collections import Counter
from sklearn.metrics import cohen_kappa_score


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


def main():
    # Load labels
    labels_a = load_labels('iaa_labeling/data/iaa_labels_A.csv')
    labels_b = load_labels('iaa_labeling/data/iaa_labels_B.csv')
    
    print("=== IAA ANALYSIS ===\n")
    
    # 2-category taxonomy
    print("2-category taxonomy (failed/partial):")
    result_2cat = compute_iaa(labels_a, labels_b, taxonomy='2cat')
    print(f"  N: {result_2cat['n']}")
    print(f"  Kappa: {result_2cat['kappa']:.3f}")
    print(f"  Agreement: {result_2cat['agreement']:.3f}")
    print(f"  Human 1: {result_2cat['a_counts']}")
    print(f"  Human 2: {result_2cat['b_counts']}")
    
    # 3-category taxonomy
    print("\n3-category taxonomy (obvious/partial/deceptive):")
    result_3cat = compute_iaa(labels_a, labels_b, taxonomy='3cat')
    print(f"  N: {result_3cat['n']}")
    print(f"  Kappa: {result_3cat['kappa']:.3f}")
    print(f"  Agreement: {result_3cat['agreement']:.3f}")
    print(f"  Human 1: {result_3cat['a_counts']}")
    print(f"  Human 2: {result_3cat['b_counts']}")
    
    # Recommendation
    print("\n=== RECOMMENDATION ===")
    if result_2cat['kappa'] < 0.6:
        print("Kappa < 0.6: Label Section 5.3 'exploratory'")
        print("Remove Finding 4 from abstract")
    else:
        print("Kappa >= 0.6: Keep Section 5.3 as is")


if __name__ == '__main__':
    main()
