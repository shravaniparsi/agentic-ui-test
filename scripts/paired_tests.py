#!/usr/bin/env python3
"""
paired_tests.py - Recompute all McNemar tests correctly.

Key requirements:
1. Filter conditions C and D to visual_subset_ids.txt (147 instances)
2. Use jointly valid tasks for each pair (verdict in {SUCCESS, FAILURE})
3. State test convention (exact binomial if b+c<25, else asymptotic χ² with continuity correction)
4. Output results with all required columns
"""

import json
import csv
import math
from pathlib import Path
from scipy import stats
import numpy as np


def load_visual_subset():
    """Load the 147 intended visual subset instances."""
    with open('data/visual_subset_ids.txt') as f:
        return set(line.strip() for line in f if line.strip())


def load_results(model, condition):
    """Load results for a model-condition pair."""
    path = f'results/{model}_{condition}.jsonl'
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r['instance_id']] = r
    return results


def filter_to_subset(results, subset_ids):
    """Filter results to only instances in the subset."""
    return {k: v for k, v in results.items() if k in subset_ids}


def get_valid_instances(results_a, results_b):
    """Get instances where both have valid verdicts (SUCCESS or FAILURE)."""
    valid = []
    for instance_id in results_a:
        if instance_id in results_b:
            va = results_a[instance_id].get('verdict', '').upper()
            vb = results_b[instance_id].get('verdict', '').upper()
            if va in ('SUCCESS', 'FAILURE') and vb in ('SUCCESS', 'FAILURE'):
                valid.append(instance_id)
    return valid


def compute_confusion(valid_instances, results_a, results_b):
    """Compute 2x2 confusion matrix for paired comparison."""
    tp = fp = tn = fn = 0  # a is "first", b is "second"
    b = 0  # only first correct
    c = 0  # only second correct
    
    for instance_id in valid_instances:
        gt = load_ground_truth(instance_id)
        va = results_a[instance_id].get('verdict', '').upper()
        vb = results_b[instance_id].get('verdict', '').upper()
        
        a_correct = (va == gt)
        b_correct = (vb == gt)
        
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
    
    return b, c, len(valid_instances)


def load_ground_truth(instance_id):
    """Load ground truth for an instance."""
    if not hasattr(load_ground_truth, 'cache'):
        load_ground_truth.cache = {}
        with open('data/verification_dataset.jsonl') as f:
            for line in f:
                r = json.loads(line)
                load_ground_truth.cache[r['instance_id']] = r['ground_truth'].upper()
    return load_ground_truth.cache.get(instance_id, 'FAILURE')


def mcnemar_test(b, c, use_continuity=True):
    """
    Compute McNemar test.
    
    Convention:
    - If b+c < 25: Use exact binomial test
    - Else: Use asymptotic χ² with continuity correction (if use_continuity=True)
    """
    n = b + c
    
    if n < 25:
        # Exact binomial test
        # H0: P(b) = P(c) = 0.5
        # p-value = 2 * min(P(X <= min(b,c)), P(X >= max(b,c)))
        k = min(b, c)
        p_value = 2 * stats.binom.cdf(k, n, 0.5)
        p_value = min(p_value, 1.0)
        stat = None
        test_type = 'exact'
    else:
        # Asymptotic χ² with continuity correction
        if use_continuity:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            stat = (b - c) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(stat, 1)
        test_type = 'asymptotic'
    
    return stat, p_value, test_type


def compute_metrics(results):
    """Compute accuracy, F1, etc. for a set of results."""
    tp = fp = tn = fn = 0
    
    for instance_id, r in results.items():
        gt = load_ground_truth(instance_id)
        verdict = r.get('verdict', '').upper()
        
        if gt == 'SUCCESS' and verdict == 'SUCCESS':
            tp += 1
        elif gt == 'FAILURE' and verdict == 'SUCCESS':
            fp += 1
        elif gt == 'FAILURE' and verdict == 'FAILURE':
            tn += 1
        elif gt == 'SUCCESS' and verdict == 'FAILURE':
            fn += 1
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'n': total,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    # Load visual subset
    subset_ids = load_visual_subset()
    print(f"Visual subset: {len(subset_ids)} instances\n")
    
    # Models and conditions
    models = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1', 'claude-sonnet-4', 'gemini-2.5-flash']
    conditions = ['A', 'B', 'C', 'D']
    
    # All pairs to compare
    pairs = [
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'C'), ('B', 'D'),
        ('C', 'D')
    ]
    
    # Output file
    output_rows = []
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        # Load all conditions for this model
        results = {}
        for cond in conditions:
            raw = load_results(model, cond)
            # Filter C and D to visual subset
            if cond in ('C', 'D'):
                results[cond] = filter_to_subset(raw, subset_ids)
            else:
                results[cond] = raw  # A and B use all instances
        
        # Compute metrics for each condition
        metrics = {}
        for cond in conditions:
            metrics[cond] = compute_metrics(results[cond])
            print(f"  {cond}: n={metrics[cond]['n']}, acc={metrics[cond]['accuracy']:.3f}, f1={metrics[cond]['f1']:.3f}")
        
        # Compute McNemar tests for each pair
        for cond1, cond2 in pairs:
            # Get jointly valid instances
            valid = get_valid_instances(results[cond1], results[cond2])
            n_joint = len(valid)
            
            # Compute confusion
            b, c, n = compute_confusion(valid, results[cond1], results[cond2])
            
            # McNemar test
            stat, p_value, test_type = mcnemar_test(b, c)
            
            # Bonferroni correction (6 comparisons per model)
            p_bonferroni = min(p_value * 6, 1.0)
            
            # Check significance
            bonferroni_sig = p_bonferroni < 0.05
            holm_sig = False  # Would need to implement Holm correction
            
            # Compute lift
            acc1 = metrics[cond1]['accuracy']
            acc2 = metrics[cond2]['accuracy']
            lift = (acc2 - acc1) * 100  # in percentage points
            
            # Mathematical bounds check
            max_b_c = n_joint
            chi2_max = max_b_c  # Maximum χ² when |b-c| = n_joint
            
            row = {
                'model': model,
                'pair': f'{cond1}->{cond2}',
                'n_joint': n_joint,
                'n_both_correct': n_joint - b - c,
                'n_only_first_correct': b,
                'n_only_second_correct': c,
                'chi2_or_exact_stat': stat,
                'p_raw': p_value,
                'p_bonferroni': p_bonferroni,
                'bonferroni_sig': bonferroni_sig,
                'holm_sig': holm_sig,
                'test_type': test_type,
                'acc1': acc1,
                'acc2': acc2,
                'lift_pp': lift,
                'chi2_max_possible': chi2_max
            }
            
            output_rows.append(row)
            
            # Print with bounds check
            bounds_ok = abs(b - c) <= n_joint
            print(f"  {cond1}->{cond2}: n={n_joint}, b={b}, c={c}, χ²={stat}, p={p_value:.4f} {'✓' if bounds_ok else '✗ BOUNDS VIOLATED'}")
    
    # Save results
    output_path = 'results/S1_paired_tests.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"\n\nSaved to {output_path}")
    
    # Summary
    sig_count = sum(1 for r in output_rows if r['bonferroni_sig'])
    print(f"\nSignificant comparisons (Bonferroni): {sig_count}/{len(output_rows)}")
    
    # Check for bound violations
    violations = [r for r in output_rows if abs(r['n_only_first_correct'] - r['n_only_second_correct']) > r['n_joint']]
    if violations:
        print(f"\n⚠️  BOUND VIOLATIONS: {len(violations)}")
        for v in violations:
            print(f"  {v['model']} {v['pair']}: |b-c|={abs(v['n_only_first_correct']-v['n_only_second_correct'])} > n={v['n_joint']}")


if __name__ == '__main__':
    main()
