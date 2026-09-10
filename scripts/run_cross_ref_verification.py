#!/usr/bin/env python3
"""
run_cross_ref_verification.py - Run all 5 verifiers on cross-model regenerated text refs.

This addresses T2.4 in the audit: Run all 5 verifiers on both regenerations.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MODELS, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS
from llm_clients import get_client
from prompts.verification_prompts import PROMPTS


def load_dataset(path):
    """Load dataset from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_subset(dataset, n=147):
    """Get a subset of instances (matching the visual subset size)."""
    # Use the same seed as other experiments for consistency
    import random
    rng = random.Random(42)
    pool = list(dataset)
    rng.shuffle(pool)
    return pool[:n]


def run_verification(model_name, dataset, condition='B', variant='xref'):
    """Run verification for a model on a dataset."""
    model_cfg = MODELS[model_name]
    client = get_client(provider=model_cfg.get('provider', 'openai'))
    
    results = []
    total_cost = 0.0
    success_count = 0
    error_count = 0
    
    for i, instance in enumerate(dataset):
        instance_id = instance['instance_id']
        task_text = instance['task_text']
        text_reference = instance.get('text_reference', '')
        actual_screenshot = instance.get('final_screenshot_path', '')
        
        if not text_reference or not actual_screenshot:
            continue
        
        # Build prompt
        system = PROMPTS[condition]['system']
        user = PROMPTS[condition]['user'].format(
            task_text=task_text,
            text_reference=text_reference
        )
        
        try:
            result = client.verify(
                model_id=model_cfg['model_id'],
                model_version=model_cfg['model_version'],
                api_version=model_cfg['api_version'],
                system_prompt=system,
                user_prompt=user,
                actual_screenshot=actual_screenshot,
                reference_screenshot=None,
                supports_temperature=model_cfg.get('supports_temperature', True),
            )
            
            # Compute cost
            in_tok = result.get('_input_tokens', 0) or 0
            out_tok = result.get('_output_tokens', 0) or 0
            in_cost = (in_tok / 1000) * COST_PER_1K_INPUT_TOKENS.get(model_name, 0.0)
            out_cost = (out_tok / 1000) * COST_PER_1K_OUTPUT_TOKENS.get(model_name, 0.0)
            total_cost += in_cost + out_cost
            
            out_record = {
                'instance_id': instance_id,
                'task_text': task_text,
                'ground_truth': instance['ground_truth'],
                'failure_type': instance.get('failure_type', 'N/A'),
                'domain': instance.get('domain', 'unknown'),
                'model': model_name,
                'condition': condition,
                'variant': variant,
                'verdict': result.get('verdict'),
                'confidence': result.get('confidence'),
                'reasoning': result.get('reasoning'),
                'latency_s': result.get('_latency_s'),
                'input_tokens': in_tok,
                'output_tokens': out_tok,
                # provenance: which model wrote the text reference this run consumed
                'text_reference_generator': instance.get('text_reference_generator'),
            }
            results.append(out_record)
            
            if result.get('verdict') in ('SUCCESS', 'FAILURE'):
                success_count += 1
            else:
                error_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"  [{model_name}] {i+1}/{len(dataset)} done, cost=${total_cost:.4f}")
            
            # Pacing. Gemini needed 2s on the free tier (20 requests/day); on a
            # billed project the limit is per-minute, so default to no extra wait.
            # Override with GEMINI_SLEEP=<seconds> if a project is rate limited.
            if model_name.startswith('gemini'):
                time.sleep(float(os.environ.get('GEMINI_SLEEP', '0')))
            else:
                time.sleep(0.1)
            
        except Exception as e:
            print(f"  [{model_name}] Error on {instance_id}: {e}")
            error_count += 1
    
    return results, total_cost, success_count, error_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run cross-ref verification')
    parser.add_argument('--dataset', required=True, help='Path to cross-ref dataset')
    parser.add_argument('--variant', default='xref', help='Variant name for output')
    parser.add_argument('--n', type=int, default=147, help='Subset size')
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()), help='Models to run')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without running')
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    subset = get_subset(dataset, args.n)
    
    print(f"=== CROSS-REF VERIFICATION PLAN ===")
    print(f"Dataset: {args.dataset}")
    print(f"Subset size: {len(subset)}")
    print(f"Models: {args.models}")
    print(f"Variant: {args.variant}")
    print()
    
    if args.dry_run:
        print("Dry run - not executing")
        return
    
    # Run verification for each model
    all_results = []
    for model_name in args.models:
        print(f"\nRunning {model_name}...")
        results, cost, success, errors = run_verification(model_name, subset, variant=args.variant)
        all_results.extend(results)
        
        # Save results
        output_path = f'results/cross_refs/{model_name}_B_{args.variant}.jsonl'
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        print(f"  {model_name}: {success} success, {errors} errors, cost=${cost:.4f}")
        print(f"  Saved to {output_path}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total results: {len(all_results)}")
    print(f"Total cost: ${sum(r.get('input_tokens', 0) for r in all_results) / 1000 * 0.0004 + sum(r.get('output_tokens', 0) for r in all_results) / 1000 * 0.0016:.4f}")


if __name__ == '__main__':
    main()
