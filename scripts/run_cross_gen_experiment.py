#!/usr/bin/env python3
"""
run_cross_gen_experiment.py - Run cross-generator experiment.
Uses Claude-generated refs on the 147-instance visual subset for all 5 verifiers.
"""

import json
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MODELS, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS
from llm_clients import get_client
from prompts.verification_prompts import PROMPTS


def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_visual_subset():
    """Load the 147-instance visual subset IDs."""
    ids = set()
    with open(ROOT / "data" / "visual_subset_ids.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def subset_to_visual(dataset, visual_ids):
    """Filter dataset to only instances in the visual subset."""
    return [r for r in dataset if r.get("instance_id") in visual_ids]


def run_verification(model_name, dataset, condition='B', variant='xref-claude'):
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
            error_count += 1
            continue
        
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
                'text_reference_generator': 'claude-sonnet-4',
            }
            results.append(out_record)
            
            if result.get('verdict') in ('SUCCESS', 'FAILURE'):
                success_count += 1
            else:
                error_count += 1
            
            if (i + 1) % 25 == 0:
                print(f"  [{model_name}] {i+1}/{len(dataset)} done, cost=${total_cost:.4f}, success={success_count}, errors={error_count}")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  [{model_name}] Error on {instance_id}: {e}")
            error_count += 1
    
    return results, total_cost, success_count, error_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cross-generator experiment')
    parser.add_argument('--dataset', default=str(ROOT / "data" / "verification_dataset_xref_claude.jsonl"))
    parser.add_argument('--variant', default='xref-claude')
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    # Load dataset and subset to visual instances
    dataset = load_dataset(args.dataset)
    visual_ids = load_visual_subset()
    subset = subset_to_visual(dataset, visual_ids)
    
    print(f"=== CROSS-GENERATOR EXPERIMENT ===")
    print(f"Dataset: {args.dataset}")
    print(f"Full dataset: {len(dataset)} instances")
    print(f"Visual subset: {len(subset)} instances")
    print(f"Models: {args.models}")
    print(f"Variant: {args.variant}")
    print()
    
    if args.dry_run:
        print("Dry run - not executing")
        return
    
    # Run verification for each model
    for model_name in args.models:
        print(f"\nRunning {model_name}...")
        results, cost, success, errors = run_verification(model_name, subset, variant=args.variant)
        
        # Save results
        output_path = ROOT / "results" / "cross_refs" / f"{model_name}_B_{args.variant}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        # Compute accuracy
        valid = [r for r in results if r.get('verdict') in ('SUCCESS', 'FAILURE')]
        correct = sum(1 for r in valid if r.get('verdict') == r.get('ground_truth'))
        acc = correct / len(valid) if valid else 0
        
        print(f"  {model_name}: {success} success, {errors} errors, cost=${cost:.4f}")
        print(f"  Accuracy: {correct}/{len(valid)} = {acc:.3f}")
        print(f"  Saved to {output_path}")
    
    print(f"\n=== DONE ===")


if __name__ == '__main__':
    main()
