#!/usr/bin/env python3
"""
Cost-aware runner for the text-reference quality ablation (Condition B variants).

Variants:
  - full   (baseline: same as the existing Condition B run)
  - short  (first sentence only)
  - noisy  (full + filler/distractor sentence)
  - wrong  (deliberately off-topic reference)

Default behavior is conservative:
  - subset of N=100 instances (fixed seed)
  - models: gpt-4.1-mini, claude-sonnet-4
  - --dry-run prints the plan and an estimated cost without making API calls

Output:
  results/textref_ablation_{model}_{variant}.jsonl

Usage:
  # Print the plan and estimated cost (no API calls):
  python3 scripts/run_text_ref_ablation.py --dry-run

  # Actually run a small subset (will spend API credits):
  python3 scripts/run_text_ref_ablation.py \
      --variants short noisy wrong \
      --models gpt-4.1-mini claude-sonnet-4 \
      --n 100

This script reuses the existing prompt for Condition B, swapping the
text reference field with the chosen variant.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATASET_PATH = DATA_DIR / "verification_dataset.jsonl"
VARIANTS_PATH = DATA_DIR / "verification_dataset_textref_variants.jsonl"

# Static pricing table used for dry-run estimates so the plan can be previewed
# without installing any Python dependencies. Keep in sync with config.py.
STATIC_COST_PER_1K_INPUT_TOKENS = {
    "gpt-4.1": 0.002,
    "gpt-4.1-mini": 0.0004,
    "gpt-4.1-nano": 0.0001,
    "claude-sonnet-4": 0.003,
    "gemini-2.5-flash": 0.00015,
}
STATIC_COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4.1": 0.008,
    "gpt-4.1-mini": 0.0016,
    "gpt-4.1-nano": 0.0004,
    "claude-sonnet-4": 0.015,
    "gemini-2.5-flash": 0.0006,
}
STATIC_MODEL_NAMES = list(STATIC_COST_PER_1K_INPUT_TOKENS.keys())

VARIANT_FIELD = {
    "full": "text_reference_full",
    "short": "text_reference_short",
    "noisy": "text_reference_noisy",
    "wrong": "text_reference_wrong",
}

DEFAULT_VARIANTS = ["short", "noisy", "wrong"]
DEFAULT_MODELS = ["gpt-4.1-mini", "claude-sonnet-4"]
DEFAULT_N = 100
SEED = 42

# Conservative per-call token estimate for cost preview only.
EST_INPUT_TOKENS_PER_CALL = 1500
EST_OUTPUT_TOKENS_PER_CALL = 150


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def estimate_cost(models: list[str], variants: list[str], n: int) -> float:
    total = 0.0
    for model in models:
        in_cost = STATIC_COST_PER_1K_INPUT_TOKENS.get(model, 0.0) * (EST_INPUT_TOKENS_PER_CALL / 1000)
        out_cost = STATIC_COST_PER_1K_OUTPUT_TOKENS.get(model, 0.0) * (EST_OUTPUT_TOKENS_PER_CALL / 1000)
        per_call = in_cost + out_cost
        total += per_call * n * len(variants)
    return total


def get_subset(dataset: list[dict], n: int) -> list[dict]:
    rng = random.Random(SEED)
    pool = list(dataset)
    rng.shuffle(pool)
    return pool[:n]


def build_prompt(task_text: str, text_reference: str) -> tuple[str, str]:
    # Lazy import so --dry-run does not require the prompts module to be importable
    from prompts.verification_prompts import PROMPTS  # noqa: WPS433
    system = PROMPTS["B"]["system"]
    user = PROMPTS["B"]["user"].format(task_text=task_text, text_reference=text_reference)
    return system, user


def get_result_path(model: str, variant: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"textref_ablation_{model}_{variant}.jsonl"


def already_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("verdict") in ("SUCCESS", "FAILURE"):
                    done.add(rec["instance_id"])
            except json.JSONDecodeError:
                continue
    return done


def run_one(model: str, variant: str, subset: list[dict], variants_by_id: dict[str, dict], dry_run: bool) -> dict:
    variant_field = VARIANT_FIELD[variant]
    out_path = get_result_path(model, variant)
    completed = already_done(out_path)
    remaining = [r for r in subset if r["instance_id"] not in completed and r["instance_id"] in variants_by_id]

    summary = {
        "model": model,
        "variant": variant,
        "subset_n": len(subset),
        "already_done": len(completed),
        "remaining": len(remaining),
        "output_file": str(out_path),
    }

    print(f"\n  [{model} / {variant}] subset={summary['subset_n']} done={summary['already_done']} remaining={summary['remaining']}")
    if dry_run:
        return summary

    # Lazy imports so --dry-run never requires runtime deps.
    from config import MODELS as RUNTIME_MODELS, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS  # noqa: WPS433
    from llm_clients import get_client  # noqa: WPS433

    model_cfg = RUNTIME_MODELS[model]
    client = get_client(provider=model_cfg.get("provider", "openai"))
    total_cost = 0.0
    success_count = 0
    error_count = 0

    with open(out_path, "a") as out_f:
        for instance in remaining:
            iid = instance["instance_id"]
            task_text = instance["task_text"]
            actual_screenshot = instance["final_screenshot_path"]
            text_reference = variants_by_id[iid].get(variant_field, "")
            if not text_reference:
                continue
            if not Path(actual_screenshot).exists():
                error_count += 1
                continue

            system, user = build_prompt(task_text, text_reference)
            result = client.verify(
                model_id=model_cfg["model_id"],
                model_version=model_cfg["model_version"],
                api_version=model_cfg["api_version"],
                system_prompt=system,
                user_prompt=user,
                actual_screenshot=actual_screenshot,
                reference_screenshot=None,
                supports_temperature=model_cfg.get("supports_temperature", True),
            )

            in_tok = result.get("_input_tokens", 0) or 0
            out_tok = result.get("_output_tokens", 0) or 0
            in_cost = (in_tok / 1000) * COST_PER_1K_INPUT_TOKENS.get(model, 0.0)
            out_cost = (out_tok / 1000) * COST_PER_1K_OUTPUT_TOKENS.get(model, 0.0)
            total_cost += in_cost + out_cost

            out_record = {
                "instance_id": iid,
                "task_text": task_text,
                "ground_truth": instance["ground_truth"],
                "failure_type": instance.get("failure_type", "N/A"),
                "domain": instance.get("domain", "unknown"),
                "model": model,
                "condition": "B",
                "variant": variant,
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
                "latency_s": result.get("_latency_s"),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
            out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()

            if result.get("verdict") in ("SUCCESS", "FAILURE"):
                success_count += 1
            else:
                error_count += 1

            time.sleep(0.1)

    summary.update({
        "valid": success_count,
        "errors": error_count,
        "estimated_cost_usd": round(total_cost, 4),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cost-aware text-reference quality ablation.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_FIELD.keys()))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=STATIC_MODEL_NAMES)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="subset size (default: 100)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not VARIANTS_PATH.exists():
        raise SystemExit(f"Variants file not found: {VARIANTS_PATH}\nRun: python3 scripts/generate_text_ref_variants.py")

    dataset = load_jsonl(DATASET_PATH)
    variants = load_jsonl(VARIANTS_PATH)
    variants_by_id = {v["instance_id"]: v for v in variants}

    subset = get_subset(dataset, args.n)

    est = estimate_cost(args.models, args.variants, len(subset))
    print("=" * 60)
    print("TEXT-REF ABLATION PLAN")
    print(f"  Models:    {args.models}")
    print(f"  Variants:  {args.variants}")
    print(f"  Subset N:  {len(subset)}")
    print(f"  Estimated cost (rough upper bound): ~${est:.2f}")
    print(f"  Dry run:   {args.dry_run}")
    print("=" * 60)

    summaries = []
    for model in args.models:
        for variant in args.variants:
            s = run_one(model, variant, subset, variants_by_id, args.dry_run)
            summaries.append(s)

    print("\nDONE")
    for s in summaries:
        print("  " + json.dumps(s))


if __name__ == "__main__":
    main()
