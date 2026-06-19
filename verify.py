"""
Core verification function and batch runner.
Supports OpenAI, Anthropic (direct API), and WLM LLM Gateway backends.

Usage:
    python verify.py --model gpt-4.1 --condition A --dataset data/verification_dataset.jsonl
    python verify.py --model claude-sonnet-4 --condition all --dataset data/verification_dataset.jsonl
    python verify.py --model all --condition all --dataset data/verification_dataset.jsonl
    python verify.py --model all --condition all --dataset data/verification_dataset.jsonl --dry-run
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import MODELS, CONDITIONS, RESULTS_DIR, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS
from llm_clients import get_client
from prompts.verification_prompts import PROMPTS


def build_prompt(
    condition: str,
    task_text: str,
    text_reference: Optional[str] = None,
) -> tuple[str, str]:
    """Build the system and user prompts for a given condition."""
    system = PROMPTS[condition]["system"]
    user_template = PROMPTS[condition]["user"]

    user = user_template.format(
        task_text=task_text,
        text_reference=text_reference or "",
    )
    return system, user


def verify_single(
    client,
    model_cfg: dict,
    condition: str,
    task_text: str,
    actual_screenshot: str | Path,
    text_reference: Optional[str] = None,
    reference_screenshot: Optional[str | Path] = None,
) -> dict:
    """Run a single verification call."""
    system, user = build_prompt(condition, task_text, text_reference)

    ref_img = None
    if condition in ("C", "D") and reference_screenshot:
        ref_img = reference_screenshot

    result = client.verify(
        model_id=model_cfg["model_id"],
        model_version=model_cfg["model_version"],
        api_version=model_cfg["api_version"],
        system_prompt=system,
        user_prompt=user,
        actual_screenshot=actual_screenshot,
        reference_screenshot=ref_img,
        supports_temperature=model_cfg.get("supports_temperature", True),
    )
    return result


def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSONL dataset."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_result_path(model_name: str, condition: str) -> Path:
    """Get the output file path for a (model, condition) run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{model_name}_{condition}.jsonl"


def load_completed(result_path: Path) -> set[str]:
    """Load already-completed instance IDs for resume capability.

    Only counts instances with a valid verdict (SUCCESS/FAILURE).
    API_ERROR and PARSE_ERROR results are excluded so they get retried.
    """
    completed = set()
    if result_path.exists():
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if item.get("verdict") in ("SUCCESS", "FAILURE"):
                            completed.add(item["instance_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def run_batch(
    model_name: str,
    condition: str,
    dataset_path: str | Path,
    dry_run: bool = False,
) -> dict:
    """
    Run verification for all instances in the dataset for a given model and condition.
    Supports resume: skips already-completed instances.
    Returns a summary dict with counts and total cost.
    """
    model_cfg = MODELS[model_name]
    result_path = get_result_path(model_name, condition)

    # Clean up: remove API_ERROR/PARSE_ERROR entries so they get retried.
    # Rewrite the file keeping only valid results to avoid duplicates on resume.
    if result_path.exists():
        valid_records = []
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if item.get("verdict") in ("SUCCESS", "FAILURE"):
                            valid_records.append(line)
                    except json.JSONDecodeError:
                        continue
        with open(result_path, "w") as f:
            for line in valid_records:
                f.write(line + "\n")

    completed = load_completed(result_path)
    dataset = load_dataset(dataset_path)

    remaining = [d for d in dataset if d["instance_id"] not in completed]
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Condition: {condition}")
    print(f"Total: {len(dataset)} | Completed: {len(completed)} | Remaining: {len(remaining)}")
    print(f"Output: {result_path}")
    print(f"{'='*60}\n")

    if dry_run:
        return {"model": model_name, "condition": condition, "total": len(dataset), "remaining": len(remaining)}

    client = get_client(provider=model_cfg.get("provider", "openai"))
    total_cost = 0.0
    success_count = 0
    error_count = 0

    with open(result_path, "a") as out_f:
        for instance in tqdm(remaining, desc=f"{model_name}/{condition}"):
            instance_id = instance["instance_id"]
            task_text = instance["task_text"]
            actual_screenshot = instance["final_screenshot_path"]
            text_reference = instance.get("text_reference")
            reference_screenshot = instance.get("reference_screenshot_path")

            if not Path(actual_screenshot).exists():
                error_count += 1
                continue

            if condition in ("C", "D"):
                if not reference_screenshot or not Path(reference_screenshot).exists():
                    error_count += 1
                    continue

            result = verify_single(
                client=client,
                model_cfg=model_cfg,
                condition=condition,
                task_text=task_text,
                actual_screenshot=actual_screenshot,
                text_reference=text_reference,
                reference_screenshot=reference_screenshot,
            )

            input_cost = result.get("_input_tokens", 0) / 1000 * COST_PER_1K_INPUT_TOKENS.get(model_name, 0)
            output_cost = result.get("_output_tokens", 0) / 1000 * COST_PER_1K_OUTPUT_TOKENS.get(model_name, 0)
            total_cost += input_cost + output_cost

            output_record = {
                "instance_id": instance_id,
                "task_text": task_text,
                "ground_truth": instance["ground_truth"],
                "failure_type": instance.get("failure_type", "N/A"),
                "domain": instance.get("domain", "unknown"),
                "model": model_name,
                "condition": condition,
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
                "latency_s": result.get("_latency_s"),
                "input_tokens": result.get("_input_tokens"),
                "output_tokens": result.get("_output_tokens"),
                "raw_response": result.get("_raw", ""),
            }

            out_f.write(json.dumps(output_record) + "\n")
            out_f.flush()

            if result.get("verdict") in ("SUCCESS", "FAILURE"):
                success_count += 1
            else:
                error_count += 1

            time.sleep(0.1)

    summary = {
        "model": model_name,
        "condition": condition,
        "total": len(dataset),
        "completed": len(completed) + success_count + error_count,
        "api_errors": error_count,
        "estimated_cost_usd": round(total_cost, 4),
    }
    print(f"\nDone: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run visual self-verification experiments")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--condition", type=str, default="all", help="Condition (A/B/C/D) or 'all'")
    parser.add_argument("--dataset", type=str, required=True, help="Path to verification_dataset.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    summaries = []
    for model_name in models:
        for condition in conditions:
            summary = run_batch(model_name, condition, args.dataset, dry_run=args.dry_run)
            summaries.append(summary)

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    print("=" * 60)
    total_cost = sum(s.get("estimated_cost_usd", 0) for s in summaries)
    print(f"Total estimated cost: ${total_cost:.4f}")
    for s in summaries:
        print(f"  {s['model']}/{s['condition']}: {s.get('completed', s.get('remaining', '?'))}/{s['total']}")


if __name__ == "__main__":
    main()
