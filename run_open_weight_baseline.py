#!/usr/bin/env python3
"""
Open-weight verifier baseline (Critique issue #10).

Reviewer concern: All five evaluated verifiers are closed-API frontier models.
Adding at least one open-weight VLM strengthens external validity, makes the
study reproducible without API access, and extends the result's shelf life.

This script runs Conditions A and B on a configurable subset using one of:
  - Qwen/Qwen2.5-VL-7B-Instruct  (default)
  - OpenGVLab/InternVL2-8B
  - llava-hf/llava-1.5-7b-hf

It writes results in the same JSONL schema as the closed-API runs
(verify.py / config.py), so analyze_results.py and revision_analysis.py
will pick them up automatically once `MODELS` in those scripts is extended.

Usage:
    # Dry-run: print what we would do without loading the model.
    python3 scripts/run_open_weight_baseline.py --dry-run --n 50 \
        --model Qwen/Qwen2.5-VL-7B-Instruct --condition A

    # Real run (requires GPU):
    python3 scripts/run_open_weight_baseline.py --n 200 \
        --model Qwen/Qwen2.5-VL-7B-Instruct --condition B \
        --out results/qwen2.5-vl-7b_B.jsonl

Dependencies (only when not in --dry-run):
    pip install torch transformers accelerate Pillow

The script does not require API keys. It does require a CUDA or MPS device for
reasonable throughput; CPU inference will be very slow but functional.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
RESULTS_DIR = PROJECT_ROOT / "results"

# Mirrors the closed-API prompts in prompts/verification_prompts.py so we are
# comparing apples to apples.
SYSTEM_PROMPT = (
    "You are an expert evaluator for web automation agents. Your job is to "
    "determine whether a web task was completed successfully by examining "
    "the final state of the web page.\n\n"
    "You must respond in EXACTLY this JSON format:\n"
    "{\n"
    '  "verdict": "SUCCESS" or "FAILURE",\n'
    '  "confidence": <integer from 1 to 10>,\n'
    '  "reasoning": "<2-3 sentences explaining your judgment>"\n'
    "}\n\n"
    "Be precise. A task is only successful if ALL aspects of the task "
    "description are satisfied in the final page state. Partial completions "
    "count as FAILURE."
)

USER_A = "## Task\n{task_text}\n\n## Instructions\nExamine the screenshot of the web page's final state. Determine whether the task described above was completed successfully.\n\nRespond in the required JSON format."
USER_B = "## Task\n{task_text}\n\n## Expected Outcome (text description)\n{text_reference}\n\n## Instructions\nExamine the screenshot of the web page's final state. Compare it against the expected outcome description above. Determine whether the task was completed successfully.\n\nRespond in the required JSON format."


def load_dataset() -> list[dict]:
    rows = []
    with open(DATA_DIR / "verification_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_response(raw: str) -> dict:
    """Same lenient parser as llm_clients._parse_response, copied to avoid a
    cross-import that pulls in API client deps."""
    import re
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass
    return {"verdict": "PARSE_ERROR", "confidence": 0, "reasoning": raw[:500]}


def dry_run(args, rows: list[dict]):
    print(f"[dry-run] model={args.model}")
    print(f"[dry-run] condition={args.condition}")
    print(f"[dry-run] would process {min(args.n, len(rows))} instances")
    print(f"[dry-run] system prompt: {len(SYSTEM_PROMPT)} chars")
    sample = rows[:3]
    for r in sample:
        ut = USER_A.format(task_text=r["task_text"])[:120]
        print(f"[dry-run] sample task: {r['instance_id']} -> {ut!r}...")
    print(f"[dry-run] output target: {args.out}")
    print("[dry-run] no GPU touched, no model loaded; install torch+transformers+accelerate to run for real.")


def real_run(args, rows: list[dict]):
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:
        print(f"error: missing deps for real run ({e}). pip install torch transformers accelerate Pillow", file=sys.stderr)
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"[run] loading model {args.model} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device in ("cuda", "mps") else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for r in rows[: args.n]:
            iid = r["instance_id"]
            screenshot = SCREENSHOTS_DIR / r["screenshot"]
            if not screenshot.exists():
                continue
            user = USER_A.format(task_text=r["task_text"]) if args.condition == "A" else USER_B.format(task_text=r["task_text"], text_reference=r.get("text_reference", ""))
            img = Image.open(screenshot).convert("RGB")
            prompt = SYSTEM_PROMPT + "\n\n" + user
            try:
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
                t_start = time.time()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                latency = time.time() - t_start
                raw = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            except Exception as e:
                raw = ""
                latency = 0.0

            parsed = parse_response(raw)
            row = {
                "instance_id": iid,
                "task_text": r["task_text"],
                "ground_truth": r["ground_truth"],
                "failure_type": r.get("failure_type", ""),
                "domain": r.get("domain", ""),
                "model": args.model,
                "condition": args.condition,
                "verdict": parsed.get("verdict", "PARSE_ERROR"),
                "confidence": parsed.get("confidence", 0),
                "reasoning": parsed.get("reasoning", ""),
                "latency_s": round(latency, 2),
                "raw_response": raw,
            }
            f.write(json.dumps(row) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"[run] {written}/{min(args.n, len(rows))} done in {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"[run] wrote {written} rows to {out_path} in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    choices=["Qwen/Qwen2.5-VL-7B-Instruct",
                             "OpenGVLab/InternVL2-8B",
                             "llava-hf/llava-1.5-7b-hf"])
    ap.add_argument("--condition", default="A", choices=["A", "B"],
                    help="Conditions C/D require dual-image handling; not yet supported here.")
    ap.add_argument("--n", type=int, default=200, help="Max instances to process")
    ap.add_argument("--out", default=None, help="Output JSONL path (default: results/<model-slug>_<condition>.jsonl)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.out is None:
        slug = args.model.split("/")[-1].lower().replace(".", "")
        args.out = str(RESULTS_DIR / f"{slug}_{args.condition}.jsonl")

    rows = load_dataset()
    if args.dry_run:
        dry_run(args, rows)
        return
    real_run(args, rows)


if __name__ == "__main__":
    main()
