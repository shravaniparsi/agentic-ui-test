#!/usr/bin/env python3
"""
Human baseline collection scaffold (Critique issue #6).

Reviewer concern: We claim 86.7% accuracy is good, but never anchor against
a human verifier on the same single-screenshot setup. This script generates a
self-contained CLI tool that walks a labeler through a stratified random
sample and records their verdict + confidence, mirroring the verifier output
schema exactly so analyze_results.py can score human performance with the
same metrics as the LMM verifiers.

Usage:
    # 1. Sample a stratified set (preserves SUCCESS/FAILURE proportion).
    python3 scripts/collect_human_baseline.py sample --n 100 \
        --out data/human_baseline_sample.jsonl

    # 2. Open a terminal and label them one at a time. The script prints the
    #    task and the path to the screenshot; you open the image in any viewer.
    python3 scripts/collect_human_baseline.py label \
        --in data/human_baseline_sample.jsonl \
        --out results/human-rater_A.jsonl \
        --rater-id alice

    # 3. Score with the standard analyzer.
    python3 scripts/analyze_results.py  # picks up the new file automatically
                                        # if MODELS in config is extended

The output JSONL schema matches the closed-API verifiers (verify.py output),
so revision_analysis.py and deep_analysis.py treat the human as just another
"verifier" -- providing the apples-to-apples comparison reviewers will want.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"


def load_dataset() -> list[dict]:
    rows = []
    with open(DATA_DIR / "verification_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_sample(args):
    random.seed(args.seed)
    rows = load_dataset()
    succ = [r for r in rows if r.get("ground_truth") == "SUCCESS"]
    fail = [r for r in rows if r.get("ground_truth") == "FAILURE"]
    p_succ = len(succ) / max(1, len(succ) + len(fail))
    n_succ = round(args.n * p_succ)
    n_fail = args.n - n_succ
    sample = random.sample(succ, min(n_succ, len(succ))) + random.sample(fail, min(n_fail, len(fail)))
    random.shuffle(sample)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(sample)} stratified instances ({n_succ} SUCCESS / {n_fail} FAILURE) to {out_path}")
    print("Now run: python3 scripts/collect_human_baseline.py label --in", out_path, "--out results/human-rater_A.jsonl --rater-id <id>")


def _prompt_label(prompt: str) -> str | None:
    while True:
        s = input(prompt).strip().upper()
        if s in ("S", "SUCCESS"):
            return "SUCCESS"
        if s in ("F", "FAILURE"):
            return "FAILURE"
        if s in ("Q", "QUIT", "EXIT"):
            return None
        print("  please type S, F, or Q to quit")


def _prompt_confidence(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if 1 <= v <= 10:
                return v
        except ValueError:
            pass
        print("  please type an integer 1-10")


def cmd_label(args):
    in_rows = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if line:
                in_rows.append(json.loads(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    continue
        print(f"resuming -- {len(done_ids)} already labeled in {out_path}")

    todo = [r for r in in_rows if r["instance_id"] not in done_ids]
    print(f"To label: {len(todo)} instances. Press Ctrl-C or type Q to stop and resume later.")
    print()
    print("LABELING PROTOCOL:")
    print("  - Open the screenshot in any image viewer.")
    print("  - Read the task and decide if the page state shows SUCCESS or FAILURE.")
    print("  - SUCCESS = ALL aspects of the task are satisfied. Partial = FAILURE.")
    print("  - Confidence: 10=certain, 7=likely, 5=uncertain, 1=guess.")
    print()

    with open(out_path, "a") as f:
        for i, r in enumerate(todo, 1):
            iid = r["instance_id"]
            print(f"--- {i}/{len(todo)}: {iid} ({r.get('domain','?')}) ---")
            print(f"Task: {r.get('task_text','')[:300]}")
            ss_path = SCREENSHOTS_DIR / r.get("screenshot", "")
            print(f"Screenshot path: {ss_path}")
            if not ss_path.exists():
                print("  (screenshot file missing; skipping)")
                continue
            try:
                verdict = _prompt_label("Verdict [S=SUCCESS, F=FAILURE, Q=quit]: ")
                if verdict is None:
                    print("Saving and exiting.")
                    return
                conf = _prompt_confidence("Confidence (1-10): ")
                reasoning = input("One-line reasoning (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaving and exiting.")
                return

            row = {
                "instance_id": iid,
                "task_text": r.get("task_text", ""),
                "ground_truth": r.get("ground_truth", ""),
                "failure_type": r.get("failure_type", ""),
                "domain": r.get("domain", ""),
                "model": f"human-rater-{args.rater_id}",
                "condition": "A",
                "verdict": verdict,
                "confidence": conf,
                "reasoning": reasoning,
                "latency_s": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
    print(f"All done. {len(todo)} new labels appended to {out_path}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Stratified sample for human labeling")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/human_baseline_sample.jsonl")
    s.set_defaults(func=cmd_sample)

    l = sub.add_parser("label", help="Walk through samples and record human verdicts")
    l.add_argument("--in", dest="inp", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--rater-id", required=True)
    l.set_defaults(func=cmd_label)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
