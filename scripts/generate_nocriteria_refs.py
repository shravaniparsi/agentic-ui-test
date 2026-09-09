#!/usr/bin/env python3
"""
generate_nocriteria_refs.py - Generate text references WITHOUT evaluation criteria.
Addresses R2-2: whether the benefit comes from eval criteria or just task description.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MODELS
from llm_clients import get_client


# Prompt WITHOUT evaluation criteria
NOCRITERIA_PROMPT = (
    "You are helping evaluate web automation agents. Given the following web "
    "task description, produce a 1-3 sentence natural-language description of "
    "what the final web page should look like if the task was completed "
    "successfully. Focus on what semantic content / state should be visible. "
    "Do NOT describe the layout (colors, fonts, positions). Do NOT include "
    "'In conclusion' or preamble. Output ONLY the description.\n\n"
    "Task: {task_text}\n\n"
    "Expected outcome description:"
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="gpt-4.1-nano")
    parser.add_argument("--n", type=int, default=0, help="0 = all")
    parser.add_argument("--out", default=str(ROOT / "data" / "verification_dataset_nocriteria.jsonl"))
    args = parser.parse_args()
    
    # Load base dataset
    dataset = []
    with open(ROOT / "data" / "verification_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    
    n = args.n if args.n > 0 else len(dataset)
    spec = MODELS[args.generator]
    client = get_client(provider=spec["provider"])
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    written = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for r in dataset[:n]:
            prompt = NOCRITERIA_PROMPT.format(task_text=r.get("task_text", ""))
            try:
                desc = client.generate_text(
                    model_id=spec["model_id"],
                    model_version=spec["model_version"],
                    api_version=spec["api_version"],
                    prompt=prompt,
                    temperature=0,
                    max_tokens=200,
                    supports_temperature=spec.get("supports_temperature", True),
                ).strip()
            except Exception as e:
                print(f"Error: {e}")
                desc = ""
            
            new_row = dict(r)
            new_row["text_reference"] = desc
            new_row["text_reference_generator"] = args.generator
            new_row["text_reference_criteria"] = "none"
            f.write(json.dumps(new_row) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  {written}/{n} done in {time.time()-t0:.0f}s")
    
    print(f"Wrote {written} rows to {out_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
