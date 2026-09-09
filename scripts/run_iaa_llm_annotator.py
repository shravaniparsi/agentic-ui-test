#!/usr/bin/env python3
"""
LLM-based IAA annotator — uses Claude/GPT to label failure types.
Generates Annotator B labels for inter-annotator agreement study.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCREENSHOTS_DIR = ROOT / "data" / "screenshots"

PROMPT_TEMPLATE = """You are labeling web agent failure types. Look at the screenshot and task, then classify the failure.

Task: {task_text}

EXAMPLES:
- Task: "Find the cheapest blue kayak" → Screenshot shows Amazon homepage → Label: obvious (wrong site entirely)
- Task: "Find the cheapest blue kayak" → Screenshot shows a red kayak listing → Label: deceptive (right site, wrong item - looks normal but wrong)
- Task: "Find the cheapest blue kayak" → Screenshot shows search results loading → Label: partial (started searching but didn't finish)

Classify into ONE of these three categories:

1. "obvious" — Agent went to the WRONG WEBSITE or page entirely. The screenshot shows something completely unrelated to the task. No one would mistake this for task completion.

2. "partial" — Agent started the task but didn't finish. Maybe it's loading, or on the right site but mid-action, or partially filled a form. Progress was made but work remains.

3. "deceptive" — Agent is on the RIGHT WEBSITE and the page looks normal/valid, but the specific thing the task asks for is NOT present or is wrong. The page "looks fine" but doesn't actually satisfy the task.

Output ONLY one word: obvious, partial, or deceptive"""


def load_sample(path):
    instances = []
    with open(path) as f:
        for line in f:
            instances.append(json.loads(line.strip()))
    return instances


def label_with_anthropic(instance, client):
    """Label using Anthropic Claude API."""
    import anthropic

    task_text = instance["task_text"]
    screenshot_path = SCREENSHOTS_DIR / f"{instance['instance_id']}.png"

    if not screenshot_path.exists():
        return "error"

    import base64
    with open(screenshot_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(task_text=task_text)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0,
    )

    answer = response.content[0].text.strip().lower()
    if answer in ("obvious", "partial", "deceptive"):
        return answer
    return "error"


def label_with_openai(instance, client):
    """Label using OpenAI GPT API."""
    import base64

    task_text = instance["task_text"]
    screenshot_path = SCREENSHOTS_DIR / f"{instance['instance_id']}.png"

    if not screenshot_path.exists():
        return "error"

    with open(screenshot_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(task_text=task_text)

    response = client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}", "detail": "high"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().lower()
    if answer in ("obvious", "partial", "deceptive"):
        return answer
    return "error"


def main():
    parser = argparse.ArgumentParser(description="LLM IAA annotator")
    parser.add_argument("--model", default="claude-sonnet-4", choices=["claude-sonnet-4", "gpt-4.1"])
    parser.add_argument("--sample", default=str(ROOT / "data" / "iaa_sample.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "iaa_labeling" / "data" / "iaa_labels_B.csv"))
    args = parser.parse_args()

    instances = load_sample(args.sample)
    print(f"Loaded {len(instances)} instances")

    # Load already done
    done = {}
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["instance_id"]] = row["label"]
    print(f"Already done: {len(done)}")

    remaining = [i for i in instances if i["instance_id"] not in done]
    print(f"Remaining: {len(remaining)}")

    # Init client
    if args.model == "claude-sonnet-4":
        import anthropic
        client = anthropic.Anthropic()
        label_fn = label_with_anthropic
    else:
        from openai import OpenAI
        client = OpenAI()
        label_fn = label_with_openai

    # Label
    file_exists = out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "label", "task_text", "domain"])
        if not file_exists:
            writer.writeheader()

        for i, inst in enumerate(remaining):
            label = label_fn(inst, client)
            writer.writerow({
                "instance_id": inst["instance_id"],
                "label": label,
                "task_text": inst["task_text"],
                "domain": inst.get("domain", ""),
            })
            f.flush()

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(remaining)} done")
            time.sleep(0.5)  # Rate limit

    print(f"\nDone! Labels saved to {out_path}")


if __name__ == "__main__":
    main()
