#!/usr/bin/env python3
"""
Generate degraded text-reference variants for the ablation study.

Produces three variants per instance from the existing `text_reference`:

  - SHORT:   first sentence only.
  - NOISY:   original text plus filler/distractor sentence(s).
  - WRONG:   replaced with text that does not describe the expected outcome
             (template-based, instance-stable for reproducibility).

Output: data/verification_dataset_textref_variants.jsonl
Each line carries:
  - instance_id
  - text_reference_full
  - text_reference_short
  - text_reference_noisy
  - text_reference_wrong

This script is purely transformational; it does NOT call any LLM API.
Use scripts/run_text_ref_ablation.py to actually run condition B with these variants.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# the base dataset carries no text_reference; variants derive from the
# generated reference set produced by scripts/regenerate_text_refs.py
SRC = DATA_DIR / "verification_dataset_textref.jsonl"
DST = DATA_DIR / "verification_dataset_textref_variants.jsonl"

NOISY_FILLER = (
    " Note: the page may include a sidebar with promotional banners, "
    "a footer with site navigation, and timestamps that vary by load."
)


def first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def make_short(text: str) -> str:
    return first_sentence(text)


def make_noisy(text: str) -> str:
    return (text or "").strip() + NOISY_FILLER


def make_wrong(instance_id: str) -> str:
    # Deterministic, instance-stable wrong reference. Reviewer-friendly because
    # it is clearly off-topic and not derived from the task itself.
    return (
        "After successful completion, the page should display a help center article "
        "with troubleshooting steps unrelated to the original task."
        f" (variant_seed={instance_id})"
    )


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source dataset not found: {SRC}")

    n_in = 0
    n_out = 0
    with open(SRC) as fin, open(DST, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            iid = rec.get("instance_id", "")
            full = rec.get("text_reference") or ""
            if not full:
                continue

            out = {
                "instance_id": iid,
                "text_reference_full": full,
                "text_reference_short": make_short(full),
                "text_reference_noisy": make_noisy(full),
                "text_reference_wrong": make_wrong(iid),
            }
            fout.write(json.dumps(out) + "\n")
            n_out += 1

    print(f"[OK] Read {n_in} dataset records, wrote {n_out} variant records to: {DST}")


if __name__ == "__main__":
    main()
