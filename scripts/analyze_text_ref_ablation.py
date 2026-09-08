#!/usr/bin/env python3
"""
Summarize text-reference quality ablation results.

Reads `results/textref_ablation_{model}_{variant}.jsonl` files (if present),
plus existing Condition B results as the `full` baseline (subset of same instance ids),
and writes a comparison CSV plus a console table.

Usage:
  python3 scripts/analyze_text_ref_ablation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

VARIANTS = ["full", "short", "noisy", "wrong"]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def metrics(records: list[dict]) -> dict:
    valid = [r for r in records if r.get("verdict") in ("SUCCESS", "FAILURE")]
    n = len(valid)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    tp = sum(1 for r in valid if r["ground_truth"] == "SUCCESS" and r["verdict"] == "SUCCESS")
    fp = sum(1 for r in valid if r["ground_truth"] == "FAILURE" and r["verdict"] == "SUCCESS")
    tn = sum(1 for r in valid if r["ground_truth"] == "FAILURE" and r["verdict"] == "FAILURE")
    fn = sum(1 for r in valid if r["ground_truth"] == "SUCCESS" and r["verdict"] == "FAILURE")
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": n, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def find_models() -> list[str]:
    models = set()
    for p in RESULTS_DIR.glob("textref_ablation_*_*.jsonl"):
        stem = p.stem  # textref_ablation_{model}_{variant}
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        # model can contain hyphens; variant is the last token
        variant = parts[-1]
        if variant not in VARIANTS:
            continue
        model = "_".join(parts[2:-1])
        # the file name uses the model name as-is (with hyphens), so reconstruct accordingly
        models.add(model)
    return sorted(models)


def baseline_records_for_subset(model: str, ablation_ids: set[str]) -> list[dict]:
    base = load_jsonl(RESULTS_DIR / f"{model}_B.jsonl")
    return [r for r in base if r.get("instance_id") in ablation_ids]


def main() -> None:
    models = find_models()
    rows = []
    print("=" * 70)
    print("  TEXT-REF ABLATION SUMMARY")
    print("=" * 70)

    if not models:
        print("  No ablation result files found yet.")
        print("  Run: python3 scripts/run_text_ref_ablation.py")
        return

    for model in models:
        # collect ablation instance ids from any variant file (they share a subset)
        ablation_ids: set[str] = set()
        for variant in VARIANTS:
            if variant == "full":
                continue
            path = RESULTS_DIR / f"textref_ablation_{model}_{variant}.jsonl"
            for r in load_jsonl(path):
                ablation_ids.add(r["instance_id"])

        # full baseline restricted to ablation subset
        full_recs = baseline_records_for_subset(model, ablation_ids)
        full_m = metrics(full_recs)
        rows.append({"model": model, "variant": "full", **full_m})
        print(f"\n  {model}")
        print(f"    full   n={full_m['n']:>4} acc={full_m['accuracy']:.3f} f1={full_m['f1']:.3f}")

        for variant in ["short", "noisy", "wrong"]:
            path = RESULTS_DIR / f"textref_ablation_{model}_{variant}.jsonl"
            recs = load_jsonl(path)
            m = metrics(recs)
            rows.append({"model": model, "variant": variant, **m})
            print(f"    {variant:<6} n={m['n']:>4} acc={m['accuracy']:.3f} f1={m['f1']:.3f}")

    out_csv = RESULTS_DIR / "textref_ablation_summary.csv"
    fields = ["model", "variant", "n", "accuracy", "f1", "precision", "recall"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
