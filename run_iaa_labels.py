#!/usr/bin/env python3
"""
Inter-annotator agreement (IAA) scaffold for failure-type labels.

Reviewer concern (Critique issue #2): The failure-type categorization
(obvious / deceptive / partial) underpins Section 5.3 and Table 3 but is
labeled by the authors with no reported IAA. This script makes it cheap to
produce a defensible IAA number.

Usage:
    # 1. Sample 100 failure instances for double labeling.
    python3 scripts/run_iaa_labels.py sample --n 100 \
        --out data/iaa_sample.jsonl

    # 2. Two annotators independently label data/iaa_sample.jsonl, producing
    #    data/iaa_labels_A.jsonl and data/iaa_labels_B.jsonl. Each line is:
    #      {"instance_id": "...", "label": "obvious|deceptive|partial"}

    # 3. Compute Cohen's kappa.
    python3 scripts/run_iaa_labels.py kappa \
        --a data/iaa_labels_A.jsonl --b data/iaa_labels_B.jsonl

This script is dependency-light: no API calls, no network. Cohen's kappa
implementation is inlined so the script also runs without scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

LABELS = ["obvious", "deceptive", "partial"]


def load_dataset() -> list[dict]:
    path = DATA_DIR / "verification_dataset.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cmd_sample(args):
    random.seed(args.seed)
    rows = load_dataset()
    failures = [r for r in rows if r.get("ground_truth") == "FAILURE"]
    if len(failures) < args.n:
        print(f"warning: only {len(failures)} failure instances available; sampling all",
              file=sys.stderr)
        sample = failures
    else:
        sample = random.sample(failures, args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in sample:
            row = {
                "instance_id": r["instance_id"],
                "task_text": r.get("task_text", ""),
                "screenshot": r.get("screenshot", ""),
                "domain": r.get("domain", ""),
                "label": "",  # to be filled by annotator
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(sample)} failure instances to {out_path}")
    print("Each annotator should fill the 'label' field with one of:")
    print(f"  {LABELS}")
    print("Save as data/iaa_labels_A.jsonl and data/iaa_labels_B.jsonl, then run 'kappa'.")


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> tuple[float, dict]:
    """Cohen's kappa for two labelers over the same items.

    Returns (kappa, {"po": observed_agreement, "pe": expected_agreement,
                     "n": n, "by_label_agreement": ..., "confusion": {...}})
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0, {"po": 0.0, "pe": 0.0, "n": 0, "confusion": {}}
    classes = sorted(set(labels_a) | set(labels_b))
    confusion = {ca: Counter() for ca in classes}
    for a, b in zip(labels_a, labels_b):
        confusion[a][b] += 1
    po = sum(confusion[c][c] for c in classes) / n
    cnt_a = Counter(labels_a)
    cnt_b = Counter(labels_b)
    pe = sum((cnt_a[c] / n) * (cnt_b[c] / n) for c in classes)
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    by_class = {c: confusion[c][c] / max(1, cnt_a[c]) for c in classes}
    return kappa, {
        "po": round(po, 4),
        "pe": round(pe, 4),
        "n": n,
        "by_label_agreement": {c: round(v, 3) for c, v in by_class.items()},
        "confusion": {ca: dict(confusion[ca]) for ca in classes},
    }


def cmd_kappa(args):
    def load_labels(path):
        out = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("label"):
                    out[r["instance_id"]] = r["label"]
        return out

    a = load_labels(args.a)
    b = load_labels(args.b)
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        print("error: no common labeled instances; check inputs", file=sys.stderr)
        sys.exit(2)
    la = [a[i] for i in common]
    lb = [b[i] for i in common]
    kappa, info = cohens_kappa(la, lb)
    print(f"Cohen's kappa: {kappa:.3f}  (n={info['n']}, p_observed={info['po']}, p_expected={info['pe']})")
    print("Per-class A-then-B agreement:", json.dumps(info["by_label_agreement"], indent=2))
    print("Confusion matrix (rows=A, cols=B):", json.dumps(info["confusion"], indent=2))
    interpretation = (
        "almost perfect" if kappa >= 0.81 else
        "substantial" if kappa >= 0.61 else
        "moderate" if kappa >= 0.41 else
        "fair" if kappa >= 0.21 else
        "slight" if kappa >= 0.0 else "worse than chance"
    )
    print(f"Landis & Koch (1977) interpretation: {interpretation}")
    print()
    print("To report in the manuscript Section 4.1 / Appendix E:")
    print(f"  Two authors independently labeled n={info['n']} failure instances.")
    print(f"  Cohen's kappa = {kappa:.2f} ({interpretation} agreement).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Sample failure instances for double labeling")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/iaa_sample.jsonl")
    s.set_defaults(func=cmd_sample)

    k = sub.add_parser("kappa", help="Compute Cohen's kappa from two label files")
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    k.set_defaults(func=cmd_kappa)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
