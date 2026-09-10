#!/usr/bin/env python3
"""
Threshold Policy Analysis (zero API cost).

Derives deployment-oriented metrics from already-collected verification results.
Treats each model x condition as a verifier and reports the trade-off between
auto-decision coverage and accuracy under a confidence threshold policy.

Policy:
  - If verifier confidence >= threshold:
      auto-accept verifier verdict (SUCCESS or FAILURE).
  - Else: escalate to human review.

For each (model, condition, threshold) we compute:
  - coverage:       fraction of instances auto-decided
  - auto_accuracy:  accuracy on the auto-decided subset
  - escalation:     fraction escalated to human (1 - coverage)

Outputs:
  - results/threshold_policy.csv
  - figures/fig7_threshold_policy.pdf
  - figures/fig7_threshold_policy.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "claude-sonnet-4", "gemini-3.6-flash"]
CONDITIONS = ["A", "B", "C", "D"]
THRESHOLDS = list(range(1, 11))  # 1..10


def load_results(path: Path) -> list[dict]:
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


def valid_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("verdict") in ("SUCCESS", "FAILURE")]


def policy_metrics(records: list[dict], threshold: int) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "coverage": 0.0, "auto_accuracy": 0.0, "escalation": 0.0,
                "auto_n": 0, "auto_correct": 0}
    auto = [r for r in records if (r.get("confidence") or 0) >= threshold]
    auto_n = len(auto)
    auto_correct = sum(1 for r in auto if r.get("verdict") == r.get("ground_truth"))
    coverage = auto_n / n
    auto_accuracy = (auto_correct / auto_n) if auto_n else 0.0
    escalation = 1.0 - coverage
    return {
        "n": n,
        "coverage": coverage,
        "auto_accuracy": auto_accuracy,
        "escalation": escalation,
        "auto_n": auto_n,
        "auto_correct": auto_correct,
    }


def run_table() -> list[dict]:
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            recs = valid_records(load_results(PROJECT_ROOT / f"{model}_{cond}.jsonl"))
            for t in THRESHOLDS:
                m = policy_metrics(recs, t)
                rows.append({
                    "model": model,
                    "condition": cond,
                    "threshold": t,
                    "n": m["n"],
                    "auto_n": m["auto_n"],
                    "auto_correct": m["auto_correct"],
                    "coverage": round(m["coverage"], 4),
                    "auto_accuracy": round(m["auto_accuracy"], 4),
                    "escalation": round(m["escalation"], 4),
                })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["model", "condition", "threshold", "n", "auto_n", "auto_correct",
              "coverage", "auto_accuracy", "escalation"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def maybe_plot(rows: list[dict]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        print("[INFO] matplotlib not available; skipping figure generation.")
        return None

    by_model_cond = {}
    for r in rows:
        by_model_cond.setdefault((r["model"], r["condition"]), []).append(r)

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 4), sharey=True)
    if len(CONDITIONS) == 1:
        axes = [axes]
    for ax, cond in zip(axes, CONDITIONS):
        for model in MODELS:
            series = by_model_cond.get((model, cond), [])
            if not series:
                continue
            series_sorted = sorted(series, key=lambda x: x["coverage"])
            xs = [s["coverage"] for s in series_sorted]
            ys = [s["auto_accuracy"] for s in series_sorted]
            ax.plot(xs, ys, marker="o", label=model)
        ax.set_title(f"Condition {cond}")
        ax.set_xlabel("Coverage (fraction auto-decided)")
        ax.set_ylabel("Auto-decision accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Confidence-Threshold Deployment Policy", y=1.02)
    fig.tight_layout()
    pdf_path = FIGURES_DIR / "fig7_threshold_policy.pdf"
    png_path = FIGURES_DIR / "fig7_threshold_policy.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")
    return pdf_path


def main():
    rows = run_table()
    out_csv = RESULTS_DIR / "threshold_policy.csv"
    write_csv(rows, out_csv)
    print(f"[OK] Wrote: {out_csv} ({len(rows)} rows)")
    maybe_plot(rows)


if __name__ == "__main__":
    main()
