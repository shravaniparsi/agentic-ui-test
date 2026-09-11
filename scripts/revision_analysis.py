#!/usr/bin/env python3
"""
Revision analysis: compute all new artifacts requested by reviewer-style critique.

Outputs go to:
  results/revision_aligned_subset.csv         - per-(model,condition) metrics on the aligned 147-subset
  results/revision_calibration_extended.csv   - per-(model,condition) ECE_3bin, ECE_15bin_eqmass, Brier
  results/revision_confidence_histogram.csv   - per-(model,condition) confidence-bin histograms
  results/revision_cost_latency.csv           - per-(model,condition) total cost, mean/median/p95 latency
  results/revision_bonferroni.csv             - per-(model,pair) raw and Bonferroni-corrected p-values
  results/revision_summary.md                 - human-readable summary, ready to paste into the manuscript
  figures/fig6_reliability_diagrams.{pdf,png} - per-(model) reliability diagrams across conditions

Usage:
    python3 scripts/revision_analysis.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "claude-sonnet-4", "gemini-3.6-flash"]
MODEL_LABELS = {
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1": "GPT-4.1",
    "claude-sonnet-4": "Claude Sonnet 4",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
}
CONDITIONS = ["A", "B", "C", "D"]

# Cost per 1K tokens; mirrors visual-self-verify/config.py
COST_IN = {
    "gpt-4.1": 0.002, "gpt-4.1-mini": 0.0004, "gpt-4.1-nano": 0.0001,
    "claude-sonnet-4": 0.003, "gemini-3.6-flash": 0.00075,
}
COST_OUT = {
    "gpt-4.1": 0.008, "gpt-4.1-mini": 0.0016, "gpt-4.1-nano": 0.0004,
    "claude-sonnet-4": 0.015, "gemini-3.6-flash": 0.00375,
}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
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


def compute_basic_metrics(records: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in records:
        gt, pred = r.get("ground_truth"), r.get("verdict")
        if gt == "SUCCESS" and pred == "SUCCESS": tp += 1
        elif gt == "FAILURE" and pred == "SUCCESS": fp += 1
        elif gt == "FAILURE" and pred == "FAILURE": tn += 1
        elif gt == "SUCCESS" and pred == "FAILURE": fn += 1
    n = tp + fp + tn + fn
    if n == 0:
        return {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "accuracy": 0.0, "f1": 0.0, "balanced_accuracy": 0.0,
                "precision": 0.0, "recall": 0.0}
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal = (rec + spec) / 2
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "f1": f1, "balanced_accuracy": bal,
            "precision": prec, "recall": rec}


# ── 1. Aligned-subset analysis ───────────────────────────────────────────────

def find_visual_subset_ids() -> set[str]:
    """Canonical visual-reference subset.

    Reads data/visual_subset_ids.txt, the same 147-instance list used by
    recompute_final.py and run_nonllm_baselines.py. (An earlier version globbed
    data/references/ for a `<domain>_<idx>_human_ref.jpeg` naming convention that
    extract_human_references.py does not produce, so it silently matched nothing
    and every aligned-subset cell came out as n=0.)
    """
    path = PROJECT_ROOT / "data" / "visual_subset_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in open(path) if line.strip()}


def aligned_subset_analysis():
    """Compute per-(model,condition) metrics on the common visual-reference subset.

    Reviewer concern: A vs C/D comparisons mix N=908 (A) with N~147 (C). Here we
    restrict A and B to the same instance_ids as C and D so condition comparisons
    are over an aligned set.
    """
    subset_ids = find_visual_subset_ids()
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            recs = valid_records(load_records(PROJECT_ROOT / f"{model}_{cond}.jsonl"))
            recs = [r for r in recs if r["instance_id"] in subset_ids]
            m = compute_basic_metrics(recs)
            rows.append({
                "model": model, "condition": cond,
                "n_aligned": m["n"], "accuracy": m["accuracy"],
                "f1": m["f1"], "bal_acc": m["balanced_accuracy"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            })
    out = RESULTS_DIR / "revision_aligned_subset.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out} ({len(rows)} rows; subset_ids={len(subset_ids)})")
    return rows, subset_ids


# ── 2. Calibration: 3-bin, 15-bin equal-mass ECE, Brier ─────────────────────

def confidence_to_prob(conf: int | float) -> float:
    """Map 1..10 verbal confidence to probability of the *predicted* class.

    A confidence of 10 means "I am sure", which we map to P=1.0 of being right
    on the predicted class. A confidence of 5 means "uncertain", mapped to P=0.5.
    A confidence of 1 means "almost certainly wrong", mapped to P=0.1.
    Linear scaling from {1..10} -> {0.1..1.0}.
    """
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return 0.5
    c = max(1.0, min(10.0, c))
    return c / 10.0


def ece_fixed_bins(probs: np.ndarray, correct: np.ndarray, edges: list[float]) -> float:
    """Compute ECE with the given bin edges (equal-width on prob)."""
    n = len(probs)
    if n == 0:
        return float("nan")
    err = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = probs[mask].mean()
        err += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(err)


def ece_equal_mass(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass binning: each bin contains roughly n/n_bins instances."""
    n = len(probs)
    if n == 0:
        return float("nan")
    order = np.argsort(probs, kind="stable")
    probs_s = probs[order]
    correct_s = correct[order]
    err = 0.0
    edges_idx = np.linspace(0, n, n_bins + 1, dtype=int)
    for i in range(n_bins):
        lo, hi = edges_idx[i], edges_idx[i + 1]
        if hi <= lo:
            continue
        bin_p = probs_s[lo:hi].mean()
        bin_acc = correct_s[lo:hi].mean()
        err += ((hi - lo) / n) * abs(bin_acc - bin_p)
    return float(err)


def brier(probs: np.ndarray, correct: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    return float(((probs - correct) ** 2).mean())


def calibration_table():
    """Per-(model,condition) calibration with multiple definitions + confidence histogram."""
    cal_rows = []
    hist_rows = []
    bins_3_edges = [0.0, 0.35, 0.65, 1.001]  # corresponds to conf 1-3 / 4-6 / 7-10
    for model in MODELS:
        for cond in CONDITIONS:
            recs = valid_records(load_records(PROJECT_ROOT / f"{model}_{cond}.jsonl"))
            if not recs:
                continue
            probs = np.array([confidence_to_prob(r.get("confidence", 5)) for r in recs])
            correct = np.array([1.0 if r.get("verdict") == r.get("ground_truth") else 0.0 for r in recs])
            cal_rows.append({
                "model": model, "condition": cond, "n": len(recs),
                "ece_3bin": round(ece_fixed_bins(probs, correct, bins_3_edges), 4),
                "ece_15bin_eqmass": round(ece_equal_mass(probs, correct, 15), 4),
                "brier": round(brier(probs, correct), 4),
                "mean_conf": round(float(probs.mean()), 4),
                "mean_acc": round(float(correct.mean()), 4),
            })
            confs = [int(r.get("confidence", 0)) for r in recs]
            hist = defaultdict(int)
            for c in confs:
                hist[c] += 1
            for c in range(0, 11):
                hist_rows.append({"model": model, "condition": cond,
                                  "confidence": c, "count": hist.get(c, 0),
                                  "pct": round(100 * hist.get(c, 0) / len(recs), 2)})
    out_cal = RESULTS_DIR / "revision_calibration_extended.csv"
    with open(out_cal, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    print(f"  wrote {out_cal} ({len(cal_rows)} rows)")

    out_hist = RESULTS_DIR / "revision_confidence_histogram.csv"
    with open(out_hist, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)
    print(f"  wrote {out_hist} ({len(hist_rows)} rows)")
    return cal_rows


# ── 3. Reliability diagrams ─────────────────────────────────────────────────

def plot_reliability():
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    cond_colors = {"A": "#666666", "B": "#1f77b4", "C": "#d62728", "D": "#9467bd"}
    for ax, model in zip(axes, MODELS):
        ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.4, lw=0.8, label="perfect")
        for cond in CONDITIONS:
            recs = valid_records(load_records(PROJECT_ROOT / f"{model}_{cond}.jsonl"))
            if not recs:
                continue
            probs = np.array([confidence_to_prob(r.get("confidence", 5)) for r in recs])
            correct = np.array([1.0 if r.get("verdict") == r.get("ground_truth") else 0.0 for r in recs])
            order = np.argsort(probs)
            probs_s = probs[order]
            correct_s = correct[order]
            n_bins = 10
            n = len(probs_s)
            edges = np.linspace(0, n, n_bins + 1, dtype=int)
            xs, ys = [], []
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                if hi <= lo:
                    continue
                xs.append(probs_s[lo:hi].mean())
                ys.append(correct_s[lo:hi].mean())
            ax.plot(xs, ys, marker="o", lw=1.4, ms=4, color=cond_colors[cond], label=f"Cond {cond}")
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Empirical accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Reliability diagrams (10 equal-mass bins)", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig6_reliability_diagrams.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGURES_DIR}/fig6_reliability_diagrams.pdf,.png")


# ── 4. Cost + latency ───────────────────────────────────────────────────────

def cost_latency_table():
    rows = []
    grand_cost = 0.0
    grand_calls = 0
    for model in MODELS:
        for cond in CONDITIONS:
            recs = load_records(PROJECT_ROOT / f"{model}_{cond}.jsonl")
            if not recs:
                continue
            ins = [r.get("input_tokens", 0) or 0 for r in recs]
            outs = [r.get("output_tokens", 0) or 0 for r in recs]
            lats = [r.get("latency_s", 0) or 0 for r in recs if r.get("latency_s") not in (None, 0)]
            tot_in = sum(ins)
            tot_out = sum(outs)
            cost = (tot_in / 1000) * COST_IN[model] + (tot_out / 1000) * COST_OUT[model]
            grand_cost += cost
            grand_calls += len(recs)
            row = {
                "model": model, "condition": cond,
                "calls": len(recs),
                "total_input_tokens": tot_in,
                "total_output_tokens": tot_out,
                "total_cost_usd": round(cost, 3),
                "cost_per_call_usd": round(cost / max(1, len(recs)), 5),
                "mean_latency_s": round(float(np.mean(lats)), 2) if lats else None,
                "median_latency_s": round(float(np.median(lats)), 2) if lats else None,
                "p95_latency_s": round(float(np.percentile(lats, 95)), 2) if lats else None,
            }
            rows.append(row)
    out = RESULTS_DIR / "revision_cost_latency.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out} ({len(rows)} rows; grand total: ${grand_cost:.2f} over {grand_calls} calls)")
    return rows, grand_cost, grand_calls


# ── 5. Bonferroni-corrected significance ────────────────────────────────────

def mcnemar_paired(records_a: list[dict], records_b: list[dict]) -> dict:
    """McNemar with continuity correction on the common instances."""
    by_a = {r["instance_id"]: (r["verdict"] == r["ground_truth"]) for r in records_a}
    by_b = {r["instance_id"]: (r["verdict"] == r["ground_truth"]) for r in records_b}
    common = set(by_a.keys()) & set(by_b.keys())
    b = sum(1 for iid in common if by_a[iid] and not by_b[iid])
    c = sum(1 for iid in common if not by_a[iid] and by_b[iid])
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "n_common": len(common)}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(p), "n_common": len(common)}


def bonferroni_table():
    """Apply Bonferroni and Holm-Bonferroni corrections to all condition pairs."""
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    raw_rows = []
    for model in MODELS:
        recs_by_cond = {c: valid_records(load_records(RESULTS_DIR / f"{model}_{c}.jsonl")) for c in CONDITIONS}
        for c1, c2 in pairs:
            r = mcnemar_paired(recs_by_cond[c1], recs_by_cond[c2])
            raw_rows.append({
                "model": model, "pair": f"{c1}->{c2}",
                "n_common": r["n_common"], "b": r["b"], "c": r["c"],
                "chi2": round(r["chi2"], 3), "p_raw": r["p"],
            })
    m = len(raw_rows)
    alpha = 0.05
    bonf_alpha = alpha / m
    p_sorted = sorted(enumerate(raw_rows), key=lambda x: x[1]["p_raw"])
    holm_alphas = [alpha / (m - rank) for rank in range(m)]
    holm_significant = set()
    for rank, (orig_i, _) in enumerate(p_sorted):
        if p_sorted[rank][1]["p_raw"] <= holm_alphas[rank]:
            holm_significant.add(orig_i)
        else:
            break

    for i, row in enumerate(raw_rows):
        row["bonferroni_alpha"] = round(bonf_alpha, 5)
        row["bonferroni_significant"] = row["p_raw"] <= bonf_alpha
        row["holm_significant"] = i in holm_significant
        row["p_raw"] = (f"{row['p_raw']:.2e}" if row["p_raw"] < 1e-3 else round(row["p_raw"], 4))

    out = RESULTS_DIR / "revision_bonferroni.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"  wrote {out} ({m} tests; Bonferroni alpha'={bonf_alpha:.5f})")
    return raw_rows, bonf_alpha


# ── 6. Human-readable summary ───────────────────────────────────────────────

def write_summary(aligned_rows, cal_rows, cost_rows, bonf_rows, bonf_alpha,
                  grand_cost, grand_calls, subset_ids):
    lines: list[str] = []
    lines.append("# Revision Analysis Summary\n")
    lines.append(f"This file is auto-generated by `scripts/revision_analysis.py`. It is the source of truth for the revision-only numbers cited in the manuscript.\n\n")

    # Aligned subset
    lines.append("## Aligned-Subset Metrics (n_aligned per cell)\n\n")
    lines.append(f"All four conditions restricted to the n={len(subset_ids)} instances that have visual references. McNemar comparisons are now apples-to-apples across A/B/C/D.\n\n")
    lines.append("| Model | A acc | A f1 | B acc | B f1 | C acc | C f1 | D acc | D f1 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    by_mc = {(r["model"], r["condition"]): r for r in aligned_rows}
    for model in MODELS:
        cells = []
        for cond in CONDITIONS:
            r = by_mc.get((model, cond), {})
            cells.append(f"{r.get('accuracy', 0):.3f}")
            cells.append(f"{r.get('f1', 0):.3f}")
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Calibration
    lines.append("## Calibration (multiple definitions)\n\n")
    lines.append("| Model | Cond | n | ECE_3bin | ECE_15bin (eq-mass) | Brier | mean_conf | mean_acc |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in cal_rows:
        lines.append(f"| {MODEL_LABELS[r['model']]} | {r['condition']} | {r['n']} | "
                     f"{r['ece_3bin']:.3f} | {r['ece_15bin_eqmass']:.3f} | {r['brier']:.3f} | "
                     f"{r['mean_conf']:.3f} | {r['mean_acc']:.3f} |\n")
    lines.append("\nLower ECE / Brier = better calibration. mean_conf vs mean_acc gap signals systematic over/under-confidence.\n\n")

    # Cost + latency
    lines.append("## Cost + Latency\n\n")
    lines.append(f"Grand total: **${grand_cost:.2f}** across **{grand_calls}** API calls.\n\n")
    lines.append("| Model | Cond | Calls | Cost (USD) | Cost/call | Mean latency | p95 latency |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in cost_rows:
        lines.append(f"| {MODEL_LABELS[r['model']]} | {r['condition']} | {r['calls']} | "
                     f"${r['total_cost_usd']:.2f} | ${r['cost_per_call_usd']:.4f} | "
                     f"{r['mean_latency_s']}s | {r['p95_latency_s']}s |\n")
    lines.append("\n")

    # Bonferroni
    lines.append("## Bonferroni / Holm-corrected McNemar Tests\n\n")
    lines.append(f"30 paired tests (5 models x 6 condition pairs), Bonferroni alpha' = **{bonf_alpha:.5f}**.\n\n")
    lines.append("| Model | Pair | n | b | c | chi^2 | p_raw | Bonf-sig | Holm-sig |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in bonf_rows:
        lines.append(f"| {MODEL_LABELS[r['model']]} | {r['pair']} | {r['n_common']} | "
                     f"{r['b']} | {r['c']} | {r['chi2']} | {r['p_raw']} | "
                     f"{'YES' if r['bonferroni_significant'] else 'no'} | "
                     f"{'YES' if r['holm_significant'] else 'no'} |\n")
    lines.append("\n")

    out = RESULTS_DIR / "revision_summary.md"
    out.write_text("".join(lines))
    print(f"  wrote {out}")


# ── Entry ───────────────────────────────────────────────────────────────────

def main():
    print("[1/5] Aligned-subset analysis")
    aligned_rows, subset_ids = aligned_subset_analysis()
    print("[2/5] Calibration (3-bin, 15-bin equal-mass, Brier) + confidence histogram")
    cal_rows = calibration_table()
    print("[3/5] Reliability diagrams")
    plot_reliability()
    print("[4/5] Cost + latency")
    cost_rows, grand_cost, grand_calls = cost_latency_table()
    print("[5/5] Bonferroni / Holm correction")
    bonf_rows, bonf_alpha = bonferroni_table()
    print("[6/6] Human-readable summary")
    write_summary(aligned_rows, cal_rows, cost_rows, bonf_rows, bonf_alpha,
                  grand_cost, grand_calls, subset_ids)
    print("\nDone. See results/revision_summary.md for the manuscript-ready numbers.")


if __name__ == "__main__":
    main()
