#!/usr/bin/env python3
"""
make_manuscript_figures.py - Regenerate the manuscript's data-bearing figures.

Produces, into figures/:
  fig2_accuracy.png        accuracy by model and condition (with majority baselines)
  fig3_f1.png              F1 by model and condition
  fig4_failure_type.png    accuracy on failure instances by type, Conditions A and B
  fig7_confusion.png       confusion matrices, 5 models x 4 conditions

Reads the per-instance JSONL result files in the repository root, so the figures
always match the tables produced by scripts/recompute_final.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "claude-sonnet-4", "gemini-3.6-flash"]
LABELS = {"gpt-4.1-nano": "GPT-4.1 Nano", "gpt-4.1-mini": "GPT-4.1 Mini", "gpt-4.1": "GPT-4.1",
          "claude-sonnet-4": "Claude Sonnet 4", "gemini-3.6-flash": "Gemini 3.6 Flash"}
CONDS = ["A", "B", "C", "D"]
CONDLAB = {"A": "A (No Ref)", "B": "B (Text)", "C": "C (Visual)", "D": "D (Dual)"}


def load(model, cond):
    p = ROOT / f"{model}_{cond}.jsonl"
    return [r for r in (json.loads(l) for l in open(p) if l.strip())
            if r["verdict"] in ("SUCCESS", "FAILURE")]


def confusion(recs):
    tp = sum(1 for r in recs if r["ground_truth"] == "SUCCESS" and r["verdict"] == "SUCCESS")
    fp = sum(1 for r in recs if r["ground_truth"] == "FAILURE" and r["verdict"] == "SUCCESS")
    tn = sum(1 for r in recs if r["ground_truth"] == "FAILURE" and r["verdict"] == "FAILURE")
    fn = sum(1 for r in recs if r["ground_truth"] == "SUCCESS" and r["verdict"] == "FAILURE")
    return tp, fp, tn, fn


def acc_f1(recs):
    tp, fp, tn, fn = confusion(recs)
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (tp + tn) / n, f1


DATA = {(m, c): load(m, c) for m in MODELS for c in CONDS}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _grouped_bar(values, ylabel, title, out, baselines=None):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w, xs = 0.2, np.arange(len(MODELS))
    for k, c in enumerate(CONDS):
        ax.bar(xs + (k - 1.5) * w, [values[(m, c)] for m in MODELS], w,
               label=CONDLAB[c], color=COLORS[k])
    if baselines:
        for i, m in enumerate(MODELS):
            ax.hlines(baselines[m], i - 0.42, i + 0.42, colors="grey",
                      linestyles="--", linewidth=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig2_and_3():
    accs = {k: acc_f1(v)[0] for k, v in DATA.items()}
    f1s = {k: acc_f1(v)[1] for k, v in DATA.items()}
    base = {}
    for m in MODELS:
        recs = DATA[(m, "A")]
        base[m] = sum(1 for r in recs if r["ground_truth"] == "FAILURE") / len(recs)
    _grouped_bar(accs, "Accuracy", "Verification Accuracy by Model and Condition",
                 FIG / "fig2_accuracy.png", baselines=base)
    _grouped_bar(f1s, "F1 (SUCCESS class)", "F1 Score by Model and Condition",
                 FIG / "fig3_f1.png")


def fig4_failure_type():
    types = ["obvious", "deceptive", "partial"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, t in zip(axes, types):
        va, vb = [], []
        for m in MODELS:
            a = {r["instance_id"]: r for r in DATA[(m, "A")]
                 if r.get("failure_type") == t and r["ground_truth"] == "FAILURE"}
            b = {r["instance_id"]: r for r in DATA[(m, "B")]
                 if r.get("failure_type") == t and r["ground_truth"] == "FAILURE"}
            ids = set(a) & set(b)
            va.append(sum(1 for i in ids if a[i]["verdict"] == "FAILURE") / len(ids))
            vb.append(sum(1 for i in ids if b[i]["verdict"] == "FAILURE") / len(ids))
        xs, w = np.arange(len(MODELS)), 0.36
        ax.bar(xs - w / 2, va, w, label="A (No Ref)", color=COLORS[0])
        ax.bar(xs + w / 2, vb, w, label="B (Text)", color=COLORS[1])
        ax.set_title(t.capitalize(), fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=30,
                           ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy on failures")
    axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy by Failure Type and Condition", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_failure_type.png", dpi=200)
    plt.close(fig)


def fig7_confusion():
    fig, axes = plt.subplots(len(MODELS), 4, figsize=(11, 13))
    fig.suptitle("Confusion Matrices", fontsize=14)
    for i, m in enumerate(MODELS):
        for j, c in enumerate(CONDS):
            tp, fp, tn, fn = confusion(DATA[(m, c)])
            mat = np.array([[tn, fp], [fn, tp]])
            ax = axes[i][j]
            ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
            for r in range(2):
                for cc in range(2):
                    ax.text(cc, r, str(mat[r][cc]), ha="center", va="center",
                            color="white" if mat[r][cc] > mat.max() * 0.55 else "black",
                            fontsize=10)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred F", "Pred S"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["GT F", "GT S"], fontsize=8)
            if i == 0:
                ax.set_title(f"Cond {c}", fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[m], fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig7_confusion.png", dpi=150)
    plt.close(fig)


def fig1_framework():
    """Schematic of the verification framework and the four conditions."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.45", linewidth=1.2)
    ax.text(0.11, 0.80, "Task description\n$t$", ha="center", va="center",
            bbox={**box, "facecolor": "#EAF0F8", "edgecolor": "#4C72B0"}, fontsize=9)
    ax.text(0.11, 0.46, "Agent final\nscreenshot $s$", ha="center", va="center",
            bbox={**box, "facecolor": "#EAF0F8", "edgecolor": "#4C72B0"}, fontsize=9)
    ax.text(0.11, 0.12, "Reference\n(text and/or image)", ha="center", va="center",
            bbox={**box, "facecolor": "#FBEDE2", "edgecolor": "#DD8452"}, fontsize=9)
    ax.text(0.40, 0.46, "Prompt\nconstruction", ha="center", va="center",
            bbox={**box, "facecolor": "#FFFFFF", "edgecolor": "#333333"}, fontsize=9)
    ax.text(0.62, 0.46, "LMM\nverifier", ha="center", va="center",
            bbox={**box, "facecolor": "#E8F2EA", "edgecolor": "#55A868"}, fontsize=9)
    ax.text(0.86, 0.46, "verdict $\\in$ {SUCCESS,\nFAILURE}\nconfidence 1-10\nreasoning",
            ha="center", va="center",
            bbox={**box, "facecolor": "#FFFFFF", "edgecolor": "#333333"}, fontsize=8.5)
    for y in (0.80, 0.46, 0.12):
        ax.annotate("", xy=(0.30, 0.46), xytext=(0.19, y),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
    for x0, x1 in ((0.50, 0.545), (0.70, 0.755)):
        ax.annotate("", xy=(x1, 0.46), xytext=(x0, 0.46),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
    conds = ("A: no reference      B: text reference      "
             "C: visual reference      D: text + visual")
    ax.text(0.5, -0.04, conds, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#999999"))
    ax.set_xlim(0.02, 0.98); ax.set_ylim(-0.10, 0.94)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_framework.png", dpi=200)
    plt.close(fig)


def fig5_reliability_ab():
    """Reliability diagrams for Conditions A and B across the five verifiers."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cond in zip(axes, ["A", "B"]):
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        for k, m in enumerate(MODELS):
            recs = [r for r in DATA[(m, cond)] if r.get("confidence") is not None]
            xs, ys = [], []
            for lo in range(1, 11):
                b = [r for r in recs if r["confidence"] == lo]
                if len(b) < 5:
                    continue
                xs.append(lo / 10.0)
                ys.append(sum(1 for r in b if r["verdict"] == r["ground_truth"]) / len(b))
            if xs:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=LABELS[m],
                        color=(COLORS + ["#8172B3"])[k])
        ax.set_title(f"Condition {cond} ({CONDLAB[cond].split('(')[1][:-1]})", fontsize=10)
        ax.set_xlabel("Self-reported confidence")
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Empirical accuracy")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Reliability Diagrams, Conditions A and B", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig5_reliability_ab.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_framework(); fig2_and_3(); fig4_failure_type()
    fig5_reliability_ab(); fig7_confusion()
    print("wrote fig1, fig2, fig3, fig4, fig5, fig7")
