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


if __name__ == "__main__":
    fig2_and_3(); fig4_failure_type(); fig7_confusion()
    print("wrote fig2_accuracy.png, fig3_f1.png, fig4_failure_type.png, fig7_confusion.png")
