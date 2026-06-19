#!/usr/bin/env python3
"""
Smoke test for repository artifact integrity.

Verifies that all required files for the published package exist and that
key data files are readable and well-formed. Does NOT call any LLM API.

Exit code:
  0 = all checks passed
  1 = one or more checks failed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = ROOT / "analysis"
PROMPTS_DIR = ROOT / "prompts"
BASELINES_DIR = ROOT / "baselines"

REQUIRED_TOP_FILES = [
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "CITATION.cff",
    ROOT / "config.py",
    ROOT / "verify.py",
    ROOT / "llm_clients.py",
    ROOT / "requirements.txt",
    ROOT / ".env.example",
]

REQUIRED_DIRS = [
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    SCRIPTS_DIR,
    ANALYSIS_DIR,
    PROMPTS_DIR,
    BASELINES_DIR,
]

REQUIRED_DATA_FILES = [
    DATA_DIR / "verification_dataset.jsonl",
    DATA_DIR / "verification_dataset_textref_variants.jsonl",
    DATA_DIR / "README.md",
]

REQUIRED_RESULT_FILES = [
    RESULTS_DIR / "analysis.csv",
    RESULTS_DIR / "threshold_policy.csv",
]

OPTIONAL_REVISION_FILES = [
    RESULTS_DIR / "revision_aligned_subset.csv",
    RESULTS_DIR / "revision_calibration_extended.csv",
    RESULTS_DIR / "revision_confidence_histogram.csv",
    RESULTS_DIR / "revision_cost_latency.csv",
    RESULTS_DIR / "revision_bonferroni.csv",
    RESULTS_DIR / "revision_summary.md",
    FIGURES_DIR / "fig8_reliability_diagrams.pdf",
    FIGURES_DIR / "fig8_reliability_diagrams.png",
]

REQUIRED_FIGURES = [
    FIGURES_DIR / "fig1_accuracy_by_condition.pdf",
    FIGURES_DIR / "fig1_accuracy_by_condition.png",
    FIGURES_DIR / "fig2_f1_by_condition.pdf",
    FIGURES_DIR / "fig3_failure_type_heatmap.pdf",
    FIGURES_DIR / "fig4_calibration.pdf",
    FIGURES_DIR / "fig5_confusion_matrices.pdf",
    FIGURES_DIR / "fig6_framework_overview.pdf",
    FIGURES_DIR / "fig7_threshold_policy.pdf",
    FIGURES_DIR / "fig7_threshold_policy.png",
]

REQUIRED_SCRIPTS = [
    SCRIPTS_DIR / "run_all_pipeline.sh",
    SCRIPTS_DIR / "threshold_policy_analysis.py",
    SCRIPTS_DIR / "generate_text_ref_variants.py",
    SCRIPTS_DIR / "run_text_ref_ablation.py",
    SCRIPTS_DIR / "analyze_text_ref_ablation.py",
    SCRIPTS_DIR / "deep_analysis.py",
    SCRIPTS_DIR / "revision_analysis.py",
    SCRIPTS_DIR / "run_iaa_labels.py",
    SCRIPTS_DIR / "run_open_weight_baseline.py",
    SCRIPTS_DIR / "regenerate_text_refs.py",
    SCRIPTS_DIR / "collect_human_baseline.py",
]


def check_exists(items: list[Path], label: str, errors: list[str]) -> None:
    missing = [str(p) for p in items if not p.exists()]
    if missing:
        errors.append(f"{label} missing: {missing}")


def check_jsonl(path: Path, min_records: int, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"JSONL missing: {path}")
        return
    n_ok = 0
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n_ok += 1
            except json.JSONDecodeError:
                n_bad += 1
    if n_ok < min_records:
        errors.append(f"JSONL underfilled: {path} ok={n_ok} expected>={min_records}")
    if n_bad > 0:
        errors.append(f"JSONL malformed lines: {path} bad={n_bad}")


def check_csv_header(path: Path, expected_columns: list[str], errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"CSV missing: {path}")
        return
    with open(path) as f:
        first = f.readline().strip().split(",")
    for col in expected_columns:
        if col not in first:
            errors.append(f"CSV missing column '{col}' in {path}")


def main() -> int:
    errors: list[str] = []

    check_exists(REQUIRED_DIRS, "Directories", errors)
    check_exists(REQUIRED_TOP_FILES, "Top-level files", errors)
    check_exists(REQUIRED_DATA_FILES, "Data files", errors)
    check_exists(REQUIRED_RESULT_FILES, "Result files", errors)
    check_exists(REQUIRED_FIGURES, "Figures", errors)
    check_exists(REQUIRED_SCRIPTS, "Scripts", errors)

    check_jsonl(DATA_DIR / "verification_dataset.jsonl", min_records=900, errors=errors)
    check_jsonl(DATA_DIR / "verification_dataset_textref_variants.jsonl", min_records=900, errors=errors)
    check_csv_header(
        RESULTS_DIR / "analysis.csv",
        expected_columns=["model", "condition", "accuracy", "f1"],
        errors=errors,
    )
    check_csv_header(
        RESULTS_DIR / "threshold_policy.csv",
        expected_columns=["model", "condition", "threshold", "coverage", "auto_accuracy"],
        errors=errors,
    )

    # Optional revision artifacts -- warn but do not fail if missing.
    missing_optional = [str(p) for p in OPTIONAL_REVISION_FILES if not p.exists()]

    print("=" * 60)
    print("REPO SMOKE TEST")
    print("=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} issue(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: all required artifacts exist and are well-formed.")
    if missing_optional:
        print(f"NOTE: {len(missing_optional)} optional revision artifact(s) not present "
              "(run scripts/revision_analysis.py to generate):")
        for p in missing_optional:
            print(f"  - {p}")
    else:
        print("NOTE: all optional revision artifacts also present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
