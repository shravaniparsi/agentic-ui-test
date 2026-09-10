#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper for verification experiments and analysis.
# This script intentionally supports dry-run mode to avoid accidental API spend.

DATASET=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_pipeline.sh --dataset data/verification_dataset.jsonl [--dry-run]

Options:
  --dataset   Path to verification dataset JSONL (required)
  --dry-run   Print commands without executing model calls
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Error: --dataset is required."
  usage
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Error: dataset not found at '$DATASET'"
  exit 1
fi

if ! python3 -c "import tqdm" >/dev/null 2>&1; then
  echo "Error: missing Python dependency 'tqdm' in current environment."
  echo "Run: python3 -m pip install -r requirements.txt"
  exit 1
fi

echo "============================================================"
echo "Visual Self-Verification Pipeline"
echo "Dataset: $DATASET"
echo "Dry run: $DRY_RUN"
echo "============================================================"

VERIFY_CMD=(python3 verify.py --model all --condition all --dataset "$DATASET")
if [[ "$DRY_RUN" -eq 1 ]]; then
  VERIFY_CMD+=(--dry-run)
fi

echo ""
echo "[1/4] Verification runs"
echo "Command: ${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "Dry run complete. Skipping analysis steps."
  exit 0
fi

echo ""
echo "[2/4] Supplement tables (S1, S2, S2b, T1/S3 + matched task-ID lists)"
python3 scripts/recompute_final.py

echo ""
echo "[3/4] Revision analyses (confusion matrices, ITT, calibration, cost, Bonferroni)"
python3 scripts/full_revision_analysis.py
python3 scripts/revision_analysis.py
python3 scripts/threshold_policy_analysis.py
python3 scripts/run_nonllm_baselines.py --search

echo ""
echo "[4/4] Manuscript figures"
python3 scripts/make_manuscript_figures.py

echo ""
echo "Pipeline complete."
echo "Outputs:"
echo "  - {model}_{condition}.jsonl   (repo root)"
echo "  - results/S1_paired_tests.csv, S2_*.csv, T1_and_S3_error_rates.csv"
echo "  - results/S1_task_ids/"
echo "  - figures/"
