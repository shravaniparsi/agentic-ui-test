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
echo "[2/4] Compute aggregate metrics"
python3 analysis/compute_metrics.py --results-dir results/

echo ""
echo "[3/4] Generate standard figures"
python3 analysis/plot_figures.py --results-dir results/

echo ""
echo "[4/4] Run deep analysis package"
python3 scripts/deep_analysis.py

echo ""
echo "Pipeline complete."
echo "Outputs:"
echo "  - results/analysis.csv"
echo "  - results/{model}_{condition}.jsonl"
echo "  - figures/"
