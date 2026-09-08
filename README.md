# Can LLM Web Agents Verify Their Own Work?

A controlled study of reference-augmented LMM self-verification for web agents across 909 VisualWebArena trajectories, four reference conditions, and five frontier closed-API models.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys
python verify.py --model gpt-4.1 --condition B --dataset data/verification_dataset.jsonl
```

## Repository Structure

```
├── config.py                  # Model configs, API keys, paths
├── verify.py                  # Core verification runner
├── llm_clients.py             # OpenAI, Anthropic, Gateway clients
├── requirements.txt
├── .env.example
├── data/
│   ├── verification_dataset.jsonl    # 909 instances
│   ├── screenshots/                  # Final-state screenshots
│   └── references/                   # 147 human reference screenshots
├── results/
│   ├── {model}_{condition}.jsonl     # Raw verification results
│   ├── analysis.csv                  # Aggregate metrics
│   ├── revision_full_analysis.csv    # All metrics incl. MCC, AUC-ROC
│   ├── revision_itt_analysis.csv     # Intent-to-treat vs per-protocol
│   ├── revision_fpr_by_failure_type.csv
│   ├── revision_mcnemar_verified.csv
│   └── revision_summary.md
├── figures/                    # Publication figures (PDF+PNG)
├── scripts/
│   ├── full_revision_analysis.py     # Comprehensive revision analysis
│   ├── revision_analysis.py          # Aligned-subset, calibration, Bonferroni
│   ├── threshold_policy_analysis.py  # Deployment policy curves
│   ├── regenerate_text_refs.py       # Cross-model ref regeneration
│   ├── run_iaa_labels.py             # Inter-annotator agreement
│   ├── run_open_weight_baseline.py   # Open-weight verifier baseline
│   └── run_all_pipeline.sh           # End-to-end pipeline
└── prompts/
    └── verification_prompts.py       # Condition A-D prompts
```

## Reproducing Results

```bash
# Full pipeline (dry-run first)
bash scripts/run_all_pipeline.sh --dataset data/verification_dataset.jsonl --dry-run

# Run all verification (requires API keys)
bash scripts/run_all_pipeline.sh --dataset data/verification_dataset.jsonl

# Regenerate text references with cross-model generator
python scripts/regenerate_text_refs.py --generator claude-sonnet-4 --dry-run

# Run comprehensive analysis (zero API cost)
python scripts/full_revision_analysis.py
```

## Citation

```bibtex
@article{thatikonda2026canllmwebagents,
  title={Can LLM Web Agents Verify Their Own Work?},
  author={Thatikonda, Vishwanath and Parsi, Shravani},
  journal={IEEE Access},
  year={2026},
  note={Manuscript ID: Access-2026-32332, under revision}
}
```

## License

This research code is released for academic replication purposes.
