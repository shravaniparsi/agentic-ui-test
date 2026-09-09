# Reproducibility Guide

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API Keys

Copy `.env.example` to `.env` and fill in:

- `OPENAI_API_KEY` — for GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano
- `ANTHROPIC_API_KEY` — for Claude Sonnet 4

## Data Preparation

1. Clone VisualWebArena: `git clone https://github.com/web-arena-x/visualwebarena`
2. Download GPT-4V + SoM agent trajectories from [Google Drive](https://drive.google.com/file/d/1-tKz5ByWa1-jwtejiFgxli8fZcBPZgAE/view?usp=sharing)
3. Run trajectory parser: `python scripts/parse_vwa_trajectories.py`
4. Extract human references: `python scripts/extract_human_references.py`
5. Generate text references: `python scripts/generate_text_references.py`

## Running Experiments

### Single model + condition
```bash
python verify.py --model gpt-4.1 --condition B --dataset data/verification_dataset.jsonl
```

### All models + conditions
```bash
python verify.py --model all --condition all --dataset data/verification_dataset.jsonl
```

### Dry run (no API calls)
```bash
python verify.py --model all --condition all --dataset data/verification_dataset.jsonl --dry-run
```

## Analysis

### Full analysis (zero API cost)
```bash
python scripts/full_revision_analysis.py
```

### Threshold policy analysis
```bash
python scripts/threshold_policy_analysis.py
```

### Cross-model reference regeneration
```bash
python scripts/regenerate_text_refs.py --generator claude-sonnet-4
```

### Inter-annotator agreement
```bash
python scripts/run_iaa_labels.py sample --n 100
# Label data/iaa_labels_A.jsonl and data/iaa_labels_B.jsonl
python scripts/run_iaa_labels.py kappa --a data/iaa_labels_A.jsonl --b data/iaa_labels_B.jsonl
```

## Determinism

- Temperature is set to 0 (0.01 for Gemini)
- Model versions are pinned per run
- Provider API outputs are nominally deterministic (not bitwise guaranteed)

## Resume Capability

The runner skips records with valid verdicts and re-attempts records with API or parse errors. To force a full re-run, delete the relevant `results/{model}_{condition}.jsonl` file.

## Cost

Total cost of the full study: $32.85 across 11,322 API calls. Per-call cost ranges from $0.0002 (Gemini 2.5 Flash) to $0.0130 (Claude Sonnet 4 in Condition C).
