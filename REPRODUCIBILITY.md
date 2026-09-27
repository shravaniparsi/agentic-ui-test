# Reproducibility Guide

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API keys

Copy `.env.example` to `.env` and fill in:

- `OPENAI_API_KEY` — GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano
- `ANTHROPIC_API_KEY` — the currently configured Claude endpoint
- `GEMINI_API_KEY` — Gemini 3.6 Flash

`.env` is gitignored and must never be committed.

## Data preparation

The per-instance result files are committed, so **every analysis runs without API keys or
image data**. The steps below are only needed to re-run verification from scratch.

1. Download the GPT-4V + SoM agent trajectories published by the VisualWebArena authors
   ([Google Drive](https://drive.google.com/file/d/1-tKz5ByWa1-jwtejiFgxli8fZcBPZgAE/view)).
   The archive is a ~7.8 GB tar of 910 `render_<task_id>.html` files.

2. Extract the agent's final screenshot for each instance. The archive is streamed, so it is
   never unpacked to disk:

   ```bash
   python3 scripts/parse_vwa_trajectories.py --archive path/to/gpt4v_som.tar
   ```

   Writes 909 PNGs to `data/screenshots/`.

3. Download the human Playwright traces
   ([Google Drive folder](https://drive.google.com/drive/folders/1S_fDzB1VUTwUphWPKZ0DdjJOAXjGz94g)),
   233 `<task_id>.trace.zip` files under `human_trajectories_<domain>/`, then extract the
   last screencast-frame candidate from each:

   ```bash
   python3 scripts/extract_human_references.py --human-dir path/to/vwa_human_trajectories
   ```

   Writes unvalidated candidates to `data/reference_candidates/`. The 147 instances in
   `data/visual_subset_ids.txt` are the historical C/D analysis pool; membership alone does
   not establish task identity, successful completion or adequate visual evidence.

   The disclosed screen and its 67 approved native JPEGs are in
   `reviewer_artifacts/visual_reference_validation/`. To prepare a new dataset, use its
   hash-verified manifest with `scripts/prepare_validated_visual_dataset.py`. A changed
   reference set requires new model calls; historical predictions cannot be reassigned to it.

4. Text references are already committed
   (`data/verification_dataset_textref.jsonl` and the two ablation variants). To regenerate:

   ```bash
   python3 scripts/regenerate_text_refs.py --generator gpt-4.1-nano \
     --out data/verification_dataset_textref.jsonl --dry-run
   ```

   Note that regeneration is not bit-reproducible: references are LLM-generated, so a new run
   produces different text and is not directly comparable to the committed results.

## Running verification

```bash
# one model, one condition
python3 verify.py --model gpt-4.1 --condition B \
  --dataset data/verification_dataset_textref.jsonl

# Conditions C and D use the 147-instance subset
python3 verify.py --model gpt-4.1 --condition C \
  --dataset data/verification_dataset_visual147.jsonl

# preview without spending anything
python3 verify.py --model all --condition all \
  --dataset data/verification_dataset_textref.jsonl --dry-run
```

Runs are resumable: `verify.py` keeps only `SUCCESS`/`FAILURE` rows and retries the rest, so
re-running the same command fills in API failures without repeating valid work.

Gemini needs no inter-call pacing on a billed project. If you hit rate limits, set
`GEMINI_SLEEP=1` (seconds) before invoking.

## Analysis (no API cost)

```bash
python3 scripts/recompute_final.py          # S1, S2, S2b, T1/S3 + results/S1_task_ids/
python3 scripts/full_revision_analysis.py   # confusion matrices, ITT bounds, per-type FPR
python3 scripts/revision_analysis.py        # aligned subset, calibration, Bonferroni, cost
python3 scripts/threshold_policy_analysis.py
python3 scripts/run_nonllm_baselines.py --search
python3 scripts/make_manuscript_figures.py
```

`scripts/recompute_final.py` is the single source of truth for the reported statistics.

The new matched reference-information control can be recomputed without API calls:

```bash
python3 experiments/reference_information/analyze.py
```

The package retains the exact generation and verification records, including requested and
provider-returned model identifiers, timestamps, request IDs, token usage, prompt text and
content hashes. `experiments/reference_information/README.md` distinguishes the preserved
as-executed files from portable analysis.

## Inter-annotator agreement

```bash
python3 iaa_labeling/app.py --annotator human1   # serves http://localhost:8080
python3 iaa_labeling/app.py --annotator human2
python3 scripts/recompute_iaa.py                 # kappa, confusion matrix, distributions
```

The interface serves the 100-instance sample in `iaa_labeling/data/iaa_sample.jsonl` over the
three-category taxonomy (obvious / deceptive / partial). Annotators should label
independently and without discussion.

## Known deviations from the original submission

- **Gemini**: `gemini-2.5-flash` was retired by the provider mid-revision and returns HTTP
  404 for newly issued keys. Results use `gemini-3.6-flash`. Aggregate statistics for the
  retired model are preserved in `archive/original_submission_gemini25/`.
- **Conditions C and D**: the prompt described the reference image as the page *before* the
  agent started. It was corrected to describe the intended successful human end state, and
  both conditions were re-run. The public trace audit later showed that a filename match and
  final recorded frame do not by themselves certify a valid successful reference. Historical
  C/D results are therefore reported with that provenance limitation; pre-correction runs are
  in `archive/cd_original_prompt/`.
