# Can LLM Web Agents Verify Their Own Work?

Replication package for *Can LLM Web Agents Verify Their Own Work? The Role of Visual and
Textual References in Post-Hoc Verification* (IEEE Access, manuscript Access-2026-32332,
under revision).

A controlled study of reference-augmented post-hoc verification: five frontier closed-API
LMMs judge whether a web task succeeded, from the agent's final screenshot alone
(Condition A) or augmented with a text reference (B), a human reference screenshot (C), or
both (D), across 909 VisualWebArena trajectories produced by a single GPT-4V + Set-of-Mark
agent.

## Version of record

The exact state of this repository behind the IEEE Access revision (Access-2026-32332) is tagged **`v1.1-ieee-access-r1`**. Every number in the manuscript can be reproduced from the code and data at that tag:

```bash
git clone https://github.com/shravaniparsi/agentic-ui-test
cd agentic-ui-test && git checkout v1.1-ieee-access-r1
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
python3 verify.py --model gpt-4.1 --condition B \
  --dataset data/verification_dataset_textref.jsonl --dry-run
```

Every analysis below runs with **no API calls** from the committed result files.

```bash
python3 scripts/recompute_final.py          # S1/S2/S2b/T1 supplement tables
python3 scripts/full_revision_analysis.py   # confusion matrices, ITT bounds, per-type FPR
python3 scripts/revision_analysis.py        # aligned subset, calibration, Bonferroni, cost
python3 scripts/run_nonllm_baselines.py --search   # SSIM / pHash threshold sweep
python3 scripts/make_manuscript_figures.py  # regenerate the manuscript figures
```

## Layout

```
config.py                 model registry, pricing, API keys
verify.py                 verification runner (Conditions A-D)
llm_clients.py            OpenAI / Anthropic / Gemini clients
prompts/                  Condition A-D prompt templates
{model}_{A..D}.jsonl      per-instance results, 5 models x 4 conditions
data/
  verification_dataset.jsonl            909 instances, ground truth, paths
  verification_dataset_textref.jsonl    + text references (GPT-4.1 Nano)
  verification_dataset_xref_claude.jsonl  + cross-generator refs (Claude Sonnet 4)
  verification_dataset_nocriteria.jsonl   + refs generated without eval criteria
  visual_subset_ids.txt                 the 147 instances with a human reference
  screenshots/ references/              rebuilt by the two extraction scripts below
results/                  supplement tables, cross-reference runs, task-ID lists
figures/                  manuscript figures
scripts/                  data preparation and analysis
manuscripts/              revised manuscript and response to reviewers
archive/                  superseded runs, kept for provenance (see each README)
```

## Rebuilding the image data

`data/screenshots/` and `data/references/` are not committed (size). Both are rebuilt from
the archives published by the VisualWebArena authors:

```bash
# agent final screenshots, from the GPT-4V + SoM trajectory archive
python3 scripts/parse_vwa_trajectories.py --archive path/to/gpt4v_som.tar

# human reference screenshots, from the human Playwright traces
python3 scripts/extract_human_references.py --human-dir path/to/vwa_human_trajectories
```

The first writes 909 agent final states; the second writes 233 human final states, of which
the 147 in `data/visual_subset_ids.txt` form the visual-reference subset used by Conditions
C and D.

## Notes for reviewers

- **`scripts/recompute_final.py` is the single source of truth** for the supplement tables.
  It applies one policy throughout: jointly valid instances only, Conditions C and D
  restricted to the 147-instance subset, exact binomial McNemar when b+c < 25 and
  asymptotic chi-square with continuity correction otherwise.
- `results/S1_task_ids/` holds the matched instance IDs behind each of the 30 paired tests.
- The Gemini verifier is `gemini-3.6-flash`. The originally submitted `gemini-2.5-flash` was
  retired by the provider and returns HTTP 404 for new API keys; aggregate statistics for
  the retired model are preserved in `archive/original_submission_gemini25/`.
- Conditions C and D were re-run after a prompt correction; the pre-correction runs are in
  `archive/cd_original_prompt/`, which documents the change.

## Citation

```bibtex
@article{thatikonda2026canllmwebagents,
  title={Can LLM Web Agents Verify Their Own Work? The Role of Visual and
         Textual References in Post-Hoc Verification},
  author={Thatikonda, Vishwak and Parsi, Shravani},
  journal={IEEE Access},
  year={2026},
  note={Manuscript ID: Access-2026-32332, under revision}
}
```

## License

Released for academic replication. VisualWebArena tasks, trajectories, and human
demonstrations remain under their original licenses.
