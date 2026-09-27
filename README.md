# Can Multimodal Models Verify Web-Agent Work?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22990531.svg)](https://doi.org/10.5281/zenodo.22990531)

Replication package for *Can Multimodal Models Verify Web-Agent Work? The Role of Visual
and Textual References in Post-Hoc Verification* (IEEE Access, manuscript Access-2026-32332,
under revision).

A controlled study of reference-augmented post-hoc verification: five frontier closed-API
LMMs judge whether a web task succeeded, from the agent's final screenshot alone
(Condition A) or augmented with a text reference (B), a human reference screenshot (C), or
both (D), across 909 VisualWebArena trajectories produced by a single GPT-4V + Set-of-Mark
agent.

## Review release

The stable reviewer-evidence release is archived at **[10.5281/zenodo.22990531](https://doi.org/10.5281/zenodo.22990531)**
and tagged **`v1.2-ieee-access-reviewer-evidence`**. It adds the matched
reference-information experiment, the visual-reference audit and the offline robustness
tables used in the revision. The earlier **`v1.1-ieee-access-r1`** tag
is preserved as the historical R1 snapshot. See [`PROVENANCE.md`](PROVENANCE.md) for the
scope and limitations of each release.

```bash
git clone https://github.com/shravaniparsi/agentic-ui-test
cd agentic-ui-test && git checkout v1.2-ieee-access-reviewer-evidence
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
  visual_subset_ids.txt                 historical 147-task analysis pool (not a validation certificate)
  screenshots/ references/              rebuilt by the two extraction scripts below
results/                  supplement tables, cross-reference runs, task-ID lists
figures/                  manuscript figures
scripts/                  data preparation and analysis
experiments/reference_information/  completed matched control with raw API records
reviewer_artifacts/visual_reference_validation/  screened reference set and audit records
reviewer_artifacts/text_reference_validation/  blinded two-author audit package
archive/                  historical runs needed to interpret the submitted results
```

## Rebuilding the image data

`data/screenshots/` and `data/references/` are not committed (size). Both are rebuilt from
the archives published by the VisualWebArena authors:

```bash
# agent final screenshots, from the GPT-4V + SoM trajectory archive
python3 scripts/parse_vwa_trajectories.py --archive path/to/gpt4v_som.tar

# unvalidated human screenshot candidates, from the human Playwright traces
python3 scripts/extract_human_references.py --human-dir path/to/vwa_human_trajectories
```

The first extracts agent final-state images. The second extracts last-frame candidates
from the available human traces into `data/reference_candidates/` by default. The full
public inventory contains 233 traces; the 147 IDs in `data/visual_subset_ids.txt` identify
the historical C/D analysis pool. Neither filename matching nor the final recorded
frame establishes successful completion or useful reference coverage. Preserve that
historical pool for reproduction; do not relabel it as a newly validated pool.

For a new run, use separately adjudicated references with task-specific identity and
completion evidence and a byte-hashed manifest. `scripts/prepare_validated_visual_dataset.py`
checks such a manifest and writes a separate dataset restricted to its approved IDs.
It refuses unapproved entries, altered images, implicit task remapping and overwriting
an existing dataset. Repaired images require fresh C/D predictions; old predictions
must not be attached to the new images. Screening does not equalize viewport, recorder
overlays, compression or provider-side image processing.

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
- `experiments/reference_information/` contains a new 147-task matched control. It compares
  criteria-enriched and task-only references while holding the generator wrapper, screenshot,
  verifier prompt and provider settings fixed. It is separate from the historical study.
- `reviewer_artifacts/visual_reference_validation/` contains the disclosed reference audit.
  Sixty-seven of 147 historical candidates passed the implemented screen; this was an
  author-confirmed, AI-assisted audit rather than an independent human annotation study.
- `reviewer_artifacts/text_reference_validation/` contains deterministic, blinded offline
  interfaces and the completed independent two-author assessment of all 294 references in
  the fresh matched control. Derived agreement, disagreement and strict-subset sensitivity
  results are released there; private author export files remain excluded.
- `iaa_labeling/` contains the separate two-author reliability study for the original
  obvious/deceptive/partial failure taxonomy. The reported kappa is 0.000, so category-level
  results are retained only as descriptions of the original single-annotator labels.
- Working manuscripts and peer-review correspondence are intentionally excluded from the
  public replication package.

## Citation

```bibtex
@article{thatikonda2026canllmwebagents,
  title={Can Multimodal Models Verify Web-Agent Work? The Role of Visual and
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
