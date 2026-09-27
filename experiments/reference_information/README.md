# Matched reference-information experiment

This directory is the completed 147-task paired control requested during review. The primary
comparison is criteria-enriched versus task-only generated references within each verifier.
See `PROTOCOL.md` for the design and `RESULTS.md` for the results.

Run the analysis without API calls:

```bash
python3 experiments/reference_information/analyze.py
```

`generation_results.jsonl` and `verification_results.jsonl` are the raw append-only response
records. `attempts.jsonl` records call attempts. The request and schedule JSONL files, exact
verification prompt, matched ID lists, output tables and integrity reports are included.

The execution scripts are preserved as executed and their hashes are recorded in
`execution_code_hashes.json` and `final_artifact_hashes.json`. Their original workspace layout
placed this experiment beside `research-repository`, so `prepare.py`, `run.py` and
`validate_outputs.py` retain those original path assumptions. They are provenance artifacts,
not a one-command portable rerun. `analyze.py` is portable within this directory. Re-running
provider calls can produce different outputs and requires credentials and current endpoints.

`tasks.jsonl` retains the as-executed local screenshot paths because its exact hash is part of
the execution record. The screenshots themselves can be rebuilt from the public GPT-4V + SoM
trajectory archive; `screenshot_recovery.json` records archive members and byte hashes.
