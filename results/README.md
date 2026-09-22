# Cached evaluation outputs

Files in this directory are the machine-readable outputs of each evaluation
run, keyed by `{eval}_{condition}_seed{S}.jsonl` or `{metric}_{condition}_seed{S}.json`.
See `PLAN.md` §5.3 for what is written by each script, and `src/stats.py`
docstring for the exact filenames expected by the analysis.

Large files are not committed; a manifest in `results/MANIFEST.md` (generated
by the notebook after each run) records the SHA-256 of every produced file.
