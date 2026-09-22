# DPO vs SFT for Medical Reasoning

**Does Direct Preference Optimization preserve safety alignment better than Supervised Fine-Tuning when adapting a general-purpose instruct model to medical multiple-choice reasoning?**

A pre-registered empirical study with parameter-space mechanistic analysis. Extension of [prior work on preference-signal ablation for GSM8K reasoning](https://github.com/avvas200/dpo-reasoning-transfer).

## Motivation in one paragraph

Medical LLMs are almost always produced by SFT-adapting a general-purpose instruct model on medical data, and this process is known to erode the base model's safety alignment (see e.g. "The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs"). DPO has a built-in KL anchor to the reference policy that SFT does not. This study asks the natural precursor question to explicit safety-grafting: does the choice of adaptation objective, on its own, protect safety? And can any such difference be explained by the parameter-space diffuseness the candidate observed on GSM8K in prior work?

## Pre-registration

Everything about this study — hypotheses, decision rules, datasets, hyperparameters, statistical protocol — is committed to `PLAN.md` and `configs/` **before** any adapted model is trained. See `PLAN.md` §10 for the pre-registration lock. The statistical protocol lives as executable code in `src/stats.py` and is tested against synthetic ground truths (`tests/test_stats.py`, 25 tests, all passing) before real data exists.

## Design

- **Fully-crossed 2 × 3.** SFT vs DPO × 3 seeds (42, 1337, 20260922).
- **Base model:** Llama-3.2-3B-Instruct (fallback Qwen2.5-3B-Instruct).
- **Adaptation set:** 4,000 MedMCQA training questions, frozen sample (`data/adaptation_ids.json`).
- **Byte-identical constraint:** everything except the adaptation loss is held identical between SFT and DPO. Only the LR (2e-4 vs 5e-6), the DPO β (0.1), and the data collator differ.
- **Safety benchmark:** 200 general (AdvBench) + 200 medical (MedSafetyBench) prompts.
- **Judge:** Llama-Guard-3-8B, with κ spot-check against manual annotation before pooled analysis.
- **Mechanistic:** per-token KL from base + per-module LoRA Frobenius audit.
- **Compute:** ~41 h wall-clock on Colab Pro A100 40GB.

## Decision rules (verbatim from PLAN.md §3.1)

**H1 (primary, confirmatory):** DPO preserves safety better than SFT at matched medical utility. Supported iff:
- mean Δ_safety ≥ +5 percentage points across 3 seeds, AND
- pooled McNemar p < 0.025 (Bonferroni for 2 planned comparisons), AND
- 95% paired-bootstrap CI on Δ_safety across seeds excludes zero.

**Utility parity precondition:** |Δ_utility| ≤ 2 pp AND 95% CI fully inside ±2 pp band. If it fails, H1 is not tested; a Pareto plot is reported per PLAN.md §3.3.

**H2 (secondary, exploratory):** DPO's smaller output-space and weight-space shifts explain the safety gap. Supported iff Δ_norm > 0 AND Δ_KL > 0 AND Spearman ρ ≥ +0.5 across seeds. n = 3, so exploratory only.

## Repository layout

```
.
├── PLAN.md · PLAN.pdf         Pre-registration (LOCKED)
├── REPORT.md                  Technical report (Week 6)
├── README.md                  This file
├── requirements.txt
├── configs/
│   ├── sft.yaml                SFT hyperparameters
│   ├── dpo.yaml                DPO hyperparameters
│   ├── seeds.yaml              All random seeds
│   └── discordant_rubric.md    Qualitative coding scheme (LOCKED)
├── data/                       Frozen datasets (adaptation_ids.json, etc.)
├── src/
│   ├── data_prep.py            SFT + DPO dataset construction
│   ├── train_sft.py            SFT training
│   ├── train_dpo.py            DPO training
│   ├── eval_medical.py         MedMCQA + MedQA evaluation
│   ├── eval_safety.py          Safety gen + Llama-Guard judgement
│   ├── mech_kl.py              Per-token KL from base
│   ├── mech_norms.py           LoRA Frobenius audit
│   └── stats.py                Pre-registered protocol (LOCKED)
├── notebooks/
│   └── run_all_colab.ipynb     Single-notebook end-to-end runner
├── tests/
│   └── test_stats.py           25 tests on the pre-registered protocol
├── results/                    Cached per-model outputs (gitignored)
└── figures/                    Plots + stats_report.json
```

## How to reproduce

1. Fork this repo. Get access to Llama-3.2-3B-Instruct and Llama-Guard-3-8B (both gated on HuggingFace).
2. Open `notebooks/run_all_colab.ipynb` in Colab Pro (A100 40GB recommended).
3. Set `REPO_URL` in cell 2 to your fork. Run cells top-to-bottom.
4. Each stage is resumable and checkpoints to Google Drive. A runtime disconnect loses only the currently-executing stage.

The full pipeline takes ~41 h across ~5 sessions and produces `figures/stats_report.json`, from which `REPORT.md` is populated in Week 6.

## Status

| Milestone | Status |
|---|---|
| Pre-registration committed | ⬜ (target: Week 1) |
| Dataset verification | ⬜ (target: Week 1) |
| Baselines cached | ⬜ (target: Week 1) |
| SFT dataset + DPO pairs built | ⬜ (target: Week 2) |
| 3 SFT runs complete | ⬜ (target: Week 3) |
| 3 DPO runs complete | ⬜ (target: Week 3) |
| Medical + safety eval on all 6 models | ⬜ (target: Week 4) |
| Judge-κ spot-check | ⬜ (target: Week 4) |
| Mechanistic analyses | ⬜ (target: Week 5) |
| `src/stats.py` executed, figures generated | ⬜ (target: Week 5) |
| REPORT.md finalized | ⬜ (target: Week 6) |

## Citation

If you use this code or refer to this study before it is written up formally:

```bibtex
@misc{sani2026dpovssft,
  title  = {DPO vs SFT for Medical Reasoning: Does Preference Optimization
            Preserve Safety Alignment Better Than Supervised Fine-Tuning?},
  author = {Sani, Abbas},
  year   = {2026},
  note   = {Pre-registered study. \url{https://github.com/USERNAME/dpo-vs-sft-medical-safety}},
}
```

## License

Code: MIT. Data: subject to upstream licences of MedMCQA, MedQA, AdvBench, MedSafetyBench, and the base models.
