# DPO vs SFT for Medical Reasoning: Does Preference Optimization Preserve Safety Alignment Better Than Supervised Fine-Tuning?

**Abbas Sani**
*Draft — TO BE FILLED IN WEEK 6 (see PLAN.md §7 M15)*

> Pre-registration commit: `<git-sha-of-PLAN-lock>` (fill in Week 1)
> Analysis code commit: `<git-sha-of-final-run>` (fill in Week 6)
> Full protocol: [PLAN.md](PLAN.md) · Results: [figures/stats_report.json](figures/stats_report.json)

---

## Abstract

*One paragraph, ~200 words. Written LAST. State (a) the question, (b) the pre-registered design in one sentence, (c) the two hypotheses, (d) the decisions against each, and (e) the mechanistic finding. If a hypothesis was not tested (utility-parity failure), state that plainly.*

---

## 1. Introduction

*~1 page. Frame the problem in three moves:*

1. *Standard medical LLM adaptation uses SFT and is known to erode safety (cite "The Forgotten Shield" and 2–3 others).*
2. *DPO has an implicit KL anchor to the reference policy that SFT does not; this suggests DPO might preserve safety better even without an explicit safety intervention.*
3. *This study tests that directly, in a pre-registered design that isolates the adaptation objective as the sole variable.*

*Close with a bulleted contribution list — three items maximum. Do NOT overclaim. If H1 was not supported, contribution is a clean null with mechanistic explanation.*

---

## 2. Background

*~1.5 pages. Three subsections.*

### 2.1 Post-training objectives: SFT and DPO
*Formal statement of both losses, one paragraph each. Highlight the reference-policy term in the DPO log-likelihood ratio.*

### 2.2 Safety alignment and catastrophic forgetting during domain adaptation
*What "safety" means operationally (refusal on harmful requests). Empirical evidence that domain fine-tuning erodes it. Point to "The Forgotten Shield" as the current state-of-the-art intervention framing.*

### 2.3 Prior work on preference optimization for medical LLMs
*Short — this is a small literature. Note the gap: no direct SFT-vs-DPO safety comparison at matched utility.*

---

## 3. Method

*~1.5 pages. This is a compressed reference to PLAN.md; do not restate the full plan. Structure:*

### 3.1 Pre-registration and design
*Cite PLAN.md commit hash. State the 2 × 3 design (SFT/DPO × 3 seeds) and the byte-identical constraint.*

### 3.2 Data
*One paragraph per: adaptation data (MedMCQA subset), safety benchmarks (general + medical), KL holdout.*

### 3.3 Training
*Table of hyperparameters (copy from PLAN.md §5.2). Note the two objective-specific values (LR, β).*

### 3.4 Evaluation and statistical protocol
*Refer to `src/stats.py` for the exact test implementations. State the decision rules verbatim from PLAN.md §3.1 in a highlighted box.*

**Pre-registered decision rules (verbatim from PLAN.md §3.1):**
> H1 supported iff mean Δ_safety ≥ +5 pp, pooled McNemar p < 0.025 (Bonferroni for 2 planned comparisons), and 95% paired-bootstrap CI on Δ_safety across seeds excludes 0.
> Utility parity: |mean Δ_utility| ≤ 2 pp AND 95% CI fully inside ±2 pp band.
> H2 (exploratory): Δ_norm > 0 AND Δ_KL > 0 AND Spearman ρ ≥ +0.5.

---

## 4. Results

*~3 pages. Every number below comes from `figures/stats_report.json`. State each pre-registered decision explicitly.*

### 4.1 Utility parity check
*Report medical accuracy for base model, per-seed SFT, per-seed DPO. State whether utility parity holds. If it fails, note that H1 will be reported as the exploratory Pareto analysis per PLAN.md §3.3.*

**Table 1: Medical accuracy (MedMCQA validation) by condition and seed.**

| Seed | SFT accuracy | DPO accuracy | Δ (DPO − SFT) pp |
|---|---|---|---|
| 42 | *fill* | *fill* | *fill* |
| 1337 | *fill* | *fill* | *fill* |
| 20260922 | *fill* | *fill* | *fill* |
| **Mean** | *fill* | *fill* | *fill* (95% CI: *fill*) |

*Utility parity: [supported / not supported]. If not supported, jump to §4.4 (Pareto analysis) and skip §4.2–§4.3.*

### 4.2 Safety retention — general safety benchmark
**Table 2: Safety-refusal rate on general safety prompts (200 prompts).**

| Seed | SFT safe rate | DPO safe rate | Δ pp | McNemar p |
|---|---|---|---|---|
| 42 | | | | |
| 1337 | | | | |
| 20260922 | | | | |
| **Pooled** | | | | *pooled McNemar p* |

*Δ_safety (general), 95% bootstrap CI across seeds: [X, Y] pp. H1_general: [supported / not supported].*

### 4.3 Safety retention — medical safety benchmark
*Same table structure. State H1_medical decision.*

### 4.4 Utility–safety Pareto plot
*One figure showing the 6 adapted models plus base model in (medical accuracy, safety rate) space, one panel per safety domain. Include base model as reference.*

### 4.5 Secondary: MedQA-USMLE external validity
*Brief. Does the SFT-vs-DPO ranking replicate on the out-of-distribution benchmark? One-paragraph note, no formal test.*

---

## 5. Mechanistic Analysis (H2, exploratory)

*~2 pages.*

### 5.1 LoRA weight-norm audit
*Table of per-module Frobenius norms averaged across seeds, SFT vs DPO. Plot.*

### 5.2 Per-token KL divergence from the base model
*Report mean per-token KL for each condition, per seed and pooled. Include a density plot over prompts.*

### 5.3 Correlation between safety retention and parameter shift
*Spearman ρ across seeds between Δ_safety and −Δ_norm, with bootstrap CI. State H2 decision. Include the reminder that n = 3.*

### 5.4 Interpretation
*If H2 supported: parameter shifts predict safety loss, and DPO shifts less. If not supported: the mechanism candidate from the GSM8K prior is not sufficient at this scale; discuss alternatives.*

---

## 6. Qualitative Analysis: Discordant Examples

*~1 page. Based on the 30 hand-coded discordant prompts using the rubric in `configs/discordant_rubric.md`.*

### 6.1 Category distribution
*Table: behavior category × surface pattern, absolute counts.*

### 6.2 Illustrative examples
*2–3 short quoted examples per behavior category. Redact any PII.*

### 6.3 Intra-annotator agreement
*Cohen's κ between pass 1 and pass 2. If κ < 0.6, downgrade to descriptive vignettes only.*

---

## 7. Sensitivity Checks and Limitations

*~1 page.*

### 7.1 Learning-rate sensitivity
*Results at LR × 0.5 and LR × 2 for both conditions (single seed each). Does the primary conclusion hold?*

### 7.2 Judge sensitivity
*Cohen's κ between Llama-Guard-3-8B and manual annotation on 100 responses. If < 0.7, describe the manual-annotation fallback.*

### 7.3 Base-model sensitivity
*If Week 6 slack allowed a full re-run on Qwen2.5-3B-Instruct, report it here. Otherwise state that the plan reserved this as external-validity check and it was not executed.*

### 7.4 Limitations
- *n = 3 seeds: sufficient for pooled McNemar; exploratory for Spearman on H2.*
- *Single base-model family (unless §7.3 was executed).*
- *One safety judge; κ spot-check mitigates but does not eliminate judge bias.*
- *4,000-example adaptation is small compared to production medical LLMs.*
- *Safety benchmarks are proxies; real-world medical safety is broader.*

---

## 8. Discussion

*~1 page. Two paragraphs.*

### 8.1 Implications for parameter-space interventions
*Connect back to "The Forgotten Shield". If DPO preserves substantially more safety than SFT (H1 supported), then explicit safety-grafting interventions may operate on a smaller residual under DPO-adapted targets, potentially requiring less aggressive intervention. If H1 is not supported, the case for explicit intervention à la Forgotten Shield is strengthened: choice of objective alone is not enough.*

### 8.2 What this study does and does not show
*Explicit about scope: single base-model family, 3B scale, MedMCQA-style multiple-choice adaptation. Does not test instruction-tuning-from-scratch or reward-model-based RLHF.*

---

## 9. Conclusion

*~0.5 page. Restate the decision on each hypothesis. State the single most important actionable finding. Point to code and data.*

---

## Reproducibility

- **Repository:** `https://github.com/USERNAME/dpo-vs-sft-medical-safety` (fill in)
- **Pre-registration commit:** `<sha>` — this is the commit hash of PLAN.md before any training was executed.
- **Final results commit:** `<sha>` — this is the commit hash after the full pipeline completed.
- **Compute:** Colab Pro, A100 40GB (fallback L4 22GB). Total ~41 h wall-clock (see PLAN.md §6.1).
- **How to reproduce:** open `notebooks/run_all_colab.ipynb`, set your GitHub URL and HF token, run cells top-to-bottom. Every stage is resumable.

## References

*Fill in. Minimum expected citations:*
- *"The Forgotten Shield" (Paper 2)*
- *Original DPO paper (Rafailov et al.)*
- *MedMCQA, MedQA benchmark papers*
- *AdvBench / HarmBench / MedSafetyBench papers*
- *Llama 3 technical report*
- *Llama Guard paper*
- *Candidate's own GSM8K DPO transfer study (self-citation as prior work)*

## Appendices

- **A.** Full hyperparameters (from configs/sft.yaml, configs/dpo.yaml)
- **B.** Full per-seed statistical tables
- **C.** Per-module LoRA Frobenius norms (full breakdown by module)
- **D.** Discordant-example coding rubric (from configs/discordant_rubric.md)
- **E.** Judge-κ spot-check details
- **F.** Deviations from pre-registration (should be empty; any entry is a labelled violation)
