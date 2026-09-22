**DPO vs SFT for Medical Reasoning:**

*Does Preference Optimization Preserve Safety Alignment Better Than
Supervised Fine-Tuning?*

A Pre-Registered Empirical Study with Parameter-Space Mechanistic
Analysis

**Research Plan v1.0**

Abbas Sani

*Prepared for HKPFS Application Supporting Work · September 2026*

**1. Executive Summary**

This document is the pre-registered research plan for a self-directed
empirical study on post-training methods for medical language models. It
is designed to (a) directly extend the candidate\'s existing DPO
reasoning-transfer study, (b) engage substantively with the prospective
supervisor\'s recent work on safety alignment in medical multimodal
LLMs, and (c) produce a defensible technical report by the HKPFS
submission deadline.

**The scientific question**

When adapting a general-purpose instruction-tuned model to medical
reasoning, does Direct Preference Optimization (DPO) preserve the base
model\'s safety alignment better than Supervised Fine-Tuning (SFT) at
matched medical utility? And is any such difference explained by DPO
producing more diffuse, lower-magnitude parameter updates than SFT ---
the mechanism the candidate previously observed on GSM8K?

**The two hypotheses**

-   **H1 (primary, confirmatory):** At matched medical utility, models
    adapted with DPO retain a higher fraction of the base model\'s
    safety-refusal behavior than models adapted with SFT.

-   **H2 (secondary, mechanistic):** The DPO-vs-SFT gap in safety
    retention is monotonically related to the magnitude and locality of
    parameter updates: SFT produces larger, more localized LoRA
    weight-norm shifts and larger per-token KL divergences from the base
    model than DPO, and models with smaller parameter shifts retain more
    safety behavior.

**Why this is the right study**

-   **Directly aligned with supervisor\'s agenda.** \"The Forgotten
    Shield\" (Paper 2) frames safety erosion during medical fine-tuning
    as a parameter-space problem. This study asks the natural precursor
    question: is the choice of adaptation objective itself a safety
    lever, before any explicit re-alignment intervention is applied?

-   **Continuous with candidate\'s prior work.** The mechanistic toolkit
    --- per-token KL, per-module LoRA Frobenius audit,
    discordant-example structural coding, exact McNemar, paired
    bootstrap CIs --- transfers wholesale. Roughly 80% of the existing
    GSM8K infrastructure is reusable.

-   **Publishable regardless of outcome.** A positive result (DPO
    preserves safety better) is a clean, actionable finding for the
    medical LLM community. A null result is equally informative: it
    would sharpen the case for the explicit parameter-space intervention
    proposed in \"The Forgotten Shield\". The study is designed so that
    either outcome is a contribution.

**Deliverables**

-   A publicly released GitHub repository containing all training,
    evaluation, and analysis code, cached artefacts, seed
    configurations, and figures.

-   A \~10--14 page technical report (PDF) targeting an ML4H- or
    NeurIPS-workshop-adjacent style, hosted in the repository and on
    arXiv.

-   This pre-registered research plan, committed to the repository
    before any adaptation training run is executed.

**Timeline and compute**

Six weeks of focused work (\~15--20 hours/week) on Colab Pro (A100 40GB
or L4 22GB, QLoRA throughout). Technical report locked in week 6;
workshop-paper adaptation optional and post-hoc. This schedule places
the report on the repository roughly 10 days before typical HKPFS
supporting-material deadlines, providing schedule margin.

**2. Motivation and Positioning**

**2.1 The problem**

Medical LLMs are almost universally produced by adapting a
general-purpose instruction-tuned base model (Llama, Qwen, Mistral) on
medical-domain data. The standard adaptation recipe is supervised
fine-tuning on question--answer pairs drawn from medical benchmarks or
synthesised from textbooks. This process is known to cause catastrophic
forgetting of the base model\'s safety alignment --- the phenomenon that
motivates \"The Forgotten Shield\" and the emerging literature on
safety-utility trade-offs in domain-adapted LLMs.

However, SFT is not the only available adaptation objective. Direct
Preference Optimization (DPO), and its close relatives (IPO, KTO, ORPO),
optimise the model against a pairwise preference dataset by reweighting
the log-likelihood ratio between chosen and rejected responses,
regularised toward a frozen reference policy via an implicit KL term.
The reference policy in the DPO loss is a first-order safety lever: it
directly penalises drift from the base model\'s distribution, whereas
SFT has no analogous term.

This suggests a testable, potentially actionable question that the
community has not, to the candidate\'s knowledge, addressed head-on:
does the built-in KL anchoring of DPO cause less safety erosion than SFT
when adapting to medical tasks?

**2.2 Prior evidence from the candidate\'s own work**

In the candidate\'s pre-registered study on preference-signal ablation
for GSM8K reasoning (Sep 2026), outcome-verified DPO on 500
reasoning-specific preference pairs produced parameter updates that were
\~2.4% larger in LoRA Frobenius norm than DPO on general-purpose
preferences, yet output-space shifts (per-token KL from the base model)
were comparable, and neither condition improved reasoning accuracy over
the base. The interpretation offered in that report was that
outcome-verified DPO at this scale acts as diffuse regularisation rather
than a directed reasoning-quality signal.

The \"diffuse regularisation\" observation is precisely the property
that would predict safety preservation. If DPO\'s weight and output
shifts are small and distributed across many parameters --- as opposed
to SFT, which is expected to produce larger, more directed shifts on the
tokens it was trained to imitate --- then the base model\'s original
refusal behaviors should be less disrupted. This study tests that
prediction directly.

**2.3 Positioning against the supervisor\'s recent work**

\"The Forgotten Shield\" (Paper 2) makes three claims of direct
relevance:

-   Current SOTA Medical MLLMs exhibit pervasive safety vulnerabilities,
    particularly against cross-modality jailbreaks.

-   Medical fine-tuning frequently induces catastrophic forgetting of
    the base model\'s safety alignment.

-   A parameter-space intervention that injects intrinsic safety
    representations from the base model into the target model, combined
    with a fine-grained parameter search for the safety--utility
    trade-off, can restore safety without additional safety data.

The present study asks a logically prior question: before applying any
explicit intervention, does the choice of adaptation objective itself
matter? If DPO preserves substantially more safety than SFT at matched
utility, this reframes the intervention problem: the parameter-space
grafting in \"The Forgotten Shield\" may need to correct a smaller
residual under a DPO-adapted target model than under an SFT-adapted one.
This is a direct experimental complement to the supervisor\'s agenda,
not a competing framing.

**3. Hypotheses, Estimands, and Decision Rules**

**3.1 Formal hypotheses**

**H1 (primary, confirmatory)**

**Estimand:** *Δ_safety = SafetyRate(DPO) − SafetyRate(SFT)*, where
SafetyRate is the proportion of unsafe prompts (from the safety
benchmark defined in §4) on which the model produces a refusal or
safe-completion response as judged by the evaluation protocol in §5.

**Constraint:** H1 is evaluated only at matched medical utility, defined
as: MedicalAccuracy(DPO) and MedicalAccuracy(SFT) both lie within ±2
percentage points of each other on the primary medical benchmark. If
this constraint is not met, H1 is not tested; a utility-vs-safety Pareto
plot is reported instead (see §3.3).

**Decision rule:** H1 is supported iff mean Δ_safety ≥ +5 percentage
points, pooled McNemar p \< 0.025 (Bonferroni-corrected for 2 planned
comparisons), and the 95% paired-bootstrap CI on Δ_safety across seeds
excludes zero.

**H2 (secondary, mechanistic)**

**Estimand:** *Δ_norm = LoRAFrobeniusNorm(SFT) − LoRAFrobeniusNorm(DPO)*
(per-module and pooled), and *Δ_KL = MeanTokenKL(SFT ‖ base) −
MeanTokenKL(DPO ‖ base)* on a held-out set of medical and general
prompts.

**Decision rule:** H2 is supported iff both Δ_norm \> 0 and Δ_KL \> 0,
and Spearman ρ between per-seed (Δ_safety, −Δ_norm) is ≥ +0.5. H2 is
exploratory and not corrected for multiple comparisons; it is reported
as evidence, not as a hypothesis test.

**3.2 Pre-specified analyses**

-   Primary: SFT vs DPO on safety benchmark, at matched utility, 3
    seeds, exact McNemar per seed, pooled McNemar with Bonferroni α =
    0.025.

-   Utility parity check: paired bootstrap CI on MedicalAccuracy(DPO) −
    MedicalAccuracy(SFT); reject utility parity if \|Δ\| \> 2 pp with
    95% CI excluding ±2 pp band.

-   Mechanistic: per-module LoRA Frobenius weight-norm audit; mean
    per-token KL divergence on 200 held-out prompts (100 medical, 100
    general); Spearman correlation across seeds between safety-retention
    and −LoRA-norm.

-   Qualitative: discordant-example coding on 30 randomly sampled
    prompts where DPO refuses and SFT does not (or vice versa), using a
    fixed 5-category rubric committed before results are seen.

**3.3 What we report if the utility-parity constraint fails**

If MedicalAccuracy(DPO) and MedicalAccuracy(SFT) differ by more than 2
pp, H1 as stated is not testable --- comparing safety across models with
different utility profiles is confounded. In that case, the report will
present a utility--safety Pareto plot across the 6 runs (3 seeds × 2
conditions), the raw safety and utility numbers with paired-bootstrap
CIs, and a clearly labelled exploratory analysis conditional on the
observed utility gap. This contingency is committed here to prevent
post-hoc rationalisation.

**4. Materials**

**4.1 Base model**

Llama-3.2-3B-Instruct is the primary base model. Justification: (a)
continuity with the candidate\'s existing GSM8K study, allowing direct
reuse of tokenization, LoRA target modules, and evaluation harness; (b)
a known non-trivial safety alignment out of the box, giving room for
observable degradation; (c) fits in QLoRA on a single Colab Pro A100
40GB with headroom for evaluation.

Fallback: Qwen2.5-3B-Instruct. Selected if the Llama licence terms are
ambiguous for the report\'s intended release or if any
model-availability issue arises. If the fallback is used, this is
disclosed in the report and the primary base is not retrained on the
same seeds to avoid multiplicity.

**4.2 Medical adaptation dataset**

Primary: MedMCQA training split (\~183k multiple-choice questions from
Indian medical entrance exams, with subject and topic labels). Selected
because it is large enough for meaningful adaptation at modest LoRA
capacity, publicly released under a permissive licence, and structurally
suited to constructing preference pairs.

Adaptation subset: A random sample of 4,000 training questions is drawn
once at the start of the study (fixed seed = 20260922) and used
identically by both SFT and DPO conditions. The sample composition is
committed to the repository as adaptation_ids.json before any training
is run.

**SFT data construction**

Each of the 4,000 adaptation questions is formatted as an
instruction--response pair. The response is the correct-answer letter
followed by the ground-truth explanation field from MedMCQA. Prompts
follow the base model\'s chat template.

**DPO data construction**

For each of the same 4,000 questions, a preference pair is constructed
as follows. The chosen response is the ground-truth explanation
(identical to the SFT target). The rejected response is generated by
prompting the base model to answer the question, then verifying against
the ground-truth key: if the base model\'s answer is incorrect, that
generation is the rejected. If the base model is correct, a distractor
is generated by conditioning the base model on a wrong-answer choice and
asking for a justification of it; the resulting incorrect justification
is the rejected. This mirrors the outcome-verified preference
construction used in the candidate\'s prior GSM8K study.

**4.3 Medical evaluation benchmark**

Primary: MedMCQA validation split (\~4,183 questions, no overlap with
the adaptation subset). Metric: accuracy on the letter answer,
exact-match after canonicalisation.

Secondary (external validity): MedQA-USMLE test split (\~1,273
questions). Reports accuracy but does not enter the primary decision
rule; used only to check whether the DPO-vs-SFT ranking is stable under
distribution shift.

**4.4 Safety benchmark**

The safety evaluation set combines (a) a general-purpose harmful-request
set and (b) a medical-specific safety set. Both are needed because the
\"safety\" that matters for a medical model includes both refusing
generic harmful instructions and refusing medically inappropriate ones
(e.g., instructions to synthesise controlled substances, provide
unsupervised dosing, or diagnose without disclaimer).

-   **General safety:** 200 prompts randomly sampled (fixed seed) from
    AdvBench-Harmful. If AdvBench is not accessible at study start,
    HarmBench-standard is the fallback; whichever is used is committed
    to the repo.

-   **Medical safety:** 200 prompts sampled from MedSafetyBench (or, if
    unavailable, hand-curated from published red-team taxonomies in the
    medical-LLM literature and cross-checked against 2 published
    sources). If hand-curation is used, the full 200-prompt set is
    committed to the repo before any evaluation is run, with source
    citations for each prompt category.

This mixed-source design is a known risk: exact reproducibility of
MedSafetyBench access should be verified in Week 1 (see §7). The
fallback path is documented so that a locked plan can be executed even
under the worst case.

**4.5 Judge for safety evaluation**

Refusal / safe-completion judgement is delegated to a fixed, cheap
judge: Llama-Guard-3-8B run locally in 4-bit on the same Colab instance,
called with the standard prompt template. Judge outputs are binary (safe
/ unsafe) at the response level. Judge agreement is spot-checked by
manual annotation of 100 randomly sampled responses (50 refusals, 50
completions) before the primary analysis is executed; if judge--human
agreement is below Cohen\'s κ = 0.7, the study switches to a full manual
annotation of the safety set. This threshold is committed here.

**5. Experimental Protocol**

**5.1 Design**

Fully-crossed 2 (adaptation objective: SFT vs DPO) × 3 (seed: 42, 1337,
20260922) design. Everything except the adaptation objective is held
byte-identical across the two conditions: base model, tokenizer, LoRA
target modules and rank, learning-rate schedule up to a scalar the two
losses require, batch size, gradient accumulation, sequence length,
evaluation harness, and judge configuration.

**5.2 Training hyperparameters**

  -------------------- ----------------------- -------------------------------
  **Hyperparameter**   **Value**               **Notes**

  Base model           Llama-3.2-3B-Instruct   Fallback: Qwen2.5-3B-Instruct

  Quantization         4-bit NF4 (bnb)         For memory only; adapters in
                                               bf16

  LoRA rank / alpha    16 / 32                 Match GSM8K study

  LoRA target modules  q,k,v,o,gate,up,down    All linear projections

  Adaptation set size  4,000 examples          Fixed sample, both conditions

  Epochs               2                       Both conditions identical

  Effective batch size 32                      micro-bs 4 × grad-accum 8

  Max sequence length  1,024 tokens            Truncate longer examples

  Learning rate        2e-4 (SFT) / 5e-6 (DPO) Objective-specific defaults

  LR schedule          cosine, 3% warmup       Identical shape

  DPO β                0.1                     Standard TRL default

  Optimizer            paged AdamW 8-bit       Memory-efficient

  Seeds                42, 1337, 20260922      3 seeds × 2 conditions = 6 runs
  -------------------- ----------------------- -------------------------------

*The differing learning rates for SFT (2e-4) and DPO (5e-6) reflect the
objectives\' distinct optimisation dynamics --- these are the
widely-used TRL defaults. A learning-rate sensitivity check (§5.5)
verifies that the primary conclusion does not depend on these specific
values.*

**5.3 Evaluation protocol**

For each of the 6 adapted models (2 conditions × 3 seeds):

-   MedMCQA validation accuracy over all 4,183 questions, greedy
    decoding, temperature 0.

-   MedQA test accuracy, same decoding.

-   Safety-set completion at temperature 0.7 with a fixed sampling seed
    (matches typical deployment), one completion per prompt, judged by
    Llama-Guard-3-8B.

-   Per-token KL divergence from the base model on a held-out 200-prompt
    mix (100 medical MedMCQA-style, 100 general Alpaca-style).

-   LoRA weight-norm audit: Frobenius norm of the delta (BA product for
    each LoRA module) reported per module and pooled.

**5.4 Statistical analysis**

The statistical protocol is committed as executable code in src/stats.py
before any adapted model is evaluated. Reported quantities:

-   Per-seed exact McNemar test comparing SFT and DPO safety-refusal
    outcomes on the 400 safety prompts.

-   Pooled McNemar across seeds (sum of discordant pairs, exact
    binomial), Bonferroni-corrected α = 0.025 for two planned
    comparisons (general safety, medical safety).

-   95% paired-bootstrap CIs (10,000 resamples, seeded) on Δ_safety per
    seed and pooled.

-   Same CI construction on Δ_utility for the parity check.

-   Spearman ρ across the 3 seeds between (Δ_safety, −Δ_norm) for H2,
    with a 95% bootstrap CI. Given n = 3, this is exploratory only; the
    report explicitly does not treat it as a confirmatory test.

**5.5 Sensitivity and robustness checks**

-   Learning-rate sensitivity: one additional seed at LR × 0.5 and LR ×
    2 for each condition, reported in an appendix; the primary decision
    rule uses only the default LR.

-   Judge sensitivity: manual annotation of 100 randomly sampled safety
    responses, Cohen\'s κ reported; primary analysis switches to manual
    if κ \< 0.7.

-   Base-model sensitivity: if time permits (Week 6 slack), the full
    pipeline is re-run on Qwen2.5-3B-Instruct as an external-validity
    check, reported as an appendix and not in the main decision rule.

**6. Compute Budget and Feasibility**

**6.1 Wall-clock estimate**

  ----------------------------- ----------- ---------------- -----------------
  **Task**                      **GPU**     **Wall-clock**   **Notes**

  MedMCQA baseline eval (base   A100 40GB   \~1.5 h          Cached, run once
  model)                                                     

  MedQA baseline eval (base     A100 40GB   \~0.5 h          Cached, run once
  model)                                                     

  DPO preference construction   A100 40GB   \~3 h            Cached, run once
  (base model gen)                                           

  Safety baseline eval (base    A100 40GB   \~1 h            Includes judge
  model)                                                     

  SFT training × 3 seeds        A100 40GB   \~6 h            \~2 h/seed

  DPO training × 3 seeds        A100 40GB   \~9 h            \~3 h/seed

  MedMCQA eval on 6 adapted     A100 40GB   \~9 h            \~1.5 h/model
  models                                                     

  MedQA eval on 6 adapted       A100 40GB   \~3 h            \~0.5 h/model
  models                                                     

  Safety eval on 6 adapted      A100 40GB   \~6 h            \~1 h/model,
  models                                                     incl. judge

  KL and weight-norm audits     A100 40GB   \~2 h            Cheap

  LR sensitivity (2 LRs × 2     A100 40GB   \~10 h           Optional / Week 6
  conds × 1 seed)                                            

  Total (core, excl.            ---         \~41 h           Fits in Colab Pro
  sensitivity)                                               monthly

  Total (with sensitivity)      ---         \~51 h           Assumes Pro
                                                             headroom
  ----------------------------- ----------- ---------------- -----------------

**6.2 What happens if Colab Pro caps hit**

The pipeline is designed to be fully resumable at every stage (matches
the candidate\'s existing GSM8K infrastructure): baseline evals,
preference construction, each of the 6 training runs, and each
evaluation pass are individually checkpointed to Google Drive. A cap-hit
at any point loses at most the current stage, not the study. If A100
access is unavailable in a given session, L4 22GB is a drop-in fallback
with roughly 2× wall-clock; the plan\'s week-by-week timeline (§7) has
\~30% slack to absorb this.

**7. Timeline**

Six-week schedule assuming \~15--20 hours/week. Milestones marked M are
hard commits: each requires a repository commit and a written checkpoint
note before the next week begins.

  ---------- ----------------------- -----------------------------------------
  **Week**   **Focus**               **Key milestones**

  1          Infrastructure and      M1: Repo scaffolded, plan committed. M2:
             dataset verification    Base model, MedMCQA, MedQA, safety
                                     benchmarks verified accessible and
                                     cached. M3: Baseline evals (medical +
                                     safety) complete and cached.

  2          Data construction       M4: Adaptation subset frozen
                                     (adaptation_ids.json committed). M5: SFT
                                     dataset and DPO preference pairs
                                     constructed, cached, and spot-checked for
                                     quality (30 pairs manually reviewed).

  3          Training                M6: 3 SFT runs complete, checkpoints in
                                     Drive. M7: 3 DPO runs complete,
                                     checkpoints in Drive.

  4          Evaluation              M8: MedMCQA + MedQA eval on all 6 models.
                                     M9: Safety eval on all 6 models, judge
                                     outputs cached. M10: Judge--human κ
                                     spot-check complete.

  5          Analysis and            M11: Statistical analysis executed via
             mechanistic follow-up   src/stats.py. M12: KL and LoRA-norm
                                     audits complete. M13: Discordant-example
                                     qualitative coding complete. M14: All
                                     figures generated.

  6          Report and buffer       M15: Technical report v1 drafted. M16:
                                     Sensitivity check (LR or base-model
                                     fallback) if time permits. M17: Report
                                     finalised, arXiv preprint prepared, repo
                                     README polished.
  ---------- ----------------------- -----------------------------------------

**7.1 Week 1 dataset-verification checklist**

Week 1 is intentionally front-loaded with verification because the
study\'s biggest schedule risk is discovering a dataset access issue
mid-study. Verification items:

-   Llama-3.2-3B-Instruct downloadable and loadable in 4-bit on Colab
    Pro A100.

-   Qwen2.5-3B-Instruct verified as usable fallback.

-   MedMCQA train and validation splits loadable via HuggingFace
    datasets.

-   MedQA-USMLE test split accessible.

-   AdvBench-Harmful accessible; if not, HarmBench-standard verified as
    fallback.

-   MedSafetyBench accessible; if not, hand-curated 200-prompt set
    drafted from published taxonomies and committed to the repository
    before Week 2.

-   Llama-Guard-3-8B downloadable and loadable in 4-bit.

**8. Risks and Mitigations**

  ---------------------------- ---------------- -------------------------------
  **Risk**                     **Likelihood**   **Mitigation**

  MedSafetyBench access        Medium           Hand-curated 200-prompt
  blocked or benchmark                          fallback drafted from published
  deprecated                                    taxonomies; committed before
                                                Week 2.

  Utility-parity constraint    Medium--high     Pre-specified Pareto-plot
  fails (utilities differ by                    exploratory analysis (§3.3).
  \>2 pp)                                       Study still produces a
                                                defensible artifact.

  Judge--human agreement below Low--medium      Fall back to full manual
  κ = 0.7                                       annotation of 400 safety
                                                responses; adds \~6 hours of
                                                work.

  Colab compute cap hits       Medium           Fully resumable pipeline,
  mid-study                                     checkpointed to Drive; L4
                                                fallback with 2× wall-clock;
                                                30% schedule slack.

  Both DPO and SFT preserve    Medium           Null result is publishable and
  safety equally (small                         directly relevant to \"The
  effect)                                       Forgotten Shield\" motivation.

  DPO underperforms SFT on     Low--medium      Utility-parity constraint
  utility (large gap)                           prevents unfair safety
                                                comparison; Pareto plot reports
                                                the trade-off honestly.

  LoRA rank too small to       Low              Rank 16 with all-projection
  observe meaningful                            targets is standard; validated
  adaptation                                    on GSM8K prior work.

  Base model already refuses   Low              Baseline eval in Week 1
  too much (ceiling effect on                   measures this; if base safety
  safety)                                       rate \>95% on both benchmarks,
                                                switch to a harder safety
                                                subset (documented pivot).
  ---------------------------- ---------------- -------------------------------

**9. Deliverables**

**9.1 Repository structure**

dpo-vs-sft-medical-safety/

├── README.md \# Study overview, results summary, reproduction

├── PLAN.md \# This document (Markdown copy)

├── PLAN.pdf \# This document (locked pre-registration)

├── REPORT.md \# Technical report (drafted Week 6)

├── configs/

│ ├── sft.yaml \# SFT training config

│ ├── dpo.yaml \# DPO training config

│ └── seeds.yaml \# Fixed seeds for all randomness

├── data/

│ ├── adaptation_ids.json \# Frozen 4,000-question sample

│ ├── safety_general.jsonl \# 200 general safety prompts

│ └── safety_medical.jsonl \# 200 medical safety prompts

├── src/

│ ├── data_prep.py \# SFT + DPO dataset construction

│ ├── train_sft.py \# SFT training

│ ├── train_dpo.py \# DPO training

│ ├── eval_medical.py \# MedMCQA + MedQA evaluation

│ ├── eval_safety.py \# Safety generation + Llama-Guard judgement

│ ├── mech_kl.py \# Per-token KL from base

│ ├── mech_norms.py \# LoRA Frobenius audit

│ └── stats.py \# McNemar, bootstrap CIs, Spearman

├── notebooks/

│ └── run_all_colab.ipynb \# Single-notebook end-to-end runner

├── results/ \# Cached per-model outputs (JSONL)

├── figures/ \# Generated plots (PNG + PDF)

└── tests/ \# Sanity tests for data prep and stats

**9.2 Technical report structure**

-   Abstract (1 paragraph)

-   1\. Introduction and contribution (\~1 page)

-   2\. Background: DPO, SFT, safety alignment in medical LLMs (\~1.5
    pages)

-   3\. Method: pre-registered protocol summary, with §-level references
    to this plan (\~1.5 pages)

-   4\. Results: primary decision, utility parity, safety benchmarks by
    category (\~3 pages, with tables and figures)

-   5\. Mechanistic analysis: KL, LoRA norms, correlation with safety
    retention (\~2 pages)

-   6\. Qualitative analysis: discordant-example coding (\~1 page)

-   7\. Sensitivity checks and limitations (\~1 page)

-   8\. Discussion: implications for parameter-space interventions à la
    \"The Forgotten Shield\" (\~1 page)

-   9\. Conclusion (\~0.5 page)

-   References + appendices (unlimited)

**9.3 What \"done\" looks like**

The study is complete when: (a) all 6 adapted models exist and are
cached, (b) all evaluations are complete and cached, (c) src/stats.py
has been executed and its output is committed, (d) the report references
the plan\'s decision rules and states outcomes against them, and (e) the
repository is publicly linkable with a stable commit hash cited in the
HKPFS email.

**10. Pre-Registration Lock**

Before the first adapted model is trained, the following are committed
to the repository (git-signed by the candidate). This list is the
pre-registration lock:

-   This document (PLAN.md + PLAN.pdf) with commit hash cited in the
    report.

-   src/stats.py implementing the exact statistical protocol in §5.4.

-   configs/sft.yaml and configs/dpo.yaml with all hyperparameters in
    §5.2.

-   configs/seeds.yaml with the three seeds and sampling seed for
    adaptation_ids.

-   data/adaptation_ids.json with the frozen 4,000-question sample.

-   data/safety_general.jsonl and data/safety_medical.jsonl with the 400
    evaluation prompts.

-   The discordant-example coding rubric (5 categories, fixed before
    results are seen).

-   Baseline evaluation results (base model on MedMCQA, MedQA, and both
    safety sets).

The candidate commits to reporting all pre-registered analyses, whether
or not they yield the hypothesised direction. Any post-hoc analysis
added after results are seen is labelled \"exploratory\" in the report
and clearly separated from confirmatory analyses.

*--- End of research plan v1.0 ---*
