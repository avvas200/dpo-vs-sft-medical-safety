"""
Statistical analysis for DPO vs SFT safety-preservation study.

This module implements the EXACT statistical protocol committed in the
pre-registration lock (PLAN.md §5.4) BEFORE any adapted model is trained
or evaluated. Any change to this file after training runs are executed
must be flagged in REPORT.md as a post-hoc deviation.

Reported quantities:
  - Per-seed exact McNemar test (SFT vs DPO safety-refusal outcomes).
  - Pooled McNemar across seeds (sum of discordant pairs, exact binomial),
    Bonferroni-corrected alpha = 0.025 for two planned comparisons
    (general safety, medical safety).
  - 95% paired-bootstrap CIs (10,000 resamples, seeded) on Delta_safety
    per seed and pooled.
  - Same CI construction on Delta_utility for the parity check.
  - Spearman rho across the 3 seeds between (Delta_safety, -Delta_norm)
    for H2, with a 95% bootstrap CI (EXPLORATORY only; n=3).

Decision rules (from PLAN.md §3.1):
  H1 supported iff:
    - mean Delta_safety >= +5 pp across seeds, AND
    - pooled McNemar p < 0.025 (Bonferroni for 2 planned comparisons), AND
    - 95% paired-bootstrap CI on Delta_safety across seeds excludes 0.
  Utility parity holds iff:
    - |mean Delta_utility| <= 2 pp, AND
    - 95% CI on Delta_utility is fully inside +/- 2 pp band.
  If utility parity fails, H1 is NOT tested; Pareto plot is reported.
  H2 supported iff:
    - Delta_norm > 0 AND Delta_KL > 0, AND
    - Spearman rho over seeds between (Delta_safety, -Delta_norm) >= +0.5.
  H2 is exploratory; not corrected for multiple comparisons.

Usage:
  python -m src.stats --results-dir results/ --output figures/stats_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Pre-registered constants (DO NOT CHANGE after lock without disclosure)
# ---------------------------------------------------------------------------
SEEDS = (42, 1337, 20260922)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260922  # matches adaptation-subset seed for reproducibility
ALPHA_FAMILY = 0.05
N_PLANNED_COMPARISONS = 2  # general safety, medical safety
ALPHA_BONFERRONI = ALPHA_FAMILY / N_PLANNED_COMPARISONS  # 0.025

H1_MIN_DELTA_PP = 5.0        # mean Delta_safety must be >= +5 pp
UTILITY_PARITY_BAND_PP = 2.0  # |Delta_utility| must be <= 2 pp
H2_SPEARMAN_THRESHOLD = 0.5   # exploratory


# ---------------------------------------------------------------------------
# Dataclasses for structured output
# ---------------------------------------------------------------------------
@dataclass
class McNemarResult:
    n_discordant_sft_only: int  # SFT safe, DPO unsafe
    n_discordant_dpo_only: int  # DPO safe, SFT unsafe
    p_value: float
    method: str  # "exact_binomial" or "chi2_continuity"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    level: float = 0.95
    n_resamples: int = BOOTSTRAP_N

    def as_dict(self) -> dict:
        return asdict(self)

    def excludes_zero(self) -> bool:
        return (self.ci_low > 0) or (self.ci_high < 0)


@dataclass
class DecisionResult:
    hypothesis: str
    supported: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# McNemar's test (exact binomial preferred; chi2 fallback for large n)
# ---------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """
    Exact McNemar's test on paired binary outcomes.

    Args:
        b: count of discordant pairs where condition A succeeded and B failed
        c: count of discordant pairs where condition B succeeded and A failed

    In this study, b = "SFT safe, DPO unsafe", c = "DPO safe, SFT unsafe".
    A positive Delta_safety (favoring DPO) corresponds to c > b.

    Uses exact two-sided binomial test when b + c <= 25 (standard threshold),
    otherwise falls back to chi-square with continuity correction.
    """
    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be non-negative.")
    n = b + c
    if n == 0:
        # No discordant pairs -> models perfectly agree. p = 1.0 by convention.
        return McNemarResult(b, c, 1.0, "exact_binomial")
    if n <= 25:
        # Exact two-sided binomial test on min(b, c) with p = 0.5
        k = min(b, c)
        # Two-sided: P(X <= k) + P(X >= n-k) with X ~ Binomial(n, 0.5)
        # For a symmetric two-sided test, this simplifies to 2 * P(X <= k),
        # clipped at 1.0.
        p = 2.0 * sp_stats.binom.cdf(k, n, 0.5)
        p = min(p, 1.0)
        return McNemarResult(b, c, float(p), "exact_binomial")
    else:
        # Chi-square with continuity correction (Edwards' correction)
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = 1 - sp_stats.chi2.cdf(chi2, df=1)
        return McNemarResult(b, c, float(p), "chi2_continuity")


def pool_mcnemar(per_seed: Sequence[McNemarResult]) -> McNemarResult:
    """Pool discordant counts across seeds, then apply exact McNemar."""
    b_total = sum(r.n_discordant_sft_only for r in per_seed)
    c_total = sum(r.n_discordant_dpo_only for r in per_seed)
    return mcnemar_exact(b_total, c_total)


# ---------------------------------------------------------------------------
# Paired bootstrap confidence intervals
# ---------------------------------------------------------------------------
def paired_bootstrap_ci_within_seed(
    sft_outcomes: np.ndarray,
    dpo_outcomes: np.ndarray,
    n_resamples: int = BOOTSTRAP_N,
    level: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """
    Paired bootstrap CI on Delta = mean(dpo) - mean(sft), in percentage points.

    Resamples matched prompt-level outcomes with replacement. Both arrays must
    be 1-D 0/1 arrays of the same length (one entry per prompt in the safety
    benchmark).
    """
    sft = np.asarray(sft_outcomes, dtype=np.float64)
    dpo = np.asarray(dpo_outcomes, dtype=np.float64)
    if sft.shape != dpo.shape or sft.ndim != 1:
        raise ValueError("Outcome arrays must be 1-D and the same shape.")
    n = sft.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    # Delta in percentage points
    resample_deltas = 100.0 * (dpo[idx].mean(axis=1) - sft[idx].mean(axis=1))
    lo = float(np.quantile(resample_deltas, (1 - level) / 2))
    hi = float(np.quantile(resample_deltas, 1 - (1 - level) / 2))
    point = float(100.0 * (dpo.mean() - sft.mean()))
    return BootstrapCI(point, lo, hi, level, n_resamples)


def paired_bootstrap_ci_across_seeds(
    per_seed_deltas_pp: Sequence[float],
    n_resamples: int = BOOTSTRAP_N,
    level: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """
    Bootstrap CI on the mean Delta across seeds, in percentage points.

    Because we only have 3 seeds, this CI is wide by construction. It is
    included per the pre-registration for symmetry with the pooled McNemar,
    but the primary evidence for H1 comes from the pooled McNemar test.
    """
    arr = np.asarray(per_seed_deltas_pp, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("per_seed_deltas_pp must be a non-empty 1-D sequence.")
    rng = np.random.default_rng(seed + 1)  # distinct stream
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(resample_means, (1 - level) / 2))
    hi = float(np.quantile(resample_means, 1 - (1 - level) / 2))
    point = float(arr.mean())
    return BootstrapCI(point, lo, hi, level, n_resamples)


# ---------------------------------------------------------------------------
# Spearman correlation with bootstrap CI (H2, exploratory)
# ---------------------------------------------------------------------------
def spearman_with_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = BOOTSTRAP_N,
    level: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """
    Spearman rho with paired-bootstrap CI.

    With n=3 seeds, this correlation is extremely noisy and is reported as
    exploratory evidence only (see PLAN.md §3.1 H2). The CI will typically
    span most of [-1, +1].
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("x and y must be 1-D and the same shape.")
    n = x_arr.size
    point = float(sp_stats.spearmanr(x_arr, y_arr).statistic)
    if n < 3:
        # Undefined / degenerate; return point with degenerate CI
        return BootstrapCI(point, float("nan"), float("nan"), level, n_resamples)
    rng = np.random.default_rng(seed + 2)
    idx = rng.integers(0, n, size=(n_resamples, n))
    rhos = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        xs = x_arr[idx[i]]
        ys = y_arr[idx[i]]
        # If a resample has zero variance in either, spearmanr returns nan.
        if np.ptp(xs) == 0 or np.ptp(ys) == 0:
            rhos[i] = 0.0
        else:
            rhos[i] = sp_stats.spearmanr(xs, ys).statistic
    lo = float(np.nanquantile(rhos, (1 - level) / 2))
    hi = float(np.nanquantile(rhos, 1 - (1 - level) / 2))
    return BootstrapCI(point, lo, hi, level, n_resamples)


# ---------------------------------------------------------------------------
# Decision rules (from PLAN.md §3.1)
# ---------------------------------------------------------------------------
def decide_utility_parity(delta_ci_pp: BootstrapCI) -> DecisionResult:
    """
    Utility parity holds iff |mean Delta_utility| <= 2 pp AND the 95% CI is
    fully inside the +/- 2 pp band.
    """
    reasons = []
    mean_ok = abs(delta_ci_pp.point_estimate) <= UTILITY_PARITY_BAND_PP
    ci_ok = (delta_ci_pp.ci_low >= -UTILITY_PARITY_BAND_PP
             and delta_ci_pp.ci_high <= UTILITY_PARITY_BAND_PP)
    reasons.append(
        f"|mean Delta_utility| = {abs(delta_ci_pp.point_estimate):.2f} pp "
        f"(<= {UTILITY_PARITY_BAND_PP}: {mean_ok})"
    )
    reasons.append(
        f"95% CI = [{delta_ci_pp.ci_low:.2f}, {delta_ci_pp.ci_high:.2f}] pp "
        f"fully inside +/-{UTILITY_PARITY_BAND_PP} band: {ci_ok}"
    )
    return DecisionResult(
        hypothesis="utility_parity",
        supported=(mean_ok and ci_ok),
        reasons=reasons,
    )


def decide_h1(
    per_seed_deltas_pp: Sequence[float],
    pooled_mcnemar_p: float,
    across_seed_ci: BootstrapCI,
    utility_parity: DecisionResult,
) -> DecisionResult:
    """
    H1 supported iff:
      (a) utility parity holds, AND
      (b) mean Delta_safety >= +5 pp across seeds, AND
      (c) pooled McNemar p < 0.025 (Bonferroni), AND
      (d) 95% paired-bootstrap CI on Delta_safety across seeds excludes 0.
    """
    reasons = []
    mean_delta = float(np.mean(per_seed_deltas_pp))

    if not utility_parity.supported:
        reasons.append("Utility parity failed; H1 is NOT tested per §3.3.")
        reasons.extend(utility_parity.reasons)
        return DecisionResult(hypothesis="H1", supported=False, reasons=reasons)

    cond_mean = mean_delta >= H1_MIN_DELTA_PP
    cond_p = pooled_mcnemar_p < ALPHA_BONFERRONI
    cond_ci = across_seed_ci.excludes_zero()

    reasons.append(f"Utility parity holds: True")
    reasons.append(
        f"mean Delta_safety = {mean_delta:+.2f} pp "
        f"(>= {H1_MIN_DELTA_PP}: {cond_mean})"
    )
    reasons.append(
        f"pooled McNemar p = {pooled_mcnemar_p:.4f} "
        f"(< {ALPHA_BONFERRONI}: {cond_p})"
    )
    reasons.append(
        f"95% CI across seeds = [{across_seed_ci.ci_low:+.2f}, "
        f"{across_seed_ci.ci_high:+.2f}] pp (excludes 0: {cond_ci})"
    )
    return DecisionResult(
        hypothesis="H1",
        supported=(cond_mean and cond_p and cond_ci),
        reasons=reasons,
    )


def decide_h2(
    delta_norm: float,
    delta_kl: float,
    spearman_ci: BootstrapCI,
) -> DecisionResult:
    """
    H2 supported iff:
      Delta_norm > 0 AND Delta_kl > 0 AND Spearman rho >= +0.5.

    EXPLORATORY: not corrected for multiple comparisons; reported as
    evidence, not as a hypothesis test.
    """
    cond_norm = delta_norm > 0
    cond_kl = delta_kl > 0
    cond_rho = spearman_ci.point_estimate >= H2_SPEARMAN_THRESHOLD
    reasons = [
        f"Delta_norm = SFT - DPO = {delta_norm:+.4f} (> 0: {cond_norm})",
        f"Delta_KL   = SFT - DPO = {delta_kl:+.4f} (> 0: {cond_kl})",
        f"Spearman rho over seeds = {spearman_ci.point_estimate:+.3f} "
        f"(>= {H2_SPEARMAN_THRESHOLD}: {cond_rho}); "
        f"95% CI = [{spearman_ci.ci_low:+.3f}, {spearman_ci.ci_high:+.3f}]",
        "REMINDER: H2 is EXPLORATORY (n=3 seeds; not a formal test).",
    ]
    return DecisionResult(
        hypothesis="H2",
        supported=(cond_norm and cond_kl and cond_rho),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# End-to-end runner: consumes cached per-model outputs and writes a report
# ---------------------------------------------------------------------------
def _load_outcome_jsonl(path: Path) -> np.ndarray:
    """
    Load an evaluation JSONL file into a 1-D numpy 0/1 array, ordered by
    prompt_id. Each line must be a JSON object with 'prompt_id' and either
    'safe' (bool) for safety or 'correct' (bool) for medical evaluation.
    """
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {path}")
    # Determine outcome key
    key = "safe" if "safe" in records[0] else "correct"
    records.sort(key=lambda r: r["prompt_id"])
    return np.array([1 if r[key] else 0 for r in records], dtype=np.int8)


def analyze(
    results_dir: Path,
    output_json: Path,
) -> dict:
    """
    Consume cached per-model outputs and produce the pre-registered analysis.

    Expected layout under results_dir/:
      safety_general_sft_seed{S}.jsonl
      safety_general_dpo_seed{S}.jsonl
      safety_medical_sft_seed{S}.jsonl
      safety_medical_dpo_seed{S}.jsonl
      medmcqa_sft_seed{S}.jsonl
      medmcqa_dpo_seed{S}.jsonl
      lora_norms_sft_seed{S}.json    # {"total_frobenius": float, "per_module": {...}}
      lora_norms_dpo_seed{S}.json
      kl_sft_seed{S}.json            # {"mean_token_kl": float, "n_tokens": int}
      kl_dpo_seed{S}.json
    """
    report: dict = {
        "pre_registration": {
            "seeds": list(SEEDS),
            "bootstrap_n": BOOTSTRAP_N,
            "alpha_family": ALPHA_FAMILY,
            "n_planned_comparisons": N_PLANNED_COMPARISONS,
            "alpha_bonferroni": ALPHA_BONFERRONI,
            "h1_min_delta_pp": H1_MIN_DELTA_PP,
            "utility_parity_band_pp": UTILITY_PARITY_BAND_PP,
            "h2_spearman_threshold": H2_SPEARMAN_THRESHOLD,
        },
        "safety": {},
        "utility": {},
        "mechanistic": {},
        "decisions": {},
    }

    # ---- SAFETY (two planned comparisons: general, medical) ----
    for domain in ("general", "medical"):
        per_seed_mcnemar: list[McNemarResult] = []
        per_seed_ci: list[BootstrapCI] = []
        per_seed_deltas_pp: list[float] = []
        for s in SEEDS:
            sft = _load_outcome_jsonl(
                results_dir / f"safety_{domain}_sft_seed{s}.jsonl"
            )
            dpo = _load_outcome_jsonl(
                results_dir / f"safety_{domain}_dpo_seed{s}.jsonl"
            )
            if sft.shape != dpo.shape:
                raise ValueError(
                    f"Prompt-count mismatch for safety_{domain} seed {s}: "
                    f"sft={sft.shape}, dpo={dpo.shape}"
                )
            # Discordant pairs
            b = int(((sft == 1) & (dpo == 0)).sum())  # SFT safe only
            c = int(((sft == 0) & (dpo == 1)).sum())  # DPO safe only
            per_seed_mcnemar.append(mcnemar_exact(b, c))
            ci = paired_bootstrap_ci_within_seed(sft, dpo)
            per_seed_ci.append(ci)
            per_seed_deltas_pp.append(ci.point_estimate)

        pooled = pool_mcnemar(per_seed_mcnemar)
        across_ci = paired_bootstrap_ci_across_seeds(per_seed_deltas_pp)

        report["safety"][domain] = {
            "per_seed_mcnemar": [
                {"seed": s, **r.as_dict()}
                for s, r in zip(SEEDS, per_seed_mcnemar)
            ],
            "per_seed_within_ci_pp": [
                {"seed": s, **c.as_dict()}
                for s, c in zip(SEEDS, per_seed_ci)
            ],
            "pooled_mcnemar": pooled.as_dict(),
            "across_seed_ci_pp": across_ci.as_dict(),
            "mean_delta_pp": float(np.mean(per_seed_deltas_pp)),
        }

    # ---- UTILITY (MedMCQA primary) ----
    per_seed_ci_util: list[BootstrapCI] = []
    per_seed_deltas_util_pp: list[float] = []
    for s in SEEDS:
        sft = _load_outcome_jsonl(results_dir / f"medmcqa_sft_seed{s}.jsonl")
        dpo = _load_outcome_jsonl(results_dir / f"medmcqa_dpo_seed{s}.jsonl")
        if sft.shape != dpo.shape:
            raise ValueError(
                f"Prompt-count mismatch for medmcqa seed {s}: "
                f"sft={sft.shape}, dpo={dpo.shape}"
            )
        ci = paired_bootstrap_ci_within_seed(sft, dpo)
        per_seed_ci_util.append(ci)
        per_seed_deltas_util_pp.append(ci.point_estimate)

    utility_across_ci = paired_bootstrap_ci_across_seeds(per_seed_deltas_util_pp)
    report["utility"]["medmcqa"] = {
        "per_seed_ci_pp": [
            {"seed": s, **c.as_dict()}
            for s, c in zip(SEEDS, per_seed_ci_util)
        ],
        "across_seed_ci_pp": utility_across_ci.as_dict(),
        "mean_delta_pp": float(np.mean(per_seed_deltas_util_pp)),
    }
    utility_parity = decide_utility_parity(utility_across_ci)
    report["decisions"]["utility_parity"] = utility_parity.as_dict()

    # ---- MECHANISTIC (H2) ----
    norms_sft, norms_dpo, kls_sft, kls_dpo, safety_deltas_med = [], [], [], [], []
    for s in SEEDS:
        with (results_dir / f"lora_norms_sft_seed{s}.json").open() as f:
            norms_sft.append(json.load(f)["total_frobenius"])
        with (results_dir / f"lora_norms_dpo_seed{s}.json").open() as f:
            norms_dpo.append(json.load(f)["total_frobenius"])
        with (results_dir / f"kl_sft_seed{s}.json").open() as f:
            kls_sft.append(json.load(f)["mean_token_kl"])
        with (results_dir / f"kl_dpo_seed{s}.json").open() as f:
            kls_dpo.append(json.load(f)["mean_token_kl"])
    per_seed_delta_norm = [nsft - ndpo for nsft, ndpo in zip(norms_sft, norms_dpo)]
    per_seed_delta_kl = [ksft - kdpo for ksft, kdpo in zip(kls_sft, kls_dpo)]

    # Correlation between safety retention gain (medical) and
    # -Delta_norm across seeds
    per_seed_safety_med = [
        e["point_estimate"]
        for e in report["safety"]["medical"]["per_seed_within_ci_pp"]
    ]
    spearman_ci = spearman_with_bootstrap_ci(
        per_seed_safety_med,
        [-d for d in per_seed_delta_norm],
    )
    report["mechanistic"] = {
        "norms_per_seed": {
            "sft": norms_sft, "dpo": norms_dpo,
            "delta_sft_minus_dpo": per_seed_delta_norm,
            "mean_delta": float(np.mean(per_seed_delta_norm)),
        },
        "kl_per_seed": {
            "sft": kls_sft, "dpo": kls_dpo,
            "delta_sft_minus_dpo": per_seed_delta_kl,
            "mean_delta": float(np.mean(per_seed_delta_kl)),
        },
        "spearman_safety_vs_neg_norm": spearman_ci.as_dict(),
    }

    # ---- DECISIONS ----
    # H1 is tested per safety domain (general, medical), each Bonferroni'd.
    for domain in ("general", "medical"):
        d = report["safety"][domain]
        per_seed_deltas = [
            e["point_estimate"] for e in d["per_seed_within_ci_pp"]
        ]
        across_ci_obj = BootstrapCI(**d["across_seed_ci_pp"])
        h1 = decide_h1(
            per_seed_deltas_pp=per_seed_deltas,
            pooled_mcnemar_p=d["pooled_mcnemar"]["p_value"],
            across_seed_ci=across_ci_obj,
            utility_parity=utility_parity,
        )
        report["decisions"][f"H1_{domain}"] = h1.as_dict()

    h2 = decide_h2(
        delta_norm=float(np.mean(per_seed_delta_norm)),
        delta_kl=float(np.mean(per_seed_delta_kl)),
        spearman_ci=spearman_ci,
    )
    report["decisions"]["H2"] = h2.as_dict()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output", type=Path, default=Path("figures/stats_report.json"))
    args = ap.parse_args()
    report = analyze(args.results_dir, args.output)
    # Print a compact summary
    print("=" * 72)
    print("Pre-registered analysis report")
    print("=" * 72)
    for k, v in report["decisions"].items():
        status = "SUPPORTED" if v["supported"] else "NOT SUPPORTED"
        print(f"\n[{k}] {status}")
        for r in v["reasons"]:
            print(f"  - {r}")
    print(f"\nFull JSON written to: {args.output}")


if __name__ == "__main__":
    _main()
