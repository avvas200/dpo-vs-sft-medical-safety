"""
Tests for src/stats.py.

These tests must pass BEFORE any real evaluation results are generated.
They verify that the pre-registered statistical protocol behaves correctly
on known ground-truth cases and synthetic data with a known effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import stats


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def test_mcnemar_no_discordant_pairs():
    """Perfect agreement -> p = 1.0 (no evidence of difference)."""
    r = stats.mcnemar_exact(0, 0)
    assert r.p_value == 1.0
    assert r.method == "exact_binomial"


def test_mcnemar_symmetric_small():
    """b = c -> two-sided p should be ~1.0 (no directional evidence)."""
    r = stats.mcnemar_exact(5, 5)
    assert r.method == "exact_binomial"
    assert r.p_value > 0.9  # Very close to 1


def test_mcnemar_strong_asymmetry_small():
    """Extreme asymmetry with small n should give small p."""
    # 0 vs 10: probability of getting all 10 flips on one side under H0
    # is 2 * (0.5)^10 = 0.00195
    r = stats.mcnemar_exact(0, 10)
    assert r.method == "exact_binomial"
    assert r.p_value < 0.01
    assert r.p_value == pytest.approx(2 * 0.5**10, rel=1e-6)


def test_mcnemar_switches_to_chi2_for_large_n():
    """b + c > 25 should use chi2 continuity."""
    r = stats.mcnemar_exact(10, 20)
    assert r.method == "chi2_continuity"
    # Sanity: chi2 with continuity for b=10, c=20 gives moderate p
    assert 0.05 < r.p_value < 0.15


def test_mcnemar_p_never_exceeds_one():
    """Exact two-sided binomial should never exceed 1.0 (clip bug guard)."""
    # b = c is the case where 2 * P(X <= k) can exceed 1 without clipping
    for n in range(0, 20):
        r = stats.mcnemar_exact(n, n)
        assert 0.0 <= r.p_value <= 1.0, f"p={r.p_value} out of range at n={n}"


def test_mcnemar_negative_counts_rejected():
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 5)


def test_pool_mcnemar_sums_correctly():
    per_seed = [
        stats.mcnemar_exact(2, 5),
        stats.mcnemar_exact(3, 4),
        stats.mcnemar_exact(1, 6),
    ]
    pooled = stats.pool_mcnemar(per_seed)
    assert pooled.n_discordant_sft_only == 6
    assert pooled.n_discordant_dpo_only == 15
    # Pooled n=21, exact test on min(6,15)=6 with n=21
    # p = 2 * P(X <= 6 | n=21, p=0.5)
    from scipy.stats import binom
    expected_p = min(2 * binom.cdf(6, 21, 0.5), 1.0)
    assert pooled.p_value == pytest.approx(expected_p, rel=1e-9)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def test_bootstrap_within_seed_zero_effect():
    """Identical outcomes -> point estimate 0 and CI containing 0."""
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    b = a.copy()
    ci = stats.paired_bootstrap_ci_within_seed(a, b, n_resamples=2000, seed=1)
    assert ci.point_estimate == 0.0
    assert ci.ci_low <= 0 <= ci.ci_high
    assert not ci.excludes_zero()


def test_bootstrap_within_seed_large_effect():
    """DPO consistently better -> positive point estimate and CI excludes 0."""
    n = 400
    rng = np.random.default_rng(0)
    # SFT safe 40%, DPO safe 80% -> Delta ~ +40 pp
    sft = (rng.random(n) < 0.4).astype(np.int8)
    dpo = (rng.random(n) < 0.8).astype(np.int8)
    ci = stats.paired_bootstrap_ci_within_seed(sft, dpo, n_resamples=2000, seed=2)
    assert 30 < ci.point_estimate < 50
    assert ci.excludes_zero()
    assert ci.ci_low > 0


def test_bootstrap_within_seed_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_ci_within_seed(
            np.array([1, 0, 1]), np.array([1, 0]),
        )


def test_bootstrap_across_seeds_reproducible():
    """Same seed -> identical CI. Guards against non-determinism regressions."""
    deltas = [2.5, 3.1, 1.8]
    ci_a = stats.paired_bootstrap_ci_across_seeds(deltas, n_resamples=1000, seed=99)
    ci_b = stats.paired_bootstrap_ci_across_seeds(deltas, n_resamples=1000, seed=99)
    assert ci_a.point_estimate == ci_b.point_estimate
    assert ci_a.ci_low == ci_b.ci_low
    assert ci_a.ci_high == ci_b.ci_high


def test_bootstrap_across_seeds_point_matches_mean():
    deltas = [2.5, 3.1, 1.8]
    ci = stats.paired_bootstrap_ci_across_seeds(deltas, n_resamples=500, seed=7)
    assert ci.point_estimate == pytest.approx(np.mean(deltas))


def test_bootstrap_within_seed_deterministic_seed():
    """Fixed seed -> identical CI on both calls."""
    a = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    b = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
    ci1 = stats.paired_bootstrap_ci_within_seed(a, b, n_resamples=500, seed=123)
    ci2 = stats.paired_bootstrap_ci_within_seed(a, b, n_resamples=500, seed=123)
    assert ci1.as_dict() == ci2.as_dict()


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
def test_utility_parity_passes_when_small_and_tight():
    ci = stats.BootstrapCI(point_estimate=0.5, ci_low=-1.2, ci_high=1.5)
    d = stats.decide_utility_parity(ci)
    assert d.supported


def test_utility_parity_fails_when_mean_too_large():
    ci = stats.BootstrapCI(point_estimate=3.0, ci_low=1.5, ci_high=4.5)
    d = stats.decide_utility_parity(ci)
    assert not d.supported


def test_utility_parity_fails_when_ci_spans_band():
    """Mean is fine, but CI leaks outside the ±2 pp band."""
    ci = stats.BootstrapCI(point_estimate=1.0, ci_low=-1.5, ci_high=3.5)
    d = stats.decide_utility_parity(ci)
    assert not d.supported


def test_h1_not_tested_if_utility_parity_fails():
    utility = stats.DecisionResult("utility_parity", False, ["failed"])
    across_ci = stats.BootstrapCI(10.0, 5.0, 15.0)
    d = stats.decide_h1([9, 10, 11], 0.001, across_ci, utility)
    assert not d.supported
    assert "not tested" in d.reasons[0].lower()


def test_h1_all_conditions_met():
    utility = stats.DecisionResult("utility_parity", True, [])
    across_ci = stats.BootstrapCI(7.0, 3.0, 11.0)  # excludes 0
    d = stats.decide_h1([6, 7, 8], 0.005, across_ci, utility)
    assert d.supported


def test_h1_fails_when_mean_below_threshold():
    utility = stats.DecisionResult("utility_parity", True, [])
    across_ci = stats.BootstrapCI(3.0, 1.0, 5.0)  # excludes 0 but mean < 5 pp
    d = stats.decide_h1([3, 3, 3], 0.001, across_ci, utility)
    assert not d.supported


def test_h1_fails_when_p_above_bonferroni():
    utility = stats.DecisionResult("utility_parity", True, [])
    across_ci = stats.BootstrapCI(6.0, 2.0, 10.0)
    d = stats.decide_h1([6, 6, 6], 0.03, across_ci, utility)  # > 0.025
    assert not d.supported


def test_h1_fails_when_ci_includes_zero():
    utility = stats.DecisionResult("utility_parity", True, [])
    across_ci = stats.BootstrapCI(6.0, -1.0, 13.0)  # spans 0
    d = stats.decide_h1([6, 6, 6], 0.005, across_ci, utility)
    assert not d.supported


def test_h2_all_conditions_met():
    spearman = stats.BootstrapCI(0.9, 0.3, 1.0)
    d = stats.decide_h2(delta_norm=0.5, delta_kl=0.1, spearman_ci=spearman)
    assert d.supported


def test_h2_fails_when_norm_gap_wrong_sign():
    spearman = stats.BootstrapCI(0.9, 0.3, 1.0)
    d = stats.decide_h2(delta_norm=-0.5, delta_kl=0.1, spearman_ci=spearman)
    assert not d.supported


def test_h2_reports_exploratory_reminder():
    spearman = stats.BootstrapCI(0.9, 0.3, 1.0)
    d = stats.decide_h2(delta_norm=0.5, delta_kl=0.1, spearman_ci=spearman)
    assert any("EXPLORATORY" in r for r in d.reasons)


# ---------------------------------------------------------------------------
# End-to-end sanity with synthetic cached results
# ---------------------------------------------------------------------------
def _write_outcome_jsonl(path: Path, arr: np.ndarray, key: str = "safe") -> None:
    with path.open("w") as f:
        for i, v in enumerate(arr):
            f.write(json.dumps({"prompt_id": i, key: bool(v)}) + "\n")


def test_analyze_end_to_end_with_synthetic_data(tmp_path):
    """
    Fabricate a full results/ directory and verify analyze() runs and produces
    the expected structure.
    """
    results = tmp_path / "results"
    results.mkdir()

    rng = np.random.default_rng(42)
    for s in stats.SEEDS:
        # Safety: DPO safer than SFT by ~10 pp on both domains
        for domain in ("general", "medical"):
            n = 200
            sft = (rng.random(n) < 0.6).astype(np.int8)
            # dpo mostly matches sft but flips ~10% more into "safe"
            dpo = sft.copy()
            flip_idx = rng.choice(np.where(sft == 0)[0], size=n // 20, replace=False)
            dpo[flip_idx] = 1
            _write_outcome_jsonl(
                results / f"safety_{domain}_sft_seed{s}.jsonl", sft, key="safe"
            )
            _write_outcome_jsonl(
                results / f"safety_{domain}_dpo_seed{s}.jsonl", dpo, key="safe"
            )
        # Utility: matched
        n = 200
        sft_u = (rng.random(n) < 0.5).astype(np.int8)
        dpo_u = sft_u.copy()
        _write_outcome_jsonl(
            results / f"medmcqa_sft_seed{s}.jsonl", sft_u, key="correct"
        )
        _write_outcome_jsonl(
            results / f"medmcqa_dpo_seed{s}.jsonl", dpo_u, key="correct"
        )
        # Mechanistic
        with (results / f"lora_norms_sft_seed{s}.json").open("w") as f:
            json.dump({"total_frobenius": 5.0, "per_module": {}}, f)
        with (results / f"lora_norms_dpo_seed{s}.json").open("w") as f:
            json.dump({"total_frobenius": 3.0, "per_module": {}}, f)
        with (results / f"kl_sft_seed{s}.json").open("w") as f:
            json.dump({"mean_token_kl": 0.20, "n_tokens": 10000}, f)
        with (results / f"kl_dpo_seed{s}.json").open("w") as f:
            json.dump({"mean_token_kl": 0.10, "n_tokens": 10000}, f)

    out = tmp_path / "figures" / "stats_report.json"
    report = stats.analyze(results, out)

    # Basic structural checks
    assert out.exists()
    assert "pre_registration" in report
    assert "H1_general" in report["decisions"]
    assert "H1_medical" in report["decisions"]
    assert "H2" in report["decisions"]
    assert "utility_parity" in report["decisions"]
    # Utility parity should hold (constructed identical)
    assert report["decisions"]["utility_parity"]["supported"]
    # Mechanistic deltas should be positive (SFT > DPO by construction)
    assert report["mechanistic"]["norms_per_seed"]["mean_delta"] > 0
    assert report["mechanistic"]["kl_per_seed"]["mean_delta"] > 0
