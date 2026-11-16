"""Peeking simulation: known answers for each rule and calibration against published constants."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.bayes import prob_b_greater_a_exact
from abkit.frequentist import diff_in_means, diff_in_proportions
from abkit.sequential import (
    DAYS,
    METHODS,
    METRICS,
    PUBLISHED_K5,
    TAU_MULTIPLIERS,
    bayes_count_boundary,
    calibrate_boundaries,
    cumulative_rejection,
    difference_and_se,
    first_rejection,
    msprt_log_likelihood_ratio,
    msprt_p_values,
    msprt_z_boundary,
    prob_b_greater_a,
    rejections,
    run_sequential,
    simulate_rate_paths,
    simulate_spend_paths,
    summarise_rule,
    z_statistics,
)

N_FAST = 4_000
N_PER_ARM = 21_306
VISIT_RATE = 0.106
VISIT_TAU = 0.0084


def within_mcse(rate: float, mcse: float, target: float, k: float = 3.0) -> bool:
    return abs(rate - target) <= k * mcse


# --- fixtures: one calibration and one A/A simulation shared by the fast tests ----------------


@pytest.fixture(scope="module")
def constants14():
    return calibrate_boundaries(DAYS, 200_000, np.random.default_rng(config.SEEDS["fixture"]))


@pytest.fixture(scope="module")
def aa_visit(constants14):
    rng = np.random.default_rng(config.SEEDS["fixture"] + 1)
    paths = simulate_rate_paths(VISIT_RATE, VISIT_RATE, N_PER_ARM, DAYS, N_FAST, rng)
    rules = rejections(paths, constants14, VISIT_TAU)
    return {name: summarise_rule(r) for name, r in rules.items()}


# --- the four required checks ------------------------------------------------------------------


def test_fixed_horizon_false_positive_rate_is_alpha(aa_visit):
    row = aa_visit["fixed"]
    assert within_mcse(row["reject_rate"], row["reject_mcse"], config.ALPHA)
    assert row["mean_stop_day"] == DAYS


def test_calibrated_constants_match_published_values_for_k5():
    c = calibrate_boundaries(5, 200_000, np.random.default_rng(config.SEEDS["fixture"] + 2))
    assert c.pocock == pytest.approx(PUBLISHED_K5["pocock"], abs=0.02)
    assert c.obf == pytest.approx(PUBLISHED_K5["obf"], abs=0.02)
    assert c.pocock_mcse < 0.01 and c.obf_mcse < 0.01
    assert c.obf < c.pocock  # the final O'Brien-Fleming boundary is the looser of the two


def test_single_look_constants_reduce_to_the_normal_quantile():
    c = calibrate_boundaries(1, 200_000, np.random.default_rng(config.SEEDS["fixture"] + 3))
    z = stats.norm.isf(config.ALPHA / 2)
    assert c.pocock == pytest.approx(z, abs=0.02) and c.obf == pytest.approx(z, abs=0.02)
    assert c.pocock == c.obf


@pytest.mark.parametrize("method", ["pocock", "obf", "msprt"])
def test_valid_sequential_methods_keep_the_false_positive_rate(aa_visit, method):
    row = aa_visit[method]
    assert row["reject_rate"] <= config.ALPHA + 3 * row["reject_mcse"]


def test_naive_peeking_with_daily_looks_inflates_the_false_positive_rate(aa_visit):
    row = aa_visit["naive"]
    assert row["reject_rate"] - 3 * row["reject_mcse"] > 0.10
    assert aa_visit["bayes"]["reject_rate"] > row["reject_rate"]  # a looser threshold still


# --- Bayesian rule: quadrature against the closed form, boundary against brute force -----------


CASES = [
    (151, 1401, 171, 1381),
    (1, 701, 5, 697),
    (1, 711, 4, 708),
    (2261, 19047, 2301, 19007),
    (10, 1513, 5, 1518),
    (1, 1, 1, 1),
    (3, 20, 7, 16),
]


@pytest.mark.parametrize(("aa", "ba", "ab", "bb"), CASES)
def test_prob_b_greater_a_matches_the_closed_form(aa, ba, ab, bb):
    assert prob_b_greater_a(aa, ba, ab, bb) == pytest.approx(
        prob_b_greater_a_exact(aa, ba, ab, bb), abs=1e-8
    )


def test_prob_b_greater_a_is_vectorised_and_symmetric():
    a = np.array([[151.0, 1.0, 3.0]]).T
    b = np.array([[1401.0, 701.0, 20.0]]).T
    forward = prob_b_greater_a(a, b, a.T + 20, b.T + 5)
    backward = prob_b_greater_a(a.T + 20, b.T + 5, a, b)
    assert forward.shape == (3, 3)
    np.testing.assert_allclose(forward + backward, 1.0, atol=1e-8)
    assert prob_b_greater_a(2, 3, 2, 3) == pytest.approx(0.5, abs=1e-8)


@pytest.mark.parametrize("x_a", [0, 1, 10, 30, 60, 150])
def test_bayes_count_boundary_is_the_first_count_over_the_threshold(x_a):
    n, threshold = 300, config.BAYES_P_BETTER
    g = int(bayes_count_boundary([x_a], n, threshold)[0])
    probs = prob_b_greater_a(1 + x_a, 1 + n - x_a, 1 + np.arange(n + 1), 1 + n - np.arange(n + 1))
    expected = int(np.argmax(probs > threshold)) if (probs > threshold).any() else n + 1
    assert g == expected


def test_bayes_rejections_match_cell_by_cell_probabilities():
    rng = np.random.default_rng(7)
    paths = simulate_rate_paths(0.02, 0.025, 2_000, 4, 300, rng)
    constants = calibrate_boundaries(4, 5_000, rng)
    rule = rejections(paths, constants, tau=0.01)["bayes"]
    n = paths.n[None, :]
    p = prob_b_greater_a(1 + paths.sum_a, 1 + n - paths.sum_a, 1 + paths.sum_b, 1 + n - paths.sum_b)
    expected = (p > config.BAYES_P_BETTER) | (p < 1 - config.BAYES_P_BETTER)
    assert rule.sum() > 0
    np.testing.assert_array_equal(rule, expected)


def test_spend_bayesian_rule_equals_naive_peeking_at_twice_the_one_sided_level():
    rng = np.random.default_rng(11)
    values = np.concatenate([np.zeros(500), rng.uniform(20, 400, 6)])
    paths = simulate_spend_paths(values, 3_000, 5, 500, rng)
    constants = calibrate_boundaries(5, 5_000, rng)
    bayes = rejections(paths, constants, tau=0.3, threshold=0.95)["bayes"]
    naive_10 = rejections(paths, constants, tau=0.3, alpha=0.10)["naive"]
    assert bayes.sum() > 0
    np.testing.assert_array_equal(bayes, naive_10)


# --- mSPRT -------------------------------------------------------------------------------------


def test_msprt_log_likelihood_ratio_hand_formula():
    diff, se, tau = 0.012, 0.004, 0.01
    v = se**2
    expected = 0.5 * math.log(v / (v + tau**2)) + tau**2 * diff**2 / (2 * v * (v + tau**2))
    assert msprt_log_likelihood_ratio(diff, se, tau) == pytest.approx(expected, rel=1e-12)
    assert msprt_log_likelihood_ratio(0.0, 0.0, tau) == 0.0
    with pytest.raises(ValueError):
        msprt_log_likelihood_ratio(diff, se, 0.0)


def test_msprt_p_values_are_running_minima_in_unit_interval():
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 0.01, (200, 6))
    se = np.full((200, 6), 0.005) / np.sqrt(np.arange(1, 7))
    p = msprt_p_values(diff, se, tau=0.01)
    assert ((p > 0) & (p <= 1)).all()
    assert (np.diff(p, axis=1) <= 0).all()
    first = np.minimum(1.0, np.exp(-msprt_log_likelihood_ratio(diff, se, 0.01)))[:, 0]
    assert (p[:, 0] == first).all()


def test_msprt_z_boundary_inverts_the_likelihood_ratio():
    v, tau, alpha = 1.2e-5, 0.0084, 0.05
    z = msprt_z_boundary(v, tau, alpha)
    log_lr = msprt_log_likelihood_ratio(z * math.sqrt(v), math.sqrt(v), tau)
    assert math.exp(log_lr) == pytest.approx(1 / alpha, rel=1e-10)


# --- simulators and statistics -----------------------------------------------------------------


def test_rate_paths_are_cumulative_binomials():
    rng = np.random.default_rng(5)
    paths = simulate_rate_paths(0.3, 1.0, 1_000, 4, 50, rng)
    np.testing.assert_array_equal(paths.n, [250, 500, 750, 1000])
    assert (np.diff(paths.sum_a, axis=1) >= 0).all()
    np.testing.assert_array_equal(paths.sum_b, np.tile(paths.n, (50, 1)))
    assert paths.binary and paths.n_runs == 50 and paths.n_looks == 4
    with pytest.raises(ValueError):
        simulate_rate_paths(0.3, 0.3, 10, 20, 5, rng)


def test_spend_paths_resample_the_control_values():
    rng = np.random.default_rng(9)
    values = np.concatenate([np.zeros(980), [30.0, 45.0, 120.0, 499.0] * 5])
    n, runs = 4_000, 2_000
    paths = simulate_spend_paths(values, n, 4, runs, rng)
    assert not paths.binary and (np.diff(paths.sumsq_a, axis=1) >= 0).all()
    # Each simulated customer is a draw from `values`, so E[sum] = n mean and E[sumsq] = n E[v²].
    for total, moment in ((paths.sum_a, values.mean()), (paths.sumsq_b, (values**2).mean())):
        last = total[:, -1]
        assert abs(last.mean() - n * moment) < 4 * last.std(ddof=1) / math.sqrt(runs)
    with pytest.raises(ValueError):
        simulate_spend_paths([-1.0, 2.0], 100, 2, 5, rng)


def test_difference_and_se_match_the_readout_tests():
    rng = np.random.default_rng(13)
    rates = simulate_rate_paths(0.1, 0.12, 5_000, 3, 20, rng)
    diff, se = difference_and_se(rates)
    ref = diff_in_proportions(rates.sum_b, rates.n, rates.sum_a, rates.n)
    np.testing.assert_allclose(z_statistics(diff, se), ref.statistic, rtol=1e-12)
    values = np.concatenate([np.zeros(400), rng.uniform(10, 300, 8)])
    spend = simulate_spend_paths(values, 6_000, 2, 10, rng)
    diff, se = difference_and_se(spend)
    n = spend.n
    mean_a, mean_b = spend.sum_a / n, spend.sum_b / n
    var_a = (spend.sumsq_a - n * mean_a**2) / (n - 1)
    var_b = (spend.sumsq_b - n * mean_b**2) / (n - 1)
    ref = diff_in_means(mean_b, var_b, n, mean_a, var_a, n)
    np.testing.assert_allclose(z_statistics(diff, se), ref.statistic, rtol=1e-10)


def test_first_rejection_and_summaries_on_a_hand_made_matrix():
    reject = np.array(
        [
            [False, True, True],
            [False, False, False],
            [True, False, True],
            [False, False, True],
        ]
    )
    np.testing.assert_array_equal(first_rejection(reject), [1, -1, 0, 2])
    row = summarise_rule(reject)
    assert row["reject_rate"] == 0.75
    assert row["reject_mcse"] == pytest.approx(math.sqrt(0.75 * 0.25 / 4))
    assert row["mean_stop_day"] == pytest.approx((2 + 3 + 1 + 3) / 4)
    assert row["mean_stop_day_rejecting"] == pytest.approx((2 + 1 + 3) / 3)
    rate, mcse = cumulative_rejection(reject)
    np.testing.assert_allclose(rate, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(mcse, np.sqrt(rate * (1 - rate) / 4))


def test_fixed_horizon_power_at_a_planted_effect_matches_the_closed_form(constants14):
    rng = np.random.default_rng(config.SEEDS["fixture"] + 4)
    p_a, delta = VISIT_RATE, 0.0084
    paths = simulate_rate_paths(p_a, p_a + delta, N_PER_ARM, DAYS, N_FAST, rng)
    row = summarise_rule(rejections(paths, constants14, VISIT_TAU)["fixed"])
    # Pooled z-test under the alternative: the null se is pooled, the true se is not.
    n, p_b = paths.n[-1], p_a + delta
    se_true = math.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / n)
    p_bar = (p_a + p_b) / 2
    crit = stats.norm.isf(config.ALPHA / 2) * math.sqrt(2 * p_bar * (1 - p_bar) / n) / se_true
    expected = stats.norm.sf(crit - delta / se_true) + stats.norm.cdf(-crit - delta / se_true)
    assert within_mcse(row["reject_rate"], row["reject_mcse"], expected)
    assert 0.7 < expected < 0.85


def test_rejections_require_matching_constants(constants14):
    rng = np.random.default_rng(1)
    paths = simulate_rate_paths(0.1, 0.1, 1_000, 5, 10, rng)
    with pytest.raises(ValueError):
        rejections(paths, constants14, tau=0.01)


def test_rejections_can_apply_a_subset_of_the_rules_unchanged():
    rng = np.random.default_rng(11)
    paths = simulate_rate_paths(0.05, 0.055, 6_000, 6, 500, rng)
    constants = calibrate_boundaries(6, 5_000, rng)
    every = rejections(paths, constants, tau=0.01)
    some = rejections(paths, constants, tau=0.01, methods=["obf", "naive"])
    assert list(some) == ["obf", "naive"]
    for name, flags in some.items():
        np.testing.assert_array_equal(flags, every[name])
    with pytest.raises(ValueError, match="Unknown methods"):
        rejections(paths, constants, tau=0.01, methods=["naive", "coin_flip"])


@pytest.mark.parametrize(
    ("p_a", "p_b", "n_per_arm"),
    [(0.10, 0.10, 7_000), (0.10, 0.11, 7_000), (0.006, 0.006, 21_000), (0.30, 0.30, 140)],
)
def test_bayes_rule_per_run_equals_the_count_boundary_route(p_a, p_b, n_per_arm):
    rng = np.random.default_rng(5)
    paths = simulate_rate_paths(p_a, p_b, n_per_arm, 14, 1_500, rng)
    constants = calibrate_boundaries(14, 5_000, rng)
    boundary = rejections(paths, constants, tau=0.01, methods=["bayes"])["bayes"]
    per_run = rejections(paths, constants, tau=0.01, methods=["bayes"], bayes_per_run=True)["bayes"]
    assert boundary.any() and not boundary.all()
    np.testing.assert_array_equal(per_run, boundary)


# --- end to end on the fixture (fast mode) -----------------------------------------------------


def test_run_sequential_writes_outputs(fixture_db, tmp_path):
    main = run_sequential(
        fixture_db, tmp_path, n_runs=200, n_runs_looks=100, n_paths=2_000, max_looks=DAYS
    )
    scenarios = {"aa": set(METRICS), "ab_observed": {"visit"}, "ab_mde": {"visit"}}
    n_rules = len(METHODS) - 1 + len(TAU_MULTIPLIERS)
    assert len(main) == sum(len(m) for m in scenarios.values()) * n_rules
    for scenario, metrics in scenarios.items():
        assert set(main.loc[main["scenario"] == scenario, "metric"]) == metrics
    assert set(main["method"]) == set(METHODS)
    assert (main["reject_mcse"] <= 0.5 / math.sqrt(200)).all()
    assert (main.loc[main["method"] == "fixed", "mean_stop_day"] == DAYS).all()
    for name in ("sequential", "sequential_looks", "sequential_by_day", "sequential_boundaries"):
        assert (tmp_path / "tables" / f"{name}.csv").stat().st_size > 0
    for name in ("fpr_vs_looks", "cumulative_rejection", "boundaries"):
        assert (tmp_path / "figures" / f"sequential_{name}.png").stat().st_size > 0
    looks = pd.read_csv(tmp_path / "tables" / "sequential_looks.csv")
    assert len(looks) == DAYS * len(METRICS) * len(METHODS)
    assert looks["n_looks"].min() == 1 and looks["n_looks"].max() == DAYS
    by_day = pd.read_csv(tmp_path / "tables" / "sequential_by_day.csv")
    assert set(by_day["method"]) == {*METHODS, "single_look"}
    cumulative = by_day[by_day["method"] != "single_look"].sort_values("day")
    steps = cumulative.groupby(["scenario", "metric", "method"])["reject_rate"].diff().dropna()
    assert (steps >= 0).all()
    boundaries = pd.read_csv(tmp_path / "tables" / "sequential_boundaries.csv")
    assert len(boundaries) == DAYS and (np.diff(boundaries["obf"]) < 0).all()
    assert boundaries["obf"].iloc[-1] == pytest.approx(boundaries["obf"].iloc[0] / math.sqrt(DAYS))
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)["sequential"]
    assert section["n_runs"] == 200 and section["days"] == DAYS
    assert section["seed"] == config.SEEDS["simulation"]
    assert set(section["tau"]) == set(METRICS)
    assert section["effects"]["ab_observed"]["treatment"] in config.TREATMENTS
    assert section["effects"]["ab_observed"]["absolute"] > section["effects"]["ab_mde"]["absolute"]
    assert {r["metric"] for r in section["small_sample_check"]} == set(METRICS)
    assert set(section["constants"]) == {"1", "5", str(DAYS)}
    assert len(section["rows"]) == len(main)
