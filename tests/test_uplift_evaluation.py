"""Uplift evaluation: hand-computed curves, oracle vs random, bootstrap, scikit-uplift check."""

import numpy as np
import pytest

from abkit.frequentist import diff_in_proportions
from abkit.plots import plot_uplift_calibration, plot_uplift_curves
from abkit.uplift.evaluation import (
    assign_bins,
    auuc,
    bootstrap_evaluation,
    qini_coefficient,
    qini_curve,
    targeting_curves,
    uplift_at_k,
    uplift_by_decile,
    uplift_curve,
)

# Ten customers already in descending score order (scores 10, 9, ..., 1):
#   row:      1  2  3  4  5  6  7  8  9  10
#   treated:  1  0  1  0  1  0  1  0  1  0
#   y:        1  0  1  1  0  0  0  1  0  0
# Cumulative (n_t, y_t, n_c, y_c) and Qini = y_t - y_c * n_t / n_c (control term 0 while n_c = 0):
#   1: (1,1,0,0) -> 1        2: (1,1,1,0) -> 1        3: (2,2,1,0) -> 2
#   4: (2,2,2,1) -> 1        5: (3,2,2,1) -> 0.5      6: (3,2,3,1) -> 1
#   7: (4,2,3,1) -> 2/3      8: (4,2,4,2) -> 0        9: (5,2,4,2) -> -0.5
#  10: (5,2,5,2) -> 0
# Uplift = (y_t / n_t - y_c / n_c) * n:
#   1, 2, 3, 2, 5/6, 2, 7/6, 0, -0.9, 0
TOY_T = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
TOY_Y = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
TOY_S = np.arange(10, 0, -1).astype(float)
TOY_QINI = [0, 1, 1, 2, 1, 0.5, 1, 2 / 3, 0, -0.5, 0]
TOY_UPLIFT = [0, 1, 2, 3, 2, 5 / 6, 2, 7 / 6, 0, -0.9, 0]
# Trapezoid areas on the share axis (step 0.1): Qini 0.05 * 2 * 6.6667 = 2/3, uplift 1.11.
# Random line ends at 0 so its area is 0. Perfect Qini ranking: the two treated responders
# (score 1), six others (score 0), two control responders (score -1), giving the points
# (0, 0), (0.2, 2), (0.8, 2), (1, 0) and area 0.2 + 1.2 + 0.2 = 1.6.
TOY_QINI_AREA, TOY_QINI_PERFECT = 2 / 3, 1.6
TOY_UPLIFT_AREA = 1.11


def _synthetic(rng, n, p0=0.2):
    """Known individual effect tau(x) = 0.4 x - 0.1 on a Bernoulli outcome."""
    x = rng.random(n)
    tau = 0.4 * x - 0.1
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < p0 + tau * treated).astype(int)
    return tau, treated, y


# --- hand-computed toy ---------------------------------------------------------------------


def test_toy_curves_match_hand_computation():
    c = targeting_curves(TOY_S, TOY_T, TOY_Y)
    np.testing.assert_allclose(c["share"], np.arange(11) / 10)
    np.testing.assert_allclose(c["n_treated"], np.cumsum(np.r_[0, TOY_T]))
    np.testing.assert_allclose(c["qini"], TOY_QINI)
    np.testing.assert_allclose(c["uplift"], TOY_UPLIFT)
    np.testing.assert_allclose(qini_curve(TOY_S, TOY_T, TOY_Y)["qini"], TOY_QINI)
    np.testing.assert_allclose(uplift_curve(TOY_S, TOY_T, TOY_Y)["uplift"], TOY_UPLIFT)


def test_toy_areas_match_hand_computation():
    q = qini_coefficient(TOY_S, TOY_T, TOY_Y)
    assert q.curve == "qini"
    assert q.model == pytest.approx(TOY_QINI_AREA)
    assert q.random == 0.0
    assert q.perfect == pytest.approx(TOY_QINI_PERFECT)
    assert q.gain == pytest.approx(TOY_QINI_AREA)
    assert q.normalised == pytest.approx(TOY_QINI_AREA / TOY_QINI_PERFECT)
    u = auuc(TOY_S, TOY_T, TOY_Y)
    assert u.model == pytest.approx(TOY_UPLIFT_AREA)
    assert u.gain == pytest.approx(TOY_UPLIFT_AREA)
    assert 0 < u.normalised < 1


def test_toy_uplift_at_k():
    # top 3: treated (1, 1) vs control (0): 1 - 0; top 5: treated 2/3 vs control 1/2.
    assert uplift_at_k(TOY_S, TOY_T, TOY_Y, 0.3) == pytest.approx(1.0)
    assert uplift_at_k(TOY_S, TOY_T, TOY_Y, 0.5) == pytest.approx(2 / 3 - 1 / 2)
    assert uplift_at_k(TOY_S, TOY_T, TOY_Y, 1.0) == pytest.approx(0.0)
    assert np.isnan(uplift_at_k(TOY_S, TOY_T, TOY_Y, 0.1))  # no control in the top 1
    with pytest.raises(ValueError, match="k must be"):
        uplift_at_k(TOY_S, TOY_T, TOY_Y, 0.0)


def test_tied_scores_enter_together():
    # Rows 1-3 share a score, rows 4-10 another: thresholds at n = 3 and n = 10 only.
    tied = np.r_[np.full(3, 2.0), np.full(7, 1.0)]
    c = targeting_curves(tied, TOY_T, TOY_Y)
    np.testing.assert_allclose(c["n"], [0, 3, 10])
    np.testing.assert_allclose(c["qini"], [0, 2, 0])


def test_curves_do_not_depend_on_row_order(rng):
    tau, t, y = _synthetic(rng, 400)
    score = np.round(tau, 2)  # ties
    perm = rng.permutation(400)
    a, b = targeting_curves(score, t, y), targeting_curves(score[perm], t[perm], y[perm])
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    assert qini_coefficient(score, t, y) == qini_coefficient(score[perm], t[perm], y[perm])


def test_input_validation():
    with pytest.raises(ValueError, match="0/1"):
        targeting_curves(TOY_S, TOY_T, TOY_Y * 2)
    with pytest.raises(ValueError, match="same length"):
        targeting_curves(TOY_S[:5], TOY_T, TOY_Y)
    with pytest.raises(ValueError, match="finite"):
        targeting_curves(np.r_[np.nan, TOY_S[1:]], TOY_T, TOY_Y)


# --- oracle, perfect and random rankings -----------------------------------------------------


def test_perfect_ranking_scores_one_and_random_near_zero(rng):
    _, t, y = _synthetic(rng, 2_000)
    perfect = (y * t - y * (1 - t)).astype(float)
    assert qini_coefficient(perfect, t, y).normalised == pytest.approx(1.0)
    assert auuc(perfect, t, y).normalised <= 1.0 + 1e-12


def test_oracle_beats_random_and_random_averages_zero(rng):
    tau, t, y = _synthetic(rng, 4_000)
    oracle_q, oracle_u = qini_coefficient(tau, t, y), auuc(tau, t, y)
    n_random = 60
    reps_q = np.array(
        [qini_coefficient(rng.random(t.size), t, y).normalised for _ in range(n_random)]
    )
    reps_u = np.array([auuc(rng.random(t.size), t, y).normalised for _ in range(n_random)])
    for reps, oracle in ((reps_q, oracle_q), (reps_u, oracle_u)):
        sd, mcse = reps.std(ddof=1), reps.std(ddof=1) / np.sqrt(n_random)
        # Random targeting: mean 0 within Monte Carlo error.
        assert abs(reps.mean()) < 4 * mcse, (reps.mean(), mcse)
        # Oracle: far above random, i.e. beyond every random draw and more than 5 SD above
        # their mean (the spread of the random scores sets the scale of "far").
        assert oracle.normalised > reps.max()
        assert (oracle.normalised - reps.mean()) / sd > 5, (oracle.normalised, reps.mean(), sd)
        assert oracle.gain > 0


# --- calibration -----------------------------------------------------------------------------


def test_assign_bins_equal_counts_and_order():
    score = np.array([0.3, 0.1, 0.9, 0.5, 0.2, 0.9])
    bins = assign_bins(score, 3)
    np.testing.assert_array_equal(bins, [2, 3, 1, 2, 3, 1])
    counts = np.bincount(assign_bins(np.arange(1003) / 7, 10))[1:]
    assert counts.max() - counts.min() <= 1


def test_uplift_by_decile_matches_direct_computation(rng):
    tau, t, y = _synthetic(rng, 2_000)
    table = uplift_by_decile(tau, t, y)
    assert list(table["decile"]) == list(range(1, 11))
    assert (table["n"] == 200).all()
    assert table["predicted"].is_monotonic_decreasing
    bins = assign_bins(tau, 10)
    for _, r in table.iterrows():
        m = bins == r["decile"]
        res = diff_in_proportions(
            y[m & (t == 1)].sum(), (m & (t == 1)).sum(), y[m & (t == 0)].sum(), (m & (t == 0)).sum()
        )
        assert r["predicted"] == pytest.approx(tau[m].mean())
        assert r["observed"] == pytest.approx(res.diff)
        assert (r["ci_low"], r["ci_high"]) == pytest.approx((res.ci_low, res.ci_high))


def test_true_effect_is_calibrated(rng):
    tau, t, y = _synthetic(rng, 10_000)
    table = uplift_by_decile(tau, t, y)
    covered = (
        (table["ci_low"] <= table["predicted"]) & (table["predicted"] <= table["ci_high"])
    ).sum()
    assert covered >= 8


# --- bootstrap -------------------------------------------------------------------------------


def test_unit_weights_reproduce_point_estimates(rng):
    tau, t, y = _synthetic(rng, 500)
    ones = np.ones(500)
    np.testing.assert_allclose(
        targeting_curves(tau, t, y, weights=ones).to_numpy(), targeting_curves(tau, t, y).to_numpy()
    )
    assert uplift_at_k(tau, t, y, 0.3, weights=ones) == uplift_at_k(tau, t, y, 0.3)


def test_bootstrap_evaluation_structure(rng):
    tau, t, y = _synthetic(rng, 1_000)
    boot = bootstrap_evaluation(tau, t, y, rng, n_resamples=80)
    s = boot.scalars.set_index("metric")
    assert list(s.index) == [
        "auuc_normalised", "auuc_gain", "qini_normalised", "qini_gain",
        "uplift_at_10", "uplift_at_20", "uplift_at_30",
    ]  # fmt: skip
    assert s.loc["qini_normalised", "estimate"] == pytest.approx(
        qini_coefficient(tau, t, y).normalised
    )
    assert s.loc["auuc_gain", "estimate"] == pytest.approx(auuc(tau, t, y).gain)
    assert s.loc["uplift_at_30", "estimate"] == pytest.approx(uplift_at_k(tau, t, y, 0.3))
    assert (s["ci_low"] <= s["estimate"]).all() and (s["estimate"] <= s["ci_high"]).all()
    assert (s[["mcse_low", "mcse_high"]] >= 0).all().all()
    assert (s["n_resamples"] == 80).all()

    c = boot.curves
    assert len(c) == 100 and c["share"].iloc[-1] == 1.0
    assert (c["qini_low"] <= c["qini"]).all() and (c["qini"] <= c["qini_high"]).all()
    assert (c["uplift_low"] <= c["uplift"]).all() and (c["uplift"] <= c["uplift_high"]).all()
    full = targeting_curves(tau, t, y)
    np.testing.assert_allclose(c["qini"], np.interp(c["share"], full["share"], full["qini"]))

    d = boot.deciles
    assert len(d) == 10
    assert (d["boot_low"] <= d["observed"]).all() and (d["observed"] <= d["boot_high"]).all()


def test_bootstrap_is_seeded(rng):
    tau, t, y = _synthetic(rng, 300)
    a = bootstrap_evaluation(tau, t, y, np.random.default_rng(9), n_resamples=30)
    b = bootstrap_evaluation(tau, t, y, np.random.default_rng(9), n_resamples=30)
    np.testing.assert_array_equal(a.scalars["ci_low"], b.scalars["ci_low"])


@pytest.mark.slow
def test_bootstrap_interval_coverage(rng):
    """Percentile interval for the Qini gain covers the large-sample value about 95% of the time."""
    n, n_sim = 2_000, 150
    reference = qini_coefficient(*_synthetic(rng, 400_000)).gain * n / 400_000
    hits = 0
    for _ in range(n_sim):
        tau, t, y = _synthetic(rng, n)
        s = bootstrap_evaluation(tau, t, y, rng, n_resamples=200).scalars.set_index("metric")
        hits += s.loc["qini_gain", "ci_low"] <= reference <= s.loc["qini_gain", "ci_high"]
    coverage = hits / n_sim
    mcse = np.sqrt(0.95 * 0.05 / n_sim)
    assert abs(coverage - 0.95) < 3 * mcse + 0.02, (coverage, mcse)


# --- plots -----------------------------------------------------------------------------------


def test_plots_write_png(rng, tmp_path):
    tau, t, y = _synthetic(rng, 500)
    boot = bootstrap_evaluation(tau, t, y, rng, n_resamples=20)
    p1 = plot_uplift_curves(boot.curves, tmp_path / "curves.png", title="mens vs control")
    p2 = plot_uplift_calibration(boot.deciles, tmp_path / "calibration.png")
    assert p1.stat().st_size > 0 and p2.stat().st_size > 0


# --- scikit-uplift cross-check ---------------------------------------------------------------


@pytest.mark.parametrize("ties", [False, True])
def test_matches_scikit_uplift(rng, ties):
    sklift_metrics = pytest.importorskip("sklift.metrics")
    tau, t, y = _synthetic(rng, 3_000)
    score = np.round(tau, 1) if ties else tau
    assert qini_coefficient(score, t, y).normalised == pytest.approx(
        sklift_metrics.qini_auc_score(y, score, t), rel=1e-9
    )
    assert auuc(score, t, y).normalised == pytest.approx(
        sklift_metrics.uplift_auc_score(y, score, t), rel=1e-9
    )
    c = targeting_curves(score, t, y)
    xq, yq = sklift_metrics.qini_curve(y, score, t)
    xu, yu = sklift_metrics.uplift_curve(y, score, t)
    np.testing.assert_allclose(c["n"], xq)
    np.testing.assert_allclose(c["n"], xu)
    np.testing.assert_allclose(c["qini"], yq)
    np.testing.assert_allclose(c["uplift"], yu)
    if not ties:  # scikit-uplift breaks ties in the opposite row order
        for k in (0.1, 0.3):
            assert uplift_at_k(score, t, y, k) == pytest.approx(
                sklift_metrics.uplift_at_k(y, score, t, strategy="overall", k=k)
            )
