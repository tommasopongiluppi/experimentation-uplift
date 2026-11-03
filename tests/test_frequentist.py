import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from abkit import config
from abkit.frequentist import (
    benjamini_hochberg,
    bootstrap_diff_in_means,
    diff_in_means,
    diff_in_proportions,
    holm,
    relative_lift_ci,
    srm_test,
    wilson_interval,
)

EQUAL = {"mens": 21_000, "womens": 21_000, "control": 21_000}


def test_srm_perfect_split_passes():
    result = srm_test(EQUAL)
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert result.passed
    assert result.dof == 2
    assert result.expected == {"mens": 21_000.0, "womens": 21_000.0, "control": 21_000.0}


def test_srm_flags_22000_21000_21000():
    counts = {"mens": 22_000, "womens": 21_000, "control": 21_000}
    result = srm_test(counts)
    # closed form: expected 64,000 / 3 each; (2000/3)^2/e + 2 (1000/3)^2/e = 31.25
    assert result.statistic == pytest.approx(31.25)
    reference = stats.chisquare(list(counts.values()))
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)
    assert result.p_value < config.SRM_ALPHA
    assert not result.passed


def test_srm_uses_plan_threshold_by_default():
    assert srm_test(EQUAL).alpha == config.SRM_ALPHA == 0.001


def test_srm_unequal_allocation_matches_scipy():
    counts = {"a": 500, "b": 260, "c": 240}
    shares = {"a": 0.5, "b": 0.25, "c": 0.25}
    result = srm_test(counts, shares, alpha=0.05)
    reference = stats.chisquare([500, 260, 240], f_exp=[500, 250, 250])
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)


@pytest.mark.parametrize(
    ("counts", "shares"),
    [
        ({"a": 10}, None),
        ({"a": 10, "b": -1}, None),
        ({"a": 10, "b": 10}, {"a": 0.6, "b": 0.6}),
        ({"a": 10, "b": 10}, {"a": 0.5, "c": 0.5}),
    ],
)
def test_srm_invalid_input_raises(counts, shares):
    with pytest.raises(ValueError):
        srm_test(counts, shares)


@pytest.mark.slow
def test_srm_false_positive_rate_on_fair_splits():
    rng = np.random.default_rng(config.SEEDS["simulation"])
    n_sims, alpha = 2_000, 0.05
    draws = rng.multinomial(config.EXPECTED_ROWS, [1 / 3] * 3, size=n_sims)
    rejections = np.array(
        [not srm_test(dict(zip(config.ARMS, d, strict=True)), alpha=alpha).passed for d in draws]
    )
    rate = rejections.mean()
    mcse = np.sqrt(alpha * (1 - alpha) / n_sims)
    assert abs(rate - alpha) < 3 * mcse, f"rejection rate {rate:.4f}, MCSE {mcse:.4f}"


# ---------------------------------------------------------------------------------------------
# Two-arm comparisons, bootstrap and multiplicity
# ---------------------------------------------------------------------------------------------

TIGHT = {"rel": 1e-8, "abs": 1e-12}
PROPORTION_CASES = [
    (3_150, 21_307, 2_262, 21_306),
    (200, 21_000, 120, 21_000),
    (12, 400, 30, 500),
    (0, 50, 5, 60),
    (45, 45, 30, 40),
]


@pytest.mark.parametrize(("xt", "nt", "xc", "nc"), PROPORTION_CASES)
def test_z_test_matches_statsmodels(xt, nt, xc, nc):
    res = diff_in_proportions(xt, nt, xc, nc)
    z, p = proportions_ztest([xt, xc], [nt, nc], alternative="two-sided")
    assert res.statistic == pytest.approx(z, **TIGHT)
    assert res.p_value == pytest.approx(p, **TIGHT)
    assert res.diff == pytest.approx(xt / nt - xc / nc, **TIGHT)
    assert res.dof == np.inf


@pytest.mark.parametrize(("xt", "nt", "xc", "nc"), PROPORTION_CASES)
def test_newcombe_interval_matches_statsmodels(xt, nt, xc, nc):
    res = diff_in_proportions(xt, nt, xc, nc)
    low, high = confint_proportions_2indep(xt, nt, xc, nc, method="newcomb", compare="diff")
    assert res.ci_low == pytest.approx(low, **TIGHT)
    assert res.ci_high == pytest.approx(high, **TIGHT)


@pytest.mark.parametrize(("x", "n"), [(3_150, 21_307), (0, 50), (50, 50), (7, 13)])
def test_wilson_interval_matches_statsmodels(x, n):
    low, high = wilson_interval(x, n)
    ref = proportion_confint(x, n, alpha=0.05, method="wilson")
    assert (low, high) == pytest.approx(ref, **TIGHT)


@pytest.mark.parametrize(("xt", "nt", "xc", "nc"), PROPORTION_CASES[:3])
def test_rate_relative_lift_matches_statsmodels_log_ratio(xt, nt, xc, nc):
    # For proportions the delta method on log(p_t / p_c) with var p (1 - p) is the Katz log
    # interval: Var = (1 - p_t) / (n_t p_t) + (1 - p_c) / (n_c p_c).
    res = diff_in_proportions(xt, nt, xc, nc)
    low, high = confint_proportions_2indep(xt, nt, xc, nc, method="log", compare="ratio")
    assert res.rel_lift == pytest.approx((xt / nt) / (xc / nc) - 1, **TIGHT)
    assert res.rel_ci_low == pytest.approx(low - 1, **TIGHT)
    assert res.rel_ci_high == pytest.approx(high - 1, **TIGHT)


def test_relative_lift_closed_form_and_undefined():
    # mean_t = 2, mean_c = 1, var 4 and 1, n = 100: Var(log ratio) = 4/400 + 1/100 = 0.02.
    rel, low, high = relative_lift_ci(2.0, 4.0, 100, 1.0, 1.0, 100)
    z = stats.norm.ppf(0.975)
    assert rel == pytest.approx(1.0)
    assert low == pytest.approx(2 * np.exp(-z * np.sqrt(0.02)) - 1, **TIGHT)
    assert high == pytest.approx(2 * np.exp(z * np.sqrt(0.02)) - 1, **TIGHT)
    assert all(np.isnan(v) for v in relative_lift_ci(1.0, 1.0, 10, 0.0, 0.0, 10))


@pytest.mark.parametrize("scale_t", [1.0, 3.0])
def test_welch_matches_scipy(rng, scale_t):
    yt = rng.exponential(scale_t, size=257) + 0.1
    yc = rng.exponential(1.0, size=401)
    res = diff_in_means(yt.mean(), yt.var(ddof=1), yt.size, yc.mean(), yc.var(ddof=1), yc.size)
    ref = stats.ttest_ind(yt, yc, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert res.statistic == pytest.approx(ref.statistic, **TIGHT)
    assert res.p_value == pytest.approx(ref.pvalue, **TIGHT)
    assert res.dof == pytest.approx(ref.df, **TIGHT)
    assert (res.ci_low, res.ci_high) == pytest.approx((ci.low, ci.high), **TIGHT)


def test_vectorised_matches_scalar():
    xt, nt, xc, nc = (np.array(c) for c in zip(*PROPORTION_CASES[:3], strict=True))
    vec = diff_in_proportions(xt, nt, xc, nc)
    for i in range(3):
        one = diff_in_proportions(int(xt[i]), int(nt[i]), int(xc[i]), int(nc[i]))
        assert vec.ci_low[i] == pytest.approx(one.ci_low, **TIGHT)
        assert vec.p_value[i] == pytest.approx(one.p_value, **TIGHT)


def _random_pvalues(rng, m=25):
    p = np.concatenate([rng.uniform(0, 0.02, 5), rng.uniform(0, 1, m - 5)])
    p[3] = p[-1]  # a tie
    return p


@pytest.mark.parametrize(("ours", "method"), [(holm, "holm"), (benjamini_hochberg, "fdr_bh")])
def test_multiplicity_matches_statsmodels(rng, ours, method):
    for m in (1, 2, 7, 25):
        p = _random_pvalues(rng, max(m, 6))[:m]
        reject, ref, _, _ = multipletests(p, alpha=0.05, method=method)
        np.testing.assert_allclose(ours(p), ref, rtol=1e-12, atol=0)
        assert ((ours(p) <= 0.05) == reject).all()


def test_multiplicity_rejects_bad_input():
    for bad in ([0.1, np.nan], [-0.1], [1.2], [[0.1]]):
        with pytest.raises(ValueError):
            holm(bad)


def test_bootstrap_se_matches_closed_form(rng):
    # The bootstrap variance of a mean is exactly var_ddof0 / n. With B resamples the sample SD
    # of the replicates has relative MCSE ~ 1 / sqrt(2 (B - 1)).
    yt = np.where(rng.uniform(size=3_000) < 0.05, rng.lognormal(4, 1, size=3_000), 0.0)
    yc = np.where(rng.uniform(size=2_000) < 0.04, rng.lognormal(4, 1, size=2_000), 0.0)
    b = 10_000
    res = bootstrap_diff_in_means(yt, yc, np.random.default_rng(1), n_resamples=b)
    se_exact = np.sqrt(yt.var() / yt.size + yc.var() / yc.size)
    assert res.se == pytest.approx(se_exact, rel=3 / np.sqrt(2 * (b - 1)))
    assert res.diff == yt.mean() - yc.mean()
    assert res.ci_low < res.diff < res.ci_high
    assert 0 < res.mcse_low < res.se and 0 < res.mcse_high < res.se


def test_bootstrap_matches_welch_on_normal_data_and_is_seeded(rng):
    yt, yc = rng.normal(1.0, 2.0, 5_000), rng.normal(0.8, 1.5, 4_000)
    a = bootstrap_diff_in_means(yt, yc, np.random.default_rng(7), chunk_cells=1_000_000)
    b = bootstrap_diff_in_means(yt, yc, np.random.default_rng(7), chunk_cells=1_000_000)
    assert a == b
    w = diff_in_means(yt.mean(), yt.var(ddof=1), yt.size, yc.mean(), yc.var(ddof=1), yc.size)
    # Endpoints agree up to Monte Carlo error (4 MCSE) plus the n vs n - 1 variance factor.
    assert a.ci_low == pytest.approx(w.ci_low, abs=4 * a.mcse_low + 1e-3)
    assert a.ci_high == pytest.approx(w.ci_high, abs=4 * a.mcse_high + 1e-3)
