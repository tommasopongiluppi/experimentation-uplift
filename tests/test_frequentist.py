import numpy as np
import pytest
from scipy import stats

from abkit import config
from abkit.frequentist import srm_test

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
