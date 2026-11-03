"""Slow Monte Carlo coverage checks for the frequentist intervals (run with -m slow)."""

import numpy as np
import pytest
from scipy import stats

from abkit import config
from abkit.frequentist import diff_in_means, diff_in_proportions

N_SIM = 2_000
N_ARM = 21_000  # about the real arm size
LEVEL = 0.95
MCSE = np.sqrt(LEVEL * (1 - LEVEL) / N_SIM)

# Spend among converters, shaped like the pooled data (reports/tables/eda_distributions.csv):
# floor $29.99, median about $81, mean about $116, long right tail top-coded at $499.
FLOOR, CAP, SIGMA = 29.99, 499.0, 1.02


def _capped_mean(mu: float) -> float:
    """E[min(FLOOR + LogNormal(mu, SIGMA), CAP)] in closed form."""
    c = CAP - FLOOR
    tail = stats.norm.cdf((np.log(c) - mu - SIGMA**2) / SIGMA)
    return FLOOR + np.exp(mu + SIGMA**2 / 2) * tail + c * stats.norm.sf((np.log(c) - mu) / SIGMA)


def _spend_stats(rng, p_conv, mu):
    """Per-experiment mean and unbiased variance of zero-inflated spend, from sufficient stats."""
    k = rng.binomial(N_ARM, p_conv, size=N_SIM)
    ids = np.repeat(np.arange(N_SIM), k)
    v = np.minimum(FLOOR + rng.lognormal(mu, SIGMA, size=ids.size), CAP)
    s = np.bincount(ids, weights=v, minlength=N_SIM)
    q = np.bincount(ids, weights=v * v, minlength=N_SIM)
    return s / N_ARM, (q - s**2 / N_ARM) / (N_ARM - 1)


def _assert_coverage(name, covered):
    rate = covered.mean()
    assert abs(rate - LEVEL) <= 3 * MCSE, f"{name}: coverage {rate:.4f}, MCSE {MCSE:.4f}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("metric", "p_t", "p_c"), [("visit", 0.17, 0.13), ("conversion", 0.011, 0.008)]
)
def test_rate_interval_coverage(metric, p_t, p_c):
    rng = np.random.default_rng(config.SEEDS["simulation"])
    xt, xc = rng.binomial(N_ARM, p_t, N_SIM), rng.binomial(N_ARM, p_c, N_SIM)
    res = diff_in_proportions(xt, N_ARM, xc, N_ARM)
    truth, rel_truth = p_t - p_c, p_t / p_c - 1
    _assert_coverage(f"{metric} diff", (res.ci_low <= truth) & (truth <= res.ci_high))
    _assert_coverage(
        f"{metric} rel", (res.rel_ci_low <= rel_truth) & (rel_truth <= res.rel_ci_high)
    )


@pytest.mark.slow
def test_spend_interval_coverage():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 1)
    (p_t, mu_t), (p_c, mu_c) = (0.011, 3.98), (0.008, 3.93)
    mt, vt = _spend_stats(rng, p_t, mu_t)
    mc, vc = _spend_stats(rng, p_c, mu_c)
    res = diff_in_means(mt, vt, N_ARM, mc, vc, N_ARM)
    true_t, true_c = p_t * _capped_mean(mu_t), p_c * _capped_mean(mu_c)
    truth, rel_truth = true_t - true_c, true_t / true_c - 1
    _assert_coverage("spend diff", (res.ci_low <= truth) & (truth <= res.ci_high))
    _assert_coverage("spend rel", (res.rel_ci_low <= rel_truth) & (rel_truth <= res.rel_ci_high))


def test_capped_mean_closed_form_matches_monte_carlo():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 2)
    draws = np.minimum(FLOOR + rng.lognormal(3.93, SIGMA, size=2_000_000), CAP)
    mcse = draws.std(ddof=1) / np.sqrt(draws.size)
    assert abs(draws.mean() - _capped_mean(3.93)) < 4 * mcse
