"""Beta-Binomial Bayesian readout: closed forms, Monte Carlo checks, decisions and calibration."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.bayes import (
    DONT_SHIP,
    FLAT_PRIOR,
    KEEP_TESTING,
    PRIOR_STRENGTHS,
    SHIP,
    BetaPosterior,
    BetaPrior,
    build_bayes_rates,
    compare_rates,
    decide,
    posterior,
    prob_b_greater_a_exact,
    prob_b_greater_a_quad,
    rate_priors,
    run_bayes_rates,
    shared_prior,
)
from abkit.frequentist import diff_in_proportions
from abkit.readout import COMPARISONS, RATE_METRICS
from tests.conftest import customers_con

N_DRAWS = 200_000
# (alpha_a, beta_a, alpha_b, beta_b): small, visit-like and conversion-like posteriors.
PARAMS = [
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (3, 7, 5, 6),
    (3001, 18000, 3201, 18100),
    (191, 21116, 231, 21077),
    (231, 21077, 191, 21116),
]


# --- exact P(p_B > p_A) ------------------------------------------------------------------------


def test_closed_form_known_values():
    # Both uniform: 1/2. B with density 2x against a uniform A: P(B > A) = E[B] = 2/3.
    assert prob_b_greater_a_exact(1, 1, 1, 1) == pytest.approx(0.5, abs=1e-12)
    assert prob_b_greater_a_exact(1, 1, 2, 1) == pytest.approx(2 / 3, abs=1e-12)
    assert prob_b_greater_a_exact(2, 1, 1, 1) == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize(("aa", "ba", "ab", "bb"), PARAMS)
def test_closed_form_is_symmetric_and_matches_quadrature(aa, ba, ab, bb):
    p = prob_b_greater_a_exact(aa, ba, ab, bb)
    assert p + prob_b_greater_a_exact(ab, bb, aa, ba) == pytest.approx(1.0, abs=1e-10)
    assert p == pytest.approx(prob_b_greater_a_quad(aa, ba, ab, bb), abs=1e-8)


def test_closed_form_rejects_bad_parameters():
    with pytest.raises(ValueError):
        prob_b_greater_a_exact(1, 1, 1.5, 1)
    with pytest.raises(ValueError):
        prob_b_greater_a_exact(0, 1, 1, 1)
    with pytest.raises(ValueError):
        prob_b_greater_a_quad(1, -1, 1, 1)
    # Non-integer alpha_b is fine for quadrature.
    assert 0 < prob_b_greater_a_quad(0.9, 99.1, 1.1, 99.3) < 1


# --- Monte Carlo against the closed forms ------------------------------------------------------


@pytest.mark.parametrize(("aa", "ba", "ab", "bb"), PARAMS)
def test_monte_carlo_p_better_matches_closed_form(aa, ba, ab, bb):
    rng = np.random.default_rng(config.SEEDS["bayes"] + 1)
    res = compare_rates(BetaPosterior(ab, bb), BetaPosterior(aa, ba), rng, N_DRAWS)
    exact = prob_b_greater_a_exact(aa, ba, ab, bb)
    k = round(res.p_better * N_DRAWS)
    p_tilde = (k + 2) / (N_DRAWS + 4)
    assert res.p_better_mcse == pytest.approx(math.sqrt(p_tilde * (1 - p_tilde) / N_DRAWS))
    assert abs(res.p_better - exact) <= 3 * res.p_better_mcse


def test_p_better_mcse_stays_positive_when_every_draw_agrees():
    # Beta(500, 10) against Beta(10, 500): P(better) is 1 to many decimals, every draw agrees,
    # and the plug-in SE sqrt(p (1 - p) / n) would be exactly 0; the adjusted SE, with
    # p~ = (n + 2) / (n + 4), is about sqrt(2) / n and still brackets the exact value.
    n = 10_000
    rng = np.random.default_rng(config.SEEDS["bayes"] + 5)
    res = compare_rates(BetaPosterior(500, 10), BetaPosterior(10, 500), rng, n)
    exact = prob_b_greater_a_exact(10, 500, 500, 10)
    p_tilde = (n + 2) / (n + 4)
    assert res.p_better == 1.0
    assert res.p_better_mcse == pytest.approx(math.sqrt(p_tilde * (1 - p_tilde) / n), rel=1e-12)
    assert abs(res.p_better - exact) <= 3 * res.p_better_mcse


def test_posterior_update_and_moments_match_beta_formulas():
    post = posterior(BetaPrior(2.5, 4.0, "x"), successes=7, n=20)
    assert (post.alpha, post.beta) == (9.5, 17.0)
    assert posterior(FLAT_PRIOR, 0, 10) == BetaPosterior(1.0, 11.0)
    for a, b in ((9.5, 17.0), (1.0, 1.0), (231.0, 21077.0)):
        p = BetaPosterior(a, b)
        assert p.mean == pytest.approx(a / (a + b), rel=1e-12)
        assert p.var == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1)), rel=1e-12)
        assert p.mean == pytest.approx(stats.beta.mean(a, b), rel=1e-12)
        assert p.var == pytest.approx(stats.beta.var(a, b), rel=1e-12)
    for bad in ((-1, 10), (11, 10), (1, -1)):
        with pytest.raises(ValueError):
            posterior(FLAT_PRIOR, *bad)


def test_monte_carlo_moments_match_exact_posterior_moments():
    rng = np.random.default_rng(config.SEEDS["bayes"] + 2)
    post_t, post_c = BetaPosterior(3201, 18100), BetaPosterior(3001, 18000)
    res = compare_rates(post_t, post_c, rng, N_DRAWS)
    assert res.diff_mean == pytest.approx(post_t.mean - post_c.mean, rel=1e-12)
    assert res.diff_sd == pytest.approx(math.sqrt(post_t.var + post_c.var), rel=1e-12)
    assert abs(res.diff_mc_mean - res.diff_mean) <= 3 * res.diff_mcse
    # The posterior difference is near normal here, so the sample sd has MCSE sd / sqrt(2 n).
    assert abs(res.diff_mc_sd - res.diff_sd) <= 3 * res.diff_sd / math.sqrt(2 * (N_DRAWS - 1))
    # E[max(d, 0)] - E[max(-d, 0)] = E[d] draw by draw, and P(better) + P(worse) = 1.
    assert res.loss_keep - res.loss_ship == pytest.approx(res.diff_mc_mean, abs=1e-15)
    assert res.ci_low < res.diff_mean < res.ci_high
    assert res.rel_ci_low < res.rel_lift_mean < res.rel_ci_high
    with pytest.raises(ValueError):
        compare_rates(post_t, post_c, rng, 1)


def test_same_seed_gives_identical_results():
    a = compare_rates(BetaPosterior(5, 6), BetaPosterior(3, 7), np.random.default_rng(9), 10_000)
    b = compare_rates(BetaPosterior(5, 6), BetaPosterior(3, 7), np.random.default_rng(9), 10_000)
    assert a == b


@pytest.mark.parametrize(("xt", "xc"), [(3500, 3000), (230, 190)])
def test_flat_prior_interval_is_close_to_newcombe_at_large_n(xt, xc):
    # Flat prior and n = 21,000: the credible and confidence intervals differ by O(1/n) while
    # their width is O(1/sqrt(n)), a relative gap of about 1/sqrt(n) < 1%; allow 2% of width.
    n = 21_000
    rng = np.random.default_rng(config.SEEDS["bayes"] + 3)
    res = compare_rates(posterior(FLAT_PRIOR, xt, n), posterior(FLAT_PRIOR, xc, n), rng, N_DRAWS)
    freq = diff_in_proportions(xt, n, xc, n)
    width = freq.ci_high - freq.ci_low
    assert res.ci_low == pytest.approx(freq.ci_low, abs=0.02 * width)
    assert res.ci_high == pytest.approx(freq.ci_high, abs=0.02 * width)
    rel_width = freq.rel_ci_high - freq.rel_ci_low
    assert res.rel_ci_low == pytest.approx(freq.rel_ci_low, abs=0.02 * rel_width)
    assert res.rel_ci_high == pytest.approx(freq.rel_ci_high, abs=0.02 * rel_width)


# --- decision rule -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_better", "loss_ship", "loss_keep", "expected"),
    [
        (0.99, 1e-6, 2e-3, SHIP),
        (0.95, 9.9e-5, 2e-3, SHIP),  # both conditions exactly at the boundary
        (0.99, 1e-4, 2e-3, KEEP_TESTING),  # confident but the expected loss is not small
        (0.949, 1e-6, 2e-3, KEEP_TESTING),
        (0.01, 2e-3, 1e-6, DONT_SHIP),
        (0.05, 2e-3, 1e-6, DONT_SHIP),
        (0.01, 2e-3, 1e-4, KEEP_TESTING),
        (0.6, 1e-6, 1e-6, KEEP_TESTING),
    ],
)
def test_decision_rule(p_better, loss_ship, loss_keep, expected):
    verdict = decide(p_better, loss_ship, loss_keep, loss_threshold=1e-4)
    assert verdict.decision == expected
    assert "P(" in verdict.reason and "0.95" in verdict.reason


# --- prior sensitivity -------------------------------------------------------------------------


def test_shared_prior_has_the_pooled_mean_and_strength():
    prior = shared_prior(0.15, 1_000)
    assert prior.mean == pytest.approx(0.15) and prior.strength == pytest.approx(1_000)
    assert (prior.alpha, prior.beta, prior.label) == (150.0, 850.0, "shared_1000")
    assert FLAT_PRIOR.mean == 0.5 and FLAT_PRIOR.strength == 2
    labels = [p.label for p in rate_priors(0.15)]
    assert labels == ["flat", *(f"shared_{s}" for s in PRIOR_STRENGTHS)]
    for bad in ((0.0, 10), (1.0, 10), (0.5, 0)):
        with pytest.raises(ValueError):
            shared_prior(*bad)


def test_shared_prior_shrinks_the_difference_toward_zero():
    # Equal n: the posterior mean difference is (x_t - x_c) / (n + n0) exactly, so it shrinks
    # monotonically with the prior strength and P(better) moves toward 1/2.
    n, xt, xc = 21_000, 3_500, 3_000
    rng = np.random.default_rng(config.SEEDS["bayes"] + 4)
    pooled = (xt + xc) / (2 * n)
    diffs, probs = [], []
    for prior in rate_priors(pooled, (100, 1_000, 10_000, 100_000, 1_000_000)):
        res = compare_rates(posterior(prior, xt, n), posterior(prior, xc, n), rng, 20_000)
        assert res.diff_mean == pytest.approx((xt - xc) / (n + prior.strength), rel=1e-12)
        diffs.append(res.diff_mean)
        probs.append(res.p_better)
    assert all(a > b > 0 for a, b in itertools.pairwise(diffs))
    assert probs[0] > 0.999 and 0.5 < probs[-1] < probs[0]


# --- readout on the fixture --------------------------------------------------------------------


@pytest.fixture
def readout(customers):
    return build_bayes_rates(customers_con(customers), np.random.default_rng(5), n_draws=50_000)


def test_readout_has_every_metric_prior_and_comparison(readout):
    table, pooled = readout
    n_priors = 1 + len(PRIOR_STRENGTHS)
    assert len(table) == len(RATE_METRICS) * n_priors * len(COMPARISONS) == 30
    keys = set(zip(table.metric, table.prior, table.comparison, strict=True))
    assert len(keys) == 30 and set(table.metric) == set(RATE_METRICS)
    assert set(table.decision) <= {SHIP, DONT_SHIP, KEEP_TESTING}
    assert (table.n_draws == 50_000).all() and (table.level == 0.95).all()
    assert set(pooled) == set(RATE_METRICS)


def test_readout_matches_row_level_counts(readout, customers):
    table, pooled = readout
    for metric in RATE_METRICS:
        assert pooled[metric] == pytest.approx(customers[metric].mean(), rel=1e-12)
    for _, r in table.iterrows():
        yt = customers.loc[customers.arm == r.treatment, r.metric]
        yc = customers.loc[customers.arm == r.control, r.metric]
        assert (r.n_treatment, r.successes_treatment) == (len(yt), yt.sum())
        assert (r.n_control, r.successes_control) == (len(yc), yc.sum())
        assert r.post_alpha_treatment == pytest.approx(r.prior_alpha + yt.sum())
        assert r.post_beta_treatment == pytest.approx(r.prior_beta + len(yt) - yt.sum())
        assert r.prior_mean == pytest.approx(0.5 if r.prior == "flat" else pooled[r.metric])
        assert r.loss_threshold == pytest.approx(config.BAYES_LOSS_FRACTION * yc.mean())
        assert r.decision == decide(r.p_better, r.loss_ship, r.loss_keep, r.loss_threshold).decision


def test_readout_monte_carlo_agrees_with_exact_p_better(readout):
    table, _ = readout
    # 30 rows checked at once: 4 MCSE keeps the chance of a spurious failure below 0.2%.
    gap = (table.p_better - table.p_better_exact).abs()
    assert (gap <= 4 * table.p_better_mcse).all(), table.loc[gap > 4 * table.p_better_mcse]
    flat = table[table.prior == "flat"]
    for _, r in flat.iterrows():
        closed = prob_b_greater_a_exact(
            r.post_alpha_control, r.post_beta_control, r.post_alpha_treatment, r.post_beta_treatment
        )
        assert r.p_better_exact == pytest.approx(closed, abs=1e-8)


def test_run_bayes_rates_writes_outputs(fixture_db, tmp_path):
    table = run_bayes_rates(fixture_db, tmp_path, n_draws=20_000)
    written = pd.read_csv(tmp_path / "tables" / "readout_bayes_rates.csv")
    assert len(written) == len(table) == 30
    assert (tmp_path / "figures" / "bayes_p_better_by_prior.png").stat().st_size > 0
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)["bayes_rates"]
    assert section["n_draws"] == 20_000 and section["seed"] == config.SEEDS["bayes"]
    assert [p["label"] for p in section["priors"]] == [
        "flat",
        "shared_100",
        "shared_1000",
        "shared_10000",
        "shared_100000",
    ]
    assert section["decision_rule"] == {
        "p_better_threshold": 0.95,
        "loss_fraction_of_control_rate": 0.01,
    }
    assert len(section["rows"]) == 30


# --- slow: calibration of the credible intervals ------------------------------------------------

N_SIM = 2_000
N_ARM = 2_000
LEVEL = 0.95
MCSE = math.sqrt(LEVEL * (1 - LEVEL) / N_SIM)


@pytest.mark.slow
def test_credible_intervals_are_calibrated_under_the_prior():
    # Draw true rates from the prior, simulate the arms, and fit with that same prior: the 95%
    # credible intervals must then contain the truth in 95% of experiments, and the average
    # P(better) must equal the share of experiments where the treatment really is better.
    rng = np.random.default_rng(config.SEEDS["simulation"] + 40)
    prior = BetaPrior(3.0, 20.0, "calibration")
    p_t = rng.beta(prior.alpha, prior.beta, N_SIM)
    p_c = rng.beta(prior.alpha, prior.beta, N_SIM)
    x_t, x_c = rng.binomial(N_ARM, p_t), rng.binomial(N_ARM, p_c)
    covered_diff, covered_rel, p_better = [], [], []
    for xt, xc, pt, pc in zip(x_t, x_c, p_t, p_c, strict=True):
        res = compare_rates(posterior(prior, xt, N_ARM), posterior(prior, xc, N_ARM), rng, 20_000)
        covered_diff.append(res.ci_low <= pt - pc <= res.ci_high)
        covered_rel.append(res.rel_ci_low <= pt / pc - 1 <= res.rel_ci_high)
        p_better.append(res.p_better)
    for name, covered in (("diff", covered_diff), ("rel", covered_rel)):
        rate = float(np.mean(covered))
        assert abs(rate - LEVEL) <= 3 * MCSE, f"{name}: coverage {rate:.4f}, MCSE {MCSE:.4f}"
    better = (p_t > p_c).astype(float)
    gap = better - np.asarray(p_better)
    assert abs(gap.mean()) <= 3 * gap.std(ddof=1) / math.sqrt(N_SIM)
