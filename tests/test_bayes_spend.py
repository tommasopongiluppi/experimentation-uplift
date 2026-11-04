"""Bayesian spend readout: collapsed Bayesian bootstrap, two-part log-normal model, PPC, readout."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.bayes import (
    BOOTSTRAP,
    DONT_SHIP,
    KEEP_TESTING,
    SHIP,
    TWO_PART,
    WEAK_NIG_PRIOR,
    NIGPrior,
    bayesian_bootstrap_means,
    bayesian_bootstrap_means_naive,
    bayesian_bootstrap_moments,
    build_bayes_spend,
    compare_means,
    decide,
    draw_nig,
    nig_posterior,
    posterior_predictive_check,
    recommend,
    run_bayes_spend,
    summarise_draws,
    two_part_draws,
)
from abkit.frequentist import _quantile_mcse
from abkit.readout import COMPARISONS, SPEND_WINSORISED
from tests.conftest import customers_con

N_DRAWS = 100_000


def _spend_like(rng: np.random.Generator, n: int, n_pos: int, mean_pos: float) -> np.ndarray:
    y = np.zeros(n)
    y[:n_pos] = np.round(rng.gamma(2.0, mean_pos / 2, n_pos), 2)
    rng.shuffle(y)
    return y


def _sd_mcse(x: np.ndarray) -> float:
    """Monte Carlo SE of a sample standard deviation: sd sqrt((kurtosis - 1) / (4 N))."""
    kurt = ((x - x.mean()) ** 4).mean() / x.var() ** 2
    return float(x.std(ddof=1) * math.sqrt((kurt - 1) / (4 * x.size)))


# --- Bayesian bootstrap -------------------------------------------------------------------------


def test_collapsed_bootstrap_matches_naive_on_small_data():
    # 40 customers, 8 converters with ties: the collapsed Dirichlet over distinct values and
    # the naive 40-dimensional Dirichlet(1, ..., 1) are the same posterior.
    rng = np.random.default_rng(config.SEEDS["bayes"] + 10)
    y = np.zeros(40)
    y[:8] = [29.99, 29.99, 50.0, 80.5, 120.0, 120.0, 250.0, 499.0]
    fast = bayesian_bootstrap_means(y, rng, N_DRAWS)
    naive = bayesian_bootstrap_means_naive(y, rng, N_DRAWS)
    exact_mean, exact_sd = bayesian_bootstrap_moments(y)
    for draws in (fast, naive):
        assert abs(draws.mean() - exact_mean) <= 3 * draws.std(ddof=1) / math.sqrt(N_DRAWS)
        assert abs(draws.std(ddof=1) - exact_sd) <= 3 * _sd_mcse(draws)
    gap_mcse = math.hypot(fast.std(ddof=1), naive.std(ddof=1)) / math.sqrt(N_DRAWS)
    assert abs(fast.mean() - naive.mean()) <= 3 * gap_mcse
    fast.sort()
    naive.sort()
    for q in (0.025, 0.25, 0.5, 0.75, 0.975):
        mcse = math.hypot(_quantile_mcse(fast, q), _quantile_mcse(naive, q))
        assert abs(np.quantile(fast, q) - np.quantile(naive, q)) <= 3 * mcse, q
    # Whole-distribution check as well: the two samples are indistinguishable.
    assert stats.ks_2samp(fast, naive).pvalue > 0.001


def test_collapse_is_exact_for_the_zero_component():
    # With only zeros and one positive value the mean is v * W, W ~ Beta(1, n - 1); its
    # quantiles are known in closed form.
    rng = np.random.default_rng(config.SEEDS["bayes"] + 11)
    n, v = 200, 100.0
    y = np.zeros(n)
    y[0] = v
    draws = np.sort(bayesian_bootstrap_means(y, rng, N_DRAWS))
    for q in (0.025, 0.5, 0.975):
        exact = v * stats.beta(1, n - 1).ppf(q)
        assert abs(np.quantile(draws, q) - exact) <= 3 * _quantile_mcse(draws, q), q
    assert bayesian_bootstrap_moments(y) == pytest.approx(
        (v / n, v * math.sqrt((n - 1) / (n * n * (n + 1))))
    )


def test_large_n_posterior_matches_sample_mean_and_standard_error():
    rng = np.random.default_rng(config.SEEDS["bayes"] + 12)
    y = _spend_like(rng, 20_000, 200, 110.0)
    draws = bayesian_bootstrap_means(y, rng, N_DRAWS)
    exact_mean, exact_sd = bayesian_bootstrap_moments(y)
    se = y.std(ddof=1) / math.sqrt(y.size)
    assert exact_mean == pytest.approx(y.mean(), rel=1e-12)
    # exact posterior sd = se * sqrt((n - 1) / (n + 1)): within 0.01% of the SE at n = 20,000.
    assert exact_sd == pytest.approx(se * math.sqrt((y.size - 1) / (y.size + 1)), rel=1e-12)
    assert exact_sd == pytest.approx(se, rel=1e-4)
    assert abs(draws.mean() - y.mean()) <= 3 * draws.std(ddof=1) / math.sqrt(N_DRAWS)
    assert abs(draws.std(ddof=1) - exact_sd) <= 3 * _sd_mcse(draws)
    assert draws.std(ddof=1) == pytest.approx(se, rel=0.02)


def test_bootstrap_rejects_bad_input_and_is_reproducible():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        bayesian_bootstrap_means(np.array([]), rng, 10)
    with pytest.raises(ValueError):
        bayesian_bootstrap_means(np.ones(3), rng, 0)
    with pytest.raises(ValueError):
        bayesian_bootstrap_means_naive(np.array([]), rng, 10)
    with pytest.raises(ValueError):
        bayesian_bootstrap_moments(np.array([]))
    y = np.array([0.0, 0.0, 5.0, 7.0])
    a = bayesian_bootstrap_means(y, np.random.default_rng(3), 1_000, chunk_cells=7)
    b = bayesian_bootstrap_means(y, np.random.default_rng(3), 1_000)
    np.testing.assert_allclose(a, b)


def test_compare_means_planted_effect_and_identities():
    rng = np.random.default_rng(config.SEEDS["bayes"] + 13)
    yc = _spend_like(rng, 20_000, 200, 110.0)
    yt = _spend_like(rng, 20_000, 300, 120.0)
    draws_t = bayesian_bootstrap_means(yt, rng, N_DRAWS)
    draws_c = bayesian_bootstrap_means(yc, rng, N_DRAWS)
    res = compare_means(draws_t, draws_c, yt, yc)
    assert res.diff_mean == pytest.approx(yt.mean() - yc.mean(), rel=1e-12)
    sd_t, sd_c = bayesian_bootstrap_moments(yt)[1], bayesian_bootstrap_moments(yc)[1]
    assert res.diff_sd == pytest.approx(math.hypot(sd_t, sd_c), rel=1e-12)
    assert abs(res.diff_mc_mean - res.diff_mean) <= 3 * res.diff_mcse
    assert res.loss_keep - res.loss_ship == pytest.approx(res.diff_mc_mean, abs=1e-12)
    assert res.ci_low < res.diff_median < res.ci_high
    assert res.p_better > 0.99
    # Same data in both arms: P(better) = 1/2 within Monte Carlo error.
    same = compare_means(
        bayesian_bootstrap_means(yc, rng, N_DRAWS),
        bayesian_bootstrap_means(yc, rng, N_DRAWS),
        yc,
        yc,
    )
    assert same.diff_mean == 0 and abs(same.p_better - 0.5) <= 3 * same.p_better_mcse
    with pytest.raises(ValueError):
        summarise_draws(np.ones(5), np.ones(4))


# --- Two-part model -----------------------------------------------------------------------------


def test_nig_update_hand_computed():
    post = nig_posterior(NIGPrior(0.0, 1.0, 1.0, 1.0), np.array([1.0, 2.0, 3.0]))
    assert (post.kappa_n, post.mu_n, post.alpha_n, post.beta_n, post.n) == (4.0, 1.5, 2.5, 3.5, 3)
    assert post.sigma2_mean == pytest.approx(3.5 / 1.5)
    assert post.mu_scale == pytest.approx(math.sqrt(3.5 / (2.5 * 4.0)))
    assert post.mu_sd == pytest.approx(post.mu_scale * math.sqrt(5 / 3))
    for bad in (NIGPrior(0, 0, 1, 1), NIGPrior(0, 1, 0, 1), NIGPrior(0, 1, 1, -1)):
        with pytest.raises(ValueError):
            nig_posterior(bad, np.array([1.0]))
    with pytest.raises(ValueError):
        nig_posterior(WEAK_NIG_PRIOR, np.array([]))


def test_nig_draws_match_the_student_t_and_inverse_gamma_marginals():
    rng = np.random.default_rng(config.SEEDS["bayes"] + 14)
    post = nig_posterior(WEAK_NIG_PRIOR, rng.normal(4.5, 0.8, 150))
    mu, sigma2 = draw_nig(post, rng, N_DRAWS)
    t_marginal = stats.t(df=2 * post.alpha_n, loc=post.mu_n, scale=post.mu_scale)
    ig_marginal = stats.invgamma(post.alpha_n, scale=post.beta_n)
    assert abs(mu.mean() - post.mu_mean) <= 3 * mu.std(ddof=1) / math.sqrt(N_DRAWS)
    assert abs(sigma2.mean() - post.sigma2_mean) <= 3 * sigma2.std(ddof=1) / math.sqrt(N_DRAWS)
    mu.sort()
    sigma2.sort()
    for q in (0.025, 0.5, 0.975):
        assert abs(np.quantile(mu, q) - t_marginal.ppf(q)) <= 3 * _quantile_mcse(mu, q)
        assert abs(np.quantile(sigma2, q) - ig_marginal.ppf(q)) <= 3 * _quantile_mcse(sigma2, q)


def test_two_part_model_recovers_known_parameters():
    # Simulated two-part data: p = 0.02, log spend ~ N(4.4, 0.7^2) among converters.
    rng = np.random.default_rng(config.SEEDS["bayes"] + 15)
    n, p_true, mu_true, sigma_true = 400_000, 0.02, 4.4, 0.7
    conv = rng.random(n) < p_true
    spend = np.where(conv, np.exp(mu_true + sigma_true * rng.standard_normal(n)), 0.0)
    tp = two_part_draws(spend, rng, N_DRAWS)
    k = int(conv.sum())
    assert (tp.beta_post.alpha, tp.beta_post.beta) == (1 + k, 1 + n - k)
    # With the weak prior the posterior centres on the MLEs ...
    log_pos = np.log(spend[spend > 0])
    assert tp.nig_post.mu_n == pytest.approx(log_pos.mean(), abs=1e-3)
    assert tp.nig_post.sigma2_mean == pytest.approx(log_pos.var(ddof=1), rel=1e-2)
    # ... and the truth lies within 4 posterior standard deviations of each posterior mean.
    assert abs(tp.mu.mean() - mu_true) <= 4 * tp.mu.std()
    assert abs(tp.sigma2.mean() - sigma_true**2) <= 4 * tp.sigma2.std()
    assert abs(tp.p.mean() - p_true) <= 4 * tp.p.std()
    truth = p_true * math.exp(mu_true + sigma_true**2 / 2)
    assert abs(tp.mean_spend.mean() - truth) <= 4 * tp.mean_spend.std()
    assert abs(np.median(tp.conv_mean_spend) - math.exp(mu_true + sigma_true**2 / 2)) <= (
        4 * tp.conv_mean_spend.std()
    )
    with pytest.raises(ValueError):
        two_part_draws(np.zeros(10), rng, 100)


def test_posterior_predictive_check_passes_on_log_normal_data_and_flags_a_floor():
    rng = np.random.default_rng(config.SEEDS["bayes"] + 16)
    n, mu_true, sigma_true = 50_000, 4.4, 0.7
    conv = rng.random(n) < 0.01
    spend = np.where(conv, np.exp(mu_true + sigma_true * rng.standard_normal(n)), 0.0)
    tp = two_part_draws(spend, rng, 20_000)
    cap, floor = 499.0, 29.99
    ppc, implied = posterior_predictive_check(spend[spend > 0], tp, rng, cap, floor, 4_000)
    assert set(ppc.statistic) == {
        "mean",
        "q0.5",
        "q0.9",
        "q0.95",
        "q0.99",
        "max",
        "share_at_or_above_cap",
        "share_at_or_below_floor",
    }
    assert ppc.pp_pvalue.between(0.01, 0.99).all(), ppc
    assert (ppc.pp_low <= ppc.observed).all() and (ppc.observed <= ppc.pp_high).all()
    true_tail = stats.norm.sf((math.log(cap) - mu_true) / sigma_true)
    assert implied == pytest.approx(true_tail, rel=0.25)
    # Heap a quarter of converters on the floor: the floor share is now far outside the
    # posterior predictive distribution while the model still reproduces the centre.
    heaped = spend.copy()
    pos = np.flatnonzero(heaped > 0)
    heaped[pos[: len(pos) // 4]] = floor
    tp2 = two_part_draws(heaped, rng, 20_000)
    ppc2, _ = posterior_predictive_check(heaped[heaped > 0], tp2, rng, cap, floor, 4_000)
    assert float(ppc2.set_index("statistic").loc["share_at_or_below_floor", "pp_pvalue"]) < 0.01
    with pytest.raises(ValueError):
        posterior_predictive_check(spend[spend > 0], tp, rng, cap, floor, 30_000)


# --- recommendation ---------------------------------------------------------------------------


def _primary_rows(decisions: dict[str, str], means: dict[str, float], p_mens: float):
    rows = [
        {
            "model": BOOTSTRAP,
            "metric": "spend",
            "treatment": t,
            "control": config.CONTROL,
            "decision": decisions[t],
            "post_mean_treatment": means[t],
            "p_better": 0.99,
        }
        for t in ("mens", "womens")
    ]
    rows.append(
        {
            "model": BOOTSTRAP,
            "metric": "spend",
            "treatment": "mens",
            "control": "womens",
            "decision": KEEP_TESTING,
            "post_mean_treatment": means["mens"],
            "p_better": p_mens,
        }
    )
    # A winsorised and a two-part row that must be ignored by the tie-break.
    rows.append({**rows[0], "metric": SPEND_WINSORISED, "decision": SHIP})
    rows.append({**rows[1], "model": TWO_PART, "decision": SHIP})
    return pd.DataFrame(rows)


def test_recommend_applies_the_plan_tie_break():
    ship_both = {"mens": SHIP, "womens": SHIP}
    both = recommend(_primary_rows(ship_both, {"mens": 1.4, "womens": 1.1}, 0.98))
    assert both["recommended_email"] == "mens"
    assert both["qualifying_emails"] == ["mens", "womens"]
    assert both["p_mens_better_than_womens"] == 0.98
    flipped = recommend(_primary_rows(ship_both, {"mens": 1.0, "womens": 1.1}, 0.4))
    assert flipped["recommended_email"] == "womens"
    one = recommend(
        _primary_rows({"mens": KEEP_TESTING, "womens": SHIP}, {"mens": 1.4, "womens": 1.1}, 0.98)
    )
    assert one["recommended_email"] == "womens" and one["qualifying_emails"] == ["womens"]
    none = recommend(
        _primary_rows({"mens": DONT_SHIP, "womens": KEEP_TESTING}, {"mens": 1, "womens": 1}, 0.5)
    )
    assert none["recommended_email"] is None and none["qualifying_emails"] == []


# --- readout on the fixture --------------------------------------------------------------------


@pytest.fixture
def spend_readout(customers):
    rng = np.random.default_rng(config.SEEDS["bayes"] + 17)
    return build_bayes_spend(customers_con(customers), rng, n_draws=50_000, n_replicates=1_000)


def test_spend_readout_has_every_row_and_matches_row_level_data(spend_readout, customers):
    table, arms, ppc, hist = spend_readout
    assert len(table) == 3 * len(COMPARISONS) == 9
    keys = set(zip(table.model, table.metric, table.comparison, strict=True))
    assert keys == {
        (m, k, f"{t}_vs_{c}")
        for m, k in ((BOOTSTRAP, "spend"), (BOOTSTRAP, SPEND_WINSORISED), (TWO_PART, "spend"))
        for t, c in COMPARISONS
    }
    conv = customers.loc[customers.conversion == 1, "spend"]
    cap = float(np.quantile(conv, 0.95))
    df = customers.assign(spend_winsorised=customers.spend.clip(upper=cap))
    for _, r in table.iterrows():
        yt = df.loc[df.arm == r.treatment, r.metric].to_numpy(dtype=float)
        yc = df.loc[df.arm == r.control, r.metric].to_numpy(dtype=float)
        assert (r.n_treatment, r.n_control) == (yt.size, yc.size)
        assert (r.converters_treatment, r.converters_control) == ((yt > 0).sum(), (yc > 0).sum())
        assert r.mean_treatment == pytest.approx(yt.mean()) and r.mean_control == pytest.approx(
            yc.mean()
        )
        assert r.loss_threshold == pytest.approx(config.BAYES_LOSS_FRACTION * yc.mean())
        verdict = decide(r.p_better, r.loss_ship, r.loss_keep, r.loss_threshold)
        assert r.decision == verdict.decision
        assert r.incremental_revenue_per_scale == pytest.approx(config.BUDGET * r.diff_mc_mean)
        assert r.loss_ship_per_scale == pytest.approx(config.BUDGET * r.loss_ship)
        assert r.ci_low <= r.diff_median <= r.ci_high
        if r.model == BOOTSTRAP:
            assert r.diff_mean == pytest.approx(yt.mean() - yc.mean(), rel=1e-12)
            assert abs(r.diff_mc_mean - r.diff_mean) <= 4 * r.diff_mcse
            assert r.post_mean_treatment == pytest.approx(
                yt.mean(), abs=4 * r.diff_sd / math.sqrt(50_000)
            )
            assert (
                math.isnan(r.spend_cap)
                if r.metric == "spend"
                else r.spend_cap == pytest.approx(cap)
            )
        else:
            assert math.isnan(r.diff_mean) and math.isnan(r.diff_sd)
    # Per-arm table: bootstrap rows carry the exact moments; two-part rows carry the NIG fit.
    assert len(arms) == 3 * 3
    boot = arms[arms.model == BOOTSTRAP]
    assert (boot.post_mean == boot.sample_mean).all()
    for _, a in arms[arms.model == TWO_PART].iterrows():
        pos = customers.loc[(customers.arm == a.arm) & (customers.spend > 0), "spend"]
        assert a.kappa_n == pytest.approx(WEAK_NIG_PRIOR.kappa0 + pos.size)
        assert a.alpha_n == pytest.approx(WEAK_NIG_PRIOR.alpha0 + pos.size / 2)
        prior = WEAK_NIG_PRIOR
        shrunk = (prior.kappa0 * prior.mu0 + pos.size * np.log(pos).mean()) / a.kappa_n
        assert a.mu_n == pytest.approx(shrunk, rel=1e-12)
        assert a.observed_share_at_cap == pytest.approx((pos >= customers.spend.max()).mean())
    # PPC: one row per arm and statistic; cap and floor are the data's extremes.
    assert len(ppc) == 3 * 8 and set(ppc.arm) == set(config.ARMS)
    assert (ppc.cap == customers.spend.max()).all()
    assert (ppc.floor == customers.loc[customers.spend > 0, "spend"].min()).all()
    # Histogram table: densities integrate to about 1 for every model and comparison.
    for _, part in hist.groupby(["model", "comparison"]):
        mass = ((part.bin_right - part.bin_left) * part.density).sum()
        assert mass == pytest.approx(1.0, abs=0.002)


def test_bootstrap_and_two_part_agree_when_the_log_normal_is_true(spend_readout):
    # On the fixture (Gamma spend, 10 to 30 converters per arm) the two models can even disagree
    # on the sign of an effect: a log-normal fitted to a handful of Gamma draws is a different
    # estimand from the sample mean. The readout keeps both, so the table must carry both rows
    # and the user can see the gap. When spend really is log-normal and the arms are large, the
    # two posteriors of the difference coincide: same mean within Monte Carlo and model error,
    # same spread, and the same P(better).
    table, _, _, _ = spend_readout
    assert set(table.loc[table.metric == "spend", "model"]) == {BOOTSTRAP, TWO_PART}

    rng = np.random.default_rng(config.SEEDS["bayes"] + 18)
    n, sigma = 200_000, 0.6
    arms = {}
    for arm, p_conv, mu in (("control", 0.01, 4.4), ("treatment", 0.012, 4.5)):
        conv = rng.random(n) < p_conv
        arms[arm] = np.where(conv, np.exp(mu + sigma * rng.standard_normal(n)), 0.0)
    boot = compare_means(
        bayesian_bootstrap_means(arms["treatment"], rng, N_DRAWS),
        bayesian_bootstrap_means(arms["control"], rng, N_DRAWS),
        arms["treatment"],
        arms["control"],
    )
    tp_t = two_part_draws(arms["treatment"], rng, N_DRAWS).mean_spend
    tp_c = two_part_draws(arms["control"], rng, N_DRAWS).mean_spend
    para = summarise_draws(tp_t, tp_c)
    # Both are root-n consistent for the same estimand, so their centres differ by O(1/n)
    # relative to a posterior sd of O(1/sqrt n); allow a tenth of a posterior sd.
    assert abs(boot.diff_mc_mean - para.diff_mc_mean) <= 0.1 * boot.diff_sd
    assert para.diff_mc_sd == pytest.approx(boot.diff_sd, rel=0.1)
    assert abs(boot.p_better - para.p_better) <= 0.02
    assert boot.ci_low == pytest.approx(para.ci_low, abs=0.15 * boot.diff_sd)
    assert boot.ci_high == pytest.approx(para.ci_high, abs=0.15 * boot.diff_sd)


def test_run_bayes_spend_writes_outputs(fixture_db, tmp_path):
    table = run_bayes_spend(fixture_db, tmp_path, n_draws=20_000, n_replicates=500)
    written = pd.read_csv(tmp_path / "tables" / "readout_bayes_spend.csv")
    assert len(written) == len(table) == 9
    assert len(pd.read_csv(tmp_path / "tables" / "bayes_spend_ppc.csv")) == 24
    assert (tmp_path / "tables" / "bayes_spend_posterior_hist.csv").stat().st_size > 0
    assert (tmp_path / "figures" / "bayes_spend_diff_posterior.png").stat().st_size > 0
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)["bayes_spend"]
    assert section["n_draws"] == 20_000 and section["seed"] == config.SEEDS["bayes"] + 1
    assert section["revenue_scale"] == config.BUDGET
    assert section["decision_rule"] == {
        "p_better_threshold": 0.95,
        "loss_fraction_of_control_mean": 0.01,
    }
    assert set(section["recommendation"]) == {
        "qualifying_emails",
        "recommended_email",
        "basis",
        "p_mens_better_than_womens",
    }
    assert len(section["rows"]) == 9 and len(section["arms"]) == 9 and len(section["ppc"]) == 24
