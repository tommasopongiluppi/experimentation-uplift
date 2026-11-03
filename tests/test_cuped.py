"""CUPED and Lin regression adjustment: known answers, planted effects and Monte Carlo checks."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from abkit import config
from abkit.cuped import (
    ArmMoments,
    cuped,
    moments_from_stats,
    pretreatment_design,
    regression_adjustment,
)
from abkit.frequentist import diff_in_means
from abkit.metrics import arm_sufficient_stats
from abkit.readout import COMPARISONS, READOUT_METRICS, build_cuped_readout, run_cuped
from tests.conftest import customers_con

TAU = 0.1  # planted average treatment effect
N_ARM = 2_000
BASE = 10.0  # control mean, kept away from zero so the relative lift is defined


def _simulate(rng, n_sim, rho, tau=TAU, n=N_ARM):
    """Y = 10 + tau D + rho X + sqrt(1 - rho²) e, X, e ~ N(0, 1): corr(X, Y) = rho per arm."""
    xt, xc = rng.standard_normal((n_sim, n)), rng.standard_normal((n_sim, n))
    noise = np.sqrt(1 - rho**2)
    yt = BASE + tau + rho * xt + noise * rng.standard_normal((n_sim, n))
    yc = BASE + rho * xc + noise * rng.standard_normal((n_sim, n))
    return yt, xt, yc, xc


def _variance_ratio(a, b):
    """Var(a) / Var(b) over paired simulations, with the ratio-estimator Monte Carlo SE."""
    da, db = (a - a.mean()) ** 2, (b - b.mean()) ** 2
    ratio = da.mean() / db.mean()
    mcse = (da - ratio * db).std(ddof=1) / (np.sqrt(a.size) * db.mean())
    return ratio, mcse


# --- closed-form and cross-checks ------------------------------------------------------------


def test_cuped_matches_row_level_adjustment(rng):
    yt, xt = rng.exponential(2.0, 500), rng.gamma(2.0, 3.0, 500)
    yc, xc = rng.exponential(1.5, 400), rng.gamma(2.0, 3.0, 400)
    yt = yt + 0.3 * xt
    yc = yc + 0.3 * xc
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))

    x, y = np.concatenate([xt, xc]), np.concatenate([yt, yc])
    theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
    adj_t, adj_c = yt - theta * (xt - x.mean()), yc - theta * (xc - x.mean())
    ref = stats.ttest_ind(adj_t, adj_c, equal_var=False)
    ci = ref.confidence_interval()
    assert res.theta == pytest.approx(theta, rel=1e-10)
    assert res.rho == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-10)
    assert res.diff == pytest.approx(adj_t.mean() - adj_c.mean(), rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert (res.ci_low, res.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-8)
    within = lambda a, b: (a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)  # noqa: E731
    assert res.variance_reduction == pytest.approx(1 - within(adj_t, adj_c) / within(yt, yc))
    raw = stats.ttest_ind(yt, yc, equal_var=False).confidence_interval()
    assert res.raw_diff == pytest.approx(yt.mean() - yc.mean(), rel=1e-10)
    assert res.ci_width_ratio == pytest.approx((ci.high - ci.low) / (raw.high - raw.low), rel=1e-8)


def test_cuped_with_exactly_linear_outcome_removes_all_noise():
    # Y = 2 X + 5 D exactly: the adjusted outcome is constant within each arm.
    x = np.arange(1.0, 11.0)
    res = cuped(ArmMoments.from_arrays(2 * x + 5, x), ArmMoments.from_arrays(2 * x, x))
    assert res.theta == pytest.approx(2.0)
    assert res.diff == pytest.approx(5.0)
    assert res.variance_reduction == pytest.approx(1.0)


def test_moments_from_stats_match_pandas(customers):
    st = arm_sufficient_stats(customers_con(customers))
    part = customers[customers.arm == "mens"]
    m = moments_from_stats(st, "mens", "spend")
    assert m.n == len(part)
    assert float(m.cov_xy()) == pytest.approx(part.spend.cov(part.history), rel=1e-9)
    assert float(m.var_x()) == pytest.approx(part.history.var(), rel=1e-9)


def test_lin_matches_statsmodels_formula(rng):
    n = 3_000
    df = pd.DataFrame(
        {
            "d": rng.integers(0, 2, n),
            "x1": rng.normal(2.0, 1.0, n),
            "x2": rng.integers(0, 3, n).astype(float),
        }
    )
    df["y"] = 1 + 0.5 * df.d + df.x1 + 0.4 * df.d * df.x1 + rng.standard_normal(n)
    res = regression_adjustment(df.y, df.d, df[["x1", "x2"]])
    for c in ("x1", "x2"):
        df[f"{c}c"] = df[c] - df[c].mean()
    ref = smf.ols("y ~ d * (x1c + x2c)", df).fit(cov_type="HC2")
    assert res.diff == pytest.approx(ref.params["d"], rel=1e-10)
    assert res.se == pytest.approx(ref.bse["d"], rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalues["d"], rel=1e-8)
    assert res.n_covariates == 2


def test_lin_without_covariate_signal_is_the_difference_in_means(rng):
    # A single binary covariate saturates the model: Lin equals the stratified estimator.
    n = 4_000
    d = rng.integers(0, 2, n)
    s = rng.integers(0, 2, n)
    y = rng.standard_normal(n) + d * (1 + s)
    res = regression_adjustment(y, d, s.astype(float))
    w = s.mean()
    strat = sum(
        (y[(d == 1) & (s == k)].mean() - y[(d == 0) & (s == k)].mean()) * (w if k else 1 - w)
        for k in (0, 1)
    )
    assert res.diff == pytest.approx(strat, rel=1e-10)


def test_pretreatment_design_is_fixed_and_pre_treatment_only(customers):
    design = pretreatment_design(customers)
    assert len(design) == len(customers)
    # 5 numeric + (7 - 1) history bands + (3 - 1) zip codes + (3 - 1) channels
    assert design.shape[1] == 15
    assert not set(design.columns) & set(config.OUTCOME_COLS)
    with pytest.raises(ValueError):
        pretreatment_design(customers, numeric=("spend",))


# --- Monte Carlo properties (fast) ------------------------------------------------------------


def test_cuped_is_unbiased():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 10)
    n_sim = 1_000
    yt, xt, yc, xc = _simulate(rng, n_sim, rho=0.6)
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))
    mcse = res.diff.std(ddof=1) / np.sqrt(n_sim)
    assert abs(res.diff.mean() - TAU) <= 3 * mcse


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_variance_reduction_is_rho_squared(rho):
    rng = np.random.default_rng(config.SEEDS["simulation"] + 11)
    n_sim = 1_000
    yt, xt, yc, xc = _simulate(rng, n_sim, rho=rho)
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))
    # reported within-arm reduction (sample rho² is unbiased to O(1/n), well below the MCSE)
    vr = res.variance_reduction
    assert abs(vr.mean() - rho**2) <= 3 * vr.std(ddof=1) / np.sqrt(n_sim)
    # realised: the variance of the estimator across experiments falls by a factor 1 - rho²
    ratio, mcse = _variance_ratio(res.diff, res.raw_diff)
    assert abs(ratio - (1 - rho**2)) <= 3 * mcse


def test_no_harm_when_covariate_is_uncorrelated():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 12)
    n_sim = 2_000
    yt, xt, yc, xc = _simulate(rng, n_sim, rho=0.0)
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))
    ratio, mcse = _variance_ratio(res.diff, res.raw_diff)
    assert abs(ratio - 1) <= 3 * mcse
    raw = diff_in_means(yt.mean(1), yt.var(1, ddof=1), N_ARM, yc.mean(1), yc.var(1, ddof=1), N_ARM)
    np.testing.assert_allclose(res.raw_diff, raw.diff, rtol=1e-10)


def test_lin_is_unbiased_with_heterogeneous_effects():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 13)
    n_sim, n = 300, 1_000
    est = np.empty(n_sim)
    for i in range(n_sim):
        d = np.repeat([1.0, 0.0], n)
        x = rng.standard_normal(2 * n)
        y = TAU * d + 0.6 * x + 0.3 * d * x + rng.standard_normal(2 * n)  # E[x] = 0: ATE = TAU
        est[i] = regression_adjustment(y, d, x).diff
    assert abs(est.mean() - TAU) <= 3 * est.std(ddof=1) / np.sqrt(n_sim)


# --- coverage (slow) --------------------------------------------------------------------------

N_COV = 2_000
LEVEL = 1 - config.ALPHA
COVERAGE_MCSE = np.sqrt(LEVEL * (1 - LEVEL) / N_COV)


@pytest.mark.slow
def test_cuped_coverage():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 20)
    yt, xt, yc, xc = _simulate(rng, N_COV, rho=0.6)
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))
    covered = (res.ci_low <= TAU) & (res.ci_high >= TAU)
    assert abs(covered.mean() - LEVEL) <= 3 * COVERAGE_MCSE


@pytest.mark.slow
def test_lin_coverage():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 21)
    n = N_ARM
    d = np.repeat([1.0, 0.0], n)
    covered = np.empty(N_COV, dtype=bool)
    for i in range(N_COV):
        x = rng.standard_normal(2 * n)
        g = rng.integers(0, 3, 2 * n).astype(float)
        # heteroskedastic noise and an effect that varies with x and g; E[x] = 0, E[g] = 1
        y = (
            TAU * d
            + 0.6 * x
            + 0.3 * d * x
            + 0.2 * d * (g - 1)
            + (1 + 0.5 * d) * rng.standard_normal(2 * n)
        )
        r = regression_adjustment(y, d, np.column_stack([x, g]))
        covered[i] = r.ci_low <= TAU <= r.ci_high
    assert abs(covered.mean() - LEVEL) <= 3 * COVERAGE_MCSE


# --- readout ----------------------------------------------------------------------------------


def test_cuped_readout_covers_every_comparison_and_metric(customers):
    table = build_cuped_readout(customers_con(customers))
    assert len(table) == len(COMPARISONS) * len(READOUT_METRICS)
    for _, r in table.iterrows():
        part = customers[customers.arm.isin([r.treatment, r.control])]
        cap = np.quantile(customers.loc[customers.conversion == 1, "spend"], 0.95)
        y = part.spend.clip(upper=cap) if r.metric == "spend_winsorised" else part[r.metric]
        y = y.astype(float)
        t = part.arm == r.treatment
        assert r.raw_diff == pytest.approx(y[t].mean() - y[~t].mean(), rel=1e-9, abs=1e-12)
        assert r.rho == pytest.approx(np.corrcoef(part.history, y)[0, 1], rel=1e-8)
        assert r.lin_n_covariates <= 15


def test_run_cuped_writes_outputs(fixture_db, tmp_path):
    table = run_cuped(fixture_db, tmp_path)
    written = pd.read_csv(tmp_path / "tables" / "readout_cuped.csv")
    assert len(written) == len(table)
    assert (tmp_path / "figures" / "intervals_raw_cuped_lin.png").stat().st_size > 0
    section = json.loads((tmp_path / "results.json").read_text())["cuped"]
    assert section["covariate"] == "history"
    assert len(section["rows"]) == len(table)
    assert set(section["variance_reduction_by_metric"]) == set(READOUT_METRICS)
