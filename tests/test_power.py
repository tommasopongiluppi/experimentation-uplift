"""MDE, sample size and power: closed forms, statsmodels cross-checks and a Monte Carlo check."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower

from abkit import config
from abkit.cuped import ArmMoments, cuped
from abkit.frequentist import diff_in_means, diff_in_proportions
from abkit.power import (
    N_PRIMARY_TESTS,
    POWER_METRICS,
    POWER_TARGET,
    RELATIVE_LIFTS,
    build_power_tables,
    design_inputs,
    holm_alpha,
    mde,
    power_for_effect,
    run_power,
    sample_size,
    standard_error,
)
from tests.conftest import customers_con

DESIGNS = [
    # baseline, variance, n_treatment, n_control, alpha, power, variance_reduction
    (0.65, 134.3, 21_306, 21_306, 0.05, 0.8, 0.0),
    (0.65, 134.3, 21_307, 21_306, 0.025, 0.8, 0.0134),
    (0.1, 0.09, 5_000, 8_000, 0.01, 0.9, 0.0),
    (12.0, 4.0, 40, 60, 0.1, 0.7, 0.36),
]


# --- closed forms ------------------------------------------------------------------------------


@pytest.mark.parametrize(("base", "var", "nt", "nc", "alpha", "power", "vr"), DESIGNS)
def test_mde_matches_hand_formula(base, var, nt, nc, alpha, power, vr):
    res = mde(base, var, nt, alpha, power, vr, n_control=nc)
    se = math.sqrt(var * (1 - vr) * (1 / nt + 1 / nc))
    expected = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se
    assert res.se == pytest.approx(se, rel=1e-12)
    assert res.absolute == pytest.approx(expected, rel=1e-12)
    assert res.relative == pytest.approx(expected / base, rel=1e-12)
    assert standard_error(var, nt, nc, vr) == se


def test_mde_at_80_power_and_5_percent_is_2_8_standard_errors():
    res = mde(1.0, 1.0, 100)
    assert res.absolute / res.se == pytest.approx(1.959964 + 0.841621, abs=1e-6)


def test_cuped_scales_the_mde_by_sqrt_one_minus_reduction():
    plain = mde(0.65, 134.3, 21_306)
    adjusted = mde(0.65, 134.3, 21_306, variance_reduction=0.36)
    assert adjusted.absolute == pytest.approx(plain.absolute * math.sqrt(1 - 0.36), rel=1e-12)


def test_sample_size_inverts_mde():
    for base, var, nt, _, alpha, power, vr in DESIGNS:
        res = mde(base, var, nt, alpha, power, vr)
        n = sample_size(base, var, res.relative, alpha, power, vr)
        assert n.n_exact == pytest.approx(nt, rel=1e-9)
        assert n.n_per_arm == math.ceil(n.n_exact)
        assert n.absolute == pytest.approx(res.absolute, rel=1e-12)


def test_sample_size_rounding_brackets_the_target_power():
    n = sample_size(0.1, 0.09, 0.10)
    assert n.n_per_arm - 1 < n.n_exact <= n.n_per_arm
    z = stats.norm.ppf(0.975)
    for k, above in ((n.n_per_arm, True), (n.n_per_arm - 1, False)):
        achieved = stats.norm.cdf(n.absolute / standard_error(0.09, k) - z)
        assert bool(achieved >= POWER_TARGET) == above


def test_holm_alpha():
    assert holm_alpha(0.05, 2) == 0.025
    assert holm_alpha(0.05, 1) == 0.05
    assert N_PRIMARY_TESTS == len(config.TREATMENTS) == 2
    with pytest.raises(ValueError):
        holm_alpha(0.05, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"power": 1.0},
        {"variance": 0.0},
        {"variance_reduction": 1.0},
        {"variance_reduction": -0.1},
        {"n_treatment": 1},
        {"baseline": 0.0},
    ],
)
def test_invalid_mde_design_raises(kwargs):
    with pytest.raises(ValueError):
        mde(**({"baseline": 1.0, "variance": 1.0, "n_treatment": 100} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"power": 0.0},
        {"variance": -1.0},
        {"variance_reduction": 1.0},
        {"relative_lift": 0.0},
        {"baseline": 0.0},
    ],
)
def test_invalid_sample_size_design_raises(kwargs):
    with pytest.raises(ValueError):
        sample_size(**({"baseline": 1.0, "variance": 1.0, "relative_lift": 0.1} | kwargs))


# --- statsmodels cross-checks ------------------------------------------------------------------


@pytest.mark.parametrize(("base", "var", "nt", "nc", "alpha", "power", "vr"), DESIGNS)
def test_power_matches_statsmodels_normal_and_t(base, var, nt, nc, alpha, power, vr):
    effects = np.array([0.0, 0.3, 1.0, 2.5]) * math.sqrt(var * (1 - vr) / nt)
    d = effects / math.sqrt(var * (1 - vr))  # standardised effect size on the adjusted outcome
    ours = power_for_effect(effects, var, nt, alpha, vr, n_control=nc)
    ref = [NormalIndPower().power(di, nt, alpha, ratio=nc / nt) for di in d]
    np.testing.assert_allclose(ours, ref, atol=1e-6, rtol=0)
    ours_t = power_for_effect(effects, var, nt, alpha, vr, n_control=nc, use_t=True)
    ref_t = [TTestIndPower().power(di, nt, alpha, ratio=nc / nt) for di in d]
    np.testing.assert_allclose(ours_t, ref_t, atol=1e-6, rtol=0)
    assert ours[0] == pytest.approx(alpha, abs=1e-6)  # size under the null


@pytest.mark.parametrize(("base", "var", "nt", "nc", "alpha", "power", "vr"), DESIGNS)
def test_mde_matches_statsmodels(base, var, nt, nc, alpha, power, vr):
    res = mde(base, var, nt, alpha, power, vr, n_control=nc)
    d = res.absolute / math.sqrt(var * (1 - vr))
    # The closed form is the one-sided equation at alpha / 2: statsmodels reproduces it exactly.
    one_sided = NormalIndPower().power(d, nt, alpha / 2, ratio=nc / nt, alternative="larger")
    assert one_sided == pytest.approx(power, abs=1e-6)
    # statsmodels' root-finder (brentq_expanding) stops at xtol = 1e-5 on the standardised
    # effect, so its solved value is compared at that tolerance; the exact check is the one above.
    solved = NormalIndPower().solve_power(
        None, nobs1=nt, alpha=alpha / 2, power=power, ratio=nc / nt, alternative="larger"
    )
    assert d == pytest.approx(solved, abs=1e-5)
    # The two-sided test adds the far tail Phi(-z_{1-alpha/2} - d sqrt(n)), so its power at the
    # MDE exceeds the target by at most that mass (about 1e-6 at 80% power, alpha 0.05).
    two_sided = NormalIndPower().power(d, nt, alpha, ratio=nc / nt)
    far_tail = stats.norm.cdf(-stats.norm.ppf(1 - alpha / 2) - stats.norm.ppf(power))
    assert power < two_sided <= power + far_tail + 1e-12


@pytest.mark.parametrize("lift", RELATIVE_LIFTS)
def test_sample_size_matches_statsmodels(lift):
    base, var, alpha, power, vr = 0.106, 0.0949, 0.05, 0.8, 0.02
    res = sample_size(base, var, lift, alpha, power, vr)
    d = res.absolute / math.sqrt(var * (1 - vr))
    assert NormalIndPower().power(d, res.n_exact, alpha / 2, alternative="larger") == pytest.approx(
        power, abs=1e-6
    )
    solved = NormalIndPower().solve_power(d, None, alpha / 2, power, alternative="larger")
    assert res.n_exact == pytest.approx(solved, rel=1e-6)
    # A pooled t-test needs slightly more customers than the normal approximation (its critical
    # value is larger); the gap shrinks with the sample size and is under 0.1% here.
    solved_t = TTestIndPower().solve_power(d, None, alpha / 2, power, alternative="larger")
    assert res.n_exact < solved_t < res.n_exact * 1.001


def test_power_curve_passes_through_the_mde():
    res = mde(0.65, 134.3, 21_306)
    at_mde = power_for_effect(res.absolute, 134.3, 21_306)
    assert at_mde == pytest.approx(POWER_TARGET, abs=1e-5)
    assert power_for_effect(res.absolute, 134.3, 21_306, use_t=True) == pytest.approx(
        POWER_TARGET, abs=1e-4
    )


# --- Monte Carlo: a test designed for 80% power rejects 80% of the time (slow) -----------------

N_SIM = 4_000
N_ARM = 1_000
MCSE = math.sqrt(POWER_TARGET * (1 - POWER_TARGET) / N_SIM)


def _assert_rejection_rate(name, rejected):
    rate = rejected.mean()
    assert abs(rate - POWER_TARGET) <= 3 * MCSE, f"{name}: rejects {rate:.4f}, MCSE {MCSE:.4f}"


@pytest.mark.slow
def test_welch_test_rejects_at_the_designed_rate():
    rng = np.random.default_rng(config.SEEDS["simulation"] + 30)
    sigma, base = 2.0, 5.0
    delta = mde(base, sigma**2, N_ARM).absolute
    yt = base + delta + sigma * rng.standard_normal((N_SIM, N_ARM))
    yc = base + sigma * rng.standard_normal((N_SIM, N_ARM))
    res = diff_in_means(yt.mean(1), yt.var(1, ddof=1), N_ARM, yc.mean(1), yc.var(1, ddof=1), N_ARM)
    _assert_rejection_rate("welch", res.p_value < config.ALPHA)


@pytest.mark.slow
def test_proportion_test_rejects_at_the_designed_rate():
    # p near 1/2 keeps p (1 - p) nearly equal in both arms, so the common-variance design holds.
    rng = np.random.default_rng(config.SEEDS["simulation"] + 31)
    p0 = 0.5
    delta = mde(p0, p0 * (1 - p0), N_ARM).absolute
    xt = rng.binomial(N_ARM, p0 + delta, N_SIM)
    xc = rng.binomial(N_ARM, p0, N_SIM)
    res = diff_in_proportions(xt, N_ARM, xc, N_ARM)
    _assert_rejection_rate("two-proportion z", res.p_value < config.ALPHA)


@pytest.mark.slow
def test_cuped_test_rejects_at_the_designed_rate_with_variance_reduction():
    # Y = base + delta D + rho X + sqrt(1 - rho²) e with X, e ~ N(0, 1): var(Y) = 1 and CUPED
    # on X removes a fraction rho² of it, which is the variance_reduction the design uses.
    rng = np.random.default_rng(config.SEEDS["simulation"] + 32)
    rho, base = 0.6, 3.0
    delta = mde(base, 1.0, N_ARM, variance_reduction=rho**2).absolute
    noise = math.sqrt(1 - rho**2)
    xt, xc = rng.standard_normal((N_SIM, N_ARM)), rng.standard_normal((N_SIM, N_ARM))
    yt = base + delta + rho * xt + noise * rng.standard_normal((N_SIM, N_ARM))
    yc = base + rho * xc + noise * rng.standard_normal((N_SIM, N_ARM))
    res = cuped(ArmMoments.from_arrays(yt, xt), ArmMoments.from_arrays(yc, xc))
    _assert_rejection_rate("cuped", res.p_value < config.ALPHA)
    # Without the adjustment the same effect is underpowered, so the reduction matters.
    raw = diff_in_means(yt.mean(1), yt.var(1, ddof=1), N_ARM, yc.mean(1), yc.var(1, ddof=1), N_ARM)
    assert (raw.p_value < config.ALPHA).mean() < POWER_TARGET - 3 * MCSE


# --- readout on the fixture --------------------------------------------------------------------


@pytest.fixture
def tables(customers):
    return build_power_tables(customers_con(customers))


def test_design_inputs_come_from_the_control_arm(customers):
    inputs = design_inputs(customers_con(customers)).set_index("metric")
    ctrl = customers[customers.arm == config.CONTROL]
    for metric in POWER_METRICS:
        y = ctrl[metric].astype(float)
        assert inputs.loc[metric, "baseline"] == pytest.approx(y.mean(), rel=1e-9)
        assert inputs.loc[metric, "variance"] == pytest.approx(y.var(ddof=1), rel=1e-9)
        assert inputs.loc[metric, "n_control"] == len(ctrl)
        lo, hi = inputs.loc[metric, ["variance_reduction_min", "variance_reduction_max"]]
        assert lo <= inputs.loc[metric, "variance_reduction"] <= hi
        assert lo >= 0 and hi < 1


def test_mde_table_layout_and_values(tables, customers):
    mdes, _, _, design = tables
    n = customers.arm.value_counts().min()
    assert design["n_per_arm"] == n and design["alpha_holm"] == config.ALPHA / 2
    assert len(mdes) == len(POWER_METRICS) * 2 * 2
    assert set(mdes.alpha_label) == {"unadjusted", "holm"}
    for _, r in mdes.iterrows():
        ref = mde(r.baseline, r.variance, n, r.alpha, POWER_TARGET, r.variance_reduction)
        assert r.mde_absolute == pytest.approx(ref.absolute, rel=1e-12)
        assert r.mde_relative == pytest.approx(ref.relative, rel=1e-12)
        assert (r.variance_reduction == 0) == (not r.cuped)
    # Holm's level is stricter and CUPED helps, so the MDE orders accordingly.
    by_level = mdes.set_index(["metric", "cuped", "alpha_label"])["mde_absolute"]
    assert (
        by_level.xs("holm", level="alpha_label") > by_level.xs("unadjusted", level="alpha_label")
    ).all()
    by_cuped = mdes.set_index(["metric", "alpha_label", "cuped"])["mde_absolute"]
    assert (by_cuped.xs(True, level="cuped") <= by_cuped.xs(False, level="cuped")).all()


def test_sample_size_table_layout_and_values(tables):
    _, sizes, _, _ = tables
    assert len(sizes) == len(POWER_METRICS) * 2 * 2 * len(RELATIVE_LIFTS)
    assert set(sizes.relative_lift) == set(RELATIVE_LIFTS)
    for _, r in sizes.iterrows():
        inputs = tables[0][(tables[0].metric == r.metric)].iloc[0]
        ref = sample_size(
            inputs.baseline,
            inputs.variance,
            r.relative_lift,
            r.alpha,
            POWER_TARGET,
            r.variance_reduction,
        )
        assert r.n_per_arm == ref.n_per_arm and r.n_total_two_arms == 2 * ref.n_per_arm
    # Larger lifts need fewer customers, by a factor of (lift ratio)² in the exact solution.
    for (_, _, _), g in sizes.groupby(["metric", "alpha_label", "cuped"]):
        g = g.sort_values("relative_lift")
        assert g.n_exact.is_monotonic_decreasing
        ratio = g.n_exact.iloc[0] / g.n_exact.iloc[-1]
        assert ratio == pytest.approx((g.relative_lift.iloc[-1] / g.relative_lift.iloc[0]) ** 2)


def test_power_curve_table(tables):
    mdes, _, curve, _ = tables
    assert set(curve.metric) == set(POWER_METRICS)
    for (metric, label, with_cuped), g in curve.groupby(["metric", "alpha_label", "cuped"]):
        assert g.relative_lift.iloc[0] == 0 and g.power.iloc[0] == pytest.approx(g.alpha.iloc[0])
        assert g.power.is_monotonic_increasing and g.power.iloc[-1] > POWER_TARGET
        row = mdes[
            (mdes.metric == metric) & (mdes.alpha_label == label) & (mdes.cuped == with_cuped)
        ]
        at_mde = np.interp(row.mde_relative.iloc[0], g.relative_lift, g.power)
        assert at_mde == pytest.approx(POWER_TARGET, abs=2e-3)  # linear interpolation on the grid


def test_run_power_writes_outputs(fixture_db, tmp_path):
    mdes = run_power(fixture_db, tmp_path)
    for name in ("power", "power_sample_size", "power_curve"):
        assert (tmp_path / "tables" / f"{name}.csv").stat().st_size > 0
    assert len(pd.read_csv(tmp_path / "tables" / "power.csv")) == len(mdes)
    assert (tmp_path / "figures" / "power_curve.png").stat().st_size > 0
    text = (tmp_path / "results.json").read_text()
    section = json.loads(text)["power"]
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    assert section["power"] == POWER_TARGET and section["alpha"] == config.ALPHA
    assert section["n_primary_tests"] == 2 and section["relative_lifts"] == list(RELATIVE_LIFTS)
    assert set(section["metrics"]) == set(POWER_METRICS)
    assert len(section["mde_rows"]) == len(mdes)
    assert len(section["sample_size_rows"]) == len(mdes) * len(RELATIVE_LIFTS)
