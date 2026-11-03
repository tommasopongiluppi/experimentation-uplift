import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.metrics import (
    CONVERTER_SPEND,
    METRICS,
    arm_metrics,
    arm_sufficient_stats,
    summarize_sufficient_stats,
)
from tests.conftest import customers_con

TOL = {"rel": 1e-9, "abs": 1e-9}


def _expected_cap(df: pd.DataFrame) -> float:
    return float(np.quantile(df.loc[df.conversion == 1, "spend"], 0.95))


def _winsorised(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spend"] = out["spend"].clip(upper=_expected_cap(df))
    return out


def _check_against_pandas(table: pd.DataFrame, df: pd.DataFrame) -> None:
    """Every statistic in the tidy table equals a pandas group-by on df."""
    df = df.copy()
    for col in METRICS:
        df[col] = df[col].astype(float)
    g = df.groupby("arm")
    conv = df[df.conversion == 1].groupby("arm")["spend"]
    assert sorted(table.arm.unique()) == sorted(config.ARMS)
    for arm in config.ARMS:
        part = df[df.arm == arm]
        for metric in METRICS:
            row = table[(table.arm == arm) & (table.metric == metric)].iloc[0]
            y = part[metric]
            assert row["n"] == len(part)
            assert row["sum"] == pytest.approx(y.sum(), **TOL)
            assert row["sum_sq"] == pytest.approx((y * y).sum(), **TOL)
            assert row["mean"] == pytest.approx(g[metric].mean()[arm], **TOL)
            assert row["var"] == pytest.approx(g[metric].var(ddof=1)[arm], **TOL)
            assert row["history_mean"] == pytest.approx(part.history.mean(), **TOL)
            assert row["history_var"] == pytest.approx(part.history.var(ddof=1), **TOL)
            assert row["cov_history"] == pytest.approx(part[metric].cov(part.history), **TOL)
        row = table[(table.arm == arm) & (table.metric == CONVERTER_SPEND)].iloc[0]
        assert row["n"] == conv.count()[arm]
        assert row["sum"] == pytest.approx(conv.sum()[arm], **TOL)
        assert row["mean"] == pytest.approx(conv.mean()[arm], **TOL)
        assert row["var"] == pytest.approx(conv.var(ddof=1)[arm], **TOL)
        assert np.isnan(row["cov_history"])


def test_arm_metrics_match_pandas_groupby(customers):
    table = arm_metrics(customers_con(customers))
    assert len(table) == len(config.ARMS) * (len(METRICS) + 1)
    assert table["spend_cap"].isna().all()
    _check_against_pandas(table, customers)


def test_winsorised_arm_metrics_match_pandas_groupby(customers):
    table = arm_metrics(customers_con(customers), winsorize=True)
    _check_against_pandas(table, _winsorised(customers))


def test_winsorisation_caps_exactly_at_converters_quantile(customers):
    cap = _expected_cap(customers)
    stats = arm_sufficient_stats(customers_con(customers), winsorize=True)
    assert stats["spend_cap"].to_numpy() == pytest.approx([cap] * 3, **TOL)
    # The cap must bind on the fixture, otherwise the test checks nothing.
    assert (customers.spend > cap).sum() > 0
    # Not the quantile over all customers, which is $0 because almost nobody converts.
    assert np.quantile(customers.spend, 0.95) == 0.0 < cap


def test_winsorisation_planted_cap_is_pooled_not_per_arm(customers):
    # 100 converters with spend 1..100 spread over the arms: mens 1..40, womens 41..70,
    # control 71..100. Pooled 95th percentile (linear) = 1 + 0.95 * 99 = 95.05.
    toy = customers.head(130).copy()
    toy["arm"] = ["mens"] * 50 + ["womens"] * 40 + ["control"] * 40
    spend = np.r_[
        np.arange(1, 41),
        np.zeros(10),
        np.arange(41, 71),
        np.zeros(10),
        np.arange(71, 101),
        np.zeros(10),
    ].astype(float)
    toy["spend"] = spend
    toy["conversion"] = (spend > 0).astype(int)
    toy["visit"] = toy["conversion"]

    raw = arm_sufficient_stats(customers_con(toy)).set_index("arm")
    win = arm_sufficient_stats(customers_con(toy), winsorize=True).set_index("arm")
    assert win["spend_cap"].to_numpy() == pytest.approx([95.05] * 3, **TOL)

    # Only control has spend above the cap (96..100): each becomes exactly 95.05.
    for arm in ("mens", "womens"):
        assert win.loc[arm, "spend_sum"] == raw.loc[arm, "spend_sum"]
        assert win.loc[arm, "spend_sum_sq"] == raw.loc[arm, "spend_sum_sq"]
    capped = np.minimum(np.arange(71, 101, dtype=float), 95.05)
    assert win.loc["control", "spend_sum"] == pytest.approx(capped.sum(), **TOL)
    assert win.loc["control", "spend_sum_sq"] == pytest.approx((capped**2).sum(), **TOL)
    assert win.loc["control", "conv_spend_sum"] == pytest.approx(capped.sum(), **TOL)
    # Zero spenders are untouched, so counts do not change.
    assert (win["n_converters"] == raw["n_converters"]).all()


def test_summarize_closed_form():
    # One arm, y = [0, 1, 1, 2], x = [1, 2, 3, 4]:
    # mean y = 1, var y = 2/3, mean x = 2.5, var x = 5/3,
    # cov(x, y) = ((-1.5)(-1) + (-0.5)(0) + (0.5)(0) + (1.5)(1)) / 3 = 1.
    # Converters: spend [1, 1, 2] -> mean 4/3, var 1/3.
    y = np.array([0.0, 1.0, 1.0, 2.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    row = {"arm": "control", "n": 4, "history_sum": x.sum(), "history_sum_sq": (x * x).sum()}
    for m in METRICS:
        row |= {f"{m}_sum": y.sum(), f"{m}_sum_sq": (y * y).sum(), f"history_x_{m}": (x * y).sum()}
    row |= {"n_converters": 3, "conv_spend_sum": 4.0, "conv_spend_sum_sq": 6.0, "spend_cap": np.nan}
    table = summarize_sufficient_stats(pd.DataFrame([row])).set_index("metric")
    assert table.loc["spend", "mean"] == pytest.approx(1.0)
    assert table.loc["spend", "var"] == pytest.approx(2 / 3)
    assert table.loc["spend", "history_mean"] == pytest.approx(2.5)
    assert table.loc["spend", "history_var"] == pytest.approx(5 / 3)
    assert table.loc["spend", "cov_history"] == pytest.approx(1.0)
    assert table.loc[CONVERTER_SPEND, "mean"] == pytest.approx(4 / 3)
    assert table.loc[CONVERTER_SPEND, "var"] == pytest.approx(1 / 3)


def test_summarize_small_n_gives_nan_variance():
    row = {"arm": "mens", "n": 1, "history_sum": 5.0, "history_sum_sq": 25.0}
    for m in METRICS:
        row |= {f"{m}_sum": 1.0, f"{m}_sum_sq": 1.0, f"history_x_{m}": 5.0}
    row |= {"n_converters": 0, "conv_spend_sum": 0.0, "conv_spend_sum_sq": 0.0, "spend_cap": 1.0}
    table = summarize_sufficient_stats(pd.DataFrame([row])).set_index("metric")
    assert table.loc["visit", "mean"] == 1.0
    assert np.isnan(table.loc["visit", "var"]) and np.isnan(table.loc["visit", "cov_history"])
    assert np.isnan(table.loc[CONVERTER_SPEND, "mean"])


def test_winsorise_without_converters_raises(customers):
    toy = customers.head(20).copy()
    toy[["conversion", "spend"]] = [0, 0.0]
    with pytest.raises(ValueError, match="no converters"):
        arm_sufficient_stats(customers_con(toy), winsorize=True)
