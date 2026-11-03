import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.balance import (
    CATEGORICAL_BALANCE_VARS,
    balance_table,
    categorical_balance_tests,
    smd,
)
from tests.conftest import customers_con


def test_smd_hand_calculation():
    # a = [1, 2, 3, 4]: mean 2.5, sample variance 5/3
    # b = [2, 4, 6, 8]: mean 5.0, sample variance 20/3
    # SMD = (2.5 - 5) / sqrt((5/3 + 20/3) / 2) = -2.5 / sqrt(25/6) = -sqrt(6)/2
    assert smd(2.5, 5 / 3, 5.0, 20 / 3) == pytest.approx(-math.sqrt(6) / 2)
    assert smd(2.5, 5 / 3, 5.0, 20 / 3) == pytest.approx(-1.2247449, abs=1e-7)


def test_smd_zero_variance():
    assert smd(1.0, 0.0, 1.0, 0.0) == 0.0
    assert smd(1.0, 0.0, 0.0, 0.0) == math.inf


def _toy(customers: pd.DataFrame) -> pd.DataFrame:
    """Take 12 cleaned rows and set arms and recency to the hand-calculation example."""
    toy = customers.head(12).copy()
    toy["arm"] = ["mens"] * 4 + ["control"] * 4 + ["womens"] * 4
    toy["recency"] = [1, 2, 3, 4, 2, 4, 6, 8, 5, 5, 5, 5]
    return toy


def test_balance_table_matches_hand_calculation_through_sql(customers):
    table = balance_table(customers_con(_toy(customers)))
    row = table[table.variable == "recency"].iloc[0]
    assert row["mean_mens"] == 2.5 and row["mean_control"] == 5.0
    assert row["smd_mens_vs_control"] == pytest.approx(-math.sqrt(6) / 2)
    # womens recency is constant 5 with variance 0: SMD vs control = 0 / sqrt((0 + 20/3) / 2)
    assert row["smd_womens_vs_control"] == pytest.approx(0.0)
    assert row["flagged"]


def test_balance_table_means_match_pandas(customers):
    table = balance_table(customers_con(customers))
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    expected = customers.groupby("arm")[numeric].mean()
    for var in numeric:
        row = table[table.variable == var].iloc[0]
        for arm in config.ARMS:
            assert row[f"mean_{arm}"] == pytest.approx(expected.loc[arm, var])
    for var in CATEGORICAL_BALANCE_VARS:
        shares = pd.crosstab(customers[var], customers["arm"], normalize="columns")
        for level in shares.index:
            row = table[(table.variable == var) & (table.level == level)].iloc[0]
            for arm in config.ARMS:
                assert row[f"mean_{arm}"] == pytest.approx(shares.loc[level, arm])


def test_balance_table_smd_matches_pandas(customers):
    table = balance_table(customers_con(customers))
    g = customers.groupby("arm")["history"]
    m, v = g.mean(), g.var(ddof=1)
    expected = (m["mens"] - m["control"]) / np.sqrt((v["mens"] + v["control"]) / 2)
    row = table[table.variable == "history"].iloc[0]
    assert row["smd_mens_vs_control"] == pytest.approx(expected)


def test_balance_table_has_one_row_per_level(customers):
    table = balance_table(customers_con(customers))
    n_levels = sum(customers[v].nunique() for v in CATEGORICAL_BALANCE_VARS)
    assert len(table) == 5 + n_levels
    assert not table.duplicated(["variable", "level"]).any()


def test_balance_contains_no_outcomes(customers):
    table = balance_table(customers_con(customers))
    assert set(table.variable) <= set(config.PRE_TREATMENT_COLS)


def test_categorical_tests_match_scipy(customers):
    tests = categorical_balance_tests(customers_con(customers)).set_index("variable")
    for var in CATEGORICAL_BALANCE_VARS:
        crosstab = pd.crosstab(customers[var], customers["arm"])[list(config.ARMS)]
        ref = stats.chi2_contingency(crosstab.to_numpy(), correction=False)
        assert tests.loc[var, "statistic"] == pytest.approx(ref.statistic)
        assert tests.loc[var, "p_value"] == pytest.approx(ref.pvalue)
        assert tests.loc[var, "dof"] == ref.dof
