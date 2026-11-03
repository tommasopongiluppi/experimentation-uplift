"""Covariate balance: standardised mean differences and chi-square tests by arm.

Only pre-treatment variables enter, never an outcome; the per-arm statistics come from
``sql/balance.sql``.
"""

import math

import duckdb
import pandas as pd
from scipy import stats

from abkit import config
from abkit.metrics import run_sql

#: Comparisons reported in the balance table, as (first arm, second arm).
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mens", "control"),
    ("womens", "control"),
    ("mens", "womens"),
)
CATEGORICAL_BALANCE_VARS: tuple[str, ...] = ("history_segment", "zip_code", "channel")


def smd(mean_a: float, var_a: float, mean_b: float, var_b: float) -> float:
    """Standardised mean difference between two groups.

    SMD = (mean_a - mean_b) / sqrt((var_a + var_b) / 2), the definition in analysis plan §5.

    Parameters
    ----------
    mean_a, var_a : float
        Mean and sample variance in group a.
    mean_b, var_b : float
        Mean and sample variance in group b.

    Returns
    -------
    float
        The SMD. If both variances are zero it is 0 when the means are equal and signed infinity
        otherwise.
    """
    diff = mean_a - mean_b
    pooled_sd = math.sqrt((var_a + var_b) / 2)
    if pooled_sd == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return diff / pooled_sd


def balance_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-arm n, mean and variance of every pre-treatment variable and category level.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    pandas.DataFrame
        Long format: variable, level, arm, n, total, mean, variance.
    """
    df = run_sql(con, "balance")
    df["level"] = df["level"].fillna("")
    return df


def balance_table(
    con: duckdb.DuckDBPyConnection, threshold: float = config.SMD_THRESHOLD
) -> pd.DataFrame:
    """Balance table: mean by arm and SMD of each comparison, one row per variable or level.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    threshold : float
        |SMD| above this is flagged (``config.SMD_THRESHOLD``, analysis plan §5).

    Returns
    -------
    pandas.DataFrame
        Columns ``variable``, ``level``, ``mean_<arm>`` for each arm,
        ``smd_<a>_vs_<b>`` for each comparison, ``max_abs_smd`` and ``flagged``.
    """
    long = balance_stats(con)
    rows = []
    for (variable, level), group in long.groupby(["variable", "level"], sort=False):
        by_arm = group.set_index("arm")
        row: dict[str, object] = {"variable": variable, "level": level}
        for arm in config.ARMS:
            row[f"mean_{arm}"] = float(by_arm.loc[arm, "mean"])
        for a, b in COMPARISONS:
            row[f"smd_{a}_vs_{b}"] = smd(
                by_arm.loc[a, "mean"],
                by_arm.loc[a, "variance"],
                by_arm.loc[b, "mean"],
                by_arm.loc[b, "variance"],
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    smd_cols = [f"smd_{a}_vs_{b}" for a, b in COMPARISONS]
    table["max_abs_smd"] = table[smd_cols].abs().max(axis=1)
    table["flagged"] = table["max_abs_smd"] > threshold
    return table


def categorical_balance_tests(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Chi-square test of independence between arm and each categorical variable.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    pandas.DataFrame
        One row per variable: n_levels, statistic, dof, p_value.
    """
    long = balance_stats(con)
    rows = []
    for variable in CATEGORICAL_BALANCE_VARS:
        counts = (
            long[long["variable"] == variable]
            .pivot_table(index="level", columns="arm", values="total", aggfunc="sum")
            .loc[:, list(config.ARMS)]
            .round()
            .astype(int)
        )
        result = stats.chi2_contingency(counts.to_numpy(), correction=False)
        rows.append(
            {
                "variable": variable,
                "n_levels": len(counts),
                "statistic": float(result.statistic),
                "dof": int(result.dof),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)
