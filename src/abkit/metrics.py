"""Run the SQL metric layer and return per-arm sufficient statistics.

The design follows production experimentation platforms: aggregate in SQL, infer in Python.
The database reduces each arm to a handful of sums (counts, sums, sums of squares and
cross-products), which are sufficient statistics for means, variances and covariances. Python
then does the inference from these few numbers, so the statistics code never needs row-level
data and the same code would work on a warehouse holding billions of rows.
"""

import logging
from collections.abc import Mapping
from importlib import resources
from typing import Any

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_sql(name: str) -> str:
    """Read a query from the package's ``sql/`` directory.

    Parameters
    ----------
    name : str
        File name without the ``.sql`` suffix, e.g. ``"arm_counts"``.

    Returns
    -------
    str
        The SQL text.
    """
    return resources.files("abkit").joinpath("sql", f"{name}.sql").read_text()


def run_sql(
    con: duckdb.DuckDBPyConnection,
    name: str,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a packaged query against a connection holding the ``customers`` table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    name : str
        Query name (file name in ``sql/`` without suffix).
    params : Mapping, optional
        Named parameters bound to ``$name`` placeholders in the query.

    Returns
    -------
    pandas.DataFrame
        Query result.
    """
    logger.debug("Running sql/%s.sql with params %s", name, params)
    if params is None:
        return con.execute(load_sql(name)).df()
    return con.execute(load_sql(name), dict(params)).df()


def arm_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count the customers in each arm.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    dict
        Arm name to customer count, ordered by arm name.
    """
    df = run_sql(con, "arm_counts")
    return {str(a): int(n) for a, n in zip(df["arm"], df["n"], strict=True)}


# Outcome metrics measured on every customer, in output order.
METRICS: tuple[str, ...] = ("visit", "conversion", "spend")
# Spend among converters: a conditional metric, its n is the number of converters.
CONVERTER_SPEND: str = "spend_converters"
COVARIATE: str = "history"


def arm_sufficient_stats(
    con: duckdb.DuckDBPyConnection, *, winsorize: bool = False
) -> pd.DataFrame:
    """Per-arm sums from ``sql/arm_metrics.sql``, one row per arm.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    winsorize : bool, default False
        If True, spend is capped at the 95th percentile of converters' spend pooled across
        arms (analysis plan §8), computed in SQL.

    Returns
    -------
    pandas.DataFrame
        Columns ``arm``, ``n``, ``{metric}_sum``, ``{metric}_sum_sq`` for visit, conversion and
        spend, ``history_sum``, ``history_sum_sq``, ``history_x_{metric}``, ``n_converters``,
        ``conv_spend_sum``, ``conv_spend_sum_sq`` and ``spend_cap`` (NaN when not winsorised).

    Raises
    ------
    ValueError
        If ``winsorize`` is True and there are no converters to define the cap.
    """
    df = run_sql(con, "arm_metrics", {"winsorize": bool(winsorize)})
    df["spend_cap"] = pd.to_numeric(df["spend_cap"], errors="coerce").astype(float)
    if winsorize and df["spend_cap"].isna().any():
        raise ValueError("Cannot winsorise spend: there are no converters to define the cap.")
    int_cols = ["n", "n_converters"]
    df[int_cols] = df[int_cols].astype("int64")
    float_cols = [c for c in df.columns if c not in ("arm", *int_cols)]
    df[float_cols] = df[float_cols].astype(float)
    df["arm"] = df["arm"].astype(str)
    return df


def _mean(total: np.ndarray, n: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, total / n, np.nan)


def _co_moment(
    sum_xy: np.ndarray, sum_x: np.ndarray, sum_y: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """Unbiased (n - 1 denominator) covariance from sums; NaN when n < 2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 1, (sum_xy - sum_x * sum_y / n) / (n - 1), np.nan)


def summarize_sufficient_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Turn per-arm sums into a tidy table of means, variances and covariances.

    For each arm and metric, with n rows, sum S, sum of squares Q and cross-product with the
    covariate P (covariate sum Sx, sum of squares Qx):

    - mean = S / n
    - var = (Q - S² / n) / (n - 1)
    - cov_history = (P - S · Sx / n) / (n - 1)
    - history_var = (Qx - Sx² / n) / (n - 1)

    Parameters
    ----------
    stats : pandas.DataFrame
        Output of :func:`arm_sufficient_stats`.

    Returns
    -------
    pandas.DataFrame
        One row per (arm, metric), metric in ``visit``, ``conversion``, ``spend`` and
        ``spend_converters`` (spend among converters, whose n is the number of converters).
        Columns: ``arm``, ``metric``, ``n``, ``sum``, ``sum_sq``, ``mean``, ``var``,
        ``history_mean``, ``history_var``, ``cov_history`` and ``spend_cap``. The history
        columns describe the arm's customers and are NaN for ``spend_converters``, which CUPED
        does not use.
    """
    n = stats["n"].to_numpy(dtype=float)
    hx_sum = stats[f"{COVARIATE}_sum"].to_numpy(dtype=float)
    hx_sum_sq = stats[f"{COVARIATE}_sum_sq"].to_numpy(dtype=float)
    hx_mean = _mean(hx_sum, n)
    hx_var = _co_moment(hx_sum_sq, hx_sum, hx_sum, n)

    frames = []
    for metric in METRICS:
        s = stats[f"{metric}_sum"].to_numpy(dtype=float)
        q = stats[f"{metric}_sum_sq"].to_numpy(dtype=float)
        p = stats[f"{COVARIATE}_x_{metric}"].to_numpy(dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "arm": stats["arm"].to_numpy(),
                    "metric": metric,
                    "n": stats["n"].to_numpy(),
                    "sum": s,
                    "sum_sq": q,
                    "mean": _mean(s, n),
                    "var": _co_moment(q, s, s, n),
                    "history_mean": hx_mean,
                    "history_var": hx_var,
                    "cov_history": _co_moment(p, hx_sum, s, n),
                }
            )
        )
    n_conv = stats["n_converters"].to_numpy(dtype=float)
    s = stats["conv_spend_sum"].to_numpy(dtype=float)
    q = stats["conv_spend_sum_sq"].to_numpy(dtype=float)
    frames.append(
        pd.DataFrame(
            {
                "arm": stats["arm"].to_numpy(),
                "metric": CONVERTER_SPEND,
                "n": stats["n_converters"].to_numpy(),
                "sum": s,
                "sum_sq": q,
                "mean": _mean(s, n_conv),
                "var": _co_moment(q, s, s, n_conv),
                "history_mean": np.nan,
                "history_var": np.nan,
                "cov_history": np.nan,
            }
        )
    )
    out = pd.concat(frames, ignore_index=True)
    caps = dict(zip(stats["arm"], stats["spend_cap"], strict=True))
    out["spend_cap"] = out["arm"].map(caps).astype(float)
    out["n"] = out["n"].astype("int64")
    order = {m: i for i, m in enumerate((*METRICS, CONVERTER_SPEND))}
    out = out.sort_values(
        ["metric", "arm"], key=lambda c: c.map(order) if c.name == "metric" else c
    )
    return out.reset_index(drop=True)


def arm_metrics(con: duckdb.DuckDBPyConnection, *, winsorize: bool = False) -> pd.DataFrame:
    """Per-arm means, unbiased variances and CUPED covariance terms for every outcome.

    Aggregation happens in SQL (``sql/arm_metrics.sql``), inference inputs are formed in
    Python, the pattern used by production experimentation platforms.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    winsorize : bool, default False
        If True, use spend capped at the 95th percentile of converters' spend pooled across
        arms (analysis plan §8 sensitivity analysis).

    Returns
    -------
    pandas.DataFrame
        Tidy table, one row per (arm, metric); see :func:`summarize_sufficient_stats`.
    """
    stats = arm_sufficient_stats(con, winsorize=winsorize)
    logger.info(
        "Computed arm metrics for %d arms (winsorize=%s, cap=%s)",
        len(stats),
        winsorize,
        stats["spend_cap"].iloc[0] if len(stats) else None,
    )
    return summarize_sufficient_stats(stats)
