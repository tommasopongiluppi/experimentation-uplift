"""Frequentist readout: every pre-registered comparison and metric (analysis plan §3, §4, §6).

Inference runs on the per-arm sufficient statistics from the SQL metric layer
(:mod:`abkit.metrics`); row-level spend is read only for the percentile bootstrap.
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from abkit import config, plots
from abkit.eda import SRMError
from abkit.frequentist import (
    benjamini_hochberg,
    bootstrap_diff_in_means,
    diff_in_means,
    diff_in_proportions,
    holm,
    srm_test,
)
from abkit.metrics import arm_counts, arm_metrics, run_sql
from abkit.report import json_records, update_results, write_table

logger = logging.getLogger(__name__)

# (treatment, control): the difference is always the first arm minus the second (plan §3).
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mens", config.CONTROL),
    ("womens", config.CONTROL),
    ("mens", "womens"),
)
SPEND_WINSORISED = "spend_winsorised"
READOUT_METRICS: tuple[str, ...] = ("spend", SPEND_WINSORISED, "conversion", "visit")
RATE_METRICS: tuple[str, ...] = ("conversion", "visit")
SPEND_METRICS: tuple[str, ...] = ("spend", SPEND_WINSORISED)
# Family -> adjustment method (plan §6; sensitivity families per Deviation D1).
FAMILY_METHODS: dict[str, str] = {
    "primary": "holm",
    "secondary": "bh",
    "sensitivity": "holm",
    "sensitivity_secondary": "none",
}
N_RESAMPLES: int = 10_000
TABLE_NAME = "readout_frequentist"
FIGURE_NAME = "forest_relative_lift.png"


def family_of(metric: str, treatment: str, control: str) -> str:
    """Multiple-testing family of one comparison (analysis plan §6, Deviation D1).

    Parameters
    ----------
    metric : str
        One of ``READOUT_METRICS``.
    treatment, control : str
        Arms compared.

    Returns
    -------
    str
        A key of ``FAMILY_METHODS``.
    """
    vs_control = control == config.CONTROL
    if metric == "spend":
        return "primary" if vs_control else "secondary"
    if metric == SPEND_WINSORISED:
        return "sensitivity" if vs_control else "sensitivity_secondary"
    if metric in RATE_METRICS:
        return "secondary"
    raise ValueError(f"Unknown metric {metric!r}.")


def adjust_pvalues(p_values: pd.Series, families: pd.Series) -> pd.Series:
    """Adjust p-values within each family with that family's method.

    Parameters
    ----------
    p_values : pandas.Series
        Unadjusted p-values.
    families : pandas.Series
        Family label per p-value (same index).

    Returns
    -------
    pandas.Series
        Adjusted p-values aligned with the input.
    """
    adjusted = pd.Series(np.nan, index=p_values.index)
    for family, idx in families.groupby(families).groups.items():
        method = FAMILY_METHODS[str(family)]
        p = p_values.loc[idx].to_numpy()
        if method == "holm":
            adjusted.loc[idx] = holm(p)
        elif method == "bh":
            adjusted.loc[idx] = benjamini_hochberg(p)
        else:
            adjusted.loc[idx] = p
    return adjusted


def _stats_lookup(con: duckdb.DuckDBPyConnection) -> tuple[dict[str, pd.DataFrame], float]:
    raw = arm_metrics(con).set_index(["metric", "arm"])
    win = arm_metrics(con, winsorize=True).set_index(["metric", "arm"])
    cap = float(win["spend_cap"].iloc[0])
    by_metric = {m: raw.loc[m] for m in ("spend", *RATE_METRICS)}
    by_metric[SPEND_WINSORISED] = win.loc["spend"]
    return by_metric, cap


def build_readout(
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    n_resamples: int = N_RESAMPLES,
    alpha: float = config.ALPHA,
) -> tuple[pd.DataFrame, float]:
    """Build the frequentist readout table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    rng : numpy.random.Generator
        Randomness for the bootstrap, consumed in row order (spend rows, then winsorised).
    n_resamples : int
        Bootstrap resamples per spend row.
    alpha : float
        Two-sided level for intervals and the ``significant`` flag.

    Returns
    -------
    tuple
        (readout table with one row per comparison and metric, winsorisation cap).
    """
    by_metric, cap = _stats_lookup(con)
    spend_rows = run_sql(con, "arm_spend")
    spend_by_arm = {
        arm: g["spend"].to_numpy(dtype=float) for arm, g in spend_rows.groupby("arm", sort=True)
    }

    rows = []
    for metric in READOUT_METRICS:
        st = by_metric[metric]
        for treatment, control in COMPARISONS:
            t, c = st.loc[treatment], st.loc[control]
            if metric in RATE_METRICS:
                res = diff_in_proportions(round(t["sum"]), t["n"], round(c["sum"]), c["n"], alpha)
                test, ci_method = "two-proportion z-test (pooled)", "Newcombe hybrid score"
            else:
                res = diff_in_means(t["mean"], t["var"], t["n"], c["mean"], c["var"], c["n"], alpha)
                test, ci_method = "Welch t-test", "Welch t"
            row = {
                "comparison": f"{treatment}_vs_{control}",
                "treatment": treatment,
                "control": control,
                "metric": metric,
                "family": family_of(metric, treatment, control),
                "n_treatment": int(t["n"]),
                "n_control": int(c["n"]),
                "mean_treatment": res.mean_t,
                "mean_control": res.mean_c,
                "diff": res.diff,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "ci_method": ci_method,
                "rel_lift": res.rel_lift,
                "rel_ci_low": res.rel_ci_low,
                "rel_ci_high": res.rel_ci_high,
                "test": test,
                "statistic": res.statistic,
                "dof": res.dof,
                "p_value": res.p_value,
            }
            if metric in SPEND_METRICS:
                yt, yc = spend_by_arm[treatment], spend_by_arm[control]
                if metric == SPEND_WINSORISED:
                    yt, yc = np.minimum(yt, cap), np.minimum(yc, cap)
                boot = bootstrap_diff_in_means(yt, yc, rng, n_resamples, alpha)
                row |= {
                    "boot_ci_low": boot.ci_low,
                    "boot_ci_high": boot.ci_high,
                    "boot_se": boot.se,
                    "boot_mcse_low": boot.mcse_low,
                    "boot_mcse_high": boot.mcse_high,
                }
            rows.append(row)

    table = pd.DataFrame(rows)
    table.insert(
        table.columns.get_loc("family") + 1,
        "adjustment",
        table["family"].map(FAMILY_METHODS),
    )
    table["p_adj"] = adjust_pvalues(table["p_value"], table["family"])
    table["significant"] = table["p_adj"] < alpha
    boot_cols = [c for c in table.columns if c.startswith("boot_")]
    order = [c for c in table.columns if c not in boot_cols] + boot_cols
    return table[order], cap


def run_frequentist(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Run the frequentist readout and write the table, results.json section and forest plot.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    n_resamples : int
        Bootstrap resamples per spend row.

    Returns
    -------
    pandas.DataFrame
        The readout table.

    Raises
    ------
    SRMError
        If the sample ratio mismatch check fails (plan §5 blocks the readout).
    """
    rng = np.random.default_rng(config.SEEDS["bootstrap"])
    with duckdb.connect(str(db_path), read_only=True) as con:
        srm = srm_test(arm_counts(con))
        if not srm.passed:
            raise SRMError(f"SRM check failed (p = {srm.p_value:.2e}); readout blocked.")
        table, cap = build_readout(con, rng, n_resamples)

    write_table(table, TABLE_NAME, reports_dir / "tables")
    plots.plot_forest(table, reports_dir / "figures" / FIGURE_NAME)
    update_results(
        "frequentist",
        {
            "alpha": config.ALPHA,
            "families": {
                f: {"adjustment": m, "n_tests": int((table["family"] == f).sum())}
                for f, m in FAMILY_METHODS.items()
            },
            "spend_winsorisation_cap": cap,
            "bootstrap": {"n_resamples": n_resamples, "seed": config.SEEDS["bootstrap"]},
            "table": f"tables/{TABLE_NAME}.csv",
            "figure": f"figures/{FIGURE_NAME}",
            "rows": json_records(table),
        },
        reports_dir / "results.json",
    )
    logger.info("Frequentist readout written: %d rows.", len(table))
    return table
