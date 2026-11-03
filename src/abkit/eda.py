"""Validity checks (SRM, covariate balance) and pooled exploratory analysis.

Validity checks compare arms on assignment counts and pre-treatment variables only. The EDA
pools all customers together and never splits an outcome by arm. Outputs: CSV tables in
reports/tables, figures in reports/figures, reports/eda.md generated from the tables, and the
``validity`` and ``eda`` sections of reports/results.json.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from abkit import config, plots
from abkit.balance import balance_table, categorical_balance_tests
from abkit.frequentist import SRMResult, srm_test
from abkit.metrics import arm_counts, run_sql
from abkit.report import read_table, update_results, write_table

logger = logging.getLogger(__name__)

N_HIST_BINS = 30
EDA_MD = "eda.md"


class SRMError(RuntimeError):
    """Raised when the sample ratio mismatch check fails (analysis plan §5 blocks the readout)."""


def srm_table(result: SRMResult) -> pd.DataFrame:
    """Per-arm observed and expected counts with the test result repeated on each row."""
    return pd.DataFrame(
        {
            "arm": list(result.counts),
            "observed": list(result.counts.values()),
            "expected": list(result.expected.values()),
            "statistic": result.statistic,
            "dof": result.dof,
            "p_value": result.p_value,
            "alpha": result.alpha,
            "passed": result.passed,
        }
    )


def log_histogram(values: np.ndarray, n_bins: int = N_HIST_BINS) -> pd.DataFrame:
    """Count values in log-spaced bins spanning their range.

    Parameters
    ----------
    values : numpy.ndarray
        Positive values.
    n_bins : int
        Number of bins.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_left``, ``bin_right``, ``n``; the first bin includes the minimum and the
        last includes the maximum.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or (values <= 0).any():
        raise ValueError("log_histogram needs a non-empty array of positive values.")
    lo, hi = values.min(), values.max()
    if lo == hi:
        lo, hi = lo / 1.01, hi * 1.01
    edges = np.geomspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "n": counts})


def compute_eda(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Pooled EDA tables (all arms together).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    dict of str to pandas.DataFrame
        Tables ``eda_rates``, ``eda_distributions``, ``eda_recency``, ``eda_correlations``,
        ``eda_spend_converters_hist`` and ``eda_history_hist``.
    """
    values = run_sql(con, "eda_values")
    converter_spend = values["converter_spend"].dropna().to_numpy()
    return {
        "eda_rates": run_sql(con, "eda_rates"),
        "eda_distributions": run_sql(con, "eda_distributions"),
        "eda_recency": run_sql(con, "eda_recency"),
        "eda_correlations": run_sql(con, "eda_correlations"),
        "eda_spend_converters_hist": log_histogram(converter_spend),
        "eda_history_hist": log_histogram(values["history"].to_numpy()),
    }


def _fmt(x: Any, digits: int = 2) -> str:
    if isinstance(x, bool | np.bool_):
        return "pass" if x else "**FAIL**"
    if isinstance(x, int | np.integer):
        return f"{x:,}"
    if isinstance(x, float | np.floating):
        if x != 0 and abs(x) < 10 ** -(digits + 1):
            return f"{x:.2e}"
        return f"{x:,.{digits}f}"
    return str(x)


def _md_table(df: pd.DataFrame, columns: dict[str, str], digits: int = 2) -> list[str]:
    header = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_fmt(row[c], digits) for c in columns) + " |"
        for row in df.to_dict(orient="records")
    ]
    return [header, rule, *body, ""]


def render_eda_markdown(tables_dir: Path = config.TABLES, figures_rel: str = "figures") -> str:
    """Render reports/eda.md from the CSV tables.

    Parameters
    ----------
    tables_dir : Path
        Directory holding the tables written by :func:`run_checks`.
    figures_rel : str
        Figure directory relative to the Markdown file.

    Returns
    -------
    str
        Markdown document.
    """
    srm = read_table("srm", tables_dir)
    balance = read_table("balance", tables_dir)
    cat_tests = read_table("balance_chi2", tables_dir)
    rates = read_table("eda_rates", tables_dir)
    dist = read_table("eda_distributions", tables_dir)
    corr = read_table("eda_correlations", tables_dir)
    recency = read_table("eda_recency", tables_dir)

    s = srm.iloc[0]
    worst = balance.loc[balance["max_abs_smd"].idxmax()]
    worst_name = f"{worst['variable']}: {worst['level']}" if worst["level"] else worst["variable"]
    n_flagged = int(balance["flagged"].sum())
    r = rates.iloc[0]

    lines = [
        "# Validity Checks and Pooled EDA",
        "",
        "Generated by `abkit.eda` from the CSV files in `reports/tables/`; do not edit by hand.",
        "Validity checks compare arms on assignment counts and pre-treatment variables only; the",
        "EDA pools all customers and never splits an outcome by arm.",
        "",
        "## 1. Sample ratio mismatch",
        "",
        f"Chi-square test against equal thirds: statistic {_fmt(s['statistic'], 3)} on "
        f"{int(s['dof'])} degrees of freedom, p = {_fmt(s['p_value'], 3)}. The check fails if "
        f"p < {s['alpha']} (analysis plan §5): **{'PASSED' if s['passed'] else 'FAILED'}**.",
        "",
        *_md_table(srm, {"arm": "Arm", "observed": "Customers", "expected": "Expected"}, 1),
        f"![Customers per arm]({figures_rel}/srm_arm_counts.png)",
        "",
        "## 2. Covariate balance",
        "",
        f"Largest |SMD| across all variables, levels and comparisons: "
        f"{_fmt(worst['max_abs_smd'], 4)} ({worst_name}). "
        f"Rows with |SMD| > {config.SMD_THRESHOLD}: {n_flagged}.",
        "",
        f"![Covariate balance]({figures_rel}/balance_smd.png)",
        "",
        *_md_table(
            balance.assign(balanced=~balance["flagged"].astype(bool)),
            {
                "variable": "Variable",
                "level": "Level",
                "mean_mens": "Mean mens",
                "mean_womens": "Mean womens",
                "mean_control": "Mean control",
                "smd_mens_vs_control": "SMD mens vs control",
                "smd_womens_vs_control": "SMD womens vs control",
                "smd_mens_vs_womens": "SMD mens vs womens",
                "balanced": f"Within ±{config.SMD_THRESHOLD}",
            },
            4,
        ),
        "Chi-square tests of independence between arm and each categorical variable:",
        "",
        *_md_table(
            cat_tests,
            {
                "variable": "Variable",
                "n_levels": "Levels",
                "statistic": "Chi-square",
                "dof": "df",
                "p_value": "p-value",
            },
            3,
        ),
        "## 3. Pooled outcomes (all arms together)",
        "",
        f"- Customers: {_fmt(int(r['n_customers']))}",
        f"- Visit rate: {r['visit_rate']:.2%} ({_fmt(int(r['n_visitors']))} visitors)",
        f"- Conversion rate: {r['conversion_rate']:.2%} ({_fmt(int(r['n_converters']))} "
        "converters)",
        f"- Share with zero spend: {r['share_zero_spend']:.2%}",
        f"- Mean spend per customer: ${r['mean_spend']:.3f}",
        "",
        "## 4. Distributions",
        "",
        *_md_table(
            dist,
            {
                "variable": "Variable",
                "population": "Population",
                "n": "n",
                "mean": "Mean",
                "sd": "SD",
                "min": "Min",
                "p50": "p50",
                "p90": "p90",
                "p99": "p99",
                "p99_9": "p99.9",
                "max": "Max",
            },
        ),
        f"![Spend among converters]({figures_rel}/eda_spend_converters_log.png)",
        "",
        f"![History]({figures_rel}/eda_history_log.png)",
        "",
        f"![Recency]({figures_rel}/eda_recency.png)",
        "",
        *_md_table(recency, {"recency": "Recency (months)", "n": "Customers", "share": "Share"}, 4),
        "## 5. Correlation of history with outcomes (input to CUPED)",
        "",
        "CUPED with `history` multiplies the variance of an outcome by about 1 - r², "
        "where r is the Pearson correlation below (analysis plan §7).",
        "",
    ]
    corr = corr.assign(r_squared=corr["pearson_r"] ** 2)
    lines += _md_table(
        corr, {"outcome": "Outcome", "n": "n", "pearson_r": "Pearson r", "r_squared": "r²"}, 4
    )
    return "\n".join(lines)


def run_checks(db_path: Path = config.DUCKDB_PATH, reports_dir: Path = config.REPORTS) -> SRMResult:
    """Run the validity checks and pooled EDA and write every output.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory (tables, figures, eda.md and results.json go under it).

    Returns
    -------
    SRMResult
        The sample ratio mismatch result.

    Raises
    ------
    SRMError
        If the SRM check fails; all outputs are written first.
    """
    tables_dir, figures_dir = reports_dir / "tables", reports_dir / "figures"
    with duckdb.connect(str(db_path), read_only=True) as con:
        srm = srm_test(arm_counts(con))
        balance = balance_table(con)
        cat_tests = categorical_balance_tests(con)
        eda = compute_eda(con)

    write_table(srm_table(srm), "srm", tables_dir)
    write_table(balance, "balance", tables_dir)
    write_table(cat_tests, "balance_chi2", tables_dir)
    for name, table in eda.items():
        write_table(table, name, tables_dir)

    plots.plot_arm_counts(srm_table(srm), figures_dir / "srm_arm_counts.png")
    plots.plot_balance(balance, figures_dir / "balance_smd.png")
    plots.plot_log_histogram(
        eda["eda_spend_converters_hist"],
        "Spend among converters (all arms pooled)",
        "Spend in the two-week window (log scale)",
        figures_dir / "eda_spend_converters_log.png",
        note="Customers who did not convert (zero spend) are excluded.",
    )
    plots.plot_log_histogram(
        eda["eda_history_hist"],
        "Past-year spend (history), all customers",
        "History (log scale)",
        figures_dir / "eda_history_log.png",
    )
    plots.plot_recency(eda["eda_recency"], figures_dir / "eda_recency.png")
    (reports_dir / EDA_MD).write_text(render_eda_markdown(tables_dir))

    worst = balance.loc[balance["max_abs_smd"].idxmax()]
    update_results(
        "validity",
        {
            "srm": asdict(srm),
            "balance": {
                "threshold": config.SMD_THRESHOLD,
                "max_abs_smd": float(worst["max_abs_smd"]),
                "max_abs_smd_variable": str(worst["variable"]),
                "max_abs_smd_level": str(worst["level"]),
                "n_flagged": int(balance["flagged"].sum()),
                "categorical_chi2": cat_tests.to_dict(orient="records"),
            },
        },
        reports_dir / config.RESULTS_JSON.name,
    )
    rates = eda["eda_rates"].iloc[0]
    update_results(
        "eda",
        {
            "pooled_rates": {k: float(v) for k, v in rates.items()},
            "history_correlations": {
                str(o): float(r)
                for o, r in zip(
                    eda["eda_correlations"]["outcome"],
                    eda["eda_correlations"]["pearson_r"],
                    strict=True,
                )
            },
        },
        reports_dir / config.RESULTS_JSON.name,
    )

    if not srm.passed:
        raise SRMError(f"Sample ratio mismatch: p = {srm.p_value:.3g} < {srm.alpha}.")
    logger.info("SRM passed (p = %.3g); max |SMD| = %.4f.", srm.p_value, worst["max_abs_smd"])
    return srm
