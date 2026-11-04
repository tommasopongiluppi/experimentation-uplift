"""Minimum detectable effect, sample size and power for pre-specified effect sizes.

Everything here is a design calculation: it takes a baseline mean and variance, a sample size
and target error rates, and says what the experiment could detect. The observed treatment
effect never enters (see docs/methodology.md on why "observed power" is meaningless).

The calculations are the standard normal-approximation formulas for a two-sample comparison of
means or rates (a two-sided test at level alpha with power 1 - beta), with an optional
variance-reduction factor for CUPED. :func:`power_for_effect` evaluates the exact power of the
two-sided test at any effect, with a normal or a t reference distribution, and is cross-checked
against ``statsmodels.stats.power``.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from abkit import config, plots
from abkit.cuped import cuped, moments_from_stats
from abkit.frequentist import _out
from abkit.metrics import arm_counts, arm_metrics, arm_sufficient_stats
from abkit.report import json_records, update_results, write_table

logger = logging.getLogger(__name__)

POWER_TARGET: float = 0.8
RELATIVE_LIFTS: tuple[float, ...] = (0.05, 0.10, 0.20)
POWER_METRICS: tuple[str, ...] = ("spend", "conversion", "visit")
# Primary family (plan §3, §6): each email vs control, Holm-corrected.
N_PRIMARY_TESTS: int = len(config.TREATMENTS)
CURVE_POINTS: int = 241
TABLE_NAME = "power"
SAMPLE_SIZE_TABLE_NAME = "power_sample_size"
CURVE_TABLE_NAME = "power_curve"
FIGURE_NAME = "power_curve.png"


@dataclass(frozen=True)
class MDEResult:
    """Minimum detectable effect of a two-sample comparison at fixed sample size.

    Attributes
    ----------
    absolute : float
        Smallest true difference (treatment minus control) detected with the target power.
    relative : float
        ``absolute / baseline``.
    se : float
        Standard error of the difference in means, after variance reduction.
    n_treatment, n_control : float
        Customers per arm.
    alpha, power : float
        Two-sided level and target power.
    variance_reduction : float
        Fraction of the outcome variance removed before the test (0 without CUPED).
    """

    absolute: float
    relative: float
    se: float
    n_treatment: float
    n_control: float
    alpha: float
    power: float
    variance_reduction: float


@dataclass(frozen=True)
class SampleSizeResult:
    """Customers per arm needed to detect a given relative lift.

    Attributes
    ----------
    n_per_arm : int
        Customers per arm, rounded up to a whole customer (equal allocation).
    n_exact : float
        The unrounded solution of the power equation.
    absolute, relative : float
        The effect designed for, in outcome units and as a fraction of the baseline.
    alpha, power, variance_reduction : float
        Design inputs.
    """

    n_per_arm: int
    n_exact: float
    absolute: float
    relative: float
    alpha: float
    power: float
    variance_reduction: float


def _check_design(alpha: float, power: float, variance: float, variance_reduction: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    if not 0 < power < 1:
        raise ValueError("power must lie in (0, 1).")
    if not variance > 0:
        raise ValueError("variance must be positive.")
    if not 0 <= variance_reduction < 1:
        raise ValueError("variance_reduction must lie in [0, 1).")


def _arm_sizes(n_treatment: float, n_control: float | None) -> tuple[float, float]:
    n_c = n_treatment if n_control is None else n_control
    if not (n_treatment > 1 and n_c > 1):
        raise ValueError("Each arm needs more than one customer.")
    return float(n_treatment), float(n_c)


def _z_sum(alpha: float, power: float) -> float:
    """z_{1 - alpha/2} + z_{power}: standard errors between the null and the alternative."""
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def standard_error(
    variance: float,
    n_treatment: float,
    n_control: float | None = None,
    variance_reduction: float = 0.0,
) -> float:
    """Compute the standard error of a difference in means with a common variance.

    Parameters
    ----------
    variance : float
        Outcome variance per customer (``p (1 - p)`` for a rate with level ``p``).
    n_treatment : float
        Customers in the treatment arm.
    n_control : float, optional
        Customers in the control arm; defaults to ``n_treatment``.
    variance_reduction : float
        Fraction of the variance removed by CUPED (about rho²); 0 for the unadjusted test.

    Returns
    -------
    float
        sqrt(variance (1 - variance_reduction) (1 / n_treatment + 1 / n_control)).
    """
    n_t, n_c = _arm_sizes(n_treatment, n_control)
    return math.sqrt(variance * (1 - variance_reduction) * (1 / n_t + 1 / n_c))


def mde(
    baseline: float,
    variance: float,
    n_treatment: float,
    alpha: float = config.ALPHA,
    power: float = POWER_TARGET,
    variance_reduction: float = 0.0,
    n_control: float | None = None,
) -> MDEResult:
    """Minimum detectable effect of a two-sided test of a difference in means or rates.

    The difference in means has standard error se (:func:`standard_error`). The test rejects
    when the estimate exceeds z_{1-alpha/2} se, and a true effect delta is detected with the
    target power when delta - z_{power} se reaches that line, so
    MDE = (z_{1-alpha/2} + z_{power}) se. The chance of rejecting in the wrong direction,
    Phi(-z_{1-alpha/2} - z_{power}), is ignored, as in every textbook version of the formula;
    :func:`power_for_effect` includes it.

    Parameters
    ----------
    baseline : float
        Control mean (or rate), the denominator of the relative effect.
    variance : float
        Outcome variance per customer, assumed common to both arms; ``p (1 - p)`` for a rate.
    n_treatment : float
        Customers in the treatment arm.
    alpha : float
        Two-sided significance level.
    power : float
        Target power, 1 - beta.
    variance_reduction : float
        Fraction of the variance CUPED removes (0 for the unadjusted test).
    n_control : float, optional
        Customers in the control arm; defaults to ``n_treatment``.

    Returns
    -------
    MDEResult
        Absolute and relative minimum detectable effect and the design inputs.

    Raises
    ------
    ValueError
        If a design input is out of range or ``baseline`` is zero.
    """
    _check_design(alpha, power, variance, variance_reduction)
    if baseline == 0:
        raise ValueError("baseline must be non-zero to express the effect relatively.")
    n_t, n_c = _arm_sizes(n_treatment, n_control)
    se = standard_error(variance, n_t, n_c, variance_reduction)
    absolute = _z_sum(alpha, power) * se
    return MDEResult(
        absolute=absolute,
        relative=absolute / baseline,
        se=se,
        n_treatment=n_t,
        n_control=n_c,
        alpha=alpha,
        power=power,
        variance_reduction=variance_reduction,
    )


def sample_size(
    baseline: float,
    variance: float,
    relative_lift: float,
    alpha: float = config.ALPHA,
    power: float = POWER_TARGET,
    variance_reduction: float = 0.0,
) -> SampleSizeResult:
    """Solve for the customers per arm (equal allocation) that detect a relative lift.

    Inverting the MDE formula with n_treatment = n_control = n gives
    n = 2 (z_{1-alpha/2} + z_{power})² variance (1 - variance_reduction) / delta², with
    delta = relative_lift · baseline. The result is rounded up to a whole customer.

    Parameters
    ----------
    baseline : float
        Control mean (or rate).
    variance : float
        Outcome variance per customer; ``p (1 - p)`` for a rate.
    relative_lift : float
        Effect to detect as a fraction of the baseline (0.05 for a 5% lift).
    alpha : float
        Two-sided significance level.
    power : float
        Target power.
    variance_reduction : float
        Fraction of the variance CUPED removes.

    Returns
    -------
    SampleSizeResult
        Customers per arm and the design inputs.

    Raises
    ------
    ValueError
        If a design input is out of range or the lift or baseline is zero.
    """
    _check_design(alpha, power, variance, variance_reduction)
    if baseline == 0 or relative_lift == 0:
        raise ValueError("baseline and relative_lift must be non-zero.")
    absolute = relative_lift * baseline
    n_exact = 2 * _z_sum(alpha, power) ** 2 * variance * (1 - variance_reduction) / absolute**2
    return SampleSizeResult(
        n_per_arm=math.ceil(n_exact),
        n_exact=n_exact,
        absolute=absolute,
        relative=relative_lift,
        alpha=alpha,
        power=power,
        variance_reduction=variance_reduction,
    )


def power_for_effect(
    effect: ArrayLike,
    variance: float,
    n_treatment: float,
    alpha: float = config.ALPHA,
    variance_reduction: float = 0.0,
    n_control: float | None = None,
    use_t: bool = False,
) -> Any:
    """Power of the two-sided test against a given true difference in means.

    With z = z_{1-alpha/2} and se from :func:`standard_error`,
    power = Phi(effect / se - z) + Phi(-effect / se - z): both rejection regions count,
    unlike the closed-form MDE. With ``use_t`` the reference distribution is the t with
    n_treatment + n_control - 2 degrees of freedom and the alternative is the non-central t
    with non-centrality effect / se (the pooled two-sample t-test), which is what
    ``statsmodels.stats.power.TTestIndPower`` computes.

    Parameters
    ----------
    effect : array_like
        True difference(s) in means, treatment minus control.
    variance : float
        Outcome variance per customer.
    n_treatment : float
        Customers in the treatment arm.
    alpha : float
        Two-sided significance level.
    variance_reduction : float
        Fraction of the variance CUPED removes.
    n_control : float, optional
        Customers in the control arm; defaults to ``n_treatment``.
    use_t : bool
        Use the t-test power instead of the normal approximation.

    Returns
    -------
    float or numpy.ndarray
        Power, one value per effect.
    """
    _check_design(alpha, 0.5, variance, variance_reduction)
    n_t, n_c = _arm_sizes(n_treatment, n_control)
    se = standard_error(variance, n_t, n_c, variance_reduction)
    shift = np.asarray(effect, dtype=float) / se
    if use_t:
        df = n_t + n_c - 2
        crit = stats.t.isf(alpha / 2, df)
        pw = stats.nct.sf(crit, df, shift) + stats.nct.cdf(-crit, df, shift)
    else:
        crit = stats.norm.isf(alpha / 2)
        pw = stats.norm.sf(crit - shift) + stats.norm.cdf(-crit - shift)
    return _out(pw)


def holm_alpha(alpha: float = config.ALPHA, n_tests: int = N_PRIMARY_TESTS) -> float:
    """Level a test must meet to be rejected by Holm whatever the other p-values do.

    Holm compares the smallest of m p-values with alpha / m, the next with alpha / (m - 1),
    and so on. Alpha / m is the threshold that guarantees a rejection regardless of the other
    tests, so it is the level to design for; it is also the Bonferroni level.

    Parameters
    ----------
    alpha : float
        Family-wise level.
    n_tests : int
        Tests in the family.

    Returns
    -------
    float
        ``alpha / n_tests``.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1.")
    return alpha / n_tests


# ---------------------------------------------------------------------------------------------
# Readout: the Hillstrom numbers
# ---------------------------------------------------------------------------------------------


def design_inputs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Baseline mean, variance and CUPED variance reduction per metric, from the metric layer.

    The baseline is the control arm (its mean and unbiased variance from
    ``sql/arm_metrics.sql``; for a rate the variance equals p (1 - p) up to the n / (n - 1)
    factor). The variance reduction is the one CUPED on ``history`` achieves on the metric,
    averaged over the two email-vs-control comparisons (its minimum and maximum are kept).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    pandas.DataFrame
        One row per metric in ``POWER_METRICS``: ``metric``, ``baseline``, ``variance``,
        ``n_control``, ``variance_reduction``, ``variance_reduction_min``,
        ``variance_reduction_max``.
    """
    metrics = arm_metrics(con).set_index(["metric", "arm"])
    sums = arm_sufficient_stats(con)
    rows = []
    for metric in POWER_METRICS:
        ctrl = metrics.loc[(metric, config.CONTROL)]
        reductions = [
            cuped(
                moments_from_stats(sums, t, metric),
                moments_from_stats(sums, config.CONTROL, metric),
            ).variance_reduction
            for t in config.TREATMENTS
        ]
        rows.append(
            {
                "metric": metric,
                "baseline": float(ctrl["mean"]),
                "variance": float(ctrl["var"]),
                "n_control": int(ctrl["n"]),
                "variance_reduction": float(np.mean(reductions)),
                "variance_reduction_min": float(np.min(reductions)),
                "variance_reduction_max": float(np.max(reductions)),
            }
        )
    return pd.DataFrame(rows)


def _reduction(inputs_row: pd.Series, with_cuped: bool) -> float:
    return float(inputs_row["variance_reduction"]) if with_cuped else 0.0


def mde_table(
    inputs: pd.DataFrame,
    n_per_arm: int,
    alphas: dict[str, float],
    power: float = POWER_TARGET,
) -> pd.DataFrame:
    """MDE of every metric at the actual sample size, per level, with and without CUPED.

    Parameters
    ----------
    inputs : pandas.DataFrame
        Output of :func:`design_inputs`.
    n_per_arm : int
        Customers per arm (equal allocation).
    alphas : dict
        Label to two-sided level, e.g. ``{"unadjusted": 0.05, "holm": 0.025}``.
    power : float
        Target power.

    Returns
    -------
    pandas.DataFrame
        One row per metric, level and CUPED flag.
    """
    rows = []
    for _, r in inputs.iterrows():
        for label, alpha in alphas.items():
            for with_cuped in (False, True):
                res = mde(
                    r["baseline"], r["variance"], n_per_arm, alpha, power, _reduction(r, with_cuped)
                )
                rows.append(
                    {
                        "metric": r["metric"],
                        "alpha_label": label,
                        "alpha": alpha,
                        "power": power,
                        "cuped": with_cuped,
                        "n_per_arm": n_per_arm,
                        "baseline": r["baseline"],
                        "variance": r["variance"],
                        "variance_reduction": res.variance_reduction,
                        "se": res.se,
                        "mde_absolute": res.absolute,
                        "mde_relative": res.relative,
                    }
                )
    return pd.DataFrame(rows)


def sample_size_table(
    inputs: pd.DataFrame,
    alphas: dict[str, float],
    lifts: tuple[float, ...] = RELATIVE_LIFTS,
    power: float = POWER_TARGET,
) -> pd.DataFrame:
    """Tabulate customers per arm for each lift, metric and level, with and without CUPED.

    Parameters
    ----------
    inputs : pandas.DataFrame
        Output of :func:`design_inputs`.
    alphas : dict
        Label to two-sided level.
    lifts : tuple of float
        Relative lifts to design for.
    power : float
        Target power.

    Returns
    -------
    pandas.DataFrame
        One row per metric, level, CUPED flag and lift.
    """
    rows = []
    for _, r in inputs.iterrows():
        for label, alpha in alphas.items():
            for with_cuped in (False, True):
                for lift in lifts:
                    res = sample_size(
                        r["baseline"], r["variance"], lift, alpha, power, _reduction(r, with_cuped)
                    )
                    rows.append(
                        {
                            "metric": r["metric"],
                            "alpha_label": label,
                            "alpha": alpha,
                            "power": power,
                            "cuped": with_cuped,
                            "relative_lift": lift,
                            "absolute_lift": res.absolute,
                            "variance_reduction": res.variance_reduction,
                            "n_per_arm": res.n_per_arm,
                            "n_total_two_arms": 2 * res.n_per_arm,
                            "n_exact": res.n_exact,
                        }
                    )
    return pd.DataFrame(rows)


def power_curve_table(
    inputs: pd.DataFrame,
    n_per_arm: int,
    alphas: dict[str, float],
    mdes: pd.DataFrame,
    n_points: int = CURVE_POINTS,
) -> pd.DataFrame:
    """Power against relative lift at the actual sample size, the table behind the figure.

    Each metric gets its own lift grid, from zero to 1.6 times its largest MDE rounded up to
    a multiple of 5%, so every panel shows the curves crossing the target.

    Parameters
    ----------
    inputs : pandas.DataFrame
        Output of :func:`design_inputs`.
    n_per_arm : int
        Customers per arm.
    alphas : dict
        Label to two-sided level.
    mdes : pandas.DataFrame
        Output of :func:`mde_table`, used to size the grid.
    n_points : int
        Grid points per curve.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``alpha_label``, ``alpha``, ``cuped``, ``relative_lift``,
        ``absolute_lift``, ``power``.
    """
    frames = []
    for _, r in inputs.iterrows():
        largest = mdes.loc[mdes["metric"] == r["metric"], "mde_relative"].max()
        x_max = 0.05 * math.ceil(1.6 * largest / 0.05)
        grid = np.linspace(0.0, x_max, n_points)
        for label, alpha in alphas.items():
            for with_cuped in (False, True):
                pw = power_for_effect(
                    grid * r["baseline"], r["variance"], n_per_arm, alpha, _reduction(r, with_cuped)
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "metric": r["metric"],
                            "alpha_label": label,
                            "alpha": alpha,
                            "cuped": with_cuped,
                            "relative_lift": grid,
                            "absolute_lift": grid * r["baseline"],
                            "power": pw,
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def build_power_tables(
    con: duckdb.DuckDBPyConnection,
    alpha: float = config.ALPHA,
    power: float = POWER_TARGET,
    lifts: tuple[float, ...] = RELATIVE_LIFTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the MDE, sample-size and power-curve tables for the experiment.

    The sample size per arm is the smallest arm (equal allocation; the three arms differ by
    under half a percent). Levels: ``alpha`` as is and the Holm level for the primary family
    (:func:`holm_alpha`).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    alpha : float
        Two-sided family-wise level.
    power : float
        Target power.
    lifts : tuple of float
        Relative lifts for the sample-size table.

    Returns
    -------
    tuple
        (MDE table, sample-size table, power-curve table, design inputs as a dict).
    """
    counts = arm_counts(con)
    n_per_arm = min(counts.values())
    alphas = {"unadjusted": alpha, "holm": holm_alpha(alpha, N_PRIMARY_TESTS)}
    inputs = design_inputs(con)
    mdes = mde_table(inputs, n_per_arm, alphas, power)
    sizes = sample_size_table(inputs, alphas, lifts, power)
    curve = power_curve_table(inputs, n_per_arm, alphas, mdes)
    design = {
        "n_per_arm": n_per_arm,
        "arm_counts": counts,
        "power": power,
        "alpha": alpha,
        "alpha_holm": alphas["holm"],
        "n_primary_tests": N_PRIMARY_TESTS,
        "relative_lifts": list(lifts),
        "baseline_arm": config.CONTROL,
        "cuped_covariate": "history",
        "metrics": {
            str(r["metric"]): {k: v for k, v in r.items() if k != "metric"}
            for r in json_records(inputs)
        },
    }
    return mdes, sizes, curve, design


def run_power(
    db_path: Path = config.DUCKDB_PATH, reports_dir: Path = config.REPORTS
) -> pd.DataFrame:
    """Run the power calculations and write the tables, figure and results.json section.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.

    Returns
    -------
    pandas.DataFrame
        The MDE table.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        mdes, sizes, curve, design = build_power_tables(con)

    tables = reports_dir / "tables"
    write_table(mdes, TABLE_NAME, tables)
    write_table(sizes, SAMPLE_SIZE_TABLE_NAME, tables)
    write_table(curve, CURVE_TABLE_NAME, tables)
    plots.plot_power_curve(curve, mdes, reports_dir / "figures" / FIGURE_NAME)
    update_results(
        "power",
        {
            **design,
            "table": f"tables/{TABLE_NAME}.csv",
            "sample_size_table": f"tables/{SAMPLE_SIZE_TABLE_NAME}.csv",
            "curve_table": f"tables/{CURVE_TABLE_NAME}.csv",
            "figure": f"figures/{FIGURE_NAME}",
            "mde_rows": json_records(mdes),
            "sample_size_rows": json_records(sizes),
        },
        reports_dir / "results.json",
    )
    logger.info("Power tables written: %d MDE rows, %d sample-size rows.", len(mdes), len(sizes))
    return mdes
