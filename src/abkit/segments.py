"""Pre-specified customer-segment (subgroup) analysis: exploratory (analysis plan §10).

"Segment" here always means a customer subgroup defined by a pre-treatment variable, never
the treatment arm. For every segment variable and level and each email vs control, the
module reports the effect on spend, visit and conversion with its interval; for every
segment variable it tests whether the effect differs across levels (OLS with arm x level
interactions and HC2 robust standard errors, one joint Wald test); and it adjusts all of one
outcome's segment tests together with Benjamini-Hochberg. Every result is labelled
exploratory: segments generate hypotheses for a future test, they are not decisions.

The module also runs a short simulation for the README: in an A/A setting, testing many
independent segments at level alpha almost always finds a "significant" one, and
Benjamini-Hochberg keeps the false discovery rate at the chosen level.
"""

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from abkit import config, plots
from abkit.frequentist import (
    ComparisonResult,
    benjamini_hochberg,
    diff_in_means,
    diff_in_proportions,
)
from abkit.metrics import arm_counts, arm_metrics, run_sql
from abkit.report import json_records, update_results, write_table

logger = logging.getLogger(__name__)

# Segment variables and their levels, exactly as pre-specified in analysis plan §10.
RECENCY_BANDS: tuple[tuple[int, int, str], ...] = (
    (1, 3, "1_3"),
    (4, 6, "4_6"),
    (7, 9, "7_9"),
    (10, 12, "10_12"),
)
SEGMENT_VARIABLES: dict[str, tuple[str, ...]] = {
    "newbie": ("0", "1"),
    "channel": tuple(config.CATEGORY_MAPS["channel"].values()),
    "zip_code": tuple(config.CATEGORY_MAPS["zip_code"].values()),
    "mens": ("0", "1"),
    "womens": ("0", "1"),
    "recency": tuple(label for _, _, label in RECENCY_BANDS),
    "history_segment": tuple(config.CATEGORY_MAPS["history_segment"].values()),
}
# Plan §10 outcomes are spend and visit; conversion is added as Deviation D5.
SEGMENT_OUTCOMES: tuple[str, ...] = ("spend", "visit", "conversion")
RATE_OUTCOMES: tuple[str, ...] = ("visit", "conversion")
COMPARISONS: tuple[tuple[str, str], ...] = tuple((t, config.CONTROL) for t in config.TREATMENTS)
LABEL: str = "exploratory"
ADJUSTMENT: str = "bh"
FDR_Q: float = config.ALPHA
# The challenge's question: did each email work better among past buyers of its merchandise?
CHALLENGE_PAIRS: tuple[tuple[str, str], ...] = (("mens", "mens"), ("womens", "womens"))
# README simulation: 20 independent segments, A/A, and a second scenario with real effects.
N_SIM_SEGMENTS: int = 20
N_SIM_TRUE: int = 5
N_SIM_RUNS: int = 10_000
SIM_POWER: float = 0.8
TABLE_NAME = "segments"
INTERACTIONS_TABLE_NAME = "segments_interactions"
FDR_TABLE_NAME = "segments_fdr_simulation"
FIGURE_NAME = "forest_segments.png"


# ---------------------------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------------------------


def recency_band(recency: ArrayLike) -> np.ndarray:
    """Map months since last purchase to the plan's four recency bands.

    Parameters
    ----------
    recency : array_like
        Integer months in ``config.RECENCY_RANGE``.

    Returns
    -------
    numpy.ndarray
        Band labels (``"1_3"``, ``"4_6"``, ``"7_9"``, ``"10_12"``).

    Raises
    ------
    ValueError
        If a value lies outside every band.
    """
    r = np.asarray(recency)
    conditions = [(r >= lo) & (r <= hi) for lo, hi, _ in RECENCY_BANDS]
    out = np.select(conditions, [label for _, _, label in RECENCY_BANDS], default="")
    if (out == "").any():
        raise ValueError("recency outside the plan's bands (1 to 12 months).")
    return out


def segment_column(df: pd.DataFrame, variable: str) -> pd.Series:
    """Level label of one segment variable for every customer.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned customers (pre-treatment columns).
    variable : str
        A key of ``SEGMENT_VARIABLES``.

    Returns
    -------
    pandas.Series
        String labels matching ``SEGMENT_VARIABLES[variable]``, same index as ``df``.
    """
    if variable not in SEGMENT_VARIABLES:
        raise ValueError(f"Not a pre-specified segment variable: {variable!r}.")
    if variable == "recency":
        values = pd.Series(recency_band(df["recency"]), index=df.index)
    elif variable in ("newbie", "mens", "womens"):
        values = df[variable].astype(int).astype(str)
    else:
        values = df[variable].astype(str)
    unknown = set(values.unique()) - set(SEGMENT_VARIABLES[variable])
    if unknown:
        raise ValueError(f"Unknown level(s) of {variable!r}: {sorted(unknown)}.")
    return values


# ---------------------------------------------------------------------------------------------
# Interaction test: OLS with arm x level interactions, HC2, joint Wald test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionResult:
    """Joint Wald test that the treatment effect is the same in every level of a segment.

    Attributes
    ----------
    statistic : float
        Wald chi-square statistic of the interaction coefficients with HC2 covariance.
    dof : int
        Number of interaction terms (levels present minus one).
    p_value : float
        Upper-tail chi-square p-value.
    n : int
        Customers in the regression.
    n_levels : int
        Levels present in the two arms.
    effects : dict
        Level -> (effect, HC2 standard error) of the treatment within that level, from the
        same regression (the reference level's effect is the treatment coefficient).
    """

    statistic: float
    dof: int
    p_value: float
    n: int
    n_levels: int
    effects: dict[str, tuple[float, float]]


def ols_hc2(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their HC2 heteroskedasticity-robust covariance.

    beta = (X'X)^-1 X'y and V = (X'X)^-1 X' diag(e_i² / (1 - h_ii)) X (X'X)^-1 with e the
    residuals and h the leverages (MacKinnon and White 1985), as ``statsmodels`` ``cov_type="HC2"``.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix, full column rank.
    y : numpy.ndarray
        Outcome.

    Returns
    -------
    tuple of numpy.ndarray
        ``beta`` and the covariance matrix ``V``.

    Raises
    ------
    ValueError
        If the design is rank deficient.
    """
    x = np.asarray(design, dtype=float)
    yv = np.asarray(y, dtype=float)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Design matrix is rank deficient.")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ yv)
    resid = yv - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    meat = (x * (resid**2 / (1 - leverage))[:, None]).T @ x
    return beta, xtx_inv @ meat @ xtx_inv


def interaction_test(y: ArrayLike, treated: ArrayLike, segment: ArrayLike) -> InteractionResult:
    """Test whether a treatment effect differs across the levels of a segment variable.

    OLS of Y on an intercept, the treatment indicator D, one indicator per level except the
    first (reference), and D times each indicator, over the customers of the two arms. The
    coefficient on D is the effect in the reference level and each interaction coefficient is
    how much the effect in that level differs from it. The joint test of the interaction
    coefficients being all zero is a Wald chi-square test with HC2 robust covariance, with as
    many degrees of freedom as interaction terms. For a binary outcome this is a linear
    probability model, which is fine for differences in rates.

    Parameters
    ----------
    y : array_like
        Outcome per customer.
    treated : array_like
        1 for the email arm, 0 for control.
    segment : array_like
        Level label per customer.

    Returns
    -------
    InteractionResult
        Statistic, degrees of freedom, p-value and the within-level effects.

    Raises
    ------
    ValueError
        If fewer than two levels are present, or a level is missing from one arm.
    """
    yv = np.asarray(y, dtype=float)
    d = np.asarray(treated, dtype=float)
    seg = pd.Categorical(np.asarray(segment).astype(str))
    if not (yv.shape[0] == d.shape[0] == len(seg)):
        raise ValueError("y, treated and segment must have the same length.")
    if not np.isin(d, (0.0, 1.0)).all() or d.min() == d.max():
        raise ValueError("treated must be 0/1 with both arms present.")
    levels = list(seg.categories)
    if len(levels) < 2:
        raise ValueError("The segment variable needs at least two levels present.")
    codes = np.asarray(seg.codes)
    dummies = (codes[:, None] == np.arange(1, len(levels))[None, :]).astype(float)
    design = np.column_stack([np.ones_like(d), d, dummies, d[:, None] * dummies])
    try:
        beta, cov = ols_hc2(design, yv)
    except ValueError as err:
        raise ValueError("A segment level is missing from one of the arms.") from err
    k = len(levels) - 1
    idx = np.arange(1 + k + 1, 2 + 2 * k)
    b = beta[idx]
    statistic = float(b @ np.linalg.solve(cov[np.ix_(idx, idx)], b))
    effects = {levels[0]: (float(beta[1]), float(math.sqrt(cov[1, 1])))}
    for j, level in enumerate(levels[1:]):
        pick = np.zeros(beta.size)
        pick[1] = 1.0
        pick[idx[j]] = 1.0
        effects[level] = (float(pick @ beta), float(math.sqrt(pick @ cov @ pick)))
    return InteractionResult(
        statistic=statistic,
        dof=k,
        p_value=float(stats.chi2.sf(statistic, k)),
        n=int(yv.size),
        n_levels=len(levels),
        effects=effects,
    )


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def _constant_outcome(res: ComparisonResult) -> ComparisonResult:
    """Welch result for two constant arms, where the test is 0 / 0: p = 1, zero-width interval."""
    return replace(
        res,
        se=0.0,
        ci_low=res.diff,
        ci_high=res.diff,
        statistic=0.0,
        dof=math.nan,
        p_value=1.0,
        rel_ci_low=res.rel_lift,
        rel_ci_high=res.rel_lift,
    )


def segment_effects(seg_stats: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """Per-level effects of each email vs control from the segment sufficient statistics.

    Spend: Welch difference in means with its interval; visit and conversion: difference in
    proportions with the pooled z-test and the Newcombe interval, as in the readout (plan §6).
    A level whose spend is constant in both arms (possible only in a tiny sample) gets p = 1
    and a zero-width interval, as the pooled z-test already does for two all-zero rates.

    Parameters
    ----------
    seg_stats : pandas.DataFrame
        Output of ``sql/segment_stats.sql`` (one row per variable, level and arm).
    alpha : float
        Two-sided level of the intervals.

    Returns
    -------
    pandas.DataFrame
        One row per outcome, segment variable, level and comparison, in the plan's order.
    """
    st = seg_stats.set_index(["variable", "level", "arm"])
    rows = []
    for outcome in SEGMENT_OUTCOMES:
        for variable, levels in SEGMENT_VARIABLES.items():
            for level in levels:
                for treatment, control in COMPARISONS:
                    t, c = st.loc[(variable, level, treatment)], st.loc[(variable, level, control)]
                    n_t, n_c = int(t["n"]), int(c["n"])
                    s_t, s_c = float(t[f"{outcome}_sum"]), float(c[f"{outcome}_sum"])
                    if outcome in RATE_OUTCOMES:
                        res = diff_in_proportions(round(s_t), n_t, round(s_c), n_c, alpha)
                        test, ci_method = "two-proportion z-test (pooled)", "Newcombe hybrid score"
                    else:
                        var_t = (float(t[f"{outcome}_sum_sq"]) - s_t**2 / n_t) / (n_t - 1)
                        var_c = (float(c[f"{outcome}_sum_sq"]) - s_c**2 / n_c) / (n_c - 1)
                        res = diff_in_means(s_t / n_t, var_t, n_t, s_c / n_c, var_c, n_c, alpha)
                        test, ci_method = "Welch t-test", "Welch t"
                        if var_t == 0 and var_c == 0:
                            # The outcome is constant in both arms (only possible in a tiny
                            # sample): no evidence either way, so p = 1 and a point interval.
                            res = _constant_outcome(res)
                    rows.append(
                        {
                            "outcome": outcome,
                            "variable": variable,
                            "level": level,
                            "treatment": treatment,
                            "control": control,
                            "comparison": f"{treatment}_vs_{control}",
                            "test_type": "level_effect",
                            "family": f"segments_{outcome}",
                            "adjustment": ADJUSTMENT,
                            "label": LABEL,
                            "n_treatment": n_t,
                            "n_control": n_c,
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
                    )
    return pd.DataFrame(rows)


def interaction_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """Joint interaction test per outcome, segment variable and comparison.

    Parameters
    ----------
    rows : pandas.DataFrame
        One row per customer with ``arm``, the pre-treatment columns and the outcomes.

    Returns
    -------
    pandas.DataFrame
        One row per outcome, variable and comparison with the Wald statistic and p-value.
    """
    out = []
    for outcome in SEGMENT_OUTCOMES:
        for variable in SEGMENT_VARIABLES:
            segment = segment_column(rows, variable)
            for treatment, control in COMPARISONS:
                mask = rows["arm"].isin((treatment, control)).to_numpy()
                res = interaction_test(
                    rows.loc[mask, outcome],
                    (rows.loc[mask, "arm"] == treatment).astype(float),
                    segment[mask],
                )
                out.append(
                    {
                        "outcome": outcome,
                        "variable": variable,
                        "treatment": treatment,
                        "control": control,
                        "comparison": f"{treatment}_vs_{control}",
                        "test_type": "interaction",
                        "family": f"segments_{outcome}",
                        "adjustment": ADJUSTMENT,
                        "label": LABEL,
                        "n": res.n,
                        "n_levels": res.n_levels,
                        "test": "Wald chi-square of arm x level interactions (HC2)",
                        "statistic": res.statistic,
                        "dof": res.dof,
                        "p_value": res.p_value,
                    }
                )
    return pd.DataFrame(out)


def adjust_families(
    effects: pd.DataFrame, interactions: pd.DataFrame, q: float = FDR_Q
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benjamini-Hochberg over each outcome's whole segment family (levels and interactions).

    Parameters
    ----------
    effects, interactions : pandas.DataFrame
        Outputs of :func:`segment_effects` and :func:`interaction_tests`.
    q : float
        False discovery rate.

    Returns
    -------
    tuple of pandas.DataFrame
        The two tables with ``p_adj``, ``significant_fdr`` and ``family_size`` columns.
    """
    effects, interactions = effects.copy(), interactions.copy()
    for table in (effects, interactions):
        table["p_adj"] = np.nan
        table["family_size"] = 0
    for family in effects["family"].unique():
        e_idx = effects.index[effects["family"] == family]
        i_idx = interactions.index[interactions["family"] == family]
        p = np.concatenate([effects.loc[e_idx, "p_value"], interactions.loc[i_idx, "p_value"]])
        adj = benjamini_hochberg(p)
        effects.loc[e_idx, "p_adj"] = adj[: len(e_idx)]
        interactions.loc[i_idx, "p_adj"] = adj[len(e_idx) :]
        effects.loc[e_idx, "family_size"] = p.size
        interactions.loc[i_idx, "family_size"] = p.size
    for table in (effects, interactions):
        table["significant_fdr"] = table["p_adj"] <= q
    return effects, interactions


def family_summary(
    effects: pd.DataFrame,
    interactions: pd.DataFrame,
    alpha: float = config.ALPHA,
    q: float = FDR_Q,
) -> dict[str, dict[str, Any]]:
    """Count each family's tests and how many are significant before and after correction.

    Parameters
    ----------
    effects, interactions : pandas.DataFrame
        Outputs of :func:`adjust_families`.
    alpha : float
        Level for the uncorrected count: a test counts when its raw ``p_value <= alpha``.
    q : float
        False discovery rate the ``significant_fdr`` flags were computed at.

    Returns
    -------
    dict
        Family -> ``adjustment``, ``q``, ``alpha``, ``n_level_tests``,
        ``n_interaction_tests``, ``n_tests``, ``n_significant_raw`` (uncorrected) and
        ``n_significant_fdr`` (after Benjamini-Hochberg). An adjusted p-value is never
        below its raw one, so with ``alpha == q`` the raw count is at least the corrected one.
    """
    out: dict[str, dict[str, Any]] = {}
    for family in effects["family"].unique():
        e = effects[effects["family"] == family]
        i = interactions[interactions["family"] == family]
        out[family] = {
            "adjustment": ADJUSTMENT,
            "q": q,
            "alpha": alpha,
            "n_level_tests": len(e),
            "n_interaction_tests": len(i),
            "n_tests": len(e) + len(i),
            "n_significant_raw": int((e["p_value"] <= alpha).sum() + (i["p_value"] <= alpha).sum()),
            "n_significant_fdr": int(e["significant_fdr"].sum() + i["significant_fdr"].sum()),
        }
    return out


def challenge_answer(effects: pd.DataFrame, interactions: pd.DataFrame) -> list[dict[str, Any]]:
    """Answer whether each email worked better among past buyers of its own merchandise.

    Reads the pre-specified rows for the ``mens`` and ``womens`` flags: the effect of the
    matching email among past buyers (flag 1) and non-buyers (flag 0) and the interaction test.

    Parameters
    ----------
    effects, interactions : pandas.DataFrame
        Adjusted tables from :func:`adjust_families`.

    Returns
    -------
    list of dict
        One record per email and outcome.
    """
    out = []
    for email, flag in CHALLENGE_PAIRS:
        for outcome in SEGMENT_OUTCOMES:
            sub = effects[
                (effects["outcome"] == outcome)
                & (effects["variable"] == flag)
                & (effects["treatment"] == email)
            ].set_index("level")
            inter = interactions[
                (interactions["outcome"] == outcome)
                & (interactions["variable"] == flag)
                & (interactions["treatment"] == email)
            ].iloc[0]
            record: dict[str, Any] = {
                "email": email,
                "segment_variable": flag,
                "outcome": outcome,
                "label": LABEL,
                "interaction_p_value": float(inter["p_value"]),
                "interaction_p_adj": float(inter["p_adj"]),
                "interaction_significant_fdr": bool(inter["significant_fdr"]),
            }
            for level, name in (("1", "past_buyers"), ("0", "non_buyers")):
                r = sub.loc[level]
                record |= {
                    f"{name}_n_treatment": int(r["n_treatment"]),
                    f"{name}_n_control": int(r["n_control"]),
                    f"{name}_diff": float(r["diff"]),
                    f"{name}_ci_low": float(r["ci_low"]),
                    f"{name}_ci_high": float(r["ci_high"]),
                    f"{name}_rel_lift": float(r["rel_lift"]),
                    f"{name}_p_value": float(r["p_value"]),
                    f"{name}_p_adj": float(r["p_adj"]),
                }
            record["larger_among_past_buyers"] = bool(
                record["past_buyers_diff"] > record["non_buyers_diff"]
            )
            out.append(record)
    return out


def build_segment_tables(
    con: duckdb.DuckDBPyConnection, alpha: float = config.ALPHA, q: float = FDR_Q
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-level effects and interaction tables, Benjamini-Hochberg adjusted.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    alpha : float
        Two-sided level of the intervals.
    q : float
        False discovery rate of the adjustment.

    Returns
    -------
    tuple of pandas.DataFrame
        Effects (one row per outcome, variable, level, comparison) and interaction tests
        (one row per outcome, variable, comparison).
    """
    effects = segment_effects(run_sql(con, "segment_stats"), alpha)
    interactions = interaction_tests(run_sql(con, "adjustment_rows"))
    return adjust_families(effects, interactions, q)


# ---------------------------------------------------------------------------------------------
# README simulation: many segments under A/A, with and without Benjamini-Hochberg
# ---------------------------------------------------------------------------------------------


def bh_rows(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for every row of a ``(runs, tests)`` array."""
    p = np.asarray(p_values, dtype=float)
    m = p.shape[1]
    order = np.argsort(p, axis=1, kind="stable")
    raw = m * np.take_along_axis(p, order, axis=1) / np.arange(1, m + 1)
    adj = np.minimum(1.0, np.minimum.accumulate(raw[:, ::-1], axis=1)[:, ::-1])
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=1)
    return out


def _rate(flags: np.ndarray) -> tuple[float, float]:
    rate = float(np.mean(flags))
    return rate, math.sqrt(rate * (1 - rate) / flags.size)


def simulate_segment_discoveries(
    n_segments: int,
    n_per_arm: int,
    rate: float,
    rng: np.random.Generator,
    n_runs: int = N_SIM_RUNS,
    n_true: int = 0,
    effect: float = 0.0,
    alpha: float = config.ALPHA,
    q: float = FDR_Q,
) -> list[dict[str, Any]]:
    """Simulate testing many independent segments, raw and Benjamini-Hochberg adjusted.

    Each run draws ``n_segments`` independent segments, each an A/B comparison of a rate
    with ``n_per_arm`` customers per arm; the first ``n_true`` segments carry a true lift of
    ``effect`` and the rest none. Every segment gets the pooled two-proportion z-test, then
    the run's p-values are adjusted by Benjamini-Hochberg. Reported per method: the share of
    runs with at least one false discovery, the false discovery rate (the mean over runs of
    false discoveries over all discoveries, 0 when there are none) and the power on the true
    segments, each with its Monte Carlo standard error, next to the analytic references:
    with m0 null segments, raw testing finds at least one false discovery with probability
    1 - (1 - alpha)^m0, and Benjamini-Hochberg has false discovery rate exactly q m0 / m under
    independence (Benjamini and Hochberg 1995). Under the global null both quantities coincide
    for each method; otherwise the other reference is left empty.

    Parameters
    ----------
    n_segments : int
        Segments per run.
    n_per_arm : int
        Customers per arm in each segment.
    rate : float
        Baseline rate of the control arm.
    rng : numpy.random.Generator
        Random generator.
    n_runs : int
        Simulated runs.
    n_true : int
        Segments with a true effect.
    effect : float
        Absolute lift in those segments.
    alpha : float
        Level of the raw tests.
    q : float
        False discovery rate of the adjustment.

    Returns
    -------
    list of dict
        Two records, ``method`` ``"raw"`` and ``"bh"``.
    """
    if not 0 <= n_true <= n_segments:
        raise ValueError("n_true must lie between 0 and n_segments.")
    p_b = np.full(n_segments, rate)
    p_b[:n_true] += effect
    x_c = rng.binomial(n_per_arm, rate, size=(n_runs, n_segments))
    x_t = rng.binomial(n_per_arm, p_b, size=(n_runs, n_segments))
    p = np.asarray(diff_in_proportions(x_t, n_per_arm, x_c, n_per_arm, alpha).p_value)
    is_null = np.arange(n_segments) >= n_true
    m0 = int(is_null.sum())
    out = []
    global_null = m0 == n_segments
    any_raw, fdr_bh = 1 - (1 - alpha) ** m0, q * m0 / n_segments
    for method, discoveries, any_reference, fdr_reference in (
        ("raw", p <= alpha, any_raw, any_raw if global_null else math.nan),
        ("bh", bh_rows(p) <= q, fdr_bh if global_null else math.nan, fdr_bh),
    ):
        false = discoveries[:, is_null].sum(axis=1)
        total = discoveries.sum(axis=1)
        any_false, any_false_mcse = _rate(false > 0)
        fdp = false / np.maximum(total, 1)
        record = {
            "scenario": "global_null" if n_true == 0 else f"{n_true}_true_effects",
            "method": method,
            "n_segments": n_segments,
            "n_true": n_true,
            "n_per_arm": n_per_arm,
            "effect": effect,
            "n_runs": n_runs,
            "any_false_discovery": any_false,
            "any_false_discovery_mcse": any_false_mcse,
            "any_false_discovery_reference": any_reference,
            "fdr": float(fdp.mean()),
            "fdr_mcse": float(fdp.std(ddof=1) / math.sqrt(n_runs)),
            "fdr_reference": fdr_reference,
            "power": math.nan,
            "power_mcse": math.nan,
        }
        if n_true:
            record["power"], record["power_mcse"] = _rate(discoveries[:, ~is_null].ravel())
        out.append(record)
    return out


def build_fdr_simulation(
    rate: float, n_per_arm: int, rng: np.random.Generator, n_runs: int = N_SIM_RUNS
) -> pd.DataFrame:
    """Run the README simulation: a global null and a scenario with a few real effects.

    The true effect in the second scenario is the lift a single segment's test detects with
    ``SIM_POWER`` power, so the power column reads against that design target.

    Parameters
    ----------
    rate : float
        Control baseline rate (the real control arm's visit rate).
    n_per_arm : int
        Customers per arm per segment (the real arm size split into ``N_SIM_SEGMENTS``).
    rng : numpy.random.Generator
        Random generator.
    n_runs : int
        Simulated runs per scenario.

    Returns
    -------
    pandas.DataFrame
        Four rows: raw and Benjamini-Hochberg under each scenario.
    """
    se = math.sqrt(2 * rate * (1 - rate) / n_per_arm)
    effect = (stats.norm.ppf(1 - config.ALPHA / 2) + stats.norm.ppf(SIM_POWER)) * se
    rows = simulate_segment_discoveries(N_SIM_SEGMENTS, n_per_arm, rate, rng, n_runs)
    rows += simulate_segment_discoveries(
        N_SIM_SEGMENTS, n_per_arm, rate, rng, n_runs, N_SIM_TRUE, effect
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------------------------


def run_segments(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    n_sim_runs: int = N_SIM_RUNS,
) -> pd.DataFrame:
    """Run the segment analysis and write its tables, figure and results.json section.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    n_sim_runs : int
        Runs per scenario of the README simulation (tests pass fewer).

    Returns
    -------
    pandas.DataFrame
        The per-level effects table.
    """
    rng = np.random.default_rng(config.SEEDS["simulation"] + 1)
    with duckdb.connect(str(db_path), read_only=True) as con:
        effects, interactions = build_segment_tables(con)
        control = arm_metrics(con).set_index(["metric", "arm"]).loc[("visit", config.CONTROL)]
        n_per_arm = min(arm_counts(con).values()) // N_SIM_SEGMENTS
    simulation = build_fdr_simulation(float(control["mean"]), n_per_arm, rng, n_sim_runs)

    tables = reports_dir / "tables"
    write_table(effects, TABLE_NAME, tables)
    write_table(interactions, INTERACTIONS_TABLE_NAME, tables)
    write_table(simulation, FDR_TABLE_NAME, tables)
    plots.plot_segment_forest(effects, reports_dir / "figures" / FIGURE_NAME, interactions)
    families = family_summary(effects, interactions)
    update_results(
        "segments",
        {
            "label": LABEL,
            "plan_section": "10",
            "deviation": "D5: conversion added as a third outcome with its own family",
            "alpha": config.ALPHA,
            "segment_variables": {k: list(v) for k, v in SEGMENT_VARIABLES.items()},
            "n_levels": sum(len(v) for v in SEGMENT_VARIABLES.values()),
            "families": families,
            "challenge_question": challenge_answer(effects, interactions),
            "fdr_simulation": {
                "seed": config.SEEDS["simulation"] + 1,
                "n_segments": N_SIM_SEGMENTS,
                "n_true": N_SIM_TRUE,
                "n_runs": n_sim_runs,
                "n_per_arm": n_per_arm,
                "baseline_rate": float(control["mean"]),
                "table": f"tables/{FDR_TABLE_NAME}.csv",
                "rows": json_records(simulation),
            },
            "table": f"tables/{TABLE_NAME}.csv",
            "interactions_table": f"tables/{INTERACTIONS_TABLE_NAME}.csv",
            "figure": f"figures/{FIGURE_NAME}",
            "rows": json_records(effects),
            "interaction_rows": json_records(interactions),
        },
        reports_dir / "results.json",
    )
    logger.info(
        "Segment tables written: %d level rows, %d interaction rows (exploratory).",
        len(effects),
        len(interactions),
    )
    return effects
