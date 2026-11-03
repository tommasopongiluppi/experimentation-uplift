"""Frequentist tests: SRM check, z-tests, Welch t-tests, CIs and multiplicity corrections."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from abkit import config


@dataclass(frozen=True)
class SRMResult:
    """Result of a sample ratio mismatch test.

    Attributes
    ----------
    counts, expected : dict
        Observed and expected customers per arm.
    statistic : float
        Pearson chi-square statistic.
    dof : int
        Degrees of freedom (number of arms minus one).
    p_value : float
        Upper-tail chi-square p-value.
    alpha : float
        Threshold below which the check fails.
    passed : bool
        True when ``p_value >= alpha``.
    """

    counts: dict[str, int]
    expected: dict[str, float]
    statistic: float
    dof: int
    p_value: float
    alpha: float
    passed: bool


def srm_test(
    counts: Mapping[str, int],
    expected_shares: Mapping[str, float] | None = None,
    alpha: float = config.SRM_ALPHA,
) -> SRMResult:
    """Chi-square goodness-of-fit test of arm counts against the planned allocation.

    The statistic is sum over arms of (observed - expected)^2 / expected, compared with a
    chi-square distribution with (number of arms - 1) degrees of freedom.

    Parameters
    ----------
    counts : mapping of str to int
        Observed customers per arm.
    expected_shares : mapping of str to float, optional
        Planned share per arm; defaults to equal shares. Must cover the same arms and sum to 1.
    alpha : float
        Failure threshold; defaults to ``config.SRM_ALPHA`` (0.001, analysis plan §5).

    Returns
    -------
    SRMResult
        Counts, expected counts, statistic, p-value and pass/fail.

    Raises
    ------
    ValueError
        If fewer than two arms are given, a count is negative, or the shares are invalid.
    """
    arms = list(counts)
    if len(arms) < 2:
        raise ValueError("Need at least two arms.")
    observed = np.array([counts[a] for a in arms], dtype=float)
    if (observed < 0).any():
        raise ValueError("Counts must be non-negative.")
    if expected_shares is None:
        shares = np.full(len(arms), 1.0 / len(arms))
    else:
        if set(expected_shares) != set(arms):
            raise ValueError("expected_shares must have the same arms as counts.")
        shares = np.array([expected_shares[a] for a in arms], dtype=float)
        if (shares <= 0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("expected_shares must be positive and sum to 1.")
    expected = observed.sum() * shares
    statistic = float(((observed - expected) ** 2 / expected).sum())
    dof = len(arms) - 1
    p_value = float(stats.chi2.sf(statistic, dof))
    return SRMResult(
        counts={a: int(n) for a, n in zip(arms, observed, strict=True)},
        expected={a: float(e) for a, e in zip(arms, expected, strict=True)},
        statistic=statistic,
        dof=dof,
        p_value=p_value,
        alpha=alpha,
        passed=p_value >= alpha,
    )


# ---------------------------------------------------------------------------------------------
# Two-arm comparisons from sufficient statistics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Difference between a treatment and a control arm, absolute and relative.

    Fields are floats for scalar inputs and numpy arrays for array inputs (vectorised use).

    Attributes
    ----------
    mean_t, mean_c : float
        Treatment and control means (rates for proportions).
    diff : float
        mean_t - mean_c.
    se : float
        Standard error used for the test statistic (pooled under H0 for proportions).
    ci_low, ci_high : float
        Confidence interval for ``diff``.
    statistic : float
        z (proportions) or Welch t (means).
    dof : float
        Welch-Satterthwaite degrees of freedom; ``inf`` for the z-test.
    p_value : float
        Two-sided p-value.
    rel_lift : float
        diff / mean_c.
    rel_ci_low, rel_ci_high : float
        Delta-method interval for the relative lift, from the log ratio of means.
    """

    mean_t: Any
    mean_c: Any
    diff: Any
    se: Any
    ci_low: Any
    ci_high: Any
    statistic: Any
    dof: Any
    p_value: Any
    rel_lift: Any
    rel_ci_low: Any
    rel_ci_high: Any


def _out(x: np.ndarray) -> Any:
    """Return a Python float for 0-d arrays and the array otherwise."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def relative_lift_ci(
    mean_t: ArrayLike,
    var_t: ArrayLike,
    n_t: ArrayLike,
    mean_c: ArrayLike,
    var_c: ArrayLike,
    n_c: ArrayLike,
    alpha: float = config.ALPHA,
) -> tuple[Any, Any, Any]:
    """Relative lift (mean_t / mean_c - 1) with a delta-method interval on the log ratio.

    Var(log(mean_t / mean_c)) ~ var_t / (n_t mean_t^2) + var_c / (n_c mean_c^2); the interval
    exp(log ratio +/- z se) - 1 is asymmetric and stays above -100%. NaN if either mean is <= 0.

    Parameters
    ----------
    mean_t, var_t, n_t, mean_c, var_c, n_c : array_like
        Per-arm means, unbiased variances and sizes.
    alpha : float
        Two-sided level; the interval has coverage 1 - alpha.

    Returns
    -------
    tuple
        (relative lift, lower bound, upper bound).
    """
    mt, vt, nt = (np.asarray(a, dtype=float) for a in (mean_t, var_t, n_t))
    mc, vc, nc = (np.asarray(a, dtype=float) for a in (mean_c, var_c, n_c))
    ok = (mt > 0) & (mc > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(ok, np.log(mt) - np.log(mc), np.nan)
        se = np.sqrt(vt / (nt * mt**2) + vc / (nc * mc**2))
        rel = np.where(mc > 0, mt / mc - 1, np.nan)
    z = _z(alpha)
    return _out(rel), _out(np.expm1(log_ratio - z * se)), _out(np.expm1(log_ratio + z * se))


def wilson_interval(
    count: ArrayLike, nobs: ArrayLike, alpha: float = config.ALPHA
) -> tuple[Any, Any]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    count, nobs : array_like
        Successes and trials.
    alpha : float
        Two-sided level.

    Returns
    -------
    tuple
        (lower, upper).
    """
    x, n = np.asarray(count, dtype=float), np.asarray(nobs, dtype=float)
    z = _z(alpha)
    p = x / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return _out(centre - half), _out(centre + half)


def diff_in_proportions(
    count_t: ArrayLike,
    nobs_t: ArrayLike,
    count_c: ArrayLike,
    nobs_c: ArrayLike,
    alpha: float = config.ALPHA,
) -> ComparisonResult:
    """Difference in rates: pooled two-proportion z-test and Newcombe hybrid score interval.

    - Test: z = (p_t - p_c) / sqrt(p (1 - p) (1/n_t + 1/n_c)), p the pooled rate under H0.
    - Interval (Newcombe 1998, method 10): with Wilson intervals (l_t, u_t) and (l_c, u_c),
      lower = d - sqrt((p_t - l_t)^2 + (u_c - p_c)^2), upper = d + sqrt((u_t - p_t)^2 +
      (p_c - l_c)^2).
    - Relative lift: delta method on the log ratio (:func:`relative_lift_ci`), with the
      binomial variance p (1 - p) per arm.

    Parameters
    ----------
    count_t, nobs_t, count_c, nobs_c : array_like
        Successes and customers in the treatment and control arms.
    alpha : float
        Two-sided level.

    Returns
    -------
    ComparisonResult
        ``dof`` is ``inf`` (normal reference distribution).
    """
    xt, nt = np.asarray(count_t, dtype=float), np.asarray(nobs_t, dtype=float)
    xc, nc = np.asarray(count_c, dtype=float), np.asarray(nobs_c, dtype=float)
    pt, pc = xt / nt, xc / nc
    diff = pt - pc
    pooled = (xt + xc) / (nt + nc)
    se = np.sqrt(pooled * (1 - pooled) * (1 / nt + 1 / nc))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stat = np.where(se > 0, diff / se, np.where(diff == 0, 0.0, np.sign(diff) * np.inf))
    p_value = 2 * stats.norm.sf(np.abs(z_stat))
    lt, ut = (np.asarray(b) for b in wilson_interval(xt, nt, alpha))
    lc, uc = (np.asarray(b) for b in wilson_interval(xc, nc, alpha))
    low = diff - np.sqrt((pt - lt) ** 2 + (uc - pc) ** 2)
    high = diff + np.sqrt((ut - pt) ** 2 + (pc - lc) ** 2)
    rel, rel_low, rel_high = relative_lift_ci(pt, pt * (1 - pt), nt, pc, pc * (1 - pc), nc, alpha)
    return ComparisonResult(
        mean_t=_out(pt),
        mean_c=_out(pc),
        diff=_out(diff),
        se=_out(se),
        ci_low=_out(low),
        ci_high=_out(high),
        statistic=_out(z_stat),
        dof=_out(np.full_like(diff, np.inf)),
        p_value=_out(p_value),
        rel_lift=rel,
        rel_ci_low=rel_low,
        rel_ci_high=rel_high,
    )


def diff_in_means(
    mean_t: ArrayLike,
    var_t: ArrayLike,
    n_t: ArrayLike,
    mean_c: ArrayLike,
    var_c: ArrayLike,
    n_c: ArrayLike,
    alpha: float = config.ALPHA,
) -> ComparisonResult:
    """Welch difference in means with Welch-Satterthwaite degrees of freedom.

    se = sqrt(var_t / n_t + var_c / n_c), t = (mean_t - mean_c) / se,
    dof = se^4 / ((var_t / n_t)^2 / (n_t - 1) + (var_c / n_c)^2 / (n_c - 1)), and the interval
    is diff +/- t_{1 - alpha/2, dof} se. Relative lift uses :func:`relative_lift_ci`.

    Parameters
    ----------
    mean_t, var_t, n_t, mean_c, var_c, n_c : array_like
        Per-arm means, unbiased (n - 1) variances and sizes, e.g. from
        :func:`abkit.metrics.arm_metrics`.
    alpha : float
        Two-sided level.

    Returns
    -------
    ComparisonResult
        Welch t statistic, dof, p-value, CI and relative lift.
    """
    mt, vt, nt = (np.asarray(a, dtype=float) for a in (mean_t, var_t, n_t))
    mc, vc, nc = (np.asarray(a, dtype=float) for a in (mean_c, var_c, n_c))
    at, ac = vt / nt, vc / nc
    se = np.sqrt(at + ac)
    diff = mt - mc
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = diff / se
        dof = (at + ac) ** 2 / (at**2 / (nt - 1) + ac**2 / (nc - 1))
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)
    half = stats.t.ppf(1 - alpha / 2, dof) * se
    rel, rel_low, rel_high = relative_lift_ci(mt, vt, nt, mc, vc, nc, alpha)
    return ComparisonResult(
        mean_t=_out(mt),
        mean_c=_out(mc),
        diff=_out(diff),
        se=_out(se),
        ci_low=_out(diff - half),
        ci_high=_out(diff + half),
        statistic=_out(t_stat),
        dof=_out(dof),
        p_value=_out(p_value),
        rel_lift=rel,
        rel_ci_low=rel_low,
        rel_ci_high=rel_high,
    )


# ---------------------------------------------------------------------------------------------
# Percentile bootstrap (row-level data)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile bootstrap interval for a difference in means.

    Attributes
    ----------
    diff : float
        Observed mean_t - mean_c.
    ci_low, ci_high : float
        Percentile interval endpoints.
    se : float
        Standard deviation of the bootstrap replicates.
    mcse_low, mcse_high : float
        Monte Carlo standard errors of the endpoints (order-statistic method).
    n_resamples : int
        Number of bootstrap resamples.
    """

    diff: float
    ci_low: float
    ci_high: float
    se: float
    mcse_low: float
    mcse_high: float
    n_resamples: int


def _bootstrap_means(
    y: np.ndarray, n_resamples: int, rng: np.random.Generator, chunk_cells: int
) -> np.ndarray:
    """Means of ``n_resamples`` with-replacement resamples of ``y``.

    A resample is fully described by how often each distinct value is drawn, which is
    Multinomial(n, value frequencies / n). Drawing these counts is the same distribution as
    drawing n indices, but needs only (resamples x distinct values) numbers; spend has few
    distinct values because almost all of it is zero.
    """
    values, freq = np.unique(y, return_counts=True)
    n = y.size
    probs = freq / n
    per_chunk = max(1, chunk_cells // values.size)
    out = np.empty(n_resamples)
    for start in range(0, n_resamples, per_chunk):
        size = min(per_chunk, n_resamples - start)
        counts = rng.multinomial(n, probs, size=size)
        out[start : start + size] = counts @ values / n
    return out


def _quantile_mcse(sorted_reps: np.ndarray, q: float) -> float:
    """Monte Carlo SE of a sample quantile from the order statistics at B q +/- sqrt(B q (1-q))."""
    b = sorted_reps.size
    half = np.sqrt(b * q * (1 - q))
    lo = int(np.clip(np.floor(b * q - half), 0, b - 1))
    hi = int(np.clip(np.ceil(b * q + half), 0, b - 1))
    return float((sorted_reps[hi] - sorted_reps[lo]) / 2)


def bootstrap_diff_in_means(
    y_t: ArrayLike,
    y_c: ArrayLike,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    alpha: float = config.ALPHA,
    chunk_cells: int = 20_000_000,
) -> BootstrapResult:
    """Percentile bootstrap interval for mean(y_t) - mean(y_c), resampling each arm separately.

    Vectorised: each resample is a multinomial count vector over the distinct values of an
    arm (see ``_bootstrap_means``), processed in chunks of at most ``chunk_cells`` numbers.

    Parameters
    ----------
    y_t, y_c : array_like
        Row-level outcomes in the treatment and control arms.
    rng : numpy.random.Generator
        Source of randomness (treatment arm drawn first, then control).
    n_resamples : int
        Bootstrap resamples (default 10,000).
    alpha : float
        Two-sided level; the interval uses the alpha/2 and 1 - alpha/2 quantiles.
    chunk_cells : int
        Memory bound per chunk.

    Returns
    -------
    BootstrapResult
        Interval, bootstrap SE and Monte Carlo SEs of the endpoints.
    """
    yt, yc = np.asarray(y_t, dtype=float), np.asarray(y_c, dtype=float)
    if yt.size == 0 or yc.size == 0:
        raise ValueError("Both arms need at least one observation.")
    reps = _bootstrap_means(yt, n_resamples, rng, chunk_cells) - _bootstrap_means(
        yc, n_resamples, rng, chunk_cells
    )
    reps.sort()
    q_low, q_high = alpha / 2, 1 - alpha / 2
    return BootstrapResult(
        diff=float(yt.mean() - yc.mean()),
        ci_low=float(np.quantile(reps, q_low)),
        ci_high=float(np.quantile(reps, q_high)),
        se=float(reps.std(ddof=1)),
        mcse_low=_quantile_mcse(reps, q_low),
        mcse_high=_quantile_mcse(reps, q_high),
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------------------------


def _check_pvalues(p_values: ArrayLike) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must lie in [0, 1] with no NaN.")
    return p


def holm(p_values: ArrayLike) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control).

    With p sorted ascending, p_adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).

    Parameters
    ----------
    p_values : array_like
        Unadjusted p-values of one family.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in the input order.
    """
    p = _check_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (false discovery rate control).

    With p sorted ascending, p_adj_(i) = min_{j >= i} min(1, m p_(j) / j).

    Parameters
    ----------
    p_values : array_like
        Unadjusted p-values of one family.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in the input order; reject at FDR q where p_adj <= q.
    """
    p = _check_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    raw = m * p[order] / np.arange(1, m + 1)
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    out = np.empty(m)
    out[order] = adj_sorted
    return out
