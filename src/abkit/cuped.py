"""Variance reduction with CUPED and the Lin (2013) regression adjustment (analysis plan §7).

- :func:`cuped` works from per-arm sufficient statistics (counts, sums, sums of squares and
  cross-products with the covariate), so like the rest of the readout it never needs
  row-level data.
- :func:`regression_adjustment` fits an OLS with treatment-by-covariate interactions and HC2
  robust standard errors, which needs one row per customer.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy import stats

from abkit import config
from abkit.frequentist import _out, diff_in_means

# Covariate pre-registered for CUPED (plan §7), used as is.
CUPED_COVARIATE: str = "history"
# Lin adjustment covariates (plan §7): every pre-treatment variable, categoricals one-hot.
LIN_NUMERIC: tuple[str, ...] = ("recency", "history", "mens", "womens", "newbie")
LIN_CATEGORICAL: tuple[str, ...] = ("history_segment", "zip_code", "channel")


# ---------------------------------------------------------------------------------------------
# CUPED from sufficient statistics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmMoments:
    """Sufficient statistics of an outcome Y and a covariate X in one arm.

    Fields may be scalars or equally shaped arrays (one entry per simulated experiment).

    Attributes
    ----------
    n : array_like
        Number of customers.
    sum_y, sum_y2 : array_like
        Sum of Y and of Y².
    sum_x, sum_x2 : array_like
        Sum of X and of X².
    sum_xy : array_like
        Sum of X·Y.
    """

    n: Any
    sum_y: Any
    sum_y2: Any
    sum_x: Any
    sum_x2: Any
    sum_xy: Any

    @classmethod
    def from_arrays(cls, y: ArrayLike, x: ArrayLike, axis: int = -1) -> "ArmMoments":
        """Compute the sums from row-level data.

        Parameters
        ----------
        y, x : array_like
            Outcome and covariate, same shape.
        axis : int
            Axis along which customers lie (other axes index simulated experiments).

        Returns
        -------
        ArmMoments
            The sufficient statistics.
        """
        y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
        if y.shape != x.shape:
            raise ValueError("y and x must have the same shape.")
        return cls(
            n=_out(np.full(np.sum(y, axis=axis).shape, y.shape[axis], dtype=float)),
            sum_y=_out(y.sum(axis=axis)),
            sum_y2=_out((y * y).sum(axis=axis)),
            sum_x=_out(x.sum(axis=axis)),
            sum_x2=_out((x * x).sum(axis=axis)),
            sum_xy=_out((x * y).sum(axis=axis)),
        )

    def __add__(self, other: "ArmMoments") -> "ArmMoments":
        """Pool two groups by adding their sums."""
        return ArmMoments(
            *(
                np.asarray(getattr(self, f)) + np.asarray(getattr(other, f))
                for f in ("n", "sum_y", "sum_y2", "sum_x", "sum_x2", "sum_xy")
            )
        )

    def mean_y(self) -> np.ndarray:
        """Mean of Y."""
        return np.asarray(self.sum_y) / np.asarray(self.n)

    def mean_x(self) -> np.ndarray:
        """Mean of X."""
        return np.asarray(self.sum_x) / np.asarray(self.n)

    def _co(self, s_ab: Any, s_a: Any, s_b: Any) -> np.ndarray:
        n = np.asarray(self.n, dtype=float)
        return (np.asarray(s_ab) - np.asarray(s_a) * np.asarray(s_b) / n) / (n - 1)

    def var_y(self) -> np.ndarray:
        """Unbiased variance of Y."""
        return self._co(self.sum_y2, self.sum_y, self.sum_y)

    def var_x(self) -> np.ndarray:
        """Unbiased variance of X."""
        return self._co(self.sum_x2, self.sum_x, self.sum_x)

    def cov_xy(self) -> np.ndarray:
        """Unbiased covariance of X and Y."""
        return self._co(self.sum_xy, self.sum_x, self.sum_y)


@dataclass(frozen=True)
class CupedResult:
    """CUPED-adjusted difference in means next to the unadjusted Welch difference.

    Fields are floats for scalar inputs and arrays for array inputs.

    Attributes
    ----------
    mean_t, mean_c : float
        Adjusted arm means, mean(Y) - theta (mean(X) - pooled mean(X)).
    diff, se, ci_low, ci_high, statistic, dof, p_value : float
        Welch difference in adjusted means with its interval and test.
    theta : float
        cov(Y, X) / var(X) on the two arms pooled.
    rho : float
        Correlation of Y and X on the two arms pooled.
    variance_reduction : float
        1 - var(Y_adj) / var(Y), with within-arm variances pooled over the two arms.
    raw_diff, raw_se, raw_ci_low, raw_ci_high : float
        Unadjusted Welch difference, standard error and interval.
    ci_width_ratio : float
        Adjusted interval width / unadjusted interval width.
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
    theta: Any
    rho: Any
    variance_reduction: Any
    raw_diff: Any
    raw_se: Any
    raw_ci_low: Any
    raw_ci_high: Any
    ci_width_ratio: Any


def cuped(treatment: ArmMoments, control: ArmMoments, alpha: float = config.ALPHA) -> CupedResult:
    """CUPED difference in means from per-arm sufficient statistics.

    With both arms pooled, theta = cov(Y, X) / var(X) and Y_adj = Y - theta (X - mean(X)).
    Per arm, mean(Y_adj) = mean(Y) - theta (mean_arm(X) - mean(X)) and
    var(Y_adj) = var(Y) - 2 theta cov(Y, X) + theta² var(X) (within-arm moments). The effect
    is the Welch difference of the adjusted means (:func:`abkit.frequentist.diff_in_means`).
    Because X is pre-treatment and theta is pooled, the estimate is unbiased for the same
    effect as the raw difference, with variance smaller by about a factor (1 - rho²).

    Parameters
    ----------
    treatment, control : ArmMoments
        Sufficient statistics of Y and X in each arm.
    alpha : float
        Two-sided level of the intervals.

    Returns
    -------
    CupedResult
        Adjusted estimate and interval, theta, rho, variance reduction and the raw estimate.
    """
    pooled = treatment + control
    var_x_p, cov_p, var_y_p = pooled.var_x(), pooled.cov_xy(), pooled.var_y()
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(var_x_p > 0, cov_p / var_x_p, 0.0)
        rho = cov_p / np.sqrt(var_x_p * var_y_p)
    mx = pooled.mean_x()

    def adjusted(a: ArmMoments) -> tuple[np.ndarray, np.ndarray]:
        mean = a.mean_y() - theta * (a.mean_x() - mx)
        var = a.var_y() - 2 * theta * a.cov_xy() + theta**2 * a.var_x()
        return mean, np.maximum(var, 0.0)

    (mt, vt), (mc, vc) = adjusted(treatment), adjusted(control)
    nt, nc = np.asarray(treatment.n, dtype=float), np.asarray(control.n, dtype=float)
    adj = diff_in_means(mt, vt, nt, mc, vc, nc, alpha)
    raw = diff_in_means(
        treatment.mean_y(), treatment.var_y(), nt, control.mean_y(), control.var_y(), nc, alpha
    )
    within_raw = (nt - 1) * treatment.var_y() + (nc - 1) * control.var_y()
    within_adj = (nt - 1) * vt + (nc - 1) * vc
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = 1 - within_adj / within_raw
        width_ratio = (np.asarray(adj.ci_high) - np.asarray(adj.ci_low)) / (
            np.asarray(raw.ci_high) - np.asarray(raw.ci_low)
        )
    return CupedResult(
        mean_t=adj.mean_t,
        mean_c=adj.mean_c,
        diff=adj.diff,
        se=adj.se,
        ci_low=adj.ci_low,
        ci_high=adj.ci_high,
        statistic=adj.statistic,
        dof=adj.dof,
        p_value=adj.p_value,
        theta=_out(theta),
        rho=_out(rho),
        variance_reduction=_out(reduction),
        raw_diff=raw.diff,
        raw_se=raw.se,
        raw_ci_low=raw.ci_low,
        raw_ci_high=raw.ci_high,
        ci_width_ratio=_out(width_ratio),
    )


def moments_from_stats(stats: pd.DataFrame, arm: str, metric: str) -> ArmMoments:
    """Pick one arm's sufficient statistics for ``metric`` and ``history`` from the SQL sums.

    Parameters
    ----------
    stats : pandas.DataFrame
        Output of :func:`abkit.metrics.arm_sufficient_stats`.
    arm : str
        Arm label.
    metric : str
        ``"visit"``, ``"conversion"`` or ``"spend"``.

    Returns
    -------
    ArmMoments
        Sums of the metric (Y) and ``history`` (X).
    """
    r = stats.set_index("arm").loc[arm]
    x = CUPED_COVARIATE
    return ArmMoments(
        n=float(r["n"]),
        sum_y=float(r[f"{metric}_sum"]),
        sum_y2=float(r[f"{metric}_sum_sq"]),
        sum_x=float(r[f"{x}_sum"]),
        sum_x2=float(r[f"{x}_sum_sq"]),
        sum_xy=float(r[f"{x}_x_{metric}"]),
    )


# ---------------------------------------------------------------------------------------------
# Lin (2013) regression adjustment
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionAdjustmentResult:
    """Lin (2013) estimate of the average treatment effect.

    Attributes
    ----------
    diff : float
        Coefficient on the treatment indicator (the average effect).
    se : float
        HC2 robust standard error.
    ci_low, ci_high : float
        Normal-based interval.
    statistic : float
        diff / se.
    p_value : float
        Two-sided normal p-value.
    n : int
        Customers in the regression.
    n_covariates : int
        Covariate columns after one-hot encoding and dropping constant columns.
    """

    diff: float
    se: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    n: int
    n_covariates: int


def pretreatment_design(
    df: pd.DataFrame,
    numeric: tuple[str, ...] = LIN_NUMERIC,
    categorical: tuple[str, ...] = LIN_CATEGORICAL,
) -> pd.DataFrame:
    """Numeric covariate matrix from pre-treatment columns, categoricals one-hot encoded.

    Each categorical gets one indicator per level except its first level in
    ``config.CATEGORY_MAPS`` (the reference), so the matrix has a fixed column set whatever
    levels the data contains.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned customers (must contain the named columns).
    numeric, categorical : tuple of str
        Pre-treatment columns to use.

    Returns
    -------
    pandas.DataFrame
        Float columns, same index as ``df``.

    Raises
    ------
    ValueError
        If a requested column is not a pre-treatment variable (no outcome is a covariate).
    """
    bad = set(numeric) | set(categorical)
    bad -= set(config.PRE_TREATMENT_COLS)
    if bad:
        raise ValueError(f"Not pre-treatment variables: {sorted(bad)}.")
    parts = [df[list(numeric)].astype(float)]
    for col in categorical:
        levels = list(config.CATEGORY_MAPS[col].values())
        cat = pd.Categorical(df[col], categories=levels)
        if pd.isna(cat).any():
            raise ValueError(f"Unknown level in {col!r}.")
        dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
        dummies.index = df.index
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def regression_adjustment(
    y: ArrayLike,
    treated: ArrayLike,
    covariates: pd.DataFrame | np.ndarray,
    alpha: float = config.ALPHA,
) -> RegressionAdjustmentResult:
    """Lin (2013) interacted regression adjustment with HC2 robust standard errors.

    OLS of Y on an intercept, the treatment indicator D, the covariates centred at their
    pooled mean (Xc) and the interactions D·Xc. Centring makes the coefficient on D the
    average treatment effect over the pooled sample; the interactions let the covariate
    slopes differ by arm, so the adjustment cannot hurt precision asymptotically, even if the
    linear model is wrong. Columns that are constant in the sample are dropped.

    Parameters
    ----------
    y : array_like
        Outcome, one entry per customer.
    treated : array_like
        1 (or True) for the treatment arm, 0 for the comparison arm.
    covariates : pandas.DataFrame or numpy.ndarray
        Pre-treatment covariates, one row per customer (e.g. :func:`pretreatment_design`).
    alpha : float
        Two-sided level of the interval.

    Returns
    -------
    RegressionAdjustmentResult
        Effect, HC2 standard error, interval and p-value.
    """
    yv = np.asarray(y, dtype=float)
    d = np.asarray(treated, dtype=float)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not (yv.shape[0] == d.shape[0] == x.shape[0]):
        raise ValueError("y, treated and covariates must have the same number of rows.")
    if not np.isin(d, (0.0, 1.0)).all() or d.min() == d.max():
        raise ValueError("treated must be 0/1 with both groups present.")
    x = x[:, x.std(axis=0) > 0]
    xc = x - x.mean(axis=0)
    design = np.column_stack([np.ones_like(d), d, xc, d[:, None] * xc])
    fit = sm.OLS(yv, design).fit(cov_type="HC2", use_t=False)
    diff, se = float(fit.params[1]), float(fit.bse[1])
    z = float(stats.norm.ppf(1 - alpha / 2))
    statistic = diff / se
    return RegressionAdjustmentResult(
        diff=diff,
        se=se,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        statistic=statistic,
        p_value=float(2 * stats.norm.sf(abs(statistic))),
        n=int(yv.size),
        n_covariates=int(x.shape[1]),
    )
