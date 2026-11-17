"""Sequential testing by simulation: peeking, group-sequential boundaries and mSPRT.

Everything in this module is a simulation built on the control arm's baseline rates and spend
values. It is not a re-analysis of the real experiment, which is a single final snapshot and is
read once (analysis plan §12). The stylised setting: a campaign rolled out over ``DAYS`` days
with the same number of new customers per arm each day, an analyst who looks at the cumulative
results every day, and six decision rules compared on their false positive rate (A/A), power
and stopping day (A/B):

1. fixed horizon, one two-sided test on the last day (the benchmark);
2. naive peeking, the same test at level alpha at every look;
3. Bayesian peeking, stop when P(B > A) leaves (1 - threshold, threshold) under flat priors;
4. Pocock and 5. O'Brien-Fleming group-sequential boundaries, calibrated by simulating the
   correlated Z-statistics so that the overall two-sided error is alpha;
6. the mixture sequential probability ratio test (mSPRT, Johari et al. 2017) with a normal
   mixing distribution, read through its always-valid p-value.

Simulations are vectorised over runs and looks: only cumulative per-arm sufficient statistics
are kept, never per-customer arrays. Every Monte Carlo rate carries its standard error.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special, stats

from abkit import config, plots
from abkit.frequentist import _quantile_mcse
from abkit.metrics import arm_counts, arm_metrics
from abkit.power import mde
from abkit.report import json_records, update_results, write_table

logger = logging.getLogger(__name__)

DAYS: int = 14
MAX_LOOKS: int = 30
N_RUNS: int = 20_000
N_RUNS_LOOKS: int = 10_000
N_PATHS: int = 500_000
PATH_CHUNK: int = 100_000
QUAD_NODES: int = 64
QUAD_TAIL: float = 1e-10
METRICS: tuple[str, ...] = ("visit", "conversion", "spend")
RATE_METRICS: tuple[str, ...] = ("visit", "conversion")
METHODS: tuple[str, ...] = ("fixed", "naive", "bayes", "pocock", "obf", "msprt")
# mSPRT sensitivity: the mixing scale tau is the metric's MDE times these factors.
TAU_MULTIPLIERS: dict[str, float] = {"msprt": 1.0, "msprt_half_tau": 0.5, "msprt_double_tau": 2.0}
SINGLE_LOOK: str = "single_look"
SCENARIO_AA: str = "aa"
SCENARIO_AB_OBSERVED: str = "ab_observed"
SCENARIO_AB_MDE: str = "ab_mde"
HEADLINE_METRIC: str = "visit"
# Jennison and Turnbull (2000), Table 2.1: two-sided alpha = 0.05, K = 5 equally spaced looks.
PUBLISHED_K5: dict[str, float] = {"pocock": 2.413, "obf": 2.040}
TABLE_NAME = "sequential"
LOOKS_TABLE_NAME = "sequential_looks"
BY_DAY_TABLE_NAME = "sequential_by_day"
BOUNDARIES_TABLE_NAME = "sequential_boundaries"
FPR_FIGURE_NAME = "sequential_fpr_vs_looks.png"
CUMULATIVE_FIGURE_NAME = "sequential_cumulative_rejection.png"
BOUNDARIES_FIGURE_NAME = "sequential_boundaries.png"


# ---------------------------------------------------------------------------------------------
# Simulated experiments
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Cumulative per-arm sufficient statistics of simulated experiments at every look.

    Attributes
    ----------
    n : numpy.ndarray
        Customers per arm at each look, shape ``(looks,)``.
    sum_a, sum_b : numpy.ndarray
        Cumulative outcome sums per run and look, shape ``(runs, looks)``.
    sumsq_a, sumsq_b : numpy.ndarray
        Cumulative sums of squared outcomes (equal to the sums for a binary outcome).
    binary : bool
        Whether the outcome is 0/1 (rates use the pooled z-test, means the Welch z-test).
    """

    n: np.ndarray
    sum_a: np.ndarray
    sum_b: np.ndarray
    sumsq_a: np.ndarray
    sumsq_b: np.ndarray
    binary: bool

    @property
    def n_runs(self) -> int:
        """Number of simulated experiments."""
        return int(self.sum_a.shape[0])

    @property
    def n_looks(self) -> int:
        """Number of looks."""
        return int(self.n.size)


def _per_look(n_per_arm: int, n_looks: int) -> int:
    if n_looks < 1 or n_per_arm < n_looks:
        raise ValueError("Need at least one look and at least one customer per arm per look.")
    return n_per_arm // n_looks


def simulate_rate_paths(
    p_a: float,
    p_b: float,
    n_per_arm: int,
    n_looks: int,
    n_runs: int,
    rng: np.random.Generator,
) -> Paths:
    """Simulate a binary outcome arriving in ``n_looks`` equal batches per arm.

    Each look adds ``n_per_arm // n_looks`` customers to each arm (the remainder, at most
    ``n_looks - 1`` customers, is dropped) and the successes are Binomial.

    Parameters
    ----------
    p_a, p_b : float
        Success rates of arm A (control) and arm B (treatment).
    n_per_arm : int
        Customers per arm at the final look.
    n_looks : int
        Number of equally spaced looks.
    n_runs : int
        Simulated experiments.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    Paths
        Cumulative counts; ``binary`` is True.
    """
    if not (0 <= p_a <= 1 and 0 <= p_b <= 1):
        raise ValueError("Rates must lie in [0, 1].")
    m = _per_look(n_per_arm, n_looks)
    x_a = rng.binomial(m, p_a, size=(n_runs, n_looks)).cumsum(axis=1).astype(float)
    x_b = rng.binomial(m, p_b, size=(n_runs, n_looks)).cumsum(axis=1).astype(float)
    n = m * np.arange(1, n_looks + 1, dtype=float)
    return Paths(n=n, sum_a=x_a, sum_b=x_b, sumsq_a=x_a, sumsq_b=x_b, binary=True)


def simulate_spend_paths(
    values: ArrayLike,
    n_per_arm: int,
    n_looks: int,
    n_runs: int,
    rng: np.random.Generator,
) -> Paths:
    """Simulate an A/A experiment on spend by resampling the control arm's values.

    Every simulated customer is an independent draw with replacement from ``values``. The
    draw is done in two stages that give the same distribution: the number of positive spends
    in each (run, look, arm) cell is Binomial(customers, share of positive values), and those
    are uniform draws among the positive values. Zero spends never need to be drawn, which is
    what makes the simulation affordable when about 99% of customers spend nothing.

    Parameters
    ----------
    values : array_like
        Spend per customer in the control arm (zeros included).
    n_per_arm : int
        Customers per arm at the final look.
    n_looks : int
        Number of equally spaced looks.
    n_runs : int
        Simulated experiments.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    Paths
        Cumulative sums and sums of squares; ``binary`` is False.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or (values < 0).any():
        raise ValueError("values must be a non-empty array of non-negative spends.")
    positives = values[values > 0]
    share = positives.size / values.size
    m = _per_look(n_per_arm, n_looks)
    counts = rng.binomial(m, share, size=(n_runs, n_looks, 2))
    draws = positives[rng.integers(0, positives.size, size=int(counts.sum()))]
    cell = np.repeat(np.arange(counts.size), counts.ravel())
    sums = np.bincount(cell, weights=draws, minlength=counts.size).reshape(counts.shape)
    sumsq = np.bincount(cell, weights=draws**2, minlength=counts.size).reshape(counts.shape)
    sums, sumsq = sums.cumsum(axis=1), sumsq.cumsum(axis=1)
    n = m * np.arange(1, n_looks + 1, dtype=float)
    return Paths(
        n=n,
        sum_a=sums[:, :, 0],
        sum_b=sums[:, :, 1],
        sumsq_a=sumsq[:, :, 0],
        sumsq_b=sumsq[:, :, 1],
        binary=False,
    )


# ---------------------------------------------------------------------------------------------
# Test statistics at every look
# ---------------------------------------------------------------------------------------------


def difference_and_se(paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    """Difference in means (B minus A) and its standard error at every look.

    For a binary outcome the standard error is the pooled one of the two-proportion z-test,
    as in :func:`abkit.frequentist.diff_in_proportions`; otherwise it is the Welch standard
    error from the two unbiased sample variances. Both are zero only when both arms are
    constant, in which case the difference is zero as well.

    Parameters
    ----------
    paths : Paths
        Simulated experiments.

    Returns
    -------
    tuple of numpy.ndarray
        ``diff`` and ``se``, each of shape ``(runs, looks)``.
    """
    n = paths.n
    mean_a, mean_b = paths.sum_a / n, paths.sum_b / n
    diff = mean_b - mean_a
    if paths.binary:
        pooled = (paths.sum_a + paths.sum_b) / (2 * n)
        var_sum = 2 * pooled * (1 - pooled)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            var_a = (paths.sumsq_a - paths.sum_a**2 / n) / (n - 1)
            var_b = (paths.sumsq_b - paths.sum_b**2 / n) / (n - 1)
        var_sum = np.nan_to_num(var_a + var_b, nan=0.0)
    se = np.sqrt(np.maximum(var_sum, 0.0) / n)
    return diff, se


def z_statistics(diff: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Z-statistics ``diff / se``, with 0 where the standard error is 0 (both arms constant)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, diff / se, 0.0)


def msprt_log_likelihood_ratio(diff: ArrayLike, se: ArrayLike, tau: float) -> Any:
    """Log of the mSPRT mixture likelihood ratio with a N(0, tau²) mixing distribution.

    For a normal estimate ``diff`` of the effect with variance V = se² (treated as known),
    Johari et al. (2017) integrate the likelihood ratio against a N(0, tau²) prior on the
    effect and obtain Λ = sqrt(V / (V + tau²)) exp(tau² diff² / (2 V (V + tau²))). Where
    V = 0 (and so diff = 0) Λ is taken as 1.

    Parameters
    ----------
    diff, se : array_like
        Estimated difference and its standard error.
    tau : float
        Standard deviation of the mixing distribution, in outcome units.

    Returns
    -------
    float or numpy.ndarray
        log Λ.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    d, v = np.asarray(diff, dtype=float), np.asarray(se, dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lr = 0.5 * np.log(v / (v + tau**2)) + tau**2 * d**2 / (2 * v * (v + tau**2))
    out = np.where(v > 0, log_lr, 0.0)
    return float(out) if out.ndim == 0 else out


def msprt_p_values(diff: np.ndarray, se: np.ndarray, tau: float) -> np.ndarray:
    """Always-valid p-values: the running minimum over looks of min(1, 1 / Λ).

    Parameters
    ----------
    diff, se : numpy.ndarray
        Differences and standard errors, shape ``(runs, looks)``.
    tau : float
        Mixing standard deviation.

    Returns
    -------
    numpy.ndarray
        p-values of shape ``(runs, looks)``, non-increasing along looks.
    """
    p = np.exp(-np.maximum(msprt_log_likelihood_ratio(diff, se, tau), 0.0))
    return np.minimum.accumulate(p, axis=1)


def msprt_z_boundary(variance: ArrayLike, tau: float, alpha: float = config.ALPHA) -> Any:
    """|z| at which the mSPRT rejects on a given look, given the estimate's variance.

    Solving Λ >= 1 / alpha with diff² = z² V gives
    z² = 2 (V + tau²) / tau² · (log(1 / alpha) + log((V + tau²) / V) / 2). The running
    minimum in :func:`msprt_p_values` can only reject earlier, so this is the boundary for a
    path that has not crossed before.

    Parameters
    ----------
    variance : array_like
        Variance V of the difference at the look.
    tau : float
        Mixing standard deviation.
    alpha : float
        Level.

    Returns
    -------
    float or numpy.ndarray
        The |z| boundary.
    """
    v = np.asarray(variance, dtype=float)
    z2 = 2 * (v + tau**2) / tau**2 * (math.log(1 / alpha) + 0.5 * np.log((v + tau**2) / v))
    out = np.sqrt(z2)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------------------------
# Bayesian peeking: exact P(p_B > p_A) for Beta posteriors, applied through a count boundary
# ---------------------------------------------------------------------------------------------


def prob_b_greater_a(
    alpha_a: ArrayLike,
    beta_a: ArrayLike,
    alpha_b: ArrayLike,
    beta_b: ArrayLike,
    nodes: int = QUAD_NODES,
) -> Any:
    """P(p_B > p_A) for independent Beta distributions, vectorised, by Gauss-Legendre quadrature.

    P(p_B > p_A) = ∫ f_B(x) F_A(x) dx, integrated over the central 1 - 2e-10 of p_B's
    distribution with ``nodes`` Gauss-Legendre nodes; the truncated tails contribute at most
    2e-10. The integral is taken over whichever of the two distributions is narrower (using
    1 - ∫ f_A(x) F_B(x) dx when that is p_A), so the other's distribution function is smooth
    on the scale of the nodes. Cross-checked against :func:`abkit.bayes.prob_b_greater_a_exact`.

    Parameters
    ----------
    alpha_a, beta_a, alpha_b, beta_b : array_like
        Beta parameters, broadcast together.
    nodes : int
        Quadrature nodes.

    Returns
    -------
    float or numpy.ndarray
        P(p_B > p_A) with the broadcast shape.
    """
    a_a, b_a, a_b, b_b = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (alpha_a, beta_a, alpha_b, beta_b))
    )
    if min(a_a.min(), b_a.min(), a_b.min(), b_b.min()) <= 0:
        raise ValueError("All Beta parameters must be positive.")
    var_a = a_a * b_a / ((a_a + b_a) ** 2 * (a_a + b_a + 1))
    var_b = a_b * b_b / ((a_b + b_b) ** 2 * (a_b + b_b + 1))
    swap = var_a < var_b
    a_int, b_int = np.where(swap, a_a, a_b), np.where(swap, b_a, b_b)
    a_cdf, b_cdf = np.where(swap, a_b, a_a), np.where(swap, b_b, b_a)
    lo = stats.beta.ppf(QUAD_TAIL, a_int, b_int)
    hi = stats.beta.isf(QUAD_TAIL, a_int, b_int)
    x, w = np.polynomial.legendre.leggauss(nodes)
    half = (hi - lo)[..., None] / 2
    t = (hi + lo)[..., None] / 2 + half * x
    log_pdf = (
        special.xlogy(a_int[..., None] - 1, t)
        + special.xlog1py(b_int[..., None] - 1, -t)
        - special.betaln(a_int, b_int)[..., None]
    )
    cdf = special.betainc(a_cdf[..., None], b_cdf[..., None], t)
    integral = (half * w * np.exp(log_pdf) * cdf).sum(axis=-1)
    out = np.where(swap, 1 - integral, integral)
    return float(out) if out.ndim == 0 else out


def bayes_count_boundary(
    x_a: ArrayLike, n: int, threshold: float = config.BAYES_P_BETTER
) -> np.ndarray:
    """Smallest number of successes in B at which P(p_B > p_A) exceeds the threshold.

    Both arms have ``n`` customers and flat Beta(1, 1) priors; P(p_B > p_A) increases in the
    successes of B, so the Bayesian stopping rule at a look is a boundary on counts. The search
    starts from the normal approximation and brackets the crossing with exact probabilities.
    Where no count reaches the threshold the boundary is ``n + 1``.

    Parameters
    ----------
    x_a : array_like
        Successes in arm A (integers).
    n : int
        Customers per arm at the look.
    threshold : float
        Probability threshold (0.95 in this project).

    Returns
    -------
    numpy.ndarray
        Boundary counts, one per entry of ``x_a``.
    """
    x_a = np.atleast_1d(np.asarray(x_a, dtype=float))
    p = (x_a + 1) / (n + 2)
    guess = np.ceil(x_a + stats.norm.ppf(threshold) * np.sqrt(2 * (n + 2) * p * (1 - p)))
    lo, width = guess - 4, 8
    for _ in range(20):
        cand = np.clip(lo[:, None] + np.arange(width + 1), 0, n)
        prob = prob_b_greater_a(1 + x_a[:, None], 1 + n - x_a[:, None], 1 + cand, 1 + n - cand)
        above = prob > threshold
        bracketed = (~above[:, 0] | (cand[:, 0] == 0)) & (above[:, -1] | (cand[:, -1] == n))
        if bracketed.all():
            break
        lo = np.where(bracketed, lo, lo - width)
        width *= 3
    else:  # pragma: no cover - the window triples each round, so this cannot happen
        raise RuntimeError("Could not bracket the Bayesian boundary.")
    first = cand[np.arange(x_a.size), above.argmax(axis=1)]
    return np.where(above.any(axis=1), first, n + 1)


def _bayes_rejections(paths: Paths, threshold: float, per_run: bool = False) -> np.ndarray:
    """Flag the looks at which P(B > A) leaves (1 - threshold, threshold).

    For a binary outcome the default route finds the boundary count once per possible count
    at each look, which is the cheaper one for many runs. ``per_run`` evaluates P(B > A) for
    every run and look directly instead: the same rule, cheaper for a few thousand runs
    because its cost does not grow with the sample size or the effect.
    """
    reject = np.empty(paths.sum_a.shape, dtype=bool)
    if paths.binary and per_run:
        prob = prob_b_greater_a(
            1 + paths.sum_a, 1 + paths.n - paths.sum_a, 1 + paths.sum_b, 1 + paths.n - paths.sum_b
        )
        return (prob > threshold) | (prob < 1 - threshold)
    if not paths.binary:
        # Large-sample normal posterior on the mean difference (flat prior):
        # P(B > A) = Phi(diff / se), so the rule is |z| > z_threshold.
        diff, se = difference_and_se(paths)
        reject[:] = np.abs(z_statistics(diff, se)) > stats.norm.ppf(threshold)
        return reject
    for k in range(paths.n_looks):
        x_a = paths.sum_a[:, k].astype(int)
        x_b = paths.sum_b[:, k].astype(int)
        lo, hi = min(x_a.min(), x_b.min()), max(x_a.max(), x_b.max())
        g = bayes_count_boundary(np.arange(lo, hi + 1), int(paths.n[k]), threshold)
        # P(B > A) < 1 - threshold is the mirror image: A's count reaches the boundary set by B.
        reject[:, k] = (x_b >= g[x_a - lo]) | (x_a >= g[x_b - lo])
    return reject


# ---------------------------------------------------------------------------------------------
# Group-sequential boundaries
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryConstants:
    """Calibrated Pocock and O'Brien-Fleming constants for K equally spaced looks.

    Attributes
    ----------
    n_looks : int
        K.
    pocock : float
        Reject at look k when |Z_k| > pocock.
    obf : float
        Reject at look k when |Z_k| > obf · sqrt(K / k); ``obf`` is the final boundary.
    pocock_mcse, obf_mcse : float
        Monte Carlo standard errors of the two quantiles.
    n_paths : int
        Simulated null paths.
    alpha : float
        Overall two-sided error.
    """

    n_looks: int
    pocock: float
    obf: float
    pocock_mcse: float
    obf_mcse: float
    n_paths: int
    alpha: float

    def pocock_boundary(self) -> np.ndarray:
        """Pocock |z| boundary at each look (constant)."""
        return np.full(self.n_looks, self.pocock)

    def obf_boundary(self) -> np.ndarray:
        """O'Brien-Fleming |z| boundary at each look, obf · sqrt(K / k)."""
        k = np.arange(1, self.n_looks + 1, dtype=float)
        return self.obf * np.sqrt(self.n_looks / k)


def calibrate_boundaries(
    n_looks: int,
    n_paths: int,
    rng: np.random.Generator,
    alpha: float = config.ALPHA,
    chunk: int = PATH_CHUNK,
) -> BoundaryConstants:
    """Calibrate Pocock and O'Brien-Fleming constants by simulating correlated null Z-paths.

    With equal increments, Z_k = S_k / sqrt(k) where S_k is a cumulative sum of k standard
    normals, so (Z_1, ..., Z_K) has the Brownian-motion correlation of a sequence of
    two-sample z-statistics. Pocock's constant is the (1 - alpha) quantile of max_k |Z_k|;
    O'Brien-Fleming's final boundary is the (1 - alpha) quantile of max_k |Z_k| sqrt(k / K),
    because the rule |Z_k| > c sqrt(K / k) is |S_k| / sqrt(K) > c.

    Parameters
    ----------
    n_looks : int
        K.
    n_paths : int
        Null paths; the quantile's Monte Carlo error falls as 1 / sqrt(n_paths).
    rng : numpy.random.Generator
        Random generator.
    alpha : float
        Overall two-sided error.
    chunk : int
        Paths simulated per batch (bounds memory).

    Returns
    -------
    BoundaryConstants
        Constants with Monte Carlo standard errors.
    """
    if n_looks < 1 or n_paths < 2:
        raise ValueError("Need at least one look and two paths.")
    k = np.arange(1, n_looks + 1, dtype=float)
    max_z = np.empty(n_paths)
    max_s = np.empty(n_paths)
    for start in range(0, n_paths, chunk):
        size = min(chunk, n_paths - start)
        s = np.abs(rng.standard_normal((size, n_looks)).cumsum(axis=1))
        max_z[start : start + size] = (s / np.sqrt(k)).max(axis=1)
        max_s[start : start + size] = s.max(axis=1) / math.sqrt(n_looks)
    max_z.sort()
    max_s.sort()
    q = 1 - alpha
    return BoundaryConstants(
        n_looks=n_looks,
        pocock=float(np.quantile(max_z, q)),
        obf=float(np.quantile(max_s, q)),
        pocock_mcse=_quantile_mcse(max_z, q),
        obf_mcse=_quantile_mcse(max_s, q),
        n_paths=n_paths,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------------------------
# Decision rules and their summaries
# ---------------------------------------------------------------------------------------------


def rejections(
    paths: Paths,
    constants: BoundaryConstants,
    tau: float,
    alpha: float = config.ALPHA,
    threshold: float = config.BAYES_P_BETTER,
    tau_multipliers: dict[str, float] | None = None,
    methods: Sequence[str] | None = None,
    bayes_per_run: bool = False,
) -> dict[str, np.ndarray]:
    """Apply every decision rule, or the ones asked for, to simulated experiments.

    Parameters
    ----------
    paths : Paths
        Simulated experiments.
    constants : BoundaryConstants
        Calibrated for ``paths.n_looks`` looks.
    tau : float
        mSPRT mixing standard deviation for the headline rule.
    alpha : float
        Two-sided level of the fixed, naive, Pocock, O'Brien-Fleming and mSPRT rules.
    threshold : float
        Bayesian P(B > A) threshold.
    tau_multipliers : dict, optional
        Method name -> factor applied to ``tau`` (default ``TAU_MULTIPLIERS``).
    methods : sequence of str, optional
        Apply only these rules (default: all of them). The rules are independent, so leaving
        some out changes nothing about the others.
    bayes_per_run : bool
        Evaluate the Bayesian rule for each run and look directly instead of through the
        count boundary. Same rule; faster when there are only a few thousand runs.

    Returns
    -------
    dict of numpy.ndarray
        Method -> boolean ``(runs, looks)`` array of looks at which the rule rejects (before
        any stopping; :func:`first_rejection` turns it into a stopping look).
    """
    if constants.n_looks != paths.n_looks:
        raise ValueError("Boundary constants were calibrated for a different number of looks.")
    diff, se = difference_and_se(paths)
    z = np.abs(z_statistics(diff, se))
    z_alpha = stats.norm.isf(alpha / 2)

    def fixed() -> np.ndarray:
        flags = np.zeros_like(z, dtype=bool)
        flags[:, -1] = z[:, -1] > z_alpha
        return flags

    rules: dict[str, Callable[[], np.ndarray]] = {
        "fixed": fixed,
        "naive": lambda: z > z_alpha,
        "bayes": lambda: _bayes_rejections(paths, threshold, bayes_per_run),
        "pocock": lambda: z > constants.pocock_boundary(),
        "obf": lambda: z > constants.obf_boundary(),
    }
    for name, factor in (tau_multipliers or TAU_MULTIPLIERS).items():
        rules[name] = lambda factor=factor: msprt_p_values(diff, se, factor * tau) <= alpha
    if methods is not None:
        unknown = sorted(set(methods) - set(rules))
        if unknown:
            raise ValueError(f"Unknown methods {unknown}; choose from {sorted(rules)}.")
        rules = {name: rules[name] for name in methods}
    return {name: rule() for name, rule in rules.items()}


def first_rejection(reject: np.ndarray) -> np.ndarray:
    """Index of the first look at which each run rejects, or -1 if it never does."""
    any_reject = reject.any(axis=1)
    return np.where(any_reject, reject.argmax(axis=1), -1)


def _rate_and_mcse(flags: np.ndarray) -> tuple[float, float]:
    rate = float(np.mean(flags))
    return rate, math.sqrt(rate * (1 - rate) / flags.size)


def _mean_and_se(x: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return (float(x.mean()) if x.size else math.nan), math.nan
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def summarise_rule(reject: np.ndarray) -> dict[str, float]:
    """Rejection rate and stopping day of one rule, with Monte Carlo standard errors.

    Parameters
    ----------
    reject : numpy.ndarray
        Boolean ``(runs, looks)`` array from :func:`rejections`.

    Returns
    -------
    dict
        ``reject_rate`` and ``reject_mcse``; ``mean_stop_day`` and its standard error, where a
        run that never rejects stops at the last look; ``mean_stop_day_rejecting`` and its
        standard error among the runs that reject (NaN if fewer than two do).
    """
    n_looks = reject.shape[1]
    stop = first_rejection(reject)
    rejected = stop >= 0
    rate, mcse = _rate_and_mcse(rejected)
    day = np.where(rejected, stop + 1, n_looks).astype(float)
    mean_day, mean_day_se = _mean_and_se(day)
    cond_day, cond_day_se = _mean_and_se(day[rejected])
    return {
        "reject_rate": rate,
        "reject_mcse": mcse,
        "mean_stop_day": mean_day,
        "mean_stop_day_se": mean_day_se,
        "mean_stop_day_rejecting": cond_day,
        "mean_stop_day_rejecting_se": cond_day_se,
    }


def cumulative_rejection(reject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of runs that have stopped by each look, with Monte Carlo standard errors."""
    stop = first_rejection(reject)
    stopped = (stop[:, None] >= 0) & (stop[:, None] <= np.arange(reject.shape[1]))
    rate = stopped.mean(axis=0)
    return rate, np.sqrt(rate * (1 - rate) / reject.shape[0])


# ---------------------------------------------------------------------------------------------
# Baselines from the data
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Baselines:
    """Control-arm inputs of the simulation.

    Attributes
    ----------
    n_per_arm : int
        Customers per arm (the smallest real arm).
    rates : dict
        Control rate per rate metric.
    variances : dict
        Control variance per metric (the power module's design input).
    means : dict
        Control mean per metric.
    spend_values : numpy.ndarray
        Spend of every control customer, resampled for the spend simulation.
    observed_visit_lift : float
        Visit rate of the better email arm minus the control's, the observed A/B effect.
    better_email : str
        The email arm with the higher visit rate.
    """

    n_per_arm: int
    rates: dict[str, float]
    variances: dict[str, float]
    means: dict[str, float]
    spend_values: np.ndarray
    observed_visit_lift: float
    better_email: str


def baselines(con: duckdb.DuckDBPyConnection) -> Baselines:
    """Read the control-arm baselines (and the observed visit lift) from the metric layer.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    Baselines
        Inputs of the simulation.
    """
    metrics = arm_metrics(con).set_index(["metric", "arm"])
    counts = arm_counts(con)
    visit = {t: float(metrics.loc[(HEADLINE_METRIC, t), "mean"]) for t in config.TREATMENTS}
    better = max(visit, key=visit.get)
    control_visit = float(metrics.loc[(HEADLINE_METRIC, config.CONTROL), "mean"])
    spend = con.execute(
        "SELECT spend FROM customers WHERE arm = ? ORDER BY customer_id", [config.CONTROL]
    ).fetchnumpy()["spend"]
    return Baselines(
        n_per_arm=min(counts.values()),
        rates={m: float(metrics.loc[(m, config.CONTROL), "mean"]) for m in RATE_METRICS},
        variances={m: float(metrics.loc[(m, config.CONTROL), "var"]) for m in METRICS},
        means={m: float(metrics.loc[(m, config.CONTROL), "mean"]) for m in METRICS},
        spend_values=np.asarray(spend, dtype=float),
        observed_visit_lift=visit[better] - control_visit,
        better_email=better,
    )


def mixing_scales(base: Baselines, alpha: float = config.ALPHA) -> dict[str, float]:
    """Choose the mSPRT mixing scale tau per metric: the MDE at 80% power and the full sample.

    The mixing distribution should put its mass on effects worth detecting. The MDE of the
    fixed-horizon design (:func:`abkit.power.mde`, the project's own "plausible effect") is
    that scale, computed from the control arm's variance and the customers per arm.
    """
    return {
        m: mde(base.means[m], base.variances[m], base.n_per_arm, alpha=alpha).absolute
        for m in METRICS
    }


def _simulate(
    base: Baselines, metric: str, effect: float, n_looks: int, n_runs: int, rng: np.random.Generator
) -> Paths:
    if metric == "spend":
        if effect != 0:
            raise ValueError("Spend is simulated under A/A only.")
        return simulate_spend_paths(base.spend_values, base.n_per_arm, n_looks, n_runs, rng)
    p = base.rates[metric]
    return simulate_rate_paths(p, p + effect, base.n_per_arm, n_looks, n_runs, rng)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def _method_label(name: str) -> str:
    return "msprt" if name in TAU_MULTIPLIERS else name


def build_sequential_tables(
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    n_runs: int = N_RUNS,
    n_runs_looks: int = N_RUNS_LOOKS,
    n_paths: int = N_PATHS,
    max_looks: int = MAX_LOOKS,
    days: int = DAYS,
    alpha: float = config.ALPHA,
    threshold: float = config.BAYES_P_BETTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the whole simulation study and build its tables.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    rng : numpy.random.Generator
        Random generator (all randomness flows through it).
    n_runs : int
        Simulated experiments per scenario at ``days`` daily looks.
    n_runs_looks : int
        Simulated A/A experiments per number of looks in the looks curve.
    n_paths : int
        Null paths per calibration of the group-sequential constants.
    max_looks : int
        The looks curve runs K = 1, ..., ``max_looks``.
    days : int
        Days of the stylised rollout (daily looks).
    alpha : float
        Two-sided level.
    threshold : float
        Bayesian P(B > A) threshold.

    Returns
    -------
    tuple
        ``main`` (one row per scenario, metric and method), ``looks`` (A/A false positive
        rate against the number of looks), ``by_day`` (cumulative rejection by day, plus the
        single-look rejection rate of the z-test on each day alone), ``boundaries`` (|z|
        boundaries of each rule at the daily looks) and a ``settings`` dict for results.json.
    """
    if days > max_looks:
        raise ValueError("max_looks must cover the daily design.")
    base = baselines(con)
    taus = mixing_scales(base, alpha)
    visit_mde = taus[HEADLINE_METRIC]
    constants = {k: calibrate_boundaries(k, n_paths, rng, alpha) for k in range(1, max_looks + 1)}
    logger.info(
        "Calibrated boundaries for K = 1..%d with %d paths; K = %d: Pocock %.3f, OBF %.3f.",
        max_looks,
        n_paths,
        days,
        constants[days].pocock,
        constants[days].obf,
    )

    scenarios = [(SCENARIO_AA, m, 0.0) for m in METRICS] + [
        (SCENARIO_AB_OBSERVED, HEADLINE_METRIC, base.observed_visit_lift),
        (SCENARIO_AB_MDE, HEADLINE_METRIC, visit_mde),
    ]
    main_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    for scenario, metric, effect in scenarios:
        paths = _simulate(base, metric, effect, days, n_runs, rng)
        rules = rejections(paths, constants[days], taus[metric], alpha, threshold)
        common = {
            "scenario": scenario,
            "metric": metric,
            "effect_absolute": effect,
            "effect_relative": effect / base.means[metric],
            "n_looks": days,
            "n_per_arm": int(paths.n[-1]),
            "n_runs": n_runs,
        }
        for name, reject in rules.items():
            main_rows.append(
                {
                    **common,
                    "method": _method_label(name),
                    "tau": TAU_MULTIPLIERS.get(name, math.nan) * taus[metric],
                    **summarise_rule(reject),
                }
            )
            if name in TAU_MULTIPLIERS and name != "msprt":
                continue
            rate, mcse = cumulative_rejection(reject)
            day_rows.extend(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "method": name,
                    "day": d + 1,
                    "n_per_arm": int(paths.n[d]),
                    "reject_rate": float(rate[d]),
                    "reject_mcse": float(mcse[d]),
                }
                for d in range(days)
            )
        # The z-test applied on each day alone, as if the experiment had ended there: the
        # small-sample check of the normal approximation (its rate should be alpha every day).
        single = rules["naive"].mean(axis=0)
        day_rows.extend(
            {
                "scenario": scenario,
                "metric": metric,
                "method": SINGLE_LOOK,
                "day": d + 1,
                "n_per_arm": int(paths.n[d]),
                "reject_rate": float(single[d]),
                "reject_mcse": math.sqrt(single[d] * (1 - single[d]) / n_runs),
            }
            for d in range(days)
        )
        logger.info("Scenario %s on %s (effect %.5f): done.", scenario, metric, effect)

    looks_rows: list[dict[str, Any]] = []
    for k in range(1, max_looks + 1):
        for metric in METRICS:
            paths = _simulate(base, metric, 0.0, k, n_runs_looks, rng)
            rules = rejections(paths, constants[k], taus[metric], alpha, threshold, {"msprt": 1.0})
            for name, reject in rules.items():
                rate, mcse = _rate_and_mcse(first_rejection(reject) >= 0)
                looks_rows.append(
                    {
                        "metric": metric,
                        "method": name,
                        "n_looks": k,
                        "n_per_arm": int(paths.n[-1]),
                        "n_runs": n_runs_looks,
                        "reject_rate": rate,
                        "reject_mcse": mcse,
                        "c_pocock": constants[k].pocock,
                        "c_obf": constants[k].obf,
                    }
                )
        logger.info("Looks curve: K = %d done.", k)

    c = constants[days]
    look = np.arange(1, days + 1)
    n_look = (base.n_per_arm // days) * look
    p = base.rates[HEADLINE_METRIC]
    boundaries = pd.DataFrame(
        {
            "n_looks": days,
            "look": look,
            "n_per_arm": n_look,
            "naive": stats.norm.isf(alpha / 2),
            "pocock": c.pocock_boundary(),
            "obf": c.obf_boundary(),
            "msprt_visit": msprt_z_boundary(2 * p * (1 - p) / n_look, visit_mde, alpha),
        }
    )
    settings = {
        "days": days,
        "n_per_arm": base.n_per_arm,
        "customers_per_arm_per_day": base.n_per_arm // days,
        "n_runs": n_runs,
        "n_runs_looks": n_runs_looks,
        "n_paths": n_paths,
        "max_looks": max_looks,
        "alpha": alpha,
        "p_better_threshold": threshold,
        "quadrature_nodes": QUAD_NODES,
        "baseline_arm": config.CONTROL,
        "baselines": {m: {"mean": base.means[m], "variance": base.variances[m]} for m in METRICS},
        "spend_positive_values": int((base.spend_values > 0).sum()),
        "effects": {
            SCENARIO_AB_OBSERVED: {
                "metric": HEADLINE_METRIC,
                "treatment": base.better_email,
                "absolute": base.observed_visit_lift,
                "relative": base.observed_visit_lift / base.means[HEADLINE_METRIC],
            },
            SCENARIO_AB_MDE: {
                "metric": HEADLINE_METRIC,
                "absolute": visit_mde,
                "relative": visit_mde / base.means[HEADLINE_METRIC],
            },
        },
        "tau": taus,
        "tau_multipliers": TAU_MULTIPLIERS,
        "constants": {
            str(k): {
                "pocock": constants[k].pocock,
                "pocock_mcse": constants[k].pocock_mcse,
                "obf": constants[k].obf,
                "obf_mcse": constants[k].obf_mcse,
            }
            for k in sorted({1, min(5, max_looks), days})
        },
        "published_k5_jennison_turnbull": PUBLISHED_K5,
    }
    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(looks_rows),
        pd.DataFrame(day_rows),
        boundaries,
        settings,
    )


def small_sample_check(looks: pd.DataFrame, by_day: pd.DataFrame) -> list[dict[str, Any]]:
    """Fixed-horizon A/A false positive rate per metric at K = 1 and on the first daily look.

    Both are a single two-sided z-test at level alpha, at the full sample (K = 1) and at one
    day's customers. A rate away from alpha here is the normal approximation failing at that
    sample size, not peeking.

    Parameters
    ----------
    looks : pandas.DataFrame
        Looks table from :func:`build_sequential_tables`.
    by_day : pandas.DataFrame
        By-day table from :func:`build_sequential_tables`.

    Returns
    -------
    list of dict
        One record per metric.
    """
    rows = []
    for metric in METRICS:
        full = looks[(looks["metric"] == metric) & (looks["method"] == "fixed")]
        full = full[full["n_looks"] == 1].iloc[0]
        first = by_day[
            (by_day["scenario"] == SCENARIO_AA)
            & (by_day["metric"] == metric)
            & (by_day["method"] == SINGLE_LOOK)
            & (by_day["day"] == 1)
        ].iloc[0]
        rows.append(
            {
                "metric": metric,
                "n_full": int(full["n_per_arm"]),
                "fpr_full": float(full["reject_rate"]),
                "fpr_full_mcse": float(full["reject_mcse"]),
                "n_first_look": int(first["n_per_arm"]),
                "fpr_first_look": float(first["reject_rate"]),
                "fpr_first_look_mcse": float(first["reject_mcse"]),
            }
        )
    return rows


def run_sequential(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    n_runs: int = N_RUNS,
    n_runs_looks: int = N_RUNS_LOOKS,
    n_paths: int = N_PATHS,
    max_looks: int = MAX_LOOKS,
) -> pd.DataFrame:
    """Run the peeking simulation study and write its tables, figures and results.json section.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    n_runs, n_runs_looks, n_paths, max_looks : int
        Simulation sizes; the defaults are the full study, tests pass smaller values.

    Returns
    -------
    pandas.DataFrame
        The main table (one row per scenario, metric and method).
    """
    rng = np.random.default_rng(config.SEEDS["simulation"])
    with duckdb.connect(str(db_path), read_only=True) as con:
        main, looks, by_day, boundaries, settings = build_sequential_tables(
            con, rng, n_runs, n_runs_looks, n_paths, max_looks
        )
    tables = reports_dir / "tables"
    figures = reports_dir / "figures"
    write_table(main, TABLE_NAME, tables)
    write_table(looks, LOOKS_TABLE_NAME, tables)
    write_table(by_day, BY_DAY_TABLE_NAME, tables)
    write_table(boundaries, BOUNDARIES_TABLE_NAME, tables)
    plots.plot_sequential_fpr_vs_looks(looks, figures / FPR_FIGURE_NAME)
    plots.plot_sequential_cumulative(by_day, figures / CUMULATIVE_FIGURE_NAME)
    plots.plot_sequential_boundaries(boundaries, figures / BOUNDARIES_FIGURE_NAME)
    update_results(
        "sequential",
        {
            **settings,
            "seed": config.SEEDS["simulation"],
            "table": f"tables/{TABLE_NAME}.csv",
            "looks_table": f"tables/{LOOKS_TABLE_NAME}.csv",
            "by_day_table": f"tables/{BY_DAY_TABLE_NAME}.csv",
            "boundaries_table": f"tables/{BOUNDARIES_TABLE_NAME}.csv",
            "figures": {
                "fpr_vs_looks": f"figures/{FPR_FIGURE_NAME}",
                "cumulative_rejection": f"figures/{CUMULATIVE_FIGURE_NAME}",
                "boundaries": f"figures/{BOUNDARIES_FIGURE_NAME}",
            },
            "small_sample_check": small_sample_check(looks, by_day),
            "rows": json_records(main),
        },
        reports_dir / "results.json",
    )
    logger.info("Sequential tables written: %d main rows, %d looks rows.", len(main), len(looks))
    return main
