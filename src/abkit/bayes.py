"""Bayesian analysis: rates, spend per customer, decision rules and prior sensitivity.

Rates (conversion, visit): each arm gets a conjugate Beta posterior (analysis plan §9).
Comparisons are summarised from posterior draws, each Monte Carlo quantity with its Monte
Carlo standard error, and P(p_B > p_A) is also computed exactly: by Miller's closed form when
the parameters are integers (the test oracle) and by numerical integration otherwise. The
readout applies the pre-registered decision rule and repeats everything under a flat prior
and under shared informative priors of increasing strength.

Spend per customer (the primary metric): the Bayesian bootstrap (Rubin 1981), collapsed over
distinct values so that the ~99% of customers who spend nothing cost one Dirichlet component,
cross-checked against a parametric two-part model (Beta-Binomial conversion x log-normal
spend among converters with a Normal-Inverse-Gamma prior) and a posterior predictive check of
that log-normal against the observed converters, including the heap at the $499 top-code.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from abkit import config, plots
from abkit.eda import SRMError
from abkit.frequentist import _quantile_mcse, srm_test
from abkit.metrics import arm_counts, arm_metrics, run_sql
from abkit.readout import COMPARISONS, RATE_METRICS, SPEND_WINSORISED
from abkit.report import json_records, update_results, write_table

logger = logging.getLogger(__name__)

N_DRAWS: int = 200_000
# Pseudo-customers in the shared informative priors (plan §9 and Deviation D3).
PRIOR_STRENGTHS: tuple[int, ...] = (100, 1_000, 10_000, 100_000)
SHIP, DONT_SHIP, KEEP_TESTING = "ship", "don't ship", "keep testing"
TABLE_NAME = "readout_bayes_rates"
FIGURE_NAME = "bayes_p_better_by_prior.png"


# ---------------------------------------------------------------------------------------------
# Priors and posteriors
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaPrior:
    """Beta(alpha, beta) prior on a rate.

    Attributes
    ----------
    alpha, beta : float
        Shape parameters; ``alpha + beta`` is the strength in pseudo-customers.
    label : str
        Short name used in tables (``"flat"`` or ``"shared_<strength>"``).
    """

    alpha: float
    beta: float
    label: str

    @property
    def strength(self) -> float:
        """Prior strength ``alpha + beta`` in pseudo-customers."""
        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        """Prior mean ``alpha / (alpha + beta)``."""
        return self.alpha / self.strength


FLAT_PRIOR = BetaPrior(1.0, 1.0, "flat")


@dataclass(frozen=True)
class BetaPosterior:
    """Beta(alpha, beta) posterior of one arm's rate.

    Attributes
    ----------
    alpha, beta : float
        Prior shape plus successes and failures respectively.
    """

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Posterior mean ``alpha / (alpha + beta)``."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        """Posterior variance ``alpha beta / ((alpha + beta)^2 (alpha + beta + 1))``."""
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s**2 * (s + 1))


def shared_prior(pooled_rate: float, strength: float) -> BetaPrior:
    """Beta prior with a given mean and strength, shared by every arm.

    alpha = strength x pooled_rate and beta = strength x (1 - pooled_rate), so the prior is
    worth ``strength`` pseudo-customers at the pooled rate. Giving the same prior to every arm
    pulls all of them toward the same value and so shrinks differences toward zero.

    Parameters
    ----------
    pooled_rate : float
        Prior mean, the rate pooled across all arms (analysis plan §9).
    strength : float
        Pseudo-customers, ``alpha + beta``.

    Returns
    -------
    BetaPrior
        Labelled ``shared_<strength>``.

    Raises
    ------
    ValueError
        If the rate is not strictly between 0 and 1 or the strength is not positive.
    """
    if not 0 < pooled_rate < 1:
        raise ValueError("pooled_rate must lie strictly between 0 and 1.")
    if not strength > 0:
        raise ValueError("strength must be positive.")
    return BetaPrior(strength * pooled_rate, strength * (1 - pooled_rate), f"shared_{strength:g}")


def posterior(prior: BetaPrior, successes: int, n: int) -> BetaPosterior:
    """Conjugate update: Beta(alpha + successes, beta + failures).

    Parameters
    ----------
    prior : BetaPrior
        Prior on the arm's rate.
    successes, n : int
        Customers with the outcome and customers in the arm.

    Returns
    -------
    BetaPosterior
        The arm's posterior.

    Raises
    ------
    ValueError
        If the counts are negative or ``successes`` exceeds ``n``.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("Need 0 <= successes <= n.")
    return BetaPosterior(prior.alpha + successes, prior.beta + (n - successes))


# ---------------------------------------------------------------------------------------------
# Exact P(p_B > p_A)
# ---------------------------------------------------------------------------------------------


def prob_b_greater_a_exact(alpha_a: float, beta_a: float, alpha_b: int, beta_b: float) -> float:
    """P(p_B > p_A) for independent Betas, in closed form (Miller 2015).

    P(p_B > p_A) = Σ_{i=0}^{alpha_B - 1} B(alpha_A + i, beta_A + beta_B)
    / ((beta_B + i) B(1 + i, beta_B) B(alpha_A, beta_A)), which holds when ``alpha_b`` is a
    positive integer; the other parameters may be any positive reals. The terms are summed on
    the log scale, so it is stable for the tens of thousands of failures seen here.

    Parameters
    ----------
    alpha_a, beta_a : float
        Parameters of p_A.
    alpha_b : int
        Shape of p_B; must be a positive integer.
    beta_b : float
        Second shape of p_B.

    Returns
    -------
    float
        P(p_B > p_A).

    Raises
    ------
    ValueError
        If ``alpha_b`` is not a positive integer or any parameter is not positive.
    """
    if min(alpha_a, beta_a, alpha_b, beta_b) <= 0:
        raise ValueError("All Beta parameters must be positive.")
    if not float(alpha_b).is_integer():
        raise ValueError("The closed form needs an integer alpha_b; use prob_b_greater_a_quad.")
    i = np.arange(int(alpha_b), dtype=float)
    log_terms = (
        special.betaln(alpha_a + i, beta_a + beta_b)
        - np.log(beta_b + i)
        - special.betaln(1 + i, beta_b)
        - special.betaln(alpha_a, beta_a)
    )
    return float(np.exp(special.logsumexp(log_terms)))


def prob_b_greater_a_quad(alpha_a: float, beta_a: float, alpha_b: float, beta_b: float) -> float:
    """P(p_B > p_A) for independent Betas by numerical integration, for any parameters.

    P(p_B > p_A) = ∫ f_B(x) F_A(x) dx over the support of p_B, where f_B is the density of
    p_B and F_A the distribution function of p_A. The integral runs between the 1e-15 and
    1 - 1e-15 quantiles of p_B (truncation error below 2e-15) with adaptive quadrature.

    Parameters
    ----------
    alpha_a, beta_a, alpha_b, beta_b : float
        Parameters of p_A and p_B.

    Returns
    -------
    float
        P(p_B > p_A), accurate to about 1e-10.
    """
    if min(alpha_a, beta_a, alpha_b, beta_b) <= 0:
        raise ValueError("All Beta parameters must be positive.")
    dist_a, dist_b = stats.beta(alpha_a, beta_a), stats.beta(alpha_b, beta_b)
    lo, hi = dist_b.ppf(1e-15), dist_b.isf(1e-15)
    value, _ = integrate.quad(
        lambda x: dist_b.pdf(x) * dist_a.cdf(x), lo, hi, epsabs=1e-12, epsrel=1e-10, limit=200
    )
    return float(value)


# ---------------------------------------------------------------------------------------------
# Monte Carlo comparison of two arms
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorComparison:
    """Posterior summary of theta_t - theta_c from ``n_draws`` joint posterior draws.

    Used for rates (theta = a Beta-distributed rate) and for spend (theta = mean spend per customer
    under the Bayesian bootstrap or the two-part model).

    Attributes
    ----------
    diff_mean, diff_sd : float
        Exact posterior mean and standard deviation of the difference when a closed form
        exists (Beta formulas for rates; Dirichlet formulas for the Bayesian bootstrap),
        otherwise NaN (two-part model, whose posterior mean has no finite closed form).
    diff_mc_mean, diff_mc_sd, diff_mcse : float
        Monte Carlo mean and standard deviation of the difference, and the Monte Carlo
        standard error of the mean, ``diff_mc_sd / sqrt(n_draws)``.
    diff_median : float
        Monte Carlo median of the difference.
    ci_low, ci_high, ci_mcse_low, ci_mcse_high : float
        Equal-tailed credible interval for the difference and the Monte Carlo standard
        errors of its endpoints.
    rel_lift_mean, rel_lift_mcse : float
        Posterior mean of the relative lift ``theta_t / theta_c - 1`` and its Monte Carlo SE.
    rel_ci_low, rel_ci_high, rel_ci_mcse_low, rel_ci_mcse_high : float
        Credible interval for the relative lift and the Monte Carlo SEs of its endpoints.
    p_better, p_better_mcse : float
        P(theta_t > theta_c) and its Monte Carlo SE, ``sqrt(p~ (1 - p~) / n_draws)`` with the
        Agresti-Coull proportion p~ = (k + 2) / (n_draws + 4), which stays positive when every
        draw agrees (the plug-in SE would be exactly zero there).
    loss_ship, loss_ship_mcse : float
        Expected loss of shipping the treatment, E[max(theta_c - theta_t, 0)], with its Monte
        Carlo SE.
    loss_keep, loss_keep_mcse : float
        Expected loss of keeping control, E[max(theta_t - theta_c, 0)], with its Monte Carlo
        SE.
    n_draws : int
        Posterior draws per arm.
    level : float
        Credible level of the intervals.
    """

    diff_mean: float
    diff_sd: float
    diff_mc_mean: float
    diff_mc_sd: float
    diff_mcse: float
    diff_median: float
    ci_low: float
    ci_high: float
    ci_mcse_low: float
    ci_mcse_high: float
    rel_lift_mean: float
    rel_lift_mcse: float
    rel_ci_low: float
    rel_ci_high: float
    rel_ci_mcse_low: float
    rel_ci_mcse_high: float
    p_better: float
    p_better_mcse: float
    loss_ship: float
    loss_ship_mcse: float
    loss_keep: float
    loss_keep_mcse: float
    n_draws: int
    level: float


def _mean_and_mcse(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def _interval(x: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    """Equal-tailed interval of a sample with the Monte Carlo SE of each endpoint."""
    s = np.sort(x)
    q_low, q_high = alpha / 2, 1 - alpha / 2
    return (
        float(np.quantile(s, q_low)),
        float(np.quantile(s, q_high)),
        _quantile_mcse(s, q_low),
        _quantile_mcse(s, q_high),
    )


def compare_rates(
    post_t: BetaPosterior,
    post_c: BetaPosterior,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
    alpha: float = config.ALPHA,
) -> PosteriorComparison:
    """Summarise the posterior of p_t - p_c from independent posterior draws.

    Draws p_t then p_c from ``rng`` (``n_draws`` each). The difference's exact posterior mean
    and standard deviation come from the Beta formulas; the credible intervals, P(better),
    the relative lift and both expected losses come from the draws, each with its Monte
    Carlo standard error (mean: sd / sqrt(n); probability: sqrt(p~ (1 - p~) / n) with the
    Agresti-Coull p~ = (k + 2) / (n + 4); quantile: order statistics at n q ± sqrt(n q (1 - q))).

    Parameters
    ----------
    post_t, post_c : BetaPosterior
        Posteriors of the treatment and control rates.
    rng : numpy.random.Generator
        Source of randomness.
    n_draws : int
        Draws per arm (at least 2).
    alpha : float
        The intervals have credible level 1 - alpha.

    Returns
    -------
    PosteriorComparison
        Posterior summaries.

    Raises
    ------
    ValueError
        If ``n_draws`` is below 2.
    """
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2.")
    pt = rng.beta(post_t.alpha, post_t.beta, n_draws)
    pc = rng.beta(post_c.alpha, post_c.beta, n_draws)
    return summarise_draws(
        pt, pc, post_t.mean - post_c.mean, float(np.sqrt(post_t.var + post_c.var)), alpha
    )


def summarise_draws(
    draws_t: np.ndarray,
    draws_c: np.ndarray,
    exact_mean: float = np.nan,
    exact_sd: float = np.nan,
    alpha: float = config.ALPHA,
) -> PosteriorComparison:
    """Summarise theta_t - theta_c from paired posterior draws of the two arms.

    Parameters
    ----------
    draws_t, draws_c : numpy.ndarray
        Posterior draws of the treatment and control quantity, same length (at least 2).
    exact_mean, exact_sd : float
        Closed-form posterior mean and standard deviation of the difference, if known.
    alpha : float
        The intervals have credible level 1 - alpha.

    Returns
    -------
    PosteriorComparison
        Posterior summaries with Monte Carlo standard errors.

    Raises
    ------
    ValueError
        If the draws differ in length or there are fewer than 2.
    """
    if draws_t.shape != draws_c.shape or draws_t.ndim != 1 or draws_t.size < 2:
        raise ValueError("draws_t and draws_c must be 1-d arrays of equal length >= 2.")
    n_draws = draws_t.size
    diff = draws_t - draws_c
    rel = draws_t / draws_c - 1
    diff_mc_mean, diff_mcse = _mean_and_mcse(diff)
    ci_low, ci_high, ci_mcse_low, ci_mcse_high = _interval(diff, alpha)
    rel_mean, rel_mcse = _mean_and_mcse(rel)
    rel_low, rel_high, rel_mcse_low, rel_mcse_high = _interval(rel, alpha)
    better = int((diff > 0).sum())
    p_better = better / n_draws
    p_tilde = (better + 2) / (n_draws + 4)
    loss_ship, loss_ship_mcse = _mean_and_mcse(np.maximum(-diff, 0.0))
    loss_keep, loss_keep_mcse = _mean_and_mcse(np.maximum(diff, 0.0))
    return PosteriorComparison(
        diff_mean=float(exact_mean),
        diff_sd=float(exact_sd),
        diff_mc_mean=diff_mc_mean,
        diff_mc_sd=float(diff.std(ddof=1)),
        diff_mcse=diff_mcse,
        diff_median=float(np.median(diff)),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_mcse_low=ci_mcse_low,
        ci_mcse_high=ci_mcse_high,
        rel_lift_mean=rel_mean,
        rel_lift_mcse=rel_mcse,
        rel_ci_low=rel_low,
        rel_ci_high=rel_high,
        rel_ci_mcse_low=rel_mcse_low,
        rel_ci_mcse_high=rel_mcse_high,
        p_better=p_better,
        p_better_mcse=float(np.sqrt(p_tilde * (1 - p_tilde) / n_draws)),
        loss_ship=loss_ship,
        loss_ship_mcse=loss_ship_mcse,
        loss_keep=loss_keep,
        loss_keep_mcse=loss_keep_mcse,
        n_draws=n_draws,
        level=1 - alpha,
    )


# ---------------------------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """Outcome of the decision rule.

    Attributes
    ----------
    decision : str
        ``"ship"``, ``"don't ship"`` or ``"keep testing"``.
    reason : str
        The conditions that produced it, with their values.
    """

    decision: str
    reason: str


def decide(
    p_better: float,
    loss_ship: float,
    loss_keep: float,
    loss_threshold: float,
    p_threshold: float = config.BAYES_P_BETTER,
) -> Decision:
    """Apply the decision rule of analysis plan §9.

    - **ship** if P(treatment better) >= ``p_threshold`` and the expected loss of shipping is
      below ``loss_threshold`` (1% of the control arm's mean).
    - **don't ship** if the mirror image holds: P(control better) >= ``p_threshold`` and the
      expected loss of keeping control is below ``loss_threshold``.
    - **keep testing** otherwise: the data settle it in neither direction.

    Parameters
    ----------
    p_better : float
        P(p_t > p_c).
    loss_ship, loss_keep : float
        Expected loss of shipping the treatment and of keeping control.
    loss_threshold : float
        Largest acceptable expected loss, in the metric's units.
    p_threshold : float
        Required probability of being better.

    Returns
    -------
    Decision
        The decision and its reason.
    """
    p_pct, thr = f"{p_better:.3f}", f"{loss_threshold:.3g}"
    if p_better >= p_threshold and loss_ship < loss_threshold:
        return Decision(
            SHIP,
            f"P(better) = {p_pct} >= {p_threshold} and expected loss of shipping "
            f"{loss_ship:.3g} < {thr}",
        )
    if 1 - p_better >= p_threshold and loss_keep < loss_threshold:
        return Decision(
            DONT_SHIP,
            f"P(control better) = {1 - p_better:.3f} >= {p_threshold} and expected loss of "
            f"keeping control {loss_keep:.3g} < {thr}",
        )
    if p_better >= p_threshold:
        reason = (
            f"P(better) = {p_pct} >= {p_threshold} but expected loss of shipping "
            f"{loss_ship:.3g} >= {thr}"
        )
    elif 1 - p_better >= p_threshold:
        reason = (
            f"P(control better) = {1 - p_better:.3f} >= {p_threshold} but expected loss of "
            f"keeping control {loss_keep:.3g} >= {thr}"
        )
    else:
        reason = f"P(better) = {p_pct} is between {1 - p_threshold:.2f} and {p_threshold}"
    return Decision(KEEP_TESTING, reason)


# ---------------------------------------------------------------------------------------------
# Readout: conversion and visit, every comparison, every prior
# ---------------------------------------------------------------------------------------------


def rate_priors(
    pooled_rate: float, strengths: tuple[int, ...] = PRIOR_STRENGTHS
) -> list[BetaPrior]:
    """List the flat prior followed by the shared priors of each strength.

    Parameters
    ----------
    pooled_rate : float
        Mean of the shared priors.
    strengths : tuple of int
        Pseudo-customers of each shared prior.

    Returns
    -------
    list of BetaPrior
        ``[flat, shared_<s1>, shared_<s2>, ...]``.
    """
    return [FLAT_PRIOR, *(shared_prior(pooled_rate, s) for s in strengths)]


def build_bayes_rates(
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
    strengths: tuple[int, ...] = PRIOR_STRENGTHS,
    alpha: float = config.ALPHA,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Beta-Binomial readout of conversion and visit for every comparison and prior.

    Successes and customers per arm come from the SQL metric layer. For each rate metric, the
    pooled rate across all arms sets the mean of the shared priors; then for each prior
    (flat first) and each comparison in :data:`abkit.readout.COMPARISONS` the two posteriors
    are compared with :func:`compare_rates`, P(better) is also computed exactly by
    quadrature, and :func:`decide` is applied with the loss threshold
    ``config.BAYES_LOSS_FRACTION`` x the observed control-arm rate.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    rng : numpy.random.Generator
        Randomness, consumed in row order (metric, then prior, then comparison).
    n_draws : int
        Posterior draws per arm and row.
    strengths : tuple of int
        Shared-prior strengths.
    alpha : float
        Intervals have credible level 1 - alpha.

    Returns
    -------
    tuple
        (table with one row per metric, prior and comparison; pooled rate per metric).
    """
    metrics = arm_metrics(con).set_index(["metric", "arm"])
    pooled: dict[str, float] = {}
    rows = []
    for metric in RATE_METRICS:
        st = metrics.loc[metric]
        pooled[metric] = float(st["sum"].sum() / st["n"].sum())
        for prior in rate_priors(pooled[metric], strengths):
            for treatment, control in COMPARISONS:
                t, c = st.loc[treatment], st.loc[control]
                x_t, n_t = round(t["sum"]), int(t["n"])
                x_c, n_c = round(c["sum"]), int(c["n"])
                post_t, post_c = posterior(prior, x_t, n_t), posterior(prior, x_c, n_c)
                res = compare_rates(post_t, post_c, rng, n_draws, alpha)
                rate_c = x_c / n_c
                threshold = config.BAYES_LOSS_FRACTION * rate_c
                verdict = decide(res.p_better, res.loss_ship, res.loss_keep, threshold)
                rows.append(
                    {
                        "comparison": f"{treatment}_vs_{control}",
                        "treatment": treatment,
                        "control": control,
                        "metric": metric,
                        "prior": prior.label,
                        "prior_strength": prior.strength,
                        "prior_mean": prior.mean,
                        "prior_alpha": prior.alpha,
                        "prior_beta": prior.beta,
                        "n_treatment": n_t,
                        "successes_treatment": x_t,
                        "n_control": n_c,
                        "successes_control": x_c,
                        "rate_treatment": x_t / n_t,
                        "rate_control": rate_c,
                        "post_alpha_treatment": post_t.alpha,
                        "post_beta_treatment": post_t.beta,
                        "post_alpha_control": post_c.alpha,
                        "post_beta_control": post_c.beta,
                        "post_mean_treatment": post_t.mean,
                        "post_mean_control": post_c.mean,
                        "diff_mean": res.diff_mean,
                        "diff_sd": res.diff_sd,
                        "diff_mc_mean": res.diff_mc_mean,
                        "diff_mcse": res.diff_mcse,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                        "ci_mcse_low": res.ci_mcse_low,
                        "ci_mcse_high": res.ci_mcse_high,
                        "rel_lift_mean": res.rel_lift_mean,
                        "rel_lift_mcse": res.rel_lift_mcse,
                        "rel_ci_low": res.rel_ci_low,
                        "rel_ci_high": res.rel_ci_high,
                        "rel_ci_mcse_low": res.rel_ci_mcse_low,
                        "rel_ci_mcse_high": res.rel_ci_mcse_high,
                        "p_better": res.p_better,
                        "p_better_mcse": res.p_better_mcse,
                        "p_better_exact": prob_b_greater_a_quad(
                            post_c.alpha, post_c.beta, post_t.alpha, post_t.beta
                        ),
                        "loss_ship": res.loss_ship,
                        "loss_ship_mcse": res.loss_ship_mcse,
                        "loss_keep": res.loss_keep,
                        "loss_keep_mcse": res.loss_keep_mcse,
                        "loss_threshold": threshold,
                        "decision": verdict.decision,
                        "reason": verdict.reason,
                        "n_draws": n_draws,
                        "level": res.level,
                    }
                )
    return pd.DataFrame(rows), pooled


def run_bayes_rates(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    n_draws: int = N_DRAWS,
) -> pd.DataFrame:
    """Run the Beta-Binomial readout and write its table, results.json section and figure.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    n_draws : int
        Posterior draws per arm and row.

    Returns
    -------
    pandas.DataFrame
        The readout table.

    Raises
    ------
    SRMError
        If the sample ratio mismatch check fails (plan §5 blocks the readout).
    """
    rng = np.random.default_rng(config.SEEDS["bayes"])
    with duckdb.connect(str(db_path), read_only=True) as con:
        srm = srm_test(arm_counts(con))
        if not srm.passed:
            raise SRMError(f"SRM check failed (p = {srm.p_value:.2e}); readout blocked.")
        table, pooled = build_bayes_rates(con, rng, n_draws)

    write_table(table, TABLE_NAME, reports_dir / "tables")
    plots.plot_p_better_by_prior(table, reports_dir / "figures" / FIGURE_NAME)
    priors = table.drop_duplicates("prior")[["prior", "prior_strength"]]
    update_results(
        "bayes_rates",
        {
            "model": "Beta-Binomial, conjugate Beta posterior per arm",
            "n_draws": n_draws,
            "seed": config.SEEDS["bayes"],
            "level": 1 - config.ALPHA,
            "priors": [
                {"label": str(p), "strength": float(s)}
                for p, s in zip(priors["prior"], priors["prior_strength"], strict=True)
            ],
            "shared_prior_mean": "rate pooled across all arms",
            "pooled_rates": pooled,
            "decision_rule": {
                "p_better_threshold": config.BAYES_P_BETTER,
                "loss_fraction_of_control_rate": config.BAYES_LOSS_FRACTION,
            },
            "table": f"tables/{TABLE_NAME}.csv",
            "figure": f"figures/{FIGURE_NAME}",
            "rows": json_records(table),
        },
        reports_dir / "results.json",
    )
    logger.info("Bayesian rate readout written: %d rows.", len(table))
    return table


# ---------------------------------------------------------------------------------------------
# Spend per customer: Bayesian bootstrap
# ---------------------------------------------------------------------------------------------

SPEND_TABLE_NAME = "readout_bayes_spend"
SPEND_PPC_TABLE_NAME = "bayes_spend_ppc"
SPEND_HIST_TABLE_NAME = "bayes_spend_posterior_hist"
SPEND_FIGURE_NAME = "bayes_spend_diff_posterior.png"
BOOTSTRAP, TWO_PART = "bayesian_bootstrap", "two_part_lognormal"
# Incremental revenue is also expressed per this many emails sent (the Q3 budget size).
REVENUE_SCALE: int = config.BUDGET
N_PPC_REPLICATES: int = 4_000
# Posterior-predictive quantiles of converters' spend compared with the observed ones.
PPC_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)


def bayesian_bootstrap_means(
    y: np.ndarray, rng: np.random.Generator, n_draws: int, chunk_cells: int = 20_000_000
) -> np.ndarray:
    """Posterior draws of the mean of ``y`` under the Bayesian bootstrap (Rubin 1981).

    The Bayesian bootstrap puts weights w ~ Dirichlet(1, ..., 1) on the n observations and
    takes Σ w_i y_i as a posterior draw of the population mean. This implementation collapses
    the observations by distinct value: if the value v occurs c_v times, the total weight on
    it is W_v = Σ_{i: y_i = v} w_i, and Σ w_i y_i = Σ_v W_v v. By the aggregation property of
    the Dirichlet distribution, summing the components of a Dirichlet(alpha_1, ..., alpha_n) over
    disjoint groups gives a Dirichlet whose parameters are the group sums, so
    (W_v)_v ~ Dirichlet(c_v)_v **exactly**. For spend, where about 99% of customers have
    y = 0, the zero-spend customers collapse into one component with parameter n_zero (and
    it carries no spend, so only its share of the total weight matters); draws are needed
    only for the distinct positive values plus that one component. Ties among converters
    (the $29.99 floor, the $499 cap) collapse the same way. The result is the same
    distribution as the naive n-dimensional Dirichlet, at a tiny fraction of the cost.

    Parameters
    ----------
    y : numpy.ndarray
        Observations (one arm's row-level spend).
    rng : numpy.random.Generator
        Source of randomness.
    n_draws : int
        Posterior draws.
    chunk_cells : int
        Memory bound: Dirichlet draws are generated in chunks of at most this many numbers.

    Returns
    -------
    numpy.ndarray
        ``n_draws`` posterior draws of the mean.

    Raises
    ------
    ValueError
        If ``y`` is empty or ``n_draws`` is below 1.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or n_draws < 1:
        raise ValueError("Need at least one observation and one draw.")
    values, counts = np.unique(y, return_counts=True)
    alpha = counts.astype(float)
    per_chunk = max(1, chunk_cells // values.size)
    out = np.empty(n_draws)
    for start in range(0, n_draws, per_chunk):
        size = min(per_chunk, n_draws - start)
        out[start : start + size] = rng.dirichlet(alpha, size=size) @ values
    return out


def bayesian_bootstrap_means_naive(
    y: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    """Draw the textbook Bayesian bootstrap, one Dirichlet(1, ..., 1) weight per observation.

    This is the textbook form, O(n) numbers per draw; it exists to test
    :func:`bayesian_bootstrap_means` on small data and should not be used on the real arms.

    Parameters
    ----------
    y : numpy.ndarray
        Observations.
    rng : numpy.random.Generator
        Source of randomness.
    n_draws : int
        Posterior draws.

    Returns
    -------
    numpy.ndarray
        ``n_draws`` posterior draws of the mean.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or n_draws < 1:
        raise ValueError("Need at least one observation and one draw.")
    return rng.dirichlet(np.ones(y.size), size=n_draws) @ y


def bayesian_bootstrap_moments(y: np.ndarray) -> tuple[float, float]:
    """Return the exact posterior mean and standard deviation of the Bayesian-bootstrap mean.

    With w ~ Dirichlet(1, ..., 1), E[w_i] = 1/n, Var(w_i) = (n - 1) / (n² (n + 1)) and
    Cov(w_i, w_j) = -1 / (n² (n + 1)), so E[Σ w_i y_i] = ȳ and
    Var(Σ w_i y_i) = Σ (y_i - ȳ)² / (n (n + 1)): the squared standard error of the mean
    times (n - 1) / (n + 1).

    Parameters
    ----------
    y : numpy.ndarray
        Observations.

    Returns
    -------
    tuple of float
        (posterior mean, posterior standard deviation).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("Need at least one observation.")
    ybar = float(y.mean())
    return ybar, float(np.sqrt(((y - ybar) ** 2).sum() / (n * (n + 1))))


def compare_means(
    draws_t: np.ndarray,
    draws_c: np.ndarray,
    y_t: np.ndarray,
    y_c: np.ndarray,
    alpha: float = config.ALPHA,
) -> PosteriorComparison:
    """Summarise the Bayesian-bootstrap posterior of mean(y_t) - mean(y_c).

    The draws come from :func:`bayesian_bootstrap_means` (one set per arm, independent); the
    exact posterior mean and standard deviation of the difference come from
    :func:`bayesian_bootstrap_moments`.

    Parameters
    ----------
    draws_t, draws_c : numpy.ndarray
        Posterior draws of each arm's mean.
    y_t, y_c : numpy.ndarray
        The row-level observations the draws were generated from.
    alpha : float
        The intervals have credible level 1 - alpha.

    Returns
    -------
    PosteriorComparison
        Posterior summaries.
    """
    mean_t, sd_t = bayesian_bootstrap_moments(y_t)
    mean_c, sd_c = bayesian_bootstrap_moments(y_c)
    return summarise_draws(draws_t, draws_c, mean_t - mean_c, float(np.hypot(sd_t, sd_c)), alpha)


# ---------------------------------------------------------------------------------------------
# Spend per customer: parametric two-part model (cross-check)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NIGPrior:
    """Normal-Inverse-Gamma prior on (mu, sigma²) of a normal sample.

    sigma² ~ Inverse-Gamma(alpha0, beta0) and mu | sigma² ~ Normal(mu0, sigma² / kappa0).
    ``kappa0`` is the prior's weight in pseudo-observations for the mean and ``2 alpha0`` for
    the variance.

    Attributes
    ----------
    mu0, kappa0, alpha0, beta0 : float
        Prior parameters; ``kappa0``, ``alpha0`` and ``beta0`` must be positive.
    """

    mu0: float
    kappa0: float
    alpha0: float
    beta0: float


# Weakly informative: a hundredth of a pseudo-observation, the same for every arm.
WEAK_NIG_PRIOR = NIGPrior(mu0=0.0, kappa0=0.01, alpha0=0.01, beta0=0.01)


@dataclass(frozen=True)
class NIGPosterior:
    """Normal-Inverse-Gamma posterior (same parametrisation as :class:`NIGPrior`).

    Attributes
    ----------
    mu_n, kappa_n, alpha_n, beta_n : float
        Posterior parameters.
    n : int
        Observations used in the update.
    """

    mu_n: float
    kappa_n: float
    alpha_n: float
    beta_n: float
    n: int

    @property
    def mu_mean(self) -> float:
        """Posterior mean of mu (``mu_n``)."""
        return self.mu_n

    @property
    def mu_sd(self) -> float:
        """Posterior standard deviation of mu: its marginal is Student-t with 2 alpha_n dof."""
        dof = 2 * self.alpha_n
        if dof <= 2:
            return float("nan")
        return float(np.sqrt(self.beta_n / (self.alpha_n * self.kappa_n) * dof / (dof - 2)))

    @property
    def mu_scale(self) -> float:
        """Scale of the Student-t marginal of mu: sqrt(beta_n / (alpha_n kappa_n))."""
        return float(np.sqrt(self.beta_n / (self.alpha_n * self.kappa_n)))

    @property
    def sigma2_mean(self) -> float:
        """Posterior mean of sigma²: beta_n / (alpha_n - 1) (infinite if alpha_n <= 1)."""
        return self.beta_n / (self.alpha_n - 1) if self.alpha_n > 1 else float("inf")


def nig_posterior(prior: NIGPrior, x: np.ndarray) -> NIGPosterior:
    """Conjugate Normal-Inverse-Gamma update for a normal sample.

    With n observations of mean x̄ and sum of squares S = Σ (x_i - x̄)²:
    kappa_n = kappa0 + n, mu_n = (kappa0 mu0 + n x̄) / kappa_n, alpha_n = alpha0 + n / 2,
    beta_n = beta0 + S / 2 + kappa0 n (x̄ - mu0)² / (2 kappa_n).

    Parameters
    ----------
    prior : NIGPrior
        Prior.
    x : numpy.ndarray
        Observations (here log spend among converters).

    Returns
    -------
    NIGPosterior
        The posterior.

    Raises
    ------
    ValueError
        If the prior parameters are not positive or ``x`` is empty.
    """
    if min(prior.kappa0, prior.alpha0, prior.beta0) <= 0:
        raise ValueError("kappa0, alpha0 and beta0 must be positive.")
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("Need at least one observation.")
    xbar = float(x.mean())
    ss = float(((x - xbar) ** 2).sum())
    kappa_n = prior.kappa0 + n
    mu_n = (prior.kappa0 * prior.mu0 + n * xbar) / kappa_n
    alpha_n = prior.alpha0 + n / 2
    beta_n = prior.beta0 + ss / 2 + prior.kappa0 * n * (xbar - prior.mu0) ** 2 / (2 * kappa_n)
    return NIGPosterior(mu_n, kappa_n, alpha_n, beta_n, n)


def draw_nig(
    post: NIGPosterior, rng: np.random.Generator, n_draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Joint posterior draws of (mu, sigma²): sigma² ~ Inverse-Gamma, then mu | sigma² ~ Normal.

    Parameters
    ----------
    post : NIGPosterior
        Posterior.
    rng : numpy.random.Generator
        Source of randomness (sigma² drawn first, then mu).
    n_draws : int
        Draws.

    Returns
    -------
    tuple of numpy.ndarray
        (mu draws, sigma² draws).
    """
    sigma2 = 1.0 / rng.gamma(post.alpha_n, 1.0 / post.beta_n, n_draws)
    mu = post.mu_n + np.sqrt(sigma2 / post.kappa_n) * rng.standard_normal(n_draws)
    return mu, sigma2


@dataclass(frozen=True)
class TwoPartDraws:
    """Posterior draws of one arm under the two-part model.

    Attributes
    ----------
    p : numpy.ndarray
        Conversion-rate draws (Beta posterior).
    mu, sigma2 : numpy.ndarray
        Draws of the log-spend mean and variance among converters.
    mean_spend : numpy.ndarray
        Draws of mean spend per customer, ``p * exp(mu + sigma2 / 2)``.
    beta_post : BetaPosterior
        The conversion posterior.
    nig_post : NIGPosterior
        The log-spend posterior.
    """

    p: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray
    mean_spend: np.ndarray
    beta_post: BetaPosterior
    nig_post: NIGPosterior

    @property
    def conv_mean_spend(self) -> np.ndarray:
        """Draws of the log-normal mean among converters, ``exp(mu + sigma2 / 2)``."""
        return np.exp(self.mu + self.sigma2 / 2)


def two_part_draws(
    spend: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
    beta_prior: BetaPrior = FLAT_PRIOR,
    nig_prior: NIGPrior = WEAK_NIG_PRIOR,
) -> TwoPartDraws:
    """Fit the two-part model to one arm and draw from its posterior.

    Conversion (spend > 0) ~ Bernoulli(p) with a Beta prior on p; log spend among converters
    ~ Normal(mu, sigma²) with a Normal-Inverse-Gamma prior. The two parts are independent given the
    data, so mean spend per customer, p · exp(mu + sigma² / 2), is drawn as a product.

    Parameters
    ----------
    spend : numpy.ndarray
        Row-level spend of the arm, zeros included.
    rng : numpy.random.Generator
        Source of randomness (p first, then sigma² and mu).
    n_draws : int
        Draws.
    beta_prior : BetaPrior
        Prior on the conversion rate.
    nig_prior : NIGPrior
        Prior on (mu, sigma²).

    Returns
    -------
    TwoPartDraws
        Draws and the two posteriors.

    Raises
    ------
    ValueError
        If the arm has no converters (the log-normal part cannot be fitted).
    """
    spend = np.asarray(spend, dtype=float)
    positive = spend[spend > 0]
    if positive.size == 0:
        raise ValueError("The two-part model needs at least one customer with positive spend.")
    beta_post = posterior(beta_prior, int(positive.size), int(spend.size))
    nig_post = nig_posterior(nig_prior, np.log(positive))
    p = rng.beta(beta_post.alpha, beta_post.beta, n_draws)
    mu, sigma2 = draw_nig(nig_post, rng, n_draws)
    return TwoPartDraws(p, mu, sigma2, p * np.exp(mu + sigma2 / 2), beta_post, nig_post)


def _ppc_statistics(x: np.ndarray, cap: float, floor: float) -> dict[str, float]:
    """Summary statistics of converters' spend used in the posterior predictive check."""
    out = {"mean": float(x.mean()), **{f"q{q:g}": float(np.quantile(x, q)) for q in PPC_QUANTILES}}
    out["max"] = float(x.max())
    out["share_at_or_above_cap"] = float((x >= cap).mean())
    out["share_at_or_below_floor"] = float((x <= floor).mean())
    return out


def posterior_predictive_check(
    positive_spend: np.ndarray,
    draws: TwoPartDraws,
    rng: np.random.Generator,
    cap: float,
    floor: float,
    n_replicates: int = N_PPC_REPLICATES,
    alpha: float = config.ALPHA,
) -> tuple[pd.DataFrame, float]:
    """Check whether a log-normal reproduces the observed converters' spend.

    For each of ``n_replicates`` posterior draws (mu, sigma²), a replicate dataset of the same size
    as the observed converters is simulated from the log-normal, and the same statistics
    (mean, quantiles, maximum, share at or above the top-code ``cap``, share at or below the
    ``floor``) are computed for the replicate and the data. The posterior predictive p-value
    is the share of replicates with a statistic at least as large as the observed one; values
    near 0 or 1 mean the model does not reproduce that feature. Separately, the model's
    implied P(spend >= cap) = E_post[1 - Φ((log cap - mu) / sigma)] is returned for comparison with
    the observed share of converters at the cap.

    Parameters
    ----------
    positive_spend : numpy.ndarray
        Observed spend of the arm's converters.
    draws : TwoPartDraws
        Posterior draws (the first ``n_replicates`` of mu and sigma² are used).
    rng : numpy.random.Generator
        Randomness for the replicate datasets.
    cap, floor : float
        The top-code and the floor of spend in the data.
    n_replicates : int
        Replicate datasets.
    alpha : float
        The posterior predictive intervals have level 1 - alpha.

    Returns
    -------
    tuple
        (table with one row per statistic: observed value, posterior predictive mean,
        interval and p-value; implied P(spend >= cap) under the model).

    Raises
    ------
    ValueError
        If more replicates are requested than there are posterior draws.
    """
    x = np.asarray(positive_spend, dtype=float)
    if n_replicates > draws.mu.size:
        raise ValueError("n_replicates cannot exceed the number of posterior draws.")
    mu, sigma = draws.mu[:n_replicates], np.sqrt(draws.sigma2[:n_replicates])
    reps = np.exp(mu[:, None] + sigma[:, None] * rng.standard_normal((n_replicates, x.size)))
    observed = _ppc_statistics(x, cap, floor)
    rep_stats = {
        "mean": reps.mean(axis=1),
        **{f"q{q:g}": np.quantile(reps, q, axis=1) for q in PPC_QUANTILES},
        "max": reps.max(axis=1),
        "share_at_or_above_cap": (reps >= cap).mean(axis=1),
        "share_at_or_below_floor": (reps <= floor).mean(axis=1),
    }
    rows = []
    for name, obs in observed.items():
        r = rep_stats[name]
        rows.append(
            {
                "statistic": name,
                "observed": obs,
                "pp_mean": float(r.mean()),
                "pp_low": float(np.quantile(r, alpha / 2)),
                "pp_high": float(np.quantile(r, 1 - alpha / 2)),
                "pp_pvalue": float((r >= obs).mean()),
                "n_converters": int(x.size),
                "n_replicates": n_replicates,
            }
        )
    implied_tail = float(stats.norm.sf((np.log(cap) - draws.mu) / np.sqrt(draws.sigma2)).mean())
    return pd.DataFrame(rows), implied_tail


# ---------------------------------------------------------------------------------------------
# Spend readout
# ---------------------------------------------------------------------------------------------


def _spend_by_arm(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    rows = run_sql(con, "arm_spend")
    return {str(a): g["spend"].to_numpy(dtype=float) for a, g in rows.groupby("arm", sort=True)}


def _comparison_row(
    treatment: str,
    control: str,
    metric: str,
    model: str,
    res: PosteriorComparison,
    y: dict[str, np.ndarray],
    arm_draws: dict[str, np.ndarray],
    cap: float,
) -> dict[str, object]:
    yt, yc = y[treatment], y[control]
    mean_c = float(yc.mean())
    threshold = config.BAYES_LOSS_FRACTION * mean_c
    verdict = decide(res.p_better, res.loss_ship, res.loss_keep, threshold)
    return {
        "comparison": f"{treatment}_vs_{control}",
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "model": model,
        "n_treatment": int(yt.size),
        "n_control": int(yc.size),
        "converters_treatment": int((yt > 0).sum()),
        "converters_control": int((yc > 0).sum()),
        "mean_treatment": float(yt.mean()),
        "mean_control": mean_c,
        "post_mean_treatment": float(arm_draws[treatment].mean()),
        "post_mean_control": float(arm_draws[control].mean()),
        "post_median_treatment": float(np.median(arm_draws[treatment])),
        "post_median_control": float(np.median(arm_draws[control])),
        "diff_mean": res.diff_mean,
        "diff_sd": res.diff_sd,
        "diff_mc_mean": res.diff_mc_mean,
        "diff_mcse": res.diff_mcse,
        "diff_median": res.diff_median,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "ci_mcse_low": res.ci_mcse_low,
        "ci_mcse_high": res.ci_mcse_high,
        "rel_lift_mean": res.rel_lift_mean,
        "rel_lift_mcse": res.rel_lift_mcse,
        "rel_ci_low": res.rel_ci_low,
        "rel_ci_high": res.rel_ci_high,
        "rel_ci_mcse_low": res.rel_ci_mcse_low,
        "rel_ci_mcse_high": res.rel_ci_mcse_high,
        "p_better": res.p_better,
        "p_better_mcse": res.p_better_mcse,
        "loss_ship": res.loss_ship,
        "loss_ship_mcse": res.loss_ship_mcse,
        "loss_keep": res.loss_keep,
        "loss_keep_mcse": res.loss_keep_mcse,
        "loss_threshold": threshold,
        "decision": verdict.decision,
        "reason": verdict.reason,
        "revenue_scale": REVENUE_SCALE,
        "incremental_revenue_per_scale": REVENUE_SCALE * res.diff_mc_mean,
        "incremental_revenue_per_scale_ci_low": REVENUE_SCALE * res.ci_low,
        "incremental_revenue_per_scale_ci_high": REVENUE_SCALE * res.ci_high,
        "loss_ship_per_scale": REVENUE_SCALE * res.loss_ship,
        "spend_cap": cap,
        "n_draws": res.n_draws,
        "level": res.level,
    }


def _posterior_hist(diffs: dict[tuple[str, str], np.ndarray], n_bins: int = 80) -> pd.DataFrame:
    """Binned densities of the difference draws, one row per model, comparison and bin."""
    lo = min(float(np.quantile(d, 0.0005)) for d in diffs.values())
    hi = max(float(np.quantile(d, 0.9995)) for d in diffs.values())
    edges = np.linspace(lo, hi, n_bins + 1)
    rows = []
    for (model, comparison), d in diffs.items():
        density, _ = np.histogram(d, bins=edges, density=True)
        rows.extend(
            {
                "model": model,
                "comparison": comparison,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "density": float(density[i]),
            }
            for i in range(n_bins)
        )
    return pd.DataFrame(rows)


def build_bayes_spend(
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
    n_replicates: int = N_PPC_REPLICATES,
    alpha: float = config.ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bayesian readout of spend per customer for every comparison.

    Rows, in order: the Bayesian bootstrap on raw spend (the primary analysis, plan §9), the
    Bayesian bootstrap on winsorised spend (the sensitivity analysis, plan §8), and the
    two-part log-normal cross-check on raw spend. Each arm's posterior is drawn once and
    reused across its comparisons, so the three comparisons share one joint posterior. The
    decision rule uses the loss threshold ``config.BAYES_LOSS_FRACTION`` x the observed
    control-arm mean of the same metric.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    rng : numpy.random.Generator
        Randomness, consumed in order: bootstrap draws per arm (raw, then winsorised), then
        the two-part draws per arm, then the posterior predictive replicates per arm.
    n_draws : int
        Posterior draws per arm and model.
    n_replicates : int
        Replicate datasets in the posterior predictive check.
    alpha : float
        Intervals have credible level 1 - alpha.

    Returns
    -------
    tuple of pandas.DataFrame
        (comparison table, per-arm posterior table, posterior predictive check table,
        histogram table of the difference draws on raw spend).
    """
    raw = _spend_by_arm(con)
    cap_win = float(arm_metrics(con, winsorise=True)["spend_cap"].iloc[0])
    win = {a: np.minimum(v, cap_win) for a, v in raw.items()}
    all_spend = np.concatenate(list(raw.values()))
    cap, floor = float(all_spend.max()), float(all_spend[all_spend > 0].min())
    arms = sorted(raw)

    rows, arm_rows, hist_draws = [], [], {}
    for metric, y, metric_cap in (("spend", raw, np.nan), (SPEND_WINSORISED, win, cap_win)):
        draws = {a: bayesian_bootstrap_means(y[a], rng, n_draws) for a in arms}
        for a in arms:
            mean_a, sd_a = bayesian_bootstrap_moments(y[a])
            arm_rows.append(
                {
                    "arm": a,
                    "metric": metric,
                    "model": BOOTSTRAP,
                    "n": int(y[a].size),
                    "converters": int((y[a] > 0).sum()),
                    "sample_mean": mean_a,
                    "post_mean": mean_a,
                    "post_sd": sd_a,
                    "post_mc_mean": float(draws[a].mean()),
                    "post_mc_sd": float(draws[a].std(ddof=1)),
                    "post_ci_low": float(np.quantile(draws[a], alpha / 2)),
                    "post_ci_high": float(np.quantile(draws[a], 1 - alpha / 2)),
                }
            )
        for treatment, control in COMPARISONS:
            res = compare_means(draws[treatment], draws[control], y[treatment], y[control], alpha)
            rows.append(
                _comparison_row(treatment, control, metric, BOOTSTRAP, res, y, draws, metric_cap)
            )
            if metric == "spend":
                hist_draws[(BOOTSTRAP, f"{treatment}_vs_{control}")] = (
                    draws[treatment] - draws[control]
                )

    two_part = {a: two_part_draws(raw[a], rng, n_draws) for a in arms}
    ppc_rows = []
    for a in arms:
        tp = two_part[a]
        ppc, implied_tail = posterior_predictive_check(
            raw[a][raw[a] > 0], tp, rng, cap, floor, n_replicates, alpha
        )
        ppc.insert(0, "arm", a)
        ppc["implied_p_at_or_above_cap"] = implied_tail
        ppc["cap"], ppc["floor"] = cap, floor
        ppc_rows.append(ppc)
        conv = raw[a][raw[a] > 0]
        arm_rows.append(
            {
                "arm": a,
                "metric": "spend",
                "model": TWO_PART,
                "n": int(raw[a].size),
                "converters": int(conv.size),
                "sample_mean": float(raw[a].mean()),
                "post_mean": np.nan,
                "post_sd": np.nan,
                "post_mc_mean": float(tp.mean_spend.mean()),
                "post_mc_sd": float(tp.mean_spend.std(ddof=1)),
                "post_ci_low": float(np.quantile(tp.mean_spend, alpha / 2)),
                "post_ci_high": float(np.quantile(tp.mean_spend, 1 - alpha / 2)),
                "conv_sample_mean": float(conv.mean()),
                "conv_post_mc_mean": float(tp.conv_mean_spend.mean()),
                "conv_post_median": float(np.median(tp.conv_mean_spend)),
                "mu_n": tp.nig_post.mu_n,
                "kappa_n": tp.nig_post.kappa_n,
                "alpha_n": tp.nig_post.alpha_n,
                "beta_n": tp.nig_post.beta_n,
                "sigma2_post_mean": tp.nig_post.sigma2_mean,
                "implied_p_at_or_above_cap": implied_tail,
                "observed_share_at_cap": float((conv >= cap).mean()),
            }
        )
    tp_draws = {a: two_part[a].mean_spend for a in arms}
    for treatment, control in COMPARISONS:
        res = summarise_draws(tp_draws[treatment], tp_draws[control], alpha=alpha)
        rows.append(
            _comparison_row(treatment, control, "spend", TWO_PART, res, raw, tp_draws, np.nan)
        )
        hist_draws[(TWO_PART, f"{treatment}_vs_{control}")] = (
            tp_draws[treatment] - tp_draws[control]
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(arm_rows),
        pd.concat(ppc_rows, ignore_index=True),
        _posterior_hist(hist_draws),
    )


def recommend(table: pd.DataFrame) -> dict[str, object]:
    """Apply the plan §9 tie-break to the primary rows (Bayesian bootstrap, raw spend).

    Ship the emails whose decision is ``"ship"`` against control; if both qualify, recommend
    the one with the higher posterior mean spend and report P(mens > womens) alongside.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`build_bayes_spend` (first element).

    Returns
    -------
    dict
        ``qualifying_emails``, ``recommended_email`` (None if none qualifies),
        ``p_mens_better_than_womens`` and the ``basis`` of the choice.
    """
    primary = table[(table["model"] == BOOTSTRAP) & (table["metric"] == "spend")]
    vs_control = primary[primary["control"] == config.CONTROL]
    shipped = vs_control[vs_control["decision"] == SHIP]
    qualifying = [str(t) for t in shipped["treatment"]]
    head_to_head = primary[(primary["treatment"] == "mens") & (primary["control"] == "womens")]
    p_mens = float(head_to_head["p_better"].iloc[0]) if not head_to_head.empty else None
    if shipped.empty:
        recommended, basis = None, "no email meets the decision rule against control"
    elif len(shipped) == 1:
        recommended = qualifying[0]
        basis = "the only email meeting the decision rule against control"
    else:
        best = shipped.loc[shipped["post_mean_treatment"].idxmax()]
        recommended = str(best["treatment"])
        basis = "both emails qualify; the higher posterior mean spend per customer wins"
    return {
        "qualifying_emails": qualifying,
        "recommended_email": recommended,
        "basis": basis,
        "p_mens_better_than_womens": p_mens,
    }


def run_bayes_spend(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    n_draws: int = N_DRAWS,
    n_replicates: int = N_PPC_REPLICATES,
) -> pd.DataFrame:
    """Run the Bayesian spend readout and write its tables, results.json section and figure.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    n_draws : int
        Posterior draws per arm and model.
    n_replicates : int
        Replicate datasets in the posterior predictive check.

    Returns
    -------
    pandas.DataFrame
        The comparison table.

    Raises
    ------
    SRMError
        If the sample ratio mismatch check fails (plan §5 blocks the readout).
    """
    rng = np.random.default_rng(config.SEEDS["bayes"] + 1)
    with duckdb.connect(str(db_path), read_only=True) as con:
        srm = srm_test(arm_counts(con))
        if not srm.passed:
            raise SRMError(f"SRM check failed (p = {srm.p_value:.2e}); readout blocked.")
        table, arms, ppc, hist = build_bayes_spend(con, rng, n_draws, n_replicates)

    tables_dir = reports_dir / "tables"
    write_table(table, SPEND_TABLE_NAME, tables_dir)
    write_table(ppc, SPEND_PPC_TABLE_NAME, tables_dir)
    write_table(hist, SPEND_HIST_TABLE_NAME, tables_dir)
    plots.plot_spend_posterior(hist, table, reports_dir / "figures" / SPEND_FIGURE_NAME)
    update_results(
        "bayes_spend",
        {
            "primary": f"{BOOTSTRAP} on raw spend",
            "sensitivity": f"{BOOTSTRAP} on winsorised spend (plan §8)",
            "cross_check": f"{TWO_PART}: Beta-Binomial conversion x log-normal spend among "
            "converters with a Normal-Inverse-Gamma prior, raw spend",
            "n_draws": n_draws,
            "ppc_replicates": n_replicates,
            "seed": config.SEEDS["bayes"] + 1,
            "level": 1 - config.ALPHA,
            "revenue_scale": REVENUE_SCALE,
            "two_part_priors": {
                "conversion": {"alpha": FLAT_PRIOR.alpha, "beta": FLAT_PRIOR.beta},
                "log_spend_nig": {
                    "mu0": WEAK_NIG_PRIOR.mu0,
                    "kappa0": WEAK_NIG_PRIOR.kappa0,
                    "alpha0": WEAK_NIG_PRIOR.alpha0,
                    "beta0": WEAK_NIG_PRIOR.beta0,
                },
            },
            "spend_cap": float(ppc["cap"].iloc[0]),
            "spend_floor": float(ppc["floor"].iloc[0]),
            "decision_rule": {
                "p_better_threshold": config.BAYES_P_BETTER,
                "loss_fraction_of_control_mean": config.BAYES_LOSS_FRACTION,
            },
            "recommendation": recommend(table),
            "table": f"tables/{SPEND_TABLE_NAME}.csv",
            "ppc_table": f"tables/{SPEND_PPC_TABLE_NAME}.csv",
            "histogram_table": f"tables/{SPEND_HIST_TABLE_NAME}.csv",
            "figure": f"figures/{SPEND_FIGURE_NAME}",
            "rows": json_records(table),
            "arms": json_records(arms),
            "ppc": json_records(ppc),
        },
        reports_dir / "results.json",
    )
    logger.info("Bayesian spend readout written: %d rows.", len(table))
    return table
