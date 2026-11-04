"""Bayesian analysis: Beta-Binomial models for rates, decision rules and prior sensitivity.

Each arm's rate gets a conjugate Beta posterior (analysis plan §9). Comparisons are summarised
from posterior draws, each Monte Carlo quantity with its Monte Carlo standard error, and
P(p_B > p_A) is also computed exactly: by Miller's closed form when the parameters are
integers (the test oracle) and by numerical integration otherwise. The readout applies the
pre-registered decision rule and repeats everything under a flat prior and under shared
informative priors of increasing strength.
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
from abkit.metrics import arm_counts, arm_metrics
from abkit.readout import COMPARISONS, RATE_METRICS
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
class RateComparison:
    """Posterior summary of p_t - p_c from ``n_draws`` joint posterior draws.

    Attributes
    ----------
    diff_mean, diff_sd : float
        Exact posterior mean and standard deviation of the difference (from the Beta
        formulas; the arms are independent).
    diff_mc_mean, diff_mc_sd, diff_mcse : float
        Monte Carlo mean and standard deviation of the difference, and the Monte Carlo
        standard error of the mean, ``diff_mc_sd / sqrt(n_draws)``.
    ci_low, ci_high, ci_mcse_low, ci_mcse_high : float
        Equal-tailed credible interval for the difference and the Monte Carlo standard
        errors of its endpoints.
    rel_lift_mean, rel_lift_mcse : float
        Posterior mean of the relative lift ``p_t / p_c - 1`` and its Monte Carlo SE.
    rel_ci_low, rel_ci_high, rel_ci_mcse_low, rel_ci_mcse_high : float
        Credible interval for the relative lift and the Monte Carlo SEs of its endpoints.
    p_better, p_better_mcse : float
        P(p_t > p_c) and its Monte Carlo SE, ``sqrt(p~ (1 - p~) / n_draws)`` with the
        Agresti-Coull proportion p~ = (k + 2) / (n_draws + 4), which stays positive when every
        draw agrees (the plug-in SE would be exactly zero there).
    loss_ship, loss_ship_mcse : float
        Expected loss of shipping the treatment, E[max(p_c - p_t, 0)], with its Monte Carlo SE.
    loss_keep, loss_keep_mcse : float
        Expected loss of keeping control, E[max(p_t - p_c, 0)], with its Monte Carlo SE.
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
) -> RateComparison:
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
    RateComparison
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
    diff = pt - pc
    rel = pt / pc - 1
    diff_mc_mean, diff_mcse = _mean_and_mcse(diff)
    ci_low, ci_high, ci_mcse_low, ci_mcse_high = _interval(diff, alpha)
    rel_mean, rel_mcse = _mean_and_mcse(rel)
    rel_low, rel_high, rel_mcse_low, rel_mcse_high = _interval(rel, alpha)
    better = int((diff > 0).sum())
    p_better = better / n_draws
    p_tilde = (better + 2) / (n_draws + 4)
    loss_ship, loss_ship_mcse = _mean_and_mcse(np.maximum(-diff, 0.0))
    loss_keep, loss_keep_mcse = _mean_and_mcse(np.maximum(diff, 0.0))
    return RateComparison(
        diff_mean=post_t.mean - post_c.mean,
        diff_sd=float(np.sqrt(post_t.var + post_c.var)),
        diff_mc_mean=diff_mc_mean,
        diff_mc_sd=float(diff.std(ddof=1)),
        diff_mcse=diff_mcse,
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

    - **ship** if P(treatment better) ≥ ``p_threshold`` and the expected loss of shipping is
      below ``loss_threshold`` (1% of the control arm's mean).
    - **don't ship** if the mirror image holds: P(control better) ≥ ``p_threshold`` and the
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
