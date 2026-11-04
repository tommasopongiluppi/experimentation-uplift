"""Pipeline stage: cross-fitted uplift learners and baselines, evaluated per email and outcome.

For each email arm (vs control) and each outcome (``visit`` primary, ``conversion``
secondary) the stage

1. builds the treatment pair and the stratified folds (``config.SEEDS["uplift"]``),
2. scores every customer out of fold with the S-, T- and X-learners and the response-model
   baseline, and adds the recency rule and random rankings,
3. evaluates every ranking with :func:`abkit.uplift.evaluation.bootstrap_evaluation`, using
   the same bootstrap seed for every ranking so that differences between rankings are paired,
4. writes ``tables/uplift_models.csv`` (metrics with intervals),
   ``tables/uplift_models_comparison.csv`` (paired differences in the Qini coefficient against
   the best ranking), the Qini and calibration figures, the ``uplift_models`` section of
   ``results.json`` and the out-of-fold scores (``uplift_scores.parquet``) for the policy stage.
"""

import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from abkit import config, plots
from abkit.report import json_records, update_results, write_table
from abkit.uplift.crossfit import (
    N_FOLDS,
    TreatmentPair,
    cross_fit_with_other,
    stratified_folds,
    treatment_pair,
)
from abkit.uplift.evaluation import (
    N_RESAMPLES,
    UPLIFT_AT_K,
    BootstrapEvaluation,
    bootstrap_evaluation,
)
from abkit.uplift.learners import (
    FIXED_PARAMS,
    GRID,
    ResponseModel,
    SLearner,
    TLearner,
    XLearner,
    random_score,
    recency_rule_score,
)
from abkit.validate import CLEAN_TABLE

logger = logging.getLogger(__name__)

LABEL: str = "uplift_models"
UPLIFT_LEARNERS: tuple[str, ...] = ("s_learner", "t_learner", "x_learner")
BASELINES: tuple[str, ...] = ("response_model", "recency_rule", "random")
RANKINGS: tuple[str, ...] = (*UPLIFT_LEARNERS, *BASELINES)
OUTCOMES: tuple[str, ...] = ("visit", "conversion")
TABLE_NAME: str = "uplift_models"
COMPARISON_TABLE_NAME: str = "uplift_models_comparison"
SCORES_NAME: str = "uplift_scores.parquet"
SELECTION_METRIC: str = "qini_normalised"
DEVIATION: str = (
    "D6: history_segment dropped from the features; D7: S-learner added and hyperparameters "
    "chosen on an inner validation split instead of fixed defaults"
)


def score_pair(pair: TreatmentPair, outcome: str, rng: np.random.Generator) -> pd.DataFrame:
    """Out-of-fold scores of every ranking for one pair and outcome.

    Parameters
    ----------
    pair : TreatmentPair
        From :func:`abkit.uplift.crossfit.treatment_pair`.
    outcome : str
        ``"visit"`` or ``"conversion"``.
    rng : numpy.random.Generator
        Source of the fold split, model seeds and the random baseline.

    Returns
    -------
    pandas.DataFrame
        One row per customer in the pair, then one per customer of the third arm. Columns
        ``customer_id``, ``arm`` (the pair's email), ``outcome``, ``in_pair``, ``treated``
        and ``fold`` (-1 outside the pair), ``y`` and one score column per name in
        ``RANKINGS``. Rows outside the pair are scored by the fold models, which never saw
        them; they are for the policy stage and take no part in the evaluation.
    """
    y = pair.outcomes[outcome].to_numpy().astype(int)
    y_other = pair.outcomes_other[outcome].to_numpy().astype(int)
    folds = stratified_folds(pair.treated, y, rng, N_FOLDS)
    learners = {
        "s_learner": SLearner(rng),
        "t_learner": TLearner(rng),
        "x_learner": XLearner(rng, pair.p_treat),
        "response_model": ResponseModel(rng),
    }
    out = pd.DataFrame(
        {
            "customer_id": np.r_[pair.customer_id, pair.other_customer_id],
            "arm": pair.arm,
            "outcome": outcome,
            "in_pair": np.r_[np.ones(pair.n, bool), np.zeros(pair.n_other, bool)],
            "treated": np.r_[pair.treated, np.full(pair.n_other, -1)],
            "y": np.r_[y, y_other],
            "fold": np.r_[folds, np.full(pair.n_other, -1)],
        }
    )
    for name, learner in learners.items():
        logger.info("Cross-fitting %s for %s / %s", name, pair.arm, outcome)
        in_pair, other = cross_fit_with_other(pair.X, pair.treated, y, learner, folds, pair.X_other)
        out[name] = np.r_[in_pair, other]
    X_all = pd.concat([pair.X, pair.X_other], ignore_index=True)
    out["recency_rule"] = recency_rule_score(
        X_all["recency"].to_numpy(), X_all["history"].to_numpy(), out["customer_id"].to_numpy()
    )
    # The third arm's random scores come from a spawned child generator, so the pair's own
    # draws, and the fold seeds of every later pair and outcome, do not depend on them.
    other_rng = rng.spawn(1)[0]
    out["random"] = np.r_[random_score(pair.n, rng), random_score(pair.n_other, other_rng)]
    return out


def evaluate_rankings(
    scores: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = config.SEEDS["bootstrap"]
) -> dict[str, BootstrapEvaluation]:
    """Bootstrap-evaluate every ranking in a :func:`score_pair` table with paired resamples.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of :func:`score_pair`; only its ``in_pair`` rows are evaluated.
    n_resamples : int
        Bootstrap resamples per ranking.
    seed : int
        The same seed is used for every ranking, so all rankings see the same resampled
        customers and their replicate differences are paired.

    Returns
    -------
    dict of str to BootstrapEvaluation
    """
    scores = scores[scores["in_pair"]]
    treated, y = scores["treated"].to_numpy(), scores["y"].to_numpy()
    return {
        name: bootstrap_evaluation(
            scores[name].to_numpy(), treated, y, np.random.default_rng(seed), n_resamples
        )
        for name in RANKINGS
    }


def metrics_table(evals: dict[str, BootstrapEvaluation], arm: str, outcome: str) -> pd.DataFrame:
    """Long table of every ranking's scalar metrics with intervals for one pair and outcome."""
    rows = []
    for name, ev in evals.items():
        rows.append(
            ev.scalars.assign(
                arm=arm,
                outcome=outcome,
                ranking=name,
                kind="uplift_model" if name in UPLIFT_LEARNERS else "baseline",
            )
        )
    table = pd.concat(rows, ignore_index=True)
    first = ["arm", "outcome", "ranking", "kind", "metric"]
    return table[first + [c for c in table.columns if c not in first]]


def compare_rankings(
    evals: dict[str, BootstrapEvaluation],
    arm: str,
    outcome: str,
    metric: str = SELECTION_METRIC,
    alpha: float = config.ALPHA,
) -> pd.DataFrame:
    """Paired bootstrap differences in ``metric`` between the best ranking and each other one.

    The best ranking is the one with the highest point estimate. A difference is
    ``meaningful`` when its percentile interval excludes zero.

    Parameters
    ----------
    evals : dict of str to BootstrapEvaluation
        From :func:`evaluate_rankings` (same seed, so replicates are paired).
    arm, outcome : str
        Labels for the rows.
    metric : str
        Scalar metric compared.
    alpha : float
        1 - interval level.

    Returns
    -------
    pandas.DataFrame
        One row per ranking: ``best``, ``ranking``, ``estimate`` (of the ranking),
        ``diff`` (best minus ranking), ``ci_low``, ``ci_high``, ``meaningful``.
    """
    estimates = {
        n: float(ev.scalars.set_index("metric").loc[metric, "estimate"]) for n, ev in evals.items()
    }
    best = max(estimates, key=lambda n: np.nan_to_num(estimates[n], nan=-np.inf))
    best_reps = evals[best].replicates[metric].to_numpy()
    rows = []
    for name, ev in evals.items():
        d = best_reps - ev.replicates[metric].to_numpy()
        lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "arm": arm,
                "outcome": outcome,
                "metric": metric,
                "best": best,
                "ranking": name,
                "estimate": estimates[name],
                "diff": estimates[best] - estimates[name],
                "ci_low": float(lo),
                "ci_high": float(hi),
                "meaningful": bool(name != best and (lo > 0 or hi < 0)),
            }
        )
    return pd.DataFrame(rows)


def load_customers(db_path: Path) -> pd.DataFrame:
    """Read the cleaned ``customers`` table ordered by ``customer_id``."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f"SELECT * FROM {CLEAN_TABLE} ORDER BY customer_id").df()


def run_uplift_models(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    scores_dir: Path = config.DATA_PROCESSED,
    n_resamples: int = N_RESAMPLES,
    outcomes: tuple[str, ...] = OUTCOMES,
) -> pd.DataFrame:
    """Run the uplift-model stage and write its tables, figures, scores and results section.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding the ``customers`` table.
    reports_dir : Path
        Reports directory.
    scores_dir : Path
        Where the out-of-fold scores parquet is written.
    n_resamples : int
        Bootstrap resamples per ranking (tests pass fewer).
    outcomes : tuple of str
        Outcomes to model.

    Returns
    -------
    pandas.DataFrame
        The metrics table.
    """
    rng = np.random.default_rng(config.SEEDS["uplift"])
    customers = load_customers(db_path)
    tables, figures = reports_dir / "tables", reports_dir / "figures"
    metric_tables, comparisons, score_tables = [], [], []
    summary: dict[str, Any] = {}
    for arm in config.TREATMENTS:
        pair = treatment_pair(customers, arm)
        for outcome in outcomes:
            scores = score_pair(pair, outcome, rng)
            score_tables.append(scores)
            evals = evaluate_rankings(scores, n_resamples)
            metric_tables.append(metrics_table(evals, arm, outcome))
            comparison = compare_rankings(evals, arm, outcome)
            comparisons.append(comparison)
            title = f"{arm} vs control, {outcome}"
            qini_fig = plots.plot_qini_comparison(
                {n: evals[n].curves for n in RANKINGS}, figures / f"qini_{arm}_{outcome}.png", title
            )
            cal_fig = plots.plot_calibration_grid(
                {n: evals[n].deciles for n in UPLIFT_LEARNERS},
                figures / f"calibration_{arm}_{outcome}.png",
                title,
            )
            best = comparison["best"].iloc[0]
            summary[f"{arm}_{outcome}"] = {
                "arm": arm,
                "outcome": outcome,
                "n": pair.n,
                "p_treat": pair.p_treat,
                "best_ranking": best,
                "best_estimate": float(
                    comparison.loc[comparison["ranking"] == best, "estimate"].iloc[0]
                ),
                "meaningfully_better_than": sorted(
                    comparison.loc[comparison["meaningful"], "ranking"]
                ),
                "not_separated_from": sorted(
                    comparison.loc[
                        ~comparison["meaningful"] & (comparison["ranking"] != best), "ranking"
                    ]
                ),
                "qini_figure": f"figures/{qini_fig.name}",
                "calibration_figure": f"figures/{cal_fig.name}",
            }
    metrics = pd.concat(metric_tables, ignore_index=True)
    comparison_table = pd.concat(comparisons, ignore_index=True)
    write_table(metrics, TABLE_NAME, tables)
    write_table(comparison_table, COMPARISON_TABLE_NAME, tables)
    scores_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(score_tables, ignore_index=True).to_parquet(scores_dir / SCORES_NAME, index=False)
    update_results(
        LABEL,
        {
            "label": LABEL,
            "plan_section": "11",
            "deviation": DEVIATION,
            "seeds": {"uplift": config.SEEDS["uplift"], "bootstrap": config.SEEDS["bootstrap"]},
            "n_folds": N_FOLDS,
            "rankings": list(RANKINGS),
            "uplift_learners": list(UPLIFT_LEARNERS),
            "baselines": list(BASELINES),
            "outcomes": list(outcomes),
            "hyperparameter_grid": list(GRID),
            "fixed_params": FIXED_PARAMS,
            "selection_metric": SELECTION_METRIC,
            "uplift_at_k": list(UPLIFT_AT_K),
            "bootstrap": {"n_resamples": n_resamples, "paired": True},
            "limitation": (
                "Bootstrap intervals resample customers with the out-of-fold scores held fixed, "
                "so they leave out model-fitting variability."
            ),
            "table": f"tables/{TABLE_NAME}.csv",
            "comparison_table": f"tables/{COMPARISON_TABLE_NAME}.csv",
            "scores_file": SCORES_NAME,
            "summary": summary,
            "rows": json_records(metrics),
            "comparison_rows": json_records(comparison_table),
        },
        reports_dir / "results.json",
    )
    logger.info("Uplift model tables written: %d metric rows.", len(metrics))
    return metrics
