"""Repeated cross-fitting: how much the uplift results move with the fold seed.

The bootstrap intervals of the uplift-model and policy stages hold the fold assignment
fixed, so they leave out the variability that comes from which customers land in which
fold (and from the base models' own seeds). Chernozhukov et al. (2018) recommend repeating
cross-fitting over several random splits and reporting the spread. This stage re-runs the
cross-fitting for the primary outcome (``visit``) and both email pairs under several fold
seeds and reports, per learner and pair, the median, minimum and maximum normalised Qini
coefficient, how often the X-learner beats the response model on the womens pair, and the
uplift policy's incremental visits at the email budget.

Seed 0 is the committed run: its out-of-fold scores are read from the scores parquet the
uplift-model stage wrote, so the seed-0 rows reproduce the headline tables exactly. The
other seeds are child generators spawned from ``config.SEEDS["uplift"]``. The committed
headline tables are not changed.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abkit import config
from abkit.report import json_records, update_results, write_table
from abkit.uplift.crossfit import TreatmentPair, assign_folds, treatment_pair
from abkit.uplift.evaluation import qini_coefficient
from abkit.uplift.models import SCORES_NAME, load_customers, score_pair
from abkit.uplift.policy import (
    POLICY_OUTCOME,
    PolicyInputs,
    ab_winner,
    build_policies,
    policy_value,
)

logger = logging.getLogger(__name__)

LABEL: str = "seed_robustness"
N_SEEDS: int = 10
LEARNERS: tuple[str, ...] = ("s_learner", "t_learner", "x_learner", "response_model")
OUTCOME: str = POLICY_OUTCOME
TABLE_NAME: str = "uplift_seed_robustness"
QINI_METRIC: str = "qini_normalised"
POLICY_METRIC: str = "uplift_policy_incremental_visits"
RESPONSE_POLICY_METRIC: str = "response_model_policy_incremental_visits"
#: The two model-based policies whose value depends on the fold seed, and their metric names.
POLICY_METRICS: dict[str, str] = {
    "uplift": POLICY_METRIC,
    "response_model": RESPONSE_POLICY_METRIC,
}
POLICY_ARM: str = "both"

Scorer = Callable[[TreatmentPair, str, np.random.Generator], pd.DataFrame]


def fold_seed_generators(
    n_seeds: int = N_SEEDS, seed: int = config.SEEDS["uplift"]
) -> list[np.random.Generator]:
    """Child generators for seeds 1 to ``n_seeds - 1``, spawned from the config seed.

    Parameters
    ----------
    n_seeds : int
        Total number of seeds including the committed seed 0.
    seed : int
        Parent seed; the committed run used ``default_rng(seed)`` directly.

    Returns
    -------
    list of numpy.random.Generator
        ``n_seeds - 1`` independent generators, deterministic in ``seed``.
    """
    return list(np.random.default_rng(seed).spawn(n_seeds - 1))


def seed_metrics(
    scores_by_arm: dict[str, pd.DataFrame],
    customers: pd.DataFrame,
    winner: str,
    seed: int,
    budget: int = config.BUDGET,
) -> pd.DataFrame:
    """Normalised Qini per learner and pair, and the uplift policy's incremental visits.

    Parameters
    ----------
    scores_by_arm : dict of str to pandas.DataFrame
        Email arm to a :func:`abkit.uplift.models.score_pair` table for ``OUTCOME`` with
        every customer scored (``in_pair`` rows are used for Qini, all rows for the policy).
    customers : pandas.DataFrame
        Cleaned customers ordered by ``customer_id``.
    winner : str
        A/B winner (needed to build the policies; only the uplift policy is reported).
    seed : int
        Seed label for the rows.
    budget : int
        Email budget.

    Returns
    -------
    pandas.DataFrame
        Long table: ``seed``, ``arm``, ``ranking``, ``metric``, ``value``.
    """
    rows = []
    ids = customers["customer_id"].to_numpy()
    tau, response = {}, {}
    for arm, scores in scores_by_arm.items():
        in_pair = scores[scores["in_pair"]]
        treated, y = in_pair["treated"].to_numpy(), in_pair["y"].to_numpy()
        for learner in LEARNERS:
            q = qini_coefficient(in_pair[learner].to_numpy(), treated, y).normalised
            rows.append(
                {"seed": seed, "arm": arm, "ranking": learner, "metric": QINI_METRIC, "value": q}
            )
        part = scores.set_index("customer_id")
        tau[arm] = part["x_learner"].reindex(ids).to_numpy()
        response[arm] = part["response_model"].reindex(ids).to_numpy()
    inputs = PolicyInputs(
        customer_id=ids,
        arm=customers[config.ARM_COL].to_numpy().astype(str),
        y=customers[list(config.OUTCOME_COLS)].reset_index(drop=True).astype(float),
        tau=pd.DataFrame(tau),
        response=pd.DataFrame(response),
        pre=customers[list(config.PRE_TREATMENT_COLS)].reset_index(drop=True),
    )
    assignments, _ = build_policies(inputs, winner, np.random.default_rng(0), budget)
    y = inputs.y[OUTCOME].to_numpy()
    nobody, _ = policy_value(assignments["nobody"], inputs.arm, y)
    for ranking, metric in POLICY_METRICS.items():
        value, _ = policy_value(assignments[ranking], inputs.arm, y)
        rows.append(
            {
                "seed": seed,
                "arm": POLICY_ARM,
                "ranking": ranking,
                "metric": metric,
                "value": float(inputs.n * (value[0] - nobody[0])),
            }
        )
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> dict[str, Any]:
    """Median, min and max per (arm, ranking, metric) and the X-learner win count.

    Parameters
    ----------
    table : pandas.DataFrame
        Concatenated :func:`seed_metrics` rows over seeds.

    Returns
    -------
    dict
        ``spread`` rows, ``x_beats_response_model_womens`` (count and seeds) and
        ``n_seeds``.
    """
    spread = (
        table.groupby(["arm", "ranking", "metric"], sort=False)["value"]
        .agg(median="median", min="min", max="max", n_seeds="size")
        .reset_index()
    )
    seed0 = table[table["seed"] == 0].set_index(["arm", "ranking", "metric"])["value"]
    spread["seed_0"] = [
        float(seed0.get((a, r, m), np.nan))
        for a, r, m in zip(spread["arm"], spread["ranking"], spread["metric"], strict=True)
    ]
    womens = table[(table["arm"] == "womens") & (table["metric"] == QINI_METRIC)].pivot_table(
        index="seed", columns="ranking", values="value"
    )
    wins = womens["x_learner"] > womens["response_model"]
    return {
        "n_seeds": int(table["seed"].nunique()),
        "spread": json_records(spread),
        "x_beats_response_model_womens": {
            "count": int(wins.sum()),
            "out_of": int(wins.size),
            "seeds": [int(s) for s in wins.index[wins]],
        },
    }


def run_seed_robustness(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    scores_dir: Path = config.DATA_PROCESSED,
    n_seeds: int = N_SEEDS,
    budget: int = config.BUDGET,
    winner: str | None = None,
    scorer: Scorer = score_pair,
) -> pd.DataFrame:
    """Repeat the visit cross-fitting over fold seeds and write the table and results section.

    Parameters
    ----------
    db_path : Path
        DuckDB database with the ``customers`` table.
    reports_dir : Path
        Reports directory.
    scores_dir : Path
        Directory with the committed scores parquet (seed 0).
    n_seeds : int
        Total seeds including the committed one.
    budget : int
        Email budget for the policy metric.
    winner : str, optional
        A/B winner; read from ``results.json`` when omitted.
    scorer : callable
        ``scorer(pair, outcome, rng)`` producing a :func:`score_pair` table (tests inject a
        fast stand-in).

    Returns
    -------
    pandas.DataFrame
        The long per-seed table.
    """
    customers = load_customers(db_path)
    if winner is None:
        winner, _ = ab_winner(reports_dir / "results.json")
    committed = pd.read_parquet(scores_dir / SCORES_NAME)
    committed = committed[committed["outcome"] == OUTCOME]
    tables = [
        seed_metrics(
            {
                arm: committed[committed["arm"] == arm].reset_index(drop=True)
                for arm in config.TREATMENTS
            },
            customers,
            winner,
            seed=0,
            budget=budget,
        )
    ]
    for k, rng in enumerate(fold_seed_generators(n_seeds), start=1):
        logger.info("Seed %d of %d", k, n_seeds - 1)
        folds = assign_folds(customers, rng)  # one assignment of all customers per seed
        pairs = {arm: treatment_pair(customers, arm, folds) for arm in config.TREATMENTS}
        scores = {arm: scorer(pairs[arm], OUTCOME, rng) for arm in config.TREATMENTS}
        tables.append(seed_metrics(scores, customers, winner, seed=k, budget=budget))
    table = pd.concat(tables, ignore_index=True)
    write_table(table, TABLE_NAME, reports_dir / "tables")
    summary = summarise(table)
    update_results(
        LABEL,
        {
            "label": LABEL,
            "plan_section": "11",
            "outcome": OUTCOME,
            "learners": list(LEARNERS),
            "seed_0": "the committed run, read from the uplift-model stage's scores parquet",
            "other_seeds": (
                f"child generators spawned from config.SEEDS['uplift'] = {config.SEEDS['uplift']}"
            ),
            "budget": budget,
            "ab_winner": winner,
            "policy_metric": POLICY_METRIC,
            "note": (
                "Bootstrap intervals elsewhere hold the folds fixed; the spread across seeds "
                "shows the fold-assignment and model-seed variability they leave out."
            ),
            **summary,
            "table": f"tables/{TABLE_NAME}.csv",
            "rows": json_records(table),
        },
        reports_dir / "results.json",
    )
    logger.info("Seed robustness written: %d rows over %d seeds.", len(table), summary["n_seeds"])
    return table
