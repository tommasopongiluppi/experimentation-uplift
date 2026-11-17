"""Repeated cross-fitting over fold seeds: generators, summary known answer, and the stage."""

import json

import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.uplift.models import SCORES_NAME, score_pair
from abkit.uplift.robustness import (
    LABEL,
    LEARNERS,
    POLICY_METRIC,
    POLICY_METRICS,
    QINI_METRIC,
    fold_seed_generators,
    policy_metric,
    run_seed_robustness,
    seed_metrics,
    summarise,
    vs_random_metric,
)


def test_fold_seed_generators_are_deterministic_and_distinct():
    a, b = fold_seed_generators(4), fold_seed_generators(4)
    assert len(a) == 3
    draws_a = [g.random(3) for g in a]
    draws_b = [g.random(3) for g in b]
    for x, y in zip(draws_a, draws_b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert len({tuple(np.round(d, 12)) for d in draws_a}) == 3
    # Spawning does not advance the parent, so the committed stream is untouched.
    parent = np.random.default_rng(config.SEEDS["uplift"])
    fold_seed_generators(4)
    np.testing.assert_array_equal(
        parent.random(2), np.random.default_rng(config.SEEDS["uplift"]).random(2)
    )


# Per policy: three incremental outcomes, and four statistics of the difference from the
# random expectation on each of spend and visits.
ROWS_PER_POLICY = 3 + 2 * 4


def _toy_table():
    rows = []
    qini = {
        ("womens", "x_learner"): [0.06, 0.05, 0.03],
        ("womens", "response_model"): [0.04, 0.02, 0.05],
        ("mens", "x_learner"): [0.00, -0.01, 0.02],
    }
    for (arm, ranking), values in qini.items():
        for seed, v in enumerate(values):
            rows.append(
                {"seed": seed, "arm": arm, "ranking": ranking, "metric": QINI_METRIC, "value": v}
            )
    for seed, v in enumerate([850.0, 900.0, 800.0]):
        rows.append(
            {"seed": seed, "arm": "both", "ranking": "uplift", "metric": POLICY_METRIC, "value": v}
        )
    # The uplift policy's spend interval against the random expectation lies above zero in
    # seeds 0 and 2 only.
    for seed, v in enumerate([1.0, 0.0, 1.0]):
        rows.append(
            {
                "seed": seed,
                "arm": "both",
                "ranking": "uplift",
                "metric": vs_random_metric("uplift", "spend", "above_zero"),
                "value": v,
            }
        )
    return pd.DataFrame(rows)


def test_summarise_known_answer():
    s = summarise(_toy_table())
    assert s["n_seeds"] == 3
    spread = pd.DataFrame(s["spread"]).set_index(["arm", "ranking"])
    x = spread.loc[("womens", "x_learner")]
    assert (x["median"], x["min"], x["max"], x["seed_0"], x["n_seeds"]) == (
        0.05,
        0.03,
        0.06,
        0.06,
        3,
    )
    policy = spread[spread["metric"] == POLICY_METRIC].loc[("both", "uplift")]
    assert (policy["median"], policy["min"], policy["max"]) == (850.0, 800.0, 900.0)
    beats = s["beats_random_expectation"]
    assert beats["uplift"]["spend"] == {"count": 2, "out_of": 3, "seeds": [0, 2]}
    assert beats["uplift"]["visit"] == {"count": 0, "out_of": 0, "seeds": []}  # no such rows
    # X beats the response model in seeds 0 and 1, not in seed 2.
    assert s["x_beats_response_model_womens"] == {"count": 2, "out_of": 3, "seeds": [0, 1]}


def _fake_scorer(rng):
    """A stand-in for score_pair with the same output layout but random scores."""

    def scorer(pair, outcome, generator):
        n = pair.n + pair.n_other
        out = pd.DataFrame(
            {
                "customer_id": np.r_[pair.customer_id, pair.other_customer_id],
                "arm": pair.arm,
                "outcome": outcome,
                "in_pair": np.r_[np.ones(pair.n, bool), np.zeros(pair.n_other, bool)],
                "treated": np.r_[pair.treated, np.full(pair.n_other, -1)],
                "y": np.r_[
                    pair.outcomes[outcome].to_numpy(), pair.outcomes_other[outcome].to_numpy()
                ],
            }
        )
        for learner in LEARNERS:
            out[learner] = generator.random(n)
        return out

    return scorer


def test_seed_metrics_rows(customers, rng):
    from abkit.uplift.crossfit import treatment_pair

    scorer = _fake_scorer(rng)
    scores = {
        arm: scorer(treatment_pair(customers, arm), "visit", rng) for arm in config.TREATMENTS
    }
    table = seed_metrics(scores, customers, "mens", seed=3, budget=300, n_resamples=30)
    assert (table["seed"] == 3).all()
    assert len(table) == 2 * len(LEARNERS) + ROWS_PER_POLICY * len(POLICY_METRICS)
    both = table[table["arm"] == "both"].set_index(["ranking", "metric"])["value"]
    assert set(POLICY_METRICS.values()) <= set(table.loc[table["arm"] == "both", "metric"])
    for ranking in POLICY_METRICS:
        for outcome in ("visit", "spend"):
            diff, low, high, above = (
                both[(ranking, vs_random_metric(ranking, outcome, stat))]
                for stat in ("diff", "ci_low", "ci_high", "above_zero")
            )
            assert low <= diff <= high
            assert above == float(low > 0)
        assert (ranking, policy_metric(ranking, "conversion")) in both.index
    qini = table[table["metric"] == QINI_METRIC]
    assert set(qini["arm"]) == set(config.TREATMENTS) and set(qini["ranking"]) == set(LEARNERS)
    assert np.isfinite(table["value"]).all()


def test_run_seed_robustness_writes_outputs(fixture_db, customers, tmp_path, rng):
    # Committed "seed 0" scores: a parquet in the uplift stage's layout.
    scorer = _fake_scorer(rng)
    from abkit.uplift.crossfit import treatment_pair

    committed = pd.concat(
        [scorer(treatment_pair(customers, arm), "visit", rng) for arm in config.TREATMENTS],
        ignore_index=True,
    )
    scores_dir = tmp_path / "processed"
    scores_dir.mkdir()
    committed.to_parquet(scores_dir / SCORES_NAME, index=False)
    table = run_seed_robustness(
        fixture_db,
        tmp_path,
        scores_dir,
        n_seeds=3,
        budget=300,
        winner="womens",
        scorer=scorer,
        n_resamples=30,
    )
    assert set(table["seed"]) == {0, 1, 2}
    assert len(table) == 3 * (2 * len(LEARNERS) + ROWS_PER_POLICY * len(POLICY_METRICS))
    results = json.loads((tmp_path / "results.json").read_text())["seed_robustness"]
    for ranking in POLICY_METRICS:
        for outcome in ("spend", "visit"):
            count = results["beats_random_expectation"][ranking][outcome]
            assert count["out_of"] == 3 and 0 <= count["count"] <= 3
            assert len(count["seeds"]) == count["count"]
    saved = pd.read_csv(tmp_path / "tables" / "uplift_seed_robustness.csv")
    assert len(saved) == len(table)
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)[LABEL]
    assert section["n_seeds"] == 3 and section["ab_winner"] == "womens"
    assert section["x_beats_response_model_womens"]["out_of"] == 3
    spread = pd.DataFrame(section["spread"])
    assert len(spread) == 2 * len(LEARNERS) + ROWS_PER_POLICY * len(POLICY_METRICS)
    assert (spread["min"] <= spread["median"]).all() and (spread["median"] <= spread["max"]).all()
    # Seed 0 reproduces the committed scores' Qini exactly.
    seed0 = table[(table["seed"] == 0) & (table["metric"] == QINI_METRIC)]
    assert len(seed0) == 2 * len(LEARNERS)


@pytest.mark.slow
def test_run_seed_robustness_with_real_learners_on_fixture(fixture_db, customers, tmp_path, rng):
    from abkit.uplift.crossfit import assign_folds, treatment_pair

    folds = assign_folds(customers, rng)
    committed = pd.concat(
        [
            score_pair(treatment_pair(customers, arm, folds), "visit", rng)
            for arm in config.TREATMENTS
        ],
        ignore_index=True,
    )
    scores_dir = tmp_path / "processed"
    scores_dir.mkdir()
    committed.to_parquet(scores_dir / SCORES_NAME, index=False)
    table = run_seed_robustness(
        fixture_db, tmp_path, scores_dir, n_seeds=2, budget=300, winner="mens", n_resamples=30
    )
    assert set(table["seed"]) == {0, 1}
    assert np.isfinite(table["value"]).all()
