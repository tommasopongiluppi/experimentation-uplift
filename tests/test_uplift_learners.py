"""Meta-learners on synthetic data with a known effect, baselines, and the pipeline stage."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.uplift import learners as learners_module
from abkit.uplift.crossfit import assign_folds, cross_fit, stratified_folds, treatment_pair
from abkit.uplift.evaluation import qini_coefficient
from abkit.uplift.features import FEATURE_NAMES, build_features
from abkit.uplift.learners import (
    GRID,
    ResponseModel,
    SLearner,
    TLearner,
    XLearner,
    fit_classifier,
    fit_regressor,
    random_score,
    recency_rule_score,
)
from abkit.uplift.models import (
    BASELINES,
    LABEL,
    RANKINGS,
    UPLIFT_LEARNERS,
    compare_rankings,
    metrics_table,
    run_uplift_models,
    score_pair,
)


def _frame(rng, n):
    """Pre-treatment columns in the cleaned schema, drawn independently of treatment."""
    history = np.round(30 + rng.lognormal(np.log(150), 0.9, n), 2)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "usd_0_100",
            "history": history,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["urban", "suburban", "rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["phone", "web", "multichannel"], n),
        }
    )


def _synthetic(rng, n, p_treat=0.5):
    """Known heterogeneous effect: tau = 0.30 [recency <= 4] + 0.10 newbie - 0.05.

    The baseline rate also varies, 0.10 + 0.15 womens, so "likely to respond" (high
    baseline) and "persuadable" (high tau) are different customers.
    """
    df = _frame(rng, n)
    X = build_features(df)
    tau = 0.30 * (df["recency"] <= 4).to_numpy() + 0.10 * df["newbie"].to_numpy() - 0.05
    p0 = 0.10 + 0.15 * df["womens"].to_numpy()
    treated = (rng.random(n) < p_treat).astype(int)
    y = (rng.random(n) < p0 + tau * treated).astype(int)
    return X, treated, y, tau


# --- learners on synthetic data ----------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    """Fit every learner once on 4,000 rows and score 3,000 held-out rows."""
    rng = np.random.default_rng(config.SEEDS["uplift"])
    X, t, y, tau = _synthetic(rng, 7_000)
    train, test = np.arange(7_000) < 4_000, np.arange(7_000) >= 4_000
    models = {
        "s_learner": SLearner(rng),
        "t_learner": TLearner(rng),
        "x_learner": XLearner(rng, float(t[train].mean())),
        "response_model": ResponseModel(rng),
    }
    scores = {name: m(X.loc[train], t[train], y[train], X.loc[test]) for name, m in models.items()}
    return SimpleNamespace(
        X_test=X.loc[test], t=t[test], y=y[test], tau=tau[test], scores=scores, models=models
    )


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "s_learner"])
def test_learners_rank_customers_by_true_effect(fitted, name):
    score = fitted.scores[name]
    rho = stats.spearmanr(score, fitted.tau).statistic
    assert rho > 0.5, (name, rho)
    q = qini_coefficient(score, fitted.t, fitted.y).normalised
    rng = np.random.default_rng(0)
    random_q = np.array(
        [qini_coefficient(rng.random(score.size), fitted.t, fitted.y).normalised for _ in range(30)]
    )
    assert q > 0.1 and q > random_q.max(), (name, q, random_q.max())


def test_learners_recover_step_effect(fitted):
    recent = (fitted.X_test["recency"] <= 4).to_numpy()
    for name in UPLIFT_LEARNERS:
        score = fitted.scores[name]
        assert score[recent].mean() - score[~recent].mean() > 0.15, name


def test_response_model_targets_responders_not_persuadables(fitted):
    """The response model follows the baseline rate (womens), which is unrelated to tau."""
    womens = fitted.X_test["womens"].to_numpy() == 1
    resp = fitted.scores["response_model"]
    assert resp[womens].mean() - resp[~womens].mean() > 0.08
    assert (
        stats.spearmanr(resp, fitted.tau).statistic
        < stats.spearmanr(fitted.scores["t_learner"], fitted.tau).statistic
    )
    assert np.all((resp >= 0) & (resp <= 1))


def test_t_learner_score_is_difference_of_its_two_models(fitted):
    mu1, mu0 = fitted.models["t_learner"].fits[-1]
    expected = (
        mu1.estimator.predict_proba(fitted.X_test)[:, 1]
        - mu0.estimator.predict_proba(fitted.X_test)[:, 1]
    )
    np.testing.assert_allclose(fitted.scores["t_learner"], expected)


def test_x_learner_combines_second_stage_with_known_propensity(fitted):
    model = fitted.models["x_learner"]
    f = model.fits[-1]
    g = model.p_treat
    expected = g * f["tau0"].estimator.predict(fitted.X_test) + (1 - g) * f[
        "tau1"
    ].estimator.predict(fitted.X_test)
    np.testing.assert_allclose(fitted.scores["x_learner"], expected)
    assert set(f) == {"mu1", "mu0", "tau1", "tau0"}


def test_s_learner_score_is_treated_minus_control_prediction(fitted):
    fit = fitted.models["s_learner"].fits[-1]
    X1 = fitted.X_test.assign(treated=1.0)
    X0 = fitted.X_test.assign(treated=0.0)
    expected = fit.estimator.predict_proba(X1)[:, 1] - fit.estimator.predict_proba(X0)[:, 1]
    np.testing.assert_allclose(fitted.scores["s_learner"], expected)
    assert "treated" in fit.estimator.feature_names_in_


def test_response_model_fits_on_treated_rows_only(monkeypatch, rng):
    X, t, y, _ = _synthetic(rng, 600)
    seen = []
    original = learners_module.fit_classifier

    def spy(Xf, yf, r, *args):
        seen.append(len(Xf))
        return original(Xf, yf, r, *args)

    monkeypatch.setattr(learners_module, "fit_classifier", spy)
    ResponseModel(rng)(X, t, y, X.iloc[:5])
    assert seen == [int(t.sum())]


def test_x_learner_rejects_degenerate_propensity(rng):
    X, t, y, _ = _synthetic(rng, 200)
    with pytest.raises(ValueError, match="p_treat"):
        XLearner(rng, 1.0)(X, t, y, X)


# --- base-model fitting ------------------------------------------------------------------------


def test_fit_classifier_selects_from_grid_with_inner_loss(rng):
    X, t, y, _ = _synthetic(rng, 2_000)
    fit = fit_classifier(X.loc[t == 1], y[t == 1], rng)
    assert fit.params in GRID
    assert np.isfinite(fit.inner_loss) and fit.inner_loss > 0
    assert fit.n_iter >= 1
    assert fit.estimator.n_features_in_ == len(FEATURE_NAMES)


def test_fit_classifier_prefers_the_grid_point_with_lower_inner_loss(rng):
    X, t, y, _ = _synthetic(rng, 3_000)
    useless = {"learning_rate": 1e-6, "max_leaf_nodes": 2}
    good = {"learning_rate": 0.1, "max_leaf_nodes": 15}
    fit = fit_classifier(X, y * t, rng, grid=(useless, good))
    assert fit.params == good


def test_fit_regressor_recovers_a_mean_shift(rng):
    X, _, _, _ = _synthetic(rng, 2_000)
    target = 0.5 * (X["recency"] <= 4).to_numpy() + 0.05 * rng.standard_normal(2_000)
    fit = fit_regressor(X, target, rng)
    pred = fit.estimator.predict(X)
    assert fit.params in GRID
    assert abs(pred[(X["recency"] <= 4).to_numpy()].mean() - 0.5) < 0.05
    assert abs(pred[(X["recency"] > 4).to_numpy()].mean()) < 0.05


def test_constant_outcome_gives_constant_predictor(rng):
    X, _, _, _ = _synthetic(rng, 100)
    fit = fit_classifier(X, np.zeros(100), rng)
    np.testing.assert_array_equal(fit.estimator.predict_proba(X)[:, 1], 0.0)
    assert fit.params == {} and np.isnan(fit.inner_loss)


def test_fits_are_seeded():
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    X, t, y, _ = _synthetic(np.random.default_rng(1), 1_500)
    a = TLearner(rng_a)(X, t, y, X.iloc[:50])
    b = TLearner(rng_b)(X, t, y, X.iloc[:50])
    np.testing.assert_array_equal(a, b)


# --- baselines ---------------------------------------------------------------------------------


def test_recency_rule_score_orders_by_recency_then_history_then_id():
    recency = np.array([3, 1, 1, 2, 1])
    history = np.array([10.0, 50.0, 50.0, 99.0, 70.0])
    ids = np.array([5, 4, 3, 2, 1])
    score = recency_rule_score(recency, history, ids)
    # Expected order: id 1 (rec 1, hist 70), id 3 (rec 1, 50, lower id), id 4, id 2 (rec 2), id 5
    assert list(np.argsort(-score)) == [4, 2, 1, 3, 0]
    assert np.unique(score).size == 5


def test_random_score_is_seeded_uniform():
    a, b = (
        random_score(1000, np.random.default_rng(2)),
        random_score(1000, np.random.default_rng(2)),
    )
    np.testing.assert_array_equal(a, b)
    assert 0.4 < a.mean() < 0.6 and a.min() >= 0 and a.max() < 1


# --- cross-fitting integration -----------------------------------------------------------------


def test_cross_fitted_t_learner_beats_random(rng):
    X, t, y, tau = _synthetic(rng, 3_000)
    folds = stratified_folds(t, y, rng, n_folds=3)
    score = cross_fit(X, t, y, TLearner(rng), folds)
    assert np.isfinite(score).all()
    assert stats.spearmanr(score, tau).statistic > 0.4
    assert qini_coefficient(score, t, y).normalised > 0.1


# --- comparison with paired replicates ---------------------------------------------------------


def _fake_eval(estimate, reps):
    return SimpleNamespace(
        scalars=pd.DataFrame({"metric": ["qini_normalised"], "estimate": [estimate]}),
        replicates=pd.DataFrame({"qini_normalised": reps}),
    )


def test_compare_rankings_uses_paired_differences():
    base = np.linspace(-1, 1, 201)
    evals = {
        "a": _fake_eval(0.30, base + 0.30),  # best
        "b": _fake_eval(0.10, base + 0.10),  # paired diff exactly 0.20 on every resample
        "c": _fake_eval(0.25, base[::-1] + 0.25),  # unpaired noise: diff spans zero
    }
    table = compare_rankings(evals, "mens", "visit")
    assert (table["best"] == "a").all()
    b = table.set_index("ranking").loc["b"]
    assert b["diff"] == pytest.approx(0.20)
    assert (b["ci_low"], b["ci_high"]) == pytest.approx((0.20, 0.20))
    assert bool(b["meaningful"])
    c = table.set_index("ranking").loc["c"]
    assert c["ci_low"] < 0 < c["ci_high"] and not bool(c["meaningful"])
    a = table.set_index("ranking").loc["a"]
    assert a["diff"] == 0 and not bool(a["meaningful"])


def test_metrics_table_layout():
    evals = {
        "a": SimpleNamespace(
            scalars=pd.DataFrame(
                {"metric": ["qini_normalised"], "estimate": [0.1], "ci_low": [0], "ci_high": [1]}
            )
        )
    }
    table = metrics_table(evals, "womens", "conversion")
    assert list(table.columns[:5]) == ["arm", "outcome", "ranking", "kind", "metric"]
    assert table.loc[0, "kind"] == "baseline"


# --- stage on the fixture ----------------------------------------------------------------------


def test_score_pair_columns(customers):
    subset = customers.head(1_200)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="folds"):
        score_pair(treatment_pair(subset, "mens"), "visit", rng)
    pair = treatment_pair(subset, "mens", assign_folds(subset, rng))
    scores = score_pair(pair, "visit", rng)
    assert len(scores) == pair.n + pair.n_other
    assert scores["in_pair"].sum() == pair.n
    assert set(RANKINGS) <= set(scores.columns)
    assert scores[list(RANKINGS)].notna().all().all()
    in_pair = scores[scores["in_pair"]]
    assert set(in_pair["fold"]) == set(range(5))
    np.testing.assert_array_equal(in_pair["y"], pair.outcomes["visit"])
    other = scores[~scores["in_pair"]]
    # Third-arm customers are held out too: each has a fold and is scored by that fold's model.
    assert set(other["fold"]) == set(range(5)) and (other["treated"] == -1).all()
    np.testing.assert_array_equal(other["fold"], pair.folds_other)
    np.testing.assert_array_equal(other["customer_id"], pair.other_customer_id)


def test_run_uplift_models_writes_outputs(fixture_db, customers, tmp_path):
    metrics = run_uplift_models(
        fixture_db, tmp_path, tmp_path / "processed", n_resamples=30, outcomes=("visit",)
    )
    assert set(metrics["ranking"]) == set(RANKINGS)
    assert set(metrics["arm"]) == set(config.TREATMENTS)
    assert set(metrics["metric"]) >= {
        "qini_normalised",
        "auuc_normalised",
        "qini_gain",
        "uplift_at_10",
        "uplift_at_30",
    }
    assert (metrics["ci_low"] <= metrics["estimate"]).all()
    table = pd.read_csv(tmp_path / "tables" / "uplift_models.csv")
    assert len(table) == len(metrics)
    comparison = pd.read_csv(tmp_path / "tables" / "uplift_models_comparison.csv")
    assert len(comparison) == 2 * len(RANKINGS)
    assert comparison["meaningful"].dtype == bool
    for arm in config.TREATMENTS:
        assert (tmp_path / "figures" / f"qini_{arm}_visit.png").stat().st_size > 0
        assert (tmp_path / "figures" / f"calibration_{arm}_visit.png").stat().st_size > 0
    scores = pd.read_parquet(tmp_path / "processed" / "uplift_scores.parquet")
    assert len(scores) == len(config.TREATMENTS) * len(customers)  # every customer, per pair
    counts = customers[config.ARM_COL].value_counts()
    n_pairs = sum(counts[arm] + counts[config.CONTROL] for arm in config.TREATMENTS)
    assert scores["in_pair"].sum() == n_pairs
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)[LABEL]
    assert section["label"] == LABEL and "D7" in section["deviation"]
    assert section["baselines"] == list(BASELINES) and section["uplift_learners"] == list(
        UPLIFT_LEARNERS
    )
    assert set(section["summary"]) == {f"{arm}_visit" for arm in config.TREATMENTS}
    for s in section["summary"].values():
        assert s["best_ranking"] in RANKINGS
        assert set(s["meaningfully_better_than"]).isdisjoint(s["not_separated_from"])
    assert len(section["rows"]) == len(metrics)
