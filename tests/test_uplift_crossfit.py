"""Treatment pairs, stratified folds and out-of-fold scoring."""

import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.uplift.crossfit import (
    N_FOLDS,
    cross_fit,
    cross_fit_with_other,
    stratified_folds,
    treatment_pair,
)
from abkit.uplift.features import FEATURE_NAMES, ForbiddenFeatureError


@pytest.mark.parametrize("arm", config.TREATMENTS)
def test_treatment_pair_uses_realised_share(customers, arm):
    pair = treatment_pair(customers, arm)
    counts = customers[config.ARM_COL].value_counts()
    n_email, n_control = counts[arm], counts[config.CONTROL]
    assert pair.arm == arm
    assert pair.n == n_email + n_control
    assert pair.p_treat == pytest.approx(n_email / (n_email + n_control))
    assert pair.treated.sum() == n_email
    assert list(pair.X.columns) == list(FEATURE_NAMES)
    assert list(pair.outcomes.columns) == list(config.OUTCOME_COLS)
    assert len(pair.X) == len(pair.outcomes) == pair.customer_id.size == pair.n
    other = next(a for a in config.TREATMENTS if a != arm)
    assert pair.n_other == counts[other]
    assert len(pair.X_other) == len(pair.outcomes_other) == pair.n_other
    assert set(pair.customer_id).isdisjoint(pair.other_customer_id)


def test_treatment_pair_rows_align_with_source(customers):
    pair = treatment_pair(customers, "womens")
    src = customers[customers[config.ARM_COL].isin(["womens", config.CONTROL])]
    np.testing.assert_array_equal(pair.customer_id, src["customer_id"].to_numpy())
    np.testing.assert_array_equal(pair.treated, (src[config.ARM_COL] == "womens").astype(int))
    np.testing.assert_array_equal(pair.outcomes["visit"].to_numpy(), src["visit"].to_numpy())
    assert "mens" not in set(src[config.ARM_COL])


def test_treatment_pair_rejects_bad_arm(customers):
    with pytest.raises(ValueError, match="arm must be one of"):
        treatment_pair(customers, "control")
    with pytest.raises(ValueError, match="both arms"):
        treatment_pair(customers[customers[config.ARM_COL] == "mens"], "mens")


def test_stratified_folds_balance_every_cell(rng):
    n = 1_237
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.15).astype(int)
    folds = stratified_folds(treated, y, rng)
    assert folds.shape == (n,)
    assert set(folds) == set(range(N_FOLDS))
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 1
    for t in (0, 1):
        for o in (0, 1):
            cell = np.bincount(folds[(treated == t) & (y == o)], minlength=N_FOLDS)
            assert cell.max() - cell.min() <= 1


def test_stratified_folds_are_seeded():
    treated = np.arange(400) % 2
    y = (np.arange(400) % 7 == 0).astype(int)
    a = stratified_folds(treated, y, np.random.default_rng(3))
    b = stratified_folds(treated, y, np.random.default_rng(3))
    c = stratified_folds(treated, y, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_folds_require_binary(rng):
    with pytest.raises(ValueError, match="y must be 0/1"):
        stratified_folds(np.array([0, 1, 0, 1]), np.array([0.0, 2.5, 0.0, 1.0]), rng)
    with pytest.raises(ValueError, match="treated must be 0/1"):
        stratified_folds(np.array([0, 2, 0, 1]), np.array([0, 1, 0, 1]), rng)


def _train_mean(X_train, treated_train, y_train, X_test):
    return np.full(len(X_test), y_train.mean())


def test_cross_fit_scores_every_row_out_of_fold(rng):
    n = 500
    X = pd.DataFrame({"recency": rng.integers(1, 13, n), "history": rng.gamma(2, 100, n)})
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3).astype(int)
    folds = stratified_folds(treated, y, rng)
    scores = cross_fit(X, treated, y, _train_mean, folds)
    assert scores.shape == (n,)
    assert np.isfinite(scores).all()
    for k in range(N_FOLDS):
        # The known answer: rows in fold k are scored with the mean of y over the other folds.
        np.testing.assert_allclose(scores[folds == k], y[folds != k].mean())


def test_cross_fit_never_shows_a_row_its_own_outcome(rng):
    n = 300
    X = pd.DataFrame({"recency": np.arange(n), "history": np.zeros(n)})
    treated = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    folds = stratified_folds(treated, y, rng)
    seen: list[tuple[set[int], set[int]]] = []

    def record(X_train, treated_train, y_train, X_test):
        seen.append((set(X_train["recency"]), set(X_test["recency"])))
        return np.zeros(len(X_test))

    cross_fit(X, treated, y, record, folds)
    assert len(seen) == N_FOLDS
    assert all(train.isdisjoint(test) for train, test in seen)
    assert set().union(*(test for _, test in seen)) == set(range(n))


def test_cross_fit_with_other_averages_fold_models(rng):
    n, n_other = 300, 40
    X = pd.DataFrame({"recency": rng.integers(1, 13, n), "history": rng.gamma(2, 100, n)})
    X_other = pd.DataFrame({"recency": np.ones(n_other), "history": np.ones(n_other)})
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3).astype(int)
    folds = stratified_folds(treated, y, rng)
    scores, other = cross_fit_with_other(X, treated, y, _train_mean, folds, X_other)
    np.testing.assert_allclose(scores, cross_fit(X, treated, y, _train_mean, folds))
    # Each fold model scores the other rows with its training mean; the result is the average.
    expected = np.mean([y[folds != k].mean() for k in range(N_FOLDS)])
    np.testing.assert_allclose(other, expected)
    assert other.shape == (n_other,)


def test_cross_fit_guards_features_and_shapes(rng):
    n = 100
    treated, y = rng.integers(0, 2, n), rng.integers(0, 2, n)
    folds = stratified_folds(treated, y, rng)
    leaky = pd.DataFrame({"recency": np.ones(n), "visit": y})
    with pytest.raises(ForbiddenFeatureError):
        cross_fit(leaky, treated, y, _train_mean, folds)
    X = pd.DataFrame({"recency": np.ones(n)})
    with pytest.raises(ValueError, match="shape"):
        cross_fit(X, treated, y, lambda *a: np.zeros(3), folds)
