"""Treatment pairs, stratified folds and out-of-fold scoring."""

import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.frequentist import srm_test
from abkit.uplift.crossfit import (
    N_FOLDS,
    assign_folds,
    cross_fit,
    cross_fit_with_other,
    stratified_folds,
    treatment_pair,
)
from abkit.uplift.features import (
    FEATURE_NAMES,
    ForbiddenFeatureError,
    assert_no_leaked_values,
)


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


def test_cross_fit_with_other_scores_each_outside_row_with_one_fold_model(rng):
    n, n_other = 300, 40
    X = pd.DataFrame({"recency": rng.integers(1, 13, n), "history": rng.gamma(2, 100, n)})
    X_other = pd.DataFrame({"recency": np.ones(n_other), "history": np.ones(n_other)})
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3).astype(int)
    folds = stratified_folds(treated, y, rng)
    folds_other = rng.integers(0, N_FOLDS, n_other)
    scores, other = cross_fit_with_other(X, treated, y, _train_mean, folds, X_other, folds_other)
    np.testing.assert_allclose(scores, cross_fit(X, treated, y, _train_mean, folds))
    # An outside row in fold k is scored by the model of fold k alone (its training mean),
    # exactly like the pair's held-out rows of that fold, never by an average over models.
    expected = np.array([y[folds != k].mean() for k in folds_other])
    np.testing.assert_allclose(other, expected)
    for k in range(N_FOLDS):
        assert set(other[folds_other == k]) == set(scores[folds == k])
    with pytest.raises(ValueError, match="folds_other"):
        cross_fit_with_other(X, treated, y, _train_mean, folds, X_other, folds_other[:-1])
    with pytest.raises(ValueError, match="folds_other"):
        cross_fit_with_other(X, treated, y, _train_mean, folds, X_other, folds_other + N_FOLDS)


def test_assign_folds_covers_every_customer_once_and_is_stratified(customers, rng):
    folds = assign_folds(customers, rng)
    assert folds.shape == (len(customers),) and set(folds) == set(range(N_FOLDS))
    # The strata are arm x visit x conversion; each is spread over the folds as evenly as
    # possible, so every pair and outcome is stratified by arm and outcome.
    strata = [customers[config.ARM_COL], customers["visit"], customers["conversion"]]
    table = pd.crosstab(strata, folds)
    assert (table.max(axis=1) - table.min(axis=1) <= 1).all()
    np.testing.assert_array_equal(folds, assign_folds(customers, np.random.default_rng(12345)))
    assert not np.array_equal(folds, assign_folds(customers, np.random.default_rng(7)))
    # The pair and the third arm take their folds from the same assignment.
    for arm in config.TREATMENTS:
        pair = treatment_pair(customers, arm, folds)
        by_id = dict(zip(customers["customer_id"], folds, strict=True))
        np.testing.assert_array_equal(pair.folds, [by_id[i] for i in pair.customer_id])
        np.testing.assert_array_equal(pair.folds_other, [by_id[i] for i in pair.other_customer_id])
    with pytest.raises(ValueError, match="one entry per row"):
        treatment_pair(customers, "mens", folds[:-1])


TOP = 30  # the rule selects the top 1 / TOP of customers by predicted uplift


def _noise_learner(X_train, treated_train, y_train, X_test):
    """A T-learner by least squares: on pure-noise features its scores are fitting noise."""
    A = np.c_[np.ones(len(X_test)), X_test.to_numpy()]
    out = np.zeros(len(X_test))
    for arm, sign in ((1, 1.0), (0, -1.0)):
        rows = treated_train == arm
        B = np.c_[np.ones(int(rows.sum())), X_train.to_numpy()[rows]]
        out += sign * (A @ np.linalg.lstsq(B, y_train[rows], rcond=None)[0])
    return out


def _selected_arm_counts(rng, average_outside_rows: bool) -> dict[str, int]:
    """Arm counts among the customers a top-decile uplift rule selects, on pure noise."""
    n = 90_000
    arm = rng.choice(np.array(config.ARMS), n)
    X = pd.DataFrame(rng.standard_normal((n, 40)), columns=[f"recency_{i}" for i in range(40)])
    y = (rng.random(n) < 0.15).astype(int)
    folds = rng.integers(0, N_FOLDS, n)
    tau = {}
    for email in config.TREATMENTS:
        in_pair = np.isin(arm, [email, config.CONTROL])
        treated = (arm[in_pair] == email).astype(int)
        scores = np.empty(n)
        if average_outside_rows:  # the former behaviour: every fold model scores every outside row
            parts = [
                cross_fit_with_other(
                    X[in_pair], treated, y[in_pair], _noise_learner, folds[in_pair],
                    X[~in_pair], np.full(int((~in_pair).sum()), k),
                )
                for k in range(N_FOLDS)
            ]  # fmt: skip
            scores[in_pair] = parts[0][0]
            scores[~in_pair] = np.mean([other for _, other in parts], axis=0)
        else:
            scores[in_pair], scores[~in_pair] = cross_fit_with_other(
                X[in_pair], treated, y[in_pair], _noise_learner, folds[in_pair],
                X[~in_pair], folds[~in_pair],
            )  # fmt: skip
        tau[email] = scores
    best = np.maximum(tau["mens"], tau["womens"])
    selected = np.argsort(-best)[: n // TOP]
    return {a: int((arm[selected] == a).sum()) for a in config.ARMS}


def test_selection_on_scores_does_not_depend_on_the_arm(rng):
    """One model per customer keeps a score-based rule independent of the realised arm."""
    assert srm_test(_selected_arm_counts(rng, average_outside_rows=False)).passed
    # Averaging the fold models for the third arm gives it less noisy scores, so a rule that
    # picks the highest scores under-selects it: the check this test relies on can fail.
    averaged = _selected_arm_counts(rng, average_outside_rows=True)
    assert not srm_test(averaged).passed, averaged


def test_value_guard_catches_a_renamed_or_recoded_outcome(rng):
    n = 400
    X = pd.DataFrame({"recency": rng.integers(1, 13, n), "history": rng.gamma(2, 100, n)})
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3).astype(int)
    forbidden = {"the treatment": treated, "the outcome": y}
    assert_no_leaked_values(X, forbidden)  # clean features pass
    assert_no_leaked_values(X.assign(flag=rng.integers(0, 2, n)), forbidden)  # unrelated 0/1
    for name, column in (
        ("the outcome", y),  # a copy under an innocent name
        ("the outcome", 10.0 - 3.0 * y),  # rescaled
        ("the treatment", np.where(treated == 1, "a", "b")),  # relabelled
    ):
        with pytest.raises(ForbiddenFeatureError, match=f"newbie reproduces {name}"):
            assert_no_leaked_values(X.assign(newbie=column), forbidden)
    # cross_fit applies the check, so a leaked column cannot reach a model.
    folds = stratified_folds(treated, y, rng)
    with pytest.raises(ForbiddenFeatureError, match="reproduces the outcome"):
        cross_fit(X.assign(newbie=y), treated, y, _train_mean, folds)


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
