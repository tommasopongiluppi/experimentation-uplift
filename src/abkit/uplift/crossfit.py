"""Treatment pairs and cross-fitting for the uplift models.

Each email is modelled against control as its own binary problem (analysis plan §11):
``mens`` vs ``control`` and ``womens`` vs ``control``. The treatment probability inside a pair
is known by design, so it is the realised share of the email arm in that pair, not a fitted
propensity.

Cross-fitting: customers are split into K folds, stratified jointly by arm and outcome. For
each fold the models are fitted on the other K-1 folds and score the held-out fold, so every
customer receives exactly one out-of-fold score from models that never saw that customer. The
fold assignment is seeded through ``config.SEEDS["uplift"]``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from abkit import config
from abkit.uplift.features import (
    assert_no_leaked_values,
    assert_pre_treatment_only,
    build_features,
)

N_FOLDS: int = 5

FitPredict = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame], np.ndarray]
"""``fit_predict(X_train, treated_train, y_train, X_test) -> scores for X_test``."""


@dataclass(frozen=True)
class TreatmentPair:
    """One email arm against control, ready for uplift modelling.

    Attributes
    ----------
    arm : str
        The email arm ("mens" or "womens").
    customer_id : numpy.ndarray
        Customer identifiers, aligned with the rows of ``X``.
    X : pandas.DataFrame
        Feature matrix from :func:`abkit.uplift.features.build_features`.
    treated : numpy.ndarray
        1 for customers in the email arm, 0 for control.
    outcomes : pandas.DataFrame
        The outcome columns (``visit``, ``conversion``, ``spend``), aligned with ``X``.
        Targets for the models and for evaluation; never features.
    p_treat : float
        Realised share of treated customers in the pair, ``treated.mean()``.
    other_customer_id, X_other, outcomes_other : numpy.ndarray, pandas.DataFrame
        The customers in the third arm (the other email). They take no part in fitting or
        evaluating this pair's models, but the policy stage needs this pair's score for them,
        so :func:`cross_fit_with_other` scores each of them with the one fold model of
        their own fold, exactly as it scores the pair's held-out customers.
    folds, folds_other : numpy.ndarray or None
        Fold of every customer in the pair and in the third arm, from one assignment of all
        customers (:func:`assign_folds`). None when the pair was built without folds.
    """

    arm: str
    customer_id: np.ndarray
    X: pd.DataFrame
    treated: np.ndarray
    outcomes: pd.DataFrame
    p_treat: float
    other_customer_id: np.ndarray
    X_other: pd.DataFrame
    outcomes_other: pd.DataFrame
    folds: np.ndarray | None = None
    folds_other: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Number of customers in the pair."""
        return int(self.treated.size)

    @property
    def n_other(self) -> int:
        """Number of customers in the third arm."""
        return int(self.other_customer_id.size)


def treatment_pair(df: pd.DataFrame, arm: str, folds: np.ndarray | None = None) -> TreatmentPair:
    """Subset a cleaned ``customers`` frame to one email arm plus control.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned ``customers`` rows with ``customer_id``, ``arm``, the pre-treatment columns
        and the outcome columns.
    arm : str
        One of ``config.TREATMENTS``.
    folds : numpy.ndarray, optional
        Fold of every row of ``df`` from :func:`assign_folds`; split into the pair's and the
        third arm's folds.

    Returns
    -------
    TreatmentPair
        Rows in the original order, with ``p_treat`` the realised treated share.

    Raises
    ------
    ValueError
        If ``arm`` is not an email arm or either arm of the pair is empty.
    """
    if arm not in config.TREATMENTS:
        raise ValueError(f"arm must be one of {config.TREATMENTS}, got {arm!r}.")
    keep = df[config.ARM_COL].isin([arm, config.CONTROL]).to_numpy()
    sub, other = df.loc[keep], df.loc[~keep]
    if folds is not None:
        folds = np.asarray(folds)
        if folds.shape != (len(df),):
            raise ValueError(f"folds must have one entry per row of df, got {folds.shape}.")
    treated = (sub[config.ARM_COL].to_numpy() == arm).astype(int)
    if treated.sum() == 0 or treated.sum() == treated.size:
        raise ValueError(f"Pair {arm!r} vs control needs customers in both arms.")
    return TreatmentPair(
        arm=arm,
        customer_id=sub["customer_id"].to_numpy(),
        X=build_features(sub),
        treated=treated,
        outcomes=sub[list(config.OUTCOME_COLS)].reset_index(drop=True),
        p_treat=float(treated.mean()),
        other_customer_id=other["customer_id"].to_numpy(),
        X_other=build_features(other),
        outcomes_other=other[list(config.OUTCOME_COLS)].reset_index(drop=True),
        folds=None if folds is None else folds[keep],
        folds_other=None if folds is None else folds[~keep],
    )


def assign_folds(df: pd.DataFrame, rng: np.random.Generator, n_folds: int = N_FOLDS) -> np.ndarray:
    """Assign every customer, whatever their arm, to one fold.

    One assignment serves both email-vs-control pairs and both outcomes. In each pair the
    model of fold ``k`` is fitted on the pair's customers outside fold ``k`` and scores
    every customer in fold ``k``: the pair's held-out customers and the third arm's. Every
    customer therefore gets exactly one score per pair, from one model that did not see
    them, and the noise of that score does not depend on the customer's arm.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned ``customers`` rows with ``arm``, ``visit`` and ``conversion``.
    rng : numpy.random.Generator
        Source of randomness; one integer is drawn from it as scikit-learn's ``random_state``.
    n_folds : int
        Number of folds.

    Returns
    -------
    numpy.ndarray
        Integer fold id in ``[0, n_folds)`` per row of ``df``, stratified jointly by arm,
        visit and conversion, so every pair and outcome is stratified by arm and outcome.
    """
    strata = (
        df[config.ARM_COL].astype(str)
        + "_"
        + df["visit"].astype(int).astype(str)
        + "_"
        + df["conversion"].astype(int).astype(str)
    ).to_numpy()
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(df), dtype=int)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros((len(df), 1)), strata)):
        folds[test_idx] = k
    return folds


def stratified_folds(
    treated: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = N_FOLDS,
) -> np.ndarray:
    """Assign every row to a fold, stratified jointly by arm and a binary outcome.

    Parameters
    ----------
    treated : numpy.ndarray
        0/1 treatment indicator.
    y : numpy.ndarray
        0/1 outcome used for stratification (``visit`` or ``conversion``).
    rng : numpy.random.Generator
        Source of randomness; one integer is drawn from it as scikit-learn's ``random_state``.
    n_folds : int
        Number of folds.

    Returns
    -------
    numpy.ndarray
        Integer fold id in ``[0, n_folds)`` per row. Each of the four (arm, outcome) cells is
        spread as evenly as possible across folds.

    Raises
    ------
    ValueError
        If ``y`` or ``treated`` is not binary.
    """
    treated, y = np.asarray(treated), np.asarray(y)
    for name, v in (("treated", treated), ("y", y)):
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f"{name} must be 0/1 for stratification.")
    strata = 2 * treated + y
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(strata.size, dtype=int)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros((strata.size, 1)), strata)):
        folds[test_idx] = k
    return folds


def cross_fit_with_other(
    X: pd.DataFrame,
    treated: np.ndarray,
    y: np.ndarray,
    fit_predict: FitPredict,
    folds: np.ndarray,
    X_other: pd.DataFrame,
    folds_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores for ``X`` and for rows outside the pair, one model per row.

    The model of fold ``k`` is fitted on the rows of ``X`` outside fold ``k`` and scores the
    rows of ``X`` in fold ``k`` and the rows of ``X_other`` whose fold is ``k``. Every row,
    inside or outside the pair, is scored by exactly one model, and no model saw a row it
    scores. Scoring the outside rows with the average of all fold models instead would give
    them less noisy scores than the pair's rows, and a rule that selects on the scores would
    then select on the arm.

    Parameters
    ----------
    X, treated, y, fit_predict, folds
        As in :func:`cross_fit`.
    X_other : pandas.DataFrame
        Feature rows that take no part in fitting (the third arm). May be empty.
    folds_other : numpy.ndarray
        Fold of every row of ``X_other``; each must be one of the folds of ``X``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(scores, other_scores)``: one out-of-fold score per row of ``X`` and one per row
        of ``X_other``.

    Raises
    ------
    ForbiddenFeatureError
        If a feature is named after, or reproduces the values of, the outcome or treatment.
    ValueError
        If ``folds_other`` has the wrong length or names a fold ``X`` does not have, or if
        ``fit_predict`` returns the wrong number of scores.
    """
    assert_pre_treatment_only(X.columns)
    assert_pre_treatment_only(X_other.columns)
    treated, y, folds = np.asarray(treated), np.asarray(y), np.asarray(folds)
    assert_no_leaked_values(X, {"the treatment": treated, "the outcome": y})
    folds_other = np.asarray(folds_other)
    fold_ids = np.unique(folds)
    if folds_other.shape != (len(X_other),) or not np.isin(folds_other, fold_ids).all():
        raise ValueError("folds_other must give every row of X_other one of the folds of X.")
    scores = np.full(len(X), np.nan)
    other = np.full(len(X_other), np.nan)
    for k in fold_ids:
        test, test_other = folds == k, folds_other == k
        train = ~test
        n_test = int(test.sum())
        X_score = pd.concat([X.loc[test], X_other.loc[test_other]], ignore_index=True)
        out = np.asarray(fit_predict(X.loc[train], treated[train], y[train], X_score), dtype=float)
        if out.shape != (len(X_score),):
            raise ValueError(f"fit_predict returned shape {out.shape} for fold {k}.")
        scores[test] = out[:n_test]
        other[test_other] = out[n_test:]
    return scores, other


def cross_fit(
    X: pd.DataFrame,
    treated: np.ndarray,
    y: np.ndarray,
    fit_predict: FitPredict,
    folds: np.ndarray,
) -> np.ndarray:
    """Out-of-fold scores: for each fold, fit on the other folds and score the held-out rows.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix; its columns pass :func:`assert_pre_treatment_only`.
    treated : numpy.ndarray
        0/1 treatment indicator.
    y : numpy.ndarray
        Outcome used to fit the models.
    fit_predict : callable
        ``fit_predict(X_train, treated_train, y_train, X_test)`` returning one score per
        row of ``X_test``. Any learner that fits this signature can be cross-fitted.
    folds : numpy.ndarray
        Fold id per row from :func:`stratified_folds`.

    Returns
    -------
    numpy.ndarray
        One score per row of ``X``, each produced by models that did not see that row.

    Raises
    ------
    ForbiddenFeatureError
        If ``X`` contains an outcome or arm column.
    ValueError
        If ``fit_predict`` returns the wrong number of scores for a fold.
    """
    scores, _ = cross_fit_with_other(
        X, treated, y, fit_predict, folds, X.iloc[:0], np.empty(0, dtype=int)
    )
    return scores
