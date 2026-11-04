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
from abkit.uplift.features import assert_pre_treatment_only, build_features

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
    """

    arm: str
    customer_id: np.ndarray
    X: pd.DataFrame
    treated: np.ndarray
    outcomes: pd.DataFrame
    p_treat: float

    @property
    def n(self) -> int:
        """Number of customers in the pair."""
        return int(self.treated.size)


def treatment_pair(df: pd.DataFrame, arm: str) -> TreatmentPair:
    """Subset a cleaned ``customers`` frame to one email arm plus control.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned ``customers`` rows with ``customer_id``, ``arm``, the pre-treatment columns
        and the outcome columns.
    arm : str
        One of ``config.TREATMENTS``.

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
    sub = df.loc[keep]
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
    )


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
    assert_pre_treatment_only(X.columns)
    treated, y, folds = np.asarray(treated), np.asarray(y), np.asarray(folds)
    scores = np.full(len(X), np.nan)
    for k in np.unique(folds):
        test = folds == k
        train = ~test
        out = np.asarray(
            fit_predict(X.loc[train], treated[train], y[train], X.loc[test]), dtype=float
        )
        if out.shape != (int(test.sum()),):
            raise ValueError(f"fit_predict returned shape {out.shape} for fold {k}.")
        scores[test] = out
    return scores
