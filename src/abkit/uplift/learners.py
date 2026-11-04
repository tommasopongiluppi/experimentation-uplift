"""Uplift meta-learners and the response-model baseline on scikit-learn gradient boosting.

Every learner is a callable with the :data:`abkit.uplift.crossfit.FitPredict` signature
``fit_predict(X_train, treated_train, y_train, X_test) -> scores``, so :func:`cross_fit`
drives all of them over the same folds. Base models are ``HistGradientBoostingClassifier``
(binary outcomes) and ``HistGradientBoostingRegressor`` (the X-learner's second stage), with
the 0/1 columns passed as categorical features via :func:`categorical_mask`.

Meta-learners, with ``mu_1`` and ``mu_0`` the outcome models under treatment and control:

- **S-learner:** one classifier with the treatment indicator as an extra feature;
  ``tau(x) = mu(x, 1) - mu(x, 0)``.
- **T-learner:** separate classifiers for the treated and the control rows;
  ``tau(x) = mu_1(x) - mu_0(x)``.
- **X-learner** (Kunzel et al., 2019): T-learner first stage, then imputed individual effects
  ``D_i = y_i - mu_0(x_i)`` for treated rows and ``D_i = mu_1(x_i) - y_i`` for control rows,
  regressed on x to give ``tau_1`` and ``tau_0``; combined with the known propensity ``g``
  (the realised treated share) as ``tau(x) = g tau_0(x) + (1 - g) tau_1(x)``.
- **Response model** (baseline): a classifier for ``P(y = 1 | treated, x)`` fitted on treated
  rows only; its score ranks customers by how likely they are to respond when emailed, not by
  how much the email changes their response.

Hyperparameters come from a small fixed grid chosen on an inner validation split of the
training fold (log loss for classifiers, squared error for regressors), and every fit uses
early stopping on its own internal validation fraction. The evaluation fold is never used
for any choice. Random seeds are drawn from the ``numpy.random.Generator`` each learner holds.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from abkit.uplift.features import assert_pre_treatment_only, categorical_mask

logger = logging.getLogger(__name__)

GRID: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 15},
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
)
FIXED_PARAMS: dict[str, Any] = {
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "min_samples_leaf": 50,
}
INNER_VALIDATION_FRACTION: float = 0.2
#: A hyperparameter grid: one dict of ``HistGradientBoosting*`` parameters per candidate.
Grid = tuple[dict[str, Any], ...]
TREATMENT_FEATURE: str = "treated"


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an integer ``random_state`` for scikit-learn from a numpy Generator."""
    return int(rng.integers(0, np.iinfo(np.int32).max))


class _Constant:
    """Stand-in estimator when the training outcome has a single value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.value)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


@dataclass
class FitResult:
    """A fitted base model with the grid point chosen and its inner validation loss.

    Attributes
    ----------
    estimator : object
        Fitted scikit-learn estimator (or a constant predictor if ``y`` had one value).
    params : dict
        The grid point selected.
    inner_loss : float
        Validation loss of that grid point on the inner split (log loss or squared error).
    n_iter : int
        Boosting iterations of the refitted model after early stopping.
    """

    estimator: Any
    params: dict[str, Any]
    inner_loss: float
    n_iter: int


def _fit_best(
    X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, classifier: bool, grid: Grid
) -> FitResult:
    y = np.asarray(y, dtype=float)
    if np.unique(y).size < 2:
        return FitResult(_Constant(float(y[0])), {}, float("nan"), 0)
    cat = categorical_mask(X.columns, extra=(TREATMENT_FEATURE,))
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=INNER_VALIDATION_FRACTION,
        stratify=y if classifier else None,
        random_state=draw_seed(rng),
    )
    best: tuple[float, dict[str, Any]] | None = None
    for params in grid:
        est = cls(**FIXED_PARAMS, **params, categorical_features=cat, random_state=draw_seed(rng))
        est.fit(X_fit, y_fit)
        if classifier:
            loss = log_loss(y_val, est.predict_proba(X_val)[:, 1], labels=[0, 1])
        else:
            loss = mean_squared_error(y_val, est.predict(X_val))
        if best is None or loss < best[0]:
            best = (float(loss), params)
    assert best is not None
    est = cls(**FIXED_PARAMS, **best[1], categorical_features=cat, random_state=draw_seed(rng))
    est.fit(X, y)
    return FitResult(est, best[1], best[0], int(est.n_iter_))


def fit_classifier(
    X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, grid: Grid = GRID
) -> FitResult:
    """Fit a gradient boosting classifier with grid selection on an inner split by log loss.

    Parameters
    ----------
    X : pandas.DataFrame
        Features; the 0/1 columns are passed as categorical.
    y : numpy.ndarray
        0/1 outcome.
    rng : numpy.random.Generator
        Source of the split and model seeds.
    grid : Grid
        Candidate hyperparameters; the pipeline's fast mode passes a single point.

    Returns
    -------
    FitResult
    """
    return _fit_best(X, y, rng, classifier=True, grid=grid)


def fit_regressor(
    X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, grid: Grid = GRID
) -> FitResult:
    """Fit a gradient boosting regressor with grid selection on an inner split by squared error.

    Parameters
    ----------
    X : pandas.DataFrame
        Features; the 0/1 columns are passed as categorical.
    y : numpy.ndarray
        Continuous target (imputed individual effects for the X-learner).
    rng : numpy.random.Generator
        Source of the split and model seeds.
    grid : Grid
        Candidate hyperparameters; the pipeline's fast mode passes a single point.

    Returns
    -------
    FitResult
    """
    return _fit_best(X, y, rng, classifier=False, grid=grid)


def _proba(fit: FitResult, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(fit.estimator.predict_proba(X))[:, 1]


@dataclass
class SLearner:
    """One classifier with the treatment indicator as a feature; score = mu(x,1) - mu(x,0).

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of seeds. ``fits`` collects one :class:`FitResult` per call.
    grid : Grid
        Hyperparameter grid searched on the inner split.
    """

    rng: np.random.Generator
    grid: Grid = GRID
    fits: list[FitResult] = field(default_factory=list)

    def __call__(
        self,
        X_train: pd.DataFrame,
        treated_train: np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Fit on the training rows and return predicted uplift for the test rows."""
        assert_pre_treatment_only(X_train.columns)
        fit = fit_classifier(
            X_train.assign(**{TREATMENT_FEATURE: treated_train.astype(float)}),
            y_train,
            self.rng,
            self.grid,
        )
        self.fits.append(fit)
        return _proba(fit, X_test.assign(**{TREATMENT_FEATURE: 1.0})) - _proba(
            fit, X_test.assign(**{TREATMENT_FEATURE: 0.0})
        )


@dataclass
class TLearner:
    """Separate classifiers for treated and control rows; score = mu_1(x) - mu_0(x).

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of seeds. ``fits`` collects ``(mu_1, mu_0)`` per call.
    """

    rng: np.random.Generator
    grid: Grid = GRID
    fits: list[tuple[FitResult, FitResult]] = field(default_factory=list)

    def first_stage(
        self, X_train: pd.DataFrame, treated_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[FitResult, FitResult]:
        """Fit ``mu_1`` on treated rows and ``mu_0`` on control rows."""
        assert_pre_treatment_only(X_train.columns)
        t = np.asarray(treated_train) == 1
        mu1 = fit_classifier(X_train.loc[t], y_train[t], self.rng, self.grid)
        mu0 = fit_classifier(X_train.loc[~t], y_train[~t], self.rng, self.grid)
        self.fits.append((mu1, mu0))
        return mu1, mu0

    def __call__(
        self,
        X_train: pd.DataFrame,
        treated_train: np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Fit on the training rows and return predicted uplift for the test rows."""
        mu1, mu0 = self.first_stage(X_train, treated_train, y_train)
        return _proba(mu1, X_test) - _proba(mu0, X_test)


@dataclass
class XLearner:
    """Kunzel et al. (2019) X-learner with the known propensity ``p_treat``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of seeds.
    p_treat : float
        Known treatment probability (the realised share in the pair), used as ``g(x)``.
    """

    rng: np.random.Generator
    p_treat: float
    grid: Grid = GRID
    fits: list[dict[str, FitResult]] = field(default_factory=list)

    def __call__(
        self,
        X_train: pd.DataFrame,
        treated_train: np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Fit both stages on the training rows and return predicted uplift for the test rows."""
        if not 0 < self.p_treat < 1:
            raise ValueError(f"p_treat must be in (0, 1), got {self.p_treat}.")
        t = np.asarray(treated_train) == 1
        mu1, mu0 = TLearner(self.rng, self.grid).first_stage(X_train, treated_train, y_train)
        d1 = y_train[t] - _proba(mu0, X_train.loc[t])  # treated: observed minus counterfactual
        d0 = _proba(mu1, X_train.loc[~t]) - y_train[~t]  # control: counterfactual minus observed
        tau1 = fit_regressor(X_train.loc[t], d1, self.rng, self.grid)
        tau0 = fit_regressor(X_train.loc[~t], d0, self.rng, self.grid)
        self.fits.append({"mu1": mu1, "mu0": mu0, "tau1": tau1, "tau0": tau0})
        g = self.p_treat
        return g * tau0.estimator.predict(X_test) + (1 - g) * tau1.estimator.predict(X_test)


@dataclass
class ResponseModel:
    """Baseline: P(y = 1 | treated, x) from the treated rows; ranks likely responders.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of seeds.
    """

    rng: np.random.Generator
    grid: Grid = GRID
    fits: list[FitResult] = field(default_factory=list)

    def __call__(
        self,
        X_train: pd.DataFrame,
        treated_train: np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Fit on the treated training rows and return P(y = 1 | treated, x) for the test rows."""
        assert_pre_treatment_only(X_train.columns)
        t = np.asarray(treated_train) == 1
        fit = fit_classifier(X_train.loc[t], y_train[t], self.rng, self.grid)
        self.fits.append(fit)
        return _proba(fit, X_test)


def recency_rule_score(
    recency: np.ndarray, history: np.ndarray, customer_id: np.ndarray
) -> np.ndarray:
    """Score for the recency rule: most recent first, ties by higher history, then customer id.

    Parameters
    ----------
    recency : numpy.ndarray
        Months since last purchase.
    history : numpy.ndarray
        Past spend.
    customer_id : numpy.ndarray
        Identifier, lowest first on remaining ties.

    Returns
    -------
    numpy.ndarray
        Distinct scores, higher for customers the rule targets earlier (minus the rank).
    """
    order = np.lexsort((customer_id, -np.asarray(history), np.asarray(recency)))
    score = np.empty(order.size, dtype=float)
    score[order] = -np.arange(order.size, dtype=float)
    return score


def random_score(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random scores, the random-targeting baseline."""
    return rng.random(n)
