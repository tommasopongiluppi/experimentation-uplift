"""Feature construction from pre-treatment variables only.

The feature matrix for the uplift models uses the pre-treatment columns of the cleaned
``customers`` table and nothing else, never an outcome. ``recency`` and ``history`` stay
numeric, the 0/1 flags ``mens``, ``womens`` and ``newbie`` are kept as they are, and
``zip_code`` and ``channel`` are one-hot encoded with every level from ``config`` so the column
set is the same whatever subset of customers is passed in.

``history_segment`` is dropped: it is ``history`` binned into the seven dollar bands of
``config.HISTORY_BANDS``, so it carries no information the numeric column does not already
carry, and a tree model can place its own splits on ``history``. Analysis plan deviation D6.

:func:`assert_pre_treatment_only` is the leakage guard. It raises if any outcome (``visit``,
``conversion``, ``spend``), the treatment assignment (``arm`` or the raw ``segment``) or a
column derived from one of them (``arm_mens``, ``visit_rate`` ...) appears among the feature
names. :func:`build_features` calls it on its own output and the cross-fitting code calls it on
any feature frame it receives.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from abkit import config

NUMERIC_COLS: tuple[str, ...] = ("recency", "history")
BINARY_COLS: tuple[str, ...] = ("mens", "womens", "newbie")
ONE_HOT_COLS: tuple[str, ...] = ("zip_code", "channel")
DROPPED_COLS: tuple[str, ...] = ("history_segment",)
FORBIDDEN_COLS: tuple[str, ...] = (*config.OUTCOME_COLS, config.ARM_COL, config.RAW_ARM_COL)

ONE_HOT_LEVELS: dict[str, tuple[str, ...]] = {
    col: tuple(config.CATEGORY_MAPS[col].values()) for col in ONE_HOT_COLS
}
FEATURE_NAMES: tuple[str, ...] = (
    *NUMERIC_COLS,
    *BINARY_COLS,
    *(f"{col}_{level}" for col in ONE_HOT_COLS for level in ONE_HOT_LEVELS[col]),
)


class ForbiddenFeatureError(ValueError):
    """Raised when an outcome or the treatment assignment appears among the features."""


def assert_pre_treatment_only(columns: Iterable[str]) -> None:
    """Raise if any column is, or derives from, an outcome or the treatment assignment.

    A column is forbidden when its name equals one of ``FORBIDDEN_COLS`` or starts with one
    of them followed by an underscore (a one-hot child such as ``arm_mens``).

    Parameters
    ----------
    columns : iterable of str
        Feature names to check.

    Raises
    ------
    ForbiddenFeatureError
        Listing every offending column.
    """
    bad = [c for c in columns if any(c == f or c.startswith(f"{f}_") for f in FORBIDDEN_COLS)]
    if bad:
        raise ForbiddenFeatureError(
            f"Outcome or treatment columns may not be features: {sorted(bad)}. "
            f"Only pre-treatment variables are allowed: {config.PRE_TREATMENT_COLS}."
        )


def assert_no_leaked_values(X: pd.DataFrame, forbidden: dict[str, np.ndarray]) -> None:
    """Raise if a feature column carries the values of an outcome or of the treatment.

    The name check of :func:`assert_pre_treatment_only` cannot see a renamed column. This
    check compares values: a feature is rejected when it equals a forbidden vector or is a
    one-to-one recoding of it (the two take the same number of distinct values and each
    feature value goes with a single forbidden value), which also catches a rescaled or
    relabelled copy. It cannot catch a noisy or partial function of an outcome; nothing
    short of knowing where a column came from can.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    forbidden : dict of str to numpy.ndarray
        Name to vector (the outcome used as the target, the treatment indicator), each
        aligned with the rows of ``X``. A constant vector is skipped.

    Raises
    ------
    ForbiddenFeatureError
        Naming every feature that reproduces a forbidden vector.
    """
    bad = []
    for name, values in forbidden.items():
        values = np.asarray(values)
        n_values = np.unique(values).size
        if len(values) != len(X) or n_values < 2:
            continue
        for col in X.columns:
            feature = X[col].to_numpy()
            if np.unique(feature).size != n_values:
                continue
            by_level = pd.Series(values).groupby(feature)
            if (by_level.min() == by_level.max()).all():
                bad.append(f"{col} reproduces {name}")
    if bad:
        raise ForbiddenFeatureError(
            f"Feature values reproduce an outcome or the treatment: {bad}. "
            f"Only pre-treatment variables are allowed: {config.PRE_TREATMENT_COLS}."
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the uplift feature matrix from a cleaned-schema DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows of the cleaned ``customers`` table (any subset). Outcome and arm columns may
        be present; they are ignored, never used.

    Returns
    -------
    pandas.DataFrame
        Float matrix with columns ``FEATURE_NAMES`` in that fixed order, same index as
        ``df``. ``history_segment`` is not included (see the module docstring).

    Raises
    ------
    KeyError
        If a required pre-treatment column is missing.
    ValueError
        If a categorical column holds a level not listed in ``config.CATEGORY_MAPS``.
    ForbiddenFeatureError
        If the output somehow contains a forbidden column (defensive; cannot happen
        through this function's own construction, but the guard runs regardless).
    """
    required = (*NUMERIC_COLS, *BINARY_COLS, *ONE_HOT_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing pre-treatment columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in (*NUMERIC_COLS, *BINARY_COLS):
        out[col] = df[col].to_numpy(dtype=float)
    for col in ONE_HOT_COLS:
        values = df[col].to_numpy()
        levels = ONE_HOT_LEVELS[col]
        unknown = set(values) - set(levels)
        if unknown:
            raise ValueError(f"Unknown {col} level(s) {sorted(unknown)}; expected {levels}.")
        for level in levels:
            out[f"{col}_{level}"] = (values == level).astype(float)
    out = out[list(FEATURE_NAMES)]
    assert_pre_treatment_only(out.columns)
    return out


def categorical_mask(
    columns: Iterable[str] = FEATURE_NAMES, extra: Iterable[str] = ()
) -> np.ndarray:
    """Boolean mask marking the 0/1 columns (flags, one-hot indicators and ``extra``).

    Meant for the ``categorical_features`` argument of scikit-learn's histogram gradient
    boosting estimators, which then treat each indicator as a two-level category. Columns
    not known to be indicators are left numeric.

    Parameters
    ----------
    columns : iterable of str
        Feature names, in matrix order.
    extra : iterable of str
        Further column names to mark as categorical (for example a treatment indicator).

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the column is a 0/1 indicator.
    """
    indicators = {*BINARY_COLS, *FEATURE_NAMES[len(NUMERIC_COLS) :], *extra}
    return np.array([c in indicators for c in columns], dtype=bool)
