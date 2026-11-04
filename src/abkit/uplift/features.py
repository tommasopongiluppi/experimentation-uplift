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


def categorical_mask(columns: Iterable[str] = FEATURE_NAMES) -> np.ndarray:
    """Boolean mask marking the 0/1 columns (flags and one-hot indicators).

    Meant for the ``categorical_features`` argument of scikit-learn's histogram gradient
    boosting estimators, which then treat each indicator as a two-level category.

    Parameters
    ----------
    columns : iterable of str
        Feature names, in matrix order.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the column is a 0/1 indicator.
    """
    return np.array([c not in NUMERIC_COLS for c in columns], dtype=bool)
