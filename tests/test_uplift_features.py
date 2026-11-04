"""Uplift feature matrix: pre-treatment columns only, one-hot levels, and the leakage guard."""

import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.uplift.features import (
    BINARY_COLS,
    DROPPED_COLS,
    FEATURE_NAMES,
    FORBIDDEN_COLS,
    NUMERIC_COLS,
    ONE_HOT_COLS,
    ONE_HOT_LEVELS,
    ForbiddenFeatureError,
    assert_pre_treatment_only,
    build_features,
    categorical_mask,
)


def test_feature_names_cover_pre_treatment_columns_except_history_segment():
    used = set(NUMERIC_COLS) | set(BINARY_COLS) | set(ONE_HOT_COLS)
    assert used | set(DROPPED_COLS) == set(config.PRE_TREATMENT_COLS)
    assert DROPPED_COLS == ("history_segment",)
    assert "history_segment" not in FEATURE_NAMES
    assert not any(c.startswith("history_segment") for c in FEATURE_NAMES)


def test_build_features_from_fixture(customers):
    X = build_features(customers)
    assert list(X.columns) == list(FEATURE_NAMES)
    assert X.index.equals(customers.index)
    assert X.dtypes.eq(float).all()
    assert not X.isna().any().any()
    for col in (*NUMERIC_COLS, *BINARY_COLS):
        np.testing.assert_array_equal(X[col].to_numpy(), customers[col].to_numpy().astype(float))
    for col in ONE_HOT_COLS:
        block = X[[f"{col}_{lvl}" for lvl in ONE_HOT_LEVELS[col]]]
        assert block.isin([0.0, 1.0]).all().all()
        np.testing.assert_array_equal(block.sum(axis=1).to_numpy(), 1.0)
        for lvl in ONE_HOT_LEVELS[col]:
            np.testing.assert_array_equal(
                X[f"{col}_{lvl}"].to_numpy(), (customers[col] == lvl).to_numpy().astype(float)
            )


def test_one_hot_levels_come_from_config_and_are_stable_on_subsets(customers):
    for col in ONE_HOT_COLS:
        assert ONE_HOT_LEVELS[col] == tuple(config.CATEGORY_MAPS[col].values())
    sub = customers[customers["zip_code"] == "urban"]
    X = build_features(sub)
    assert list(X.columns) == list(FEATURE_NAMES)
    assert X["zip_code_rural"].eq(0.0).all() and X["zip_code_urban"].eq(1.0).all()


def test_outcome_and_arm_columns_are_ignored_not_used(customers):
    X = build_features(customers)
    for col in FORBIDDEN_COLS:
        assert col not in X.columns
    assert_pre_treatment_only(X.columns)  # does not raise


def test_unknown_level_raises(customers):
    bad = customers.copy()
    bad.loc[bad.index[0], "channel"] = "carrier_pigeon"
    with pytest.raises(ValueError, match="Unknown channel"):
        build_features(bad)


def test_missing_column_raises(customers):
    with pytest.raises(KeyError, match="recency"):
        build_features(customers.drop(columns=["recency"]))


@pytest.mark.parametrize("col", FORBIDDEN_COLS)
def test_guard_rejects_each_forbidden_column(col):
    with pytest.raises(ForbiddenFeatureError, match=col):
        assert_pre_treatment_only([*FEATURE_NAMES, col])


@pytest.mark.parametrize("col", ["arm_mens", "visit_rate", "spend_log", "segment_womens"])
def test_guard_rejects_derived_columns(col):
    with pytest.raises(ForbiddenFeatureError):
        assert_pre_treatment_only([col])


def test_guard_accepts_pre_treatment_names():
    assert_pre_treatment_only(FEATURE_NAMES)
    assert_pre_treatment_only(config.PRE_TREATMENT_COLS)
    assert_pre_treatment_only(["mens", "womens"])  # purchase flags, not the arm


def test_guard_lists_every_offender():
    with pytest.raises(ForbiddenFeatureError) as info:
        assert_pre_treatment_only(["recency", "visit", "arm", "spend"])
    assert "['arm', 'spend', 'visit']" in str(info.value)


def test_categorical_mask():
    mask = categorical_mask()
    assert mask.shape == (len(FEATURE_NAMES),)
    for name, flag in zip(FEATURE_NAMES, mask, strict=True):
        assert flag == (name not in NUMERIC_COLS)
    assert mask.sum() == len(FEATURE_NAMES) - len(NUMERIC_COLS)


def test_build_features_accepts_any_row_order(customers):
    shuffled = customers.sample(frac=1.0, random_state=0)
    X = build_features(shuffled)
    pd.testing.assert_frame_equal(X.sort_index(), build_features(customers))
