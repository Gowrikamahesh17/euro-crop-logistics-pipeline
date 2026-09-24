import numpy as np
import pytest


def test_engineer_features_drops_date_columns_adds_derived(raw_df):
    from preprocessor import engineer_features

    df = engineer_features(raw_df.copy())
    assert "Harvest_Date" not in df.columns
    assert "Event_Timestamp" not in df.columns
    for col in ["Days_Since_Harvest", "Event_Month", "Event_Hour", "Event_Dayofweek"]:
        assert col in df.columns


def test_encode_categoricals_one_hots_crop_type(raw_df):
    from preprocessor import encode_categoricals, engineer_features

    df = encode_categoricals(engineer_features(raw_df.copy()))
    assert "Crop_Type" not in df.columns
    assert {"Crop_Type_Wheat", "Crop_Type_Corn", "Crop_Type_Rice"} <= set(df.columns)


def test_split_features_targets_excludes_all_targets(raw_df):
    from preprocessor import (
        CLASSIFICATION_TARGET,
        REGRESSION_TARGETS,
        encode_categoricals,
        engineer_features,
        split_features_targets,
    )

    df = encode_categoricals(engineer_features(raw_df.copy()))
    X, y = split_features_targets(df)
    for target in REGRESSION_TARGETS + [CLASSIFICATION_TARGET]:
        assert target not in X.columns
        assert target in y.columns


def test_preprocess_produces_consistent_split_sizes(raw_df):
    from preprocessor import preprocess

    splits = preprocess(raw_df.copy())
    n_total = len(raw_df)
    assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == n_total
    assert len(splits["X_train"]) == len(splits["y_train"])
    assert len(splits["X_val"]) == len(splits["y_val"])
    assert len(splits["X_test"]) == len(splits["y_test"])


def test_preprocess_scales_train_features_to_zero_mean(raw_df):
    from preprocessor import preprocess

    splits = preprocess(raw_df.copy())
    numeric = splits["X_train"].select_dtypes(include=[np.number])
    assert np.allclose(numeric.mean(), 0, atol=0.15)


def test_preprocess_does_not_leak_targets_into_features(raw_df):
    """Regression guard: every target, including Vehicle_Type, must be absent
    from X in every split. Leaving a target in X is the easiest way to get a
    fake near-perfect score that means nothing."""
    from preprocessor import CLASSIFICATION_TARGET, REGRESSION_TARGETS, preprocess

    splits = preprocess(raw_df.copy())
    all_targets = REGRESSION_TARGETS + [CLASSIFICATION_TARGET]
    for split_name in ["X_train", "X_val", "X_test"]:
        for target in all_targets:
            assert target not in splits[split_name].columns


def test_preprocess_is_deterministic_for_fixed_seed(raw_df):
    from preprocessor import preprocess

    a = preprocess(raw_df.copy(), seed=7)
    b = preprocess(raw_df.copy(), seed=7)
    assert a["X_train"].equals(b["X_train"])
