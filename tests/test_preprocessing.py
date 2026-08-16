"""Tests for the preprocessing pipeline (duplicates, outliers, ColumnTransformer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.preprocess import remove_duplicates, winsorize_outliers


class TestRemoveDuplicates:
    def test_removes_exact_duplicate_rows(self, synthetic_df: pd.DataFrame):
        with_dupes = pd.concat([synthetic_df, synthetic_df.iloc[[0, 1]]], ignore_index=True)
        cleaned = remove_duplicates(with_dupes)
        assert cleaned["patient_id"].duplicated().sum() == 0

    def test_row_index_reset(self, synthetic_df: pd.DataFrame):
        cleaned = remove_duplicates(synthetic_df)
        assert list(cleaned.index) == list(range(len(cleaned)))


class TestWinsorizeOutliers:
    def test_caps_extreme_values(self, synthetic_df: pd.DataFrame):
        broken = synthetic_df.copy()
        broken.loc[0, "blood_glucose"] = 10000
        result = winsorize_outliers(broken, ["blood_glucose"], multiplier=1.5)
        assert result["blood_glucose"].max() < 10000

    def test_does_not_change_row_count(self, synthetic_df: pd.DataFrame):
        result = winsorize_outliers(synthetic_df, ["age", "bmi"], multiplier=1.5)
        assert len(result) == len(synthetic_df)

    def test_within_iqr_values_unchanged(self, synthetic_df: pd.DataFrame):
        col = "heart_rate"
        result = winsorize_outliers(synthetic_df, [col], multiplier=3.0)
        q1, q3 = synthetic_df[col].quantile(0.25), synthetic_df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        in_range_mask = synthetic_df[col].between(lower, upper)
        np.testing.assert_array_almost_equal(
            result.loc[in_range_mask, col].values, synthetic_df.loc[in_range_mask, col].values
        )
