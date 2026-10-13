"""Unit tests for the SafeCleaner module."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import SafeCleaner


def test_cleaner_immutability():
    df_orig = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    cleaner = SafeCleaner(df_orig)

    # Apply changes
    cleaner.apply_drop_rows([1])

    # Ensure original remains unchanged
    assert len(cleaner.raw_data) == 3
    assert len(cleaner.current_data) == 2
    assert 1 not in cleaner.current_data.index
    assert 1 in cleaner.raw_data.index


def test_cleaner_impute_missing():
    df = pd.DataFrame({"num": [10.0, np.nan, 30.0], "cat": ["A", None, "A"]})
    cleaner = SafeCleaner(df)

    # Impute numeric with median
    preview_df, diffs = cleaner.preview_impute_missing(column="num", strategy="median")
    assert len(diffs) == 1
    assert preview_df.loc[1, "num"] == 20.0
    assert cleaner.current_data["num"].isna().sum() == 1  # Not yet committed

    # Commit
    cleaner.apply_impute_missing(column="num", strategy="median")
    assert cleaner.current_data.loc[1, "num"] == 20.0
    assert cleaner.current_data["num"].isna().sum() == 0

    # Impute categorical with mode
    cleaner.apply_impute_missing(column="cat", strategy="mode")
    assert cleaner.current_data.loc[1, "cat"] == "A"


def test_cleaner_cap_outliers():
    df = pd.DataFrame({"val": [-100.0, 10.0, 20.0, 500.0]})
    cleaner = SafeCleaner(df)

    cleaner.apply_cap_outliers("val", lower=0.0, upper=100.0)
    current = cleaner.current_data["val"].tolist()
    assert current == [0.0, 10.0, 20.0, 100.0]


def test_cleaner_replace_values():
    df = pd.DataFrame({"dept": ["Engneering", "Sales", "engneering", "Finance"]})
    cleaner = SafeCleaner(df)

    cleaner.apply_replace_values("dept", mapping={"Engneering": "Engineering"})
    assert cleaner.current_data.loc[0, "dept"] == "Engineering"
    assert cleaner.current_data.loc[2, "dept"] == "Engineering"


def test_cleaner_drop_duplicates():
    df = pd.DataFrame({"id": [1, 2, 2, 3], "val": ["a", "b", "b", "c"]})
    cleaner = SafeCleaner(df)

    cleaner.apply_drop_duplicates()
    assert len(cleaner.current_data) == 3


def test_cleaner_undo_and_reset():
    df = pd.DataFrame({"val": [1, 2, 3]})
    cleaner = SafeCleaner(df)

    cleaner.apply_drop_rows([0])
    assert len(cleaner.current_data) == 2

    # Undo
    assert cleaner.undo() is True
    assert len(cleaner.current_data) == 3

    # Apply multiple and reset
    cleaner.apply_drop_rows([0])
    cleaner.apply_drop_rows([1])
    assert len(cleaner.current_data) == 1

    cleaner.reset_to_raw()
    assert len(cleaner.current_data) == 3
    assert len(cleaner.audit_trail) == 0


def test_cleaner_audit_report():
    df = pd.DataFrame({"val": [np.nan, 2.0, 3.0]})
    cleaner = SafeCleaner(df)
    cleaner.apply_impute_missing("val", strategy="mean")

    report = cleaner.generate_audit_report()
    assert "Transformation Audit Report" in report
    assert "impute_missing_mean" in report
    assert "Step 1" in report
