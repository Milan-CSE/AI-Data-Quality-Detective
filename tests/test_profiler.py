"""Unit tests for the DataProfiler module."""

import numpy as np
import pandas as pd
import pytest

from src.profiler import DataProfiler


def test_profiler_basic_dataset():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, np.nan],
            "salary": [50000.0, 60000.0, 75000.0, 80000.0, 95000.0],
            "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
            "is_active": [True, True, False, True, False],
        }
    )
    profiler = DataProfiler()
    report = profiler.profile(df)

    assert "overview" in report
    assert "columns" in report
    assert "correlations" in report
    assert "warnings" in report

    overview = report["overview"]
    assert overview["total_rows"] == 5
    assert overview["total_columns"] == 5
    assert overview["total_missing_cells"] == 1
    assert overview["duplicate_rows_count"] == 0
    assert 0.0 <= overview["health_score"] <= 100.0

    cols = report["columns"]
    assert cols["age"]["missing_count"] == 1
    assert cols["age"]["inferred_type"] == "numeric"
    assert cols["salary"]["min"] == 50000.0
    assert cols["salary"]["max"] == 95000.0
    assert cols["department"]["inferred_type"] == "categorical"
    assert len(cols["department"]["top_values"]) > 0


def test_profiler_empty_dataframe():
    df_empty = pd.DataFrame()
    profiler = DataProfiler()
    report = profiler.profile(df_empty)

    assert report["overview"]["total_rows"] == 0
    assert report["overview"]["total_columns"] == 0
    assert report["overview"]["health_score"] == 0.0
    assert "DataFrame is empty." in report["warnings"]


def test_profiler_all_null_column():
    df = pd.DataFrame(
        {
            "numeric_valid": [10, 20, 30],
            "all_null": [np.nan, np.nan, np.nan],
        }
    )
    profiler = DataProfiler()
    report = profiler.profile(df)

    assert report["columns"]["all_null"]["missing_count"] == 3
    assert report["columns"]["all_null"]["missing_pct"] == 100.0


def test_profiler_duplicates():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": ["x", "x", "y"],
        }
    )
    profiler = DataProfiler()
    report = profiler.profile(df)

    assert report["overview"]["duplicate_rows_count"] == 1
    assert report["overview"]["duplicate_rows_pct"] == 33.33
    assert any("duplicate" in w.lower() for w in report["warnings"])


def test_profiler_invalid_input_type():
    profiler = DataProfiler()
    with pytest.raises(TypeError):
        profiler.profile([1, 2, 3])  # type: ignore
