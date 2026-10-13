"""Unit tests for the error injection and evaluation benchmarking modules."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import ErrorInjector, EvaluationEngine


@pytest.fixture
def clean_sample_df():
    # Deterministic clean sample
    return pd.DataFrame(
        {
            "employee_id": [f"EMP-{1000 + i}" for i in range(100)],
            "age": [25 + (i % 35) for i in range(100)],
            "years_experience": [(i % 35) * 0.8 for i in range(100)],
            "department": ["Engineering", "Sales", "HR", "Finance"] * 25,
            "education": ["Bachelor", "Master"] * 50,
            "job_level": ["Entry", "Mid", "Senior", "Lead"] * 25,
            "salary": [50000.0 + (i * 1000) for i in range(100)],
            "performance_score": [3.0 + (i % 20) * 0.1 for i in range(100)],
            "credit_score": [650 + (i % 150) for i in range(100)],
            "remote_ratio": [50] * 100,
        }
    )


def test_error_injector_injection_integrity(clean_sample_df):
    injector = ErrorInjector(random_seed=42)
    corrupted_df, errors = injector.inject(
        clean_df=clean_sample_df,
        univariate_count=5,
        multivariate_count=5,
        category_error_count=5,
        range_violation_count=5,
        missing_count=5,
        duplicate_count=3,
    )

    # Injected count must match requested
    assert len(errors) == (5 + 5 + 5 + 5 + 5 + 3)
    assert len(corrupted_df) == len(clean_sample_df) + 3

    # Check error types
    types = {e.error_type for e in errors}
    assert "univariate_outlier" in types
    assert "multivariate_outlier" in types
    assert "invalid_category" in types
    assert "range_violation" in types
    assert "missing_value" in types
    assert "duplicate_row" in types


def test_evaluation_engine_metrics_calculation():
    flagged = {1, 2, 3, 4, 10}
    ground_truth = {1, 2, 3, 5, 6}
    total = 100

    metrics = EvaluationEngine.compute_metrics(
        detector_name="TestDetector",
        target_scope="test_scope",
        flagged_row_ids=flagged,
        ground_truth_row_ids=ground_truth,
        total_samples=total,
        runtime_ms=12.5,
    )

    # TP = {1, 2, 3} -> 3
    # FP = {4, 10} -> 2
    # FN = {5, 6} -> 2
    # TN = 100 - (3 + 2 + 2) = 93
    assert metrics.true_positives == 3
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 2
    assert metrics.true_negatives == 93

    assert pytest.approx(metrics.precision, 0.001) == (3 / 5)
    assert pytest.approx(metrics.recall, 0.001) == (3 / 5)
    assert pytest.approx(metrics.f1_score, 0.001) == (3 / 5)


def test_evaluation_engine_full_run(clean_sample_df):
    injector = ErrorInjector(random_seed=42)
    corrupted_df, errors = injector.inject(
        clean_df=clean_sample_df,
        univariate_count=5,
        multivariate_count=5,
        category_error_count=5,
        range_violation_count=5,
        missing_count=5,
        duplicate_count=2,
    )

    results = EvaluationEngine.evaluate_all(
        df_corrupted=corrupted_df,
        injected_errors=errors,
        clean_df=clean_sample_df,
        iqr_k=1.5,
        if_contamination=0.08,
        random_state=42,
    )

    assert "metrics" in results
    assert "clean_test" in results
    assert "counts" in results
    assert len(results["metrics"]) >= 5
    assert results["clean_test"]["clean_samples"] == len(clean_sample_df)
