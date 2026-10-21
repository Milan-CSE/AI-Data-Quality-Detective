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


def test_confusion_matrix_invariants_strictly_hold(clean_sample_df):
    """Regression test: verify every flagged and GT row is consistently accounted for."""
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
        if_contamination=0.05,
        random_state=42,
    )

    for m in results["metrics"]:
        # 1. Every flagged row must be either TP or FP
        assert m["total_flagged"] == m["true_positives"] + m["false_positives"], (
            f"Flagged invariant violated for {m['detector_name']} ({m['target_scope']}): "
            f"flagged={m['total_flagged']} != TP({m['true_positives']}) + FP({m['false_positives']})"
        )

        # 2. Every ground truth anomaly must be either TP or FN
        assert m["total_ground_truth_anomalies"] == m["true_positives"] + m["false_negatives"], (
            f"GT invariant violated for {m['detector_name']} ({m['target_scope']}): "
            f"GT={m['total_ground_truth_anomalies']} != TP({m['true_positives']}) + FN({m['false_negatives']})"
        )

        # 3. All population items must sum to total_samples
        assert m["total_samples"] == (
            m["true_positives"] + m["false_positives"] + m["false_negatives"] + m["true_negatives"]
        ), (
            f"Total samples invariant violated for {m['detector_name']} ({m['target_scope']}): "
            f"samples={m['total_samples']} != sum of matrix"
        )

        # 4. Precision, Recall, F1 and FPR formulas
        if m["total_flagged"] > 0:
            expected_p = round(m["true_positives"] / m["total_flagged"], 4)
            assert m["precision"] == expected_p
        if m["total_ground_truth_anomalies"] > 0:
            expected_r = round(m["true_positives"] / m["total_ground_truth_anomalies"], 4)
            assert m["recall"] == expected_r
        if (m["false_positives"] + m["true_negatives"]) > 0:
            expected_fpr = round(m["false_positives"] / (m["false_positives"] + m["true_negatives"]), 4)
            assert m["false_positive_rate"] == expected_fpr


def test_scoped_population_excludes_out_of_scope_injections():
    """Verify that scoped evaluation populations correctly account for flagged rows without loss."""
    # Population: 50 total samples (10 target anomalies, 40 clean)
    target_gt = set(range(10))
    clean_pop = set(range(10, 50))
    pop = target_gt.union(clean_pop)

    # Detector flags: 8 target anomalies (TP), 2 clean rows (FP), and 5 out-of-scope rows (50-54)
    flagged = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 50, 51, 52, 53, 54}

    m = EvaluationEngine.compute_metrics(
        detector_name="MockDetector",
        target_scope="ScopedTest",
        flagged_row_ids=flagged,
        ground_truth_row_ids=target_gt,
        total_samples=100,  # whole-dataset size, should be overridden by len(pop)
        runtime_ms=5.0,
        population_row_ids=pop,
    )

    assert m.total_samples == 50
    assert m.total_flagged == 10
    assert m.true_positives == 8
    assert m.false_positives == 2
    assert m.false_negatives == 2
    assert m.true_negatives == 38
    assert m.precision == 0.8000
    assert m.recall == 0.8000
    assert m.false_positive_rate == round(2 / 40, 4)
    assert m.total_flagged == m.true_positives + m.false_positives
    assert m.total_ground_truth_anomalies == m.true_positives + m.false_negatives
    assert m.total_samples == m.true_positives + m.false_positives + m.false_negatives + m.true_negatives


def test_numerical_scopes_consistency_and_range_violations(clean_sample_df):
    """Verify univariate and multivariate scopes consistently partition numerical anomalies."""
    injector = ErrorInjector(random_seed=42)
    corrupted_df, errors = injector.inject(
        clean_df=clean_sample_df,
        univariate_count=6,
        multivariate_count=4,
        category_error_count=2,
        range_violation_count=3,
        missing_count=2,
        duplicate_count=1,
    )

    results = EvaluationEngine.evaluate_all(
        df_corrupted=corrupted_df,
        injected_errors=errors,
        clean_df=clean_sample_df,
    )

    counts = results["counts"]
    # Univariate count must include univariate outliers (6) + range violations (3) = 9
    assert counts["univariate_count"] == 9
    assert counts["multivariate_count"] == 4

    metrics_by_scope = {m["target_scope"]: m for m in results["metrics"] if m["detector_name"] == "IQR Baseline"}
    assert metrics_by_scope["Univariate Outliers"]["total_ground_truth_anomalies"] == 9
    assert metrics_by_scope["Multivariate Outliers"]["total_ground_truth_anomalies"] == 4
    assert metrics_by_scope["All Numerical Outliers"]["total_ground_truth_anomalies"] == 13


def test_clean_baseline_false_alarms_unique_rows():
    """Verify clean baseline false alarms count unique rows, not multiple findings on one row."""
    # Create clean DataFrame with repeated rules
    df_clean = pd.DataFrame(
        {
            "employee_id": [f"EMP-{1000 + i}" for i in range(20)],
            "age": [30] * 20,
            "years_experience": [5.0] * 20,
            "department": ["Engineering"] * 20,
            "education": ["Bachelor"] * 20,
            "job_level": ["Mid"] * 20,
            "salary": [60000.0] * 20,
            "performance_score": [3.5] * 20,
            "credit_score": [700] * 20,
            "remote_ratio": [50] * 20,
        }
    )
    injector = ErrorInjector(random_seed=42)
    corrupted_df, errors = injector.inject(
        clean_df=df_clean,
        univariate_count=2,
        multivariate_count=2,
        category_error_count=2,
        range_violation_count=2,
        missing_count=2,
        duplicate_count=1,
    )

    results = EvaluationEngine.evaluate_all(
        df_corrupted=corrupted_df,
        injected_errors=errors,
        clean_df=df_clean,
    )

    clean_test = results["clean_test"]
    assert clean_test["clean_samples"] == 20
    assert clean_test["rules_false_alarms"] == 0
    assert clean_test["rules_false_alarm_rate"] == 0.0
    assert clean_test["iqr_false_alarms"] <= 20
    assert clean_test["iso_forest_false_alarms"] <= 20


def test_contamination_sensitivity_standardized_grid(clean_sample_df):
    """Verify default contamination grid matches [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]."""
    from src.evaluation import DEFAULT_CONTAMINATIONS

    expected_grid = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
    assert DEFAULT_CONTAMINATIONS == expected_grid

    injector = ErrorInjector(random_seed=42)
    corrupted_df, errors = injector.inject(
        clean_df=clean_sample_df,
        univariate_count=4,
        multivariate_count=4,
        category_error_count=2,
        range_violation_count=2,
        missing_count=2,
        duplicate_count=1,
    )

    res = EvaluationEngine.run_contamination_sensitivity(
        df_corrupted=corrupted_df,
        injected_errors=errors,
    )

    assert len(res) == len(expected_grid)
    for row, expected_c in zip(res, expected_grid):
        assert row["contamination"] == expected_c
        assert 0.0 <= row["precision_all"] <= 1.0
        assert 0.0 <= row["recall_all"] <= 1.0
        assert 0.0 <= row["f1_all"] <= 1.0

