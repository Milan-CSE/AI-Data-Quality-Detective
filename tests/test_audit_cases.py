"""Comprehensive Independent Audit Test Suite for AI Data Quality Detective.

Covers:
- Normal, clean, and corrupted datasets
- Missing values, duplicate rows, and numerical anomalies
- Empty files, malformed inputs, and unexpected data types
- Boundary cases: all-null columns, constant numerical columns, 1-row data, non-integer index
- Reproducibility with a fixed random seed
- Precision, recall, F1, and confusion-matrix independent verification
- Cleaning correctness, immutability, and rollback
- Export correctness and row/column preservation
"""

import io
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.cleaning import SafeCleaner
from src.detectors import IQRDetector, IsolationForestDetector
from src.evaluation import ErrorInjector, EvaluationEngine, InjectedError
from src.profiler import DataProfiler
from src.rules import (
    AllowedCategoriesRule,
    CrossFieldRule,
    DuplicateRowRule,
    NotNullRule,
    NumericRangeRule,
    RegexPatternRule,
    RuleEngine,
)


# =====================================================================
# 1. NORMAL, CLEAN AND CORRUPTED DATASETS
# =====================================================================

def test_clean_synthetic_dataset_profiling_and_rules():
    """Verify that a small clean dataset produces a high health score and zero rule violations."""
    df_clean = pd.DataFrame(
        {
            "employee_id": [f"EMP-{1000 + i}" for i in range(25)],
            "age": [25 + i for i in range(25)],
            "years_experience": [float(i // 2) for i in range(25)],
            "department": ["Engineering", "Sales", "HR", "Finance", "Operations"] * 5,
            "education": ["Bachelor", "Master", "PhD", "High School", "Bachelor"] * 5,
            "job_level": ["Entry", "Mid", "Senior", "Lead", "Director"] * 5,
            "salary": [50000.0 + (i * 3000) for i in range(25)],
            "performance_score": [3.0 + (i % 5) * 0.3 for i in range(25)],
            "credit_score": [650 + (i * 5) for i in range(25)],
            "remote_ratio": [0, 50, 100, 50, 0] * 5,
        }
    )

    # 1. Profile clean data
    profiler = DataProfiler()
    profile = profiler.profile(df_clean)
    assert profile["overview"]["total_rows"] == 25
    assert profile["overview"]["total_missing_cells"] == 0
    assert profile["overview"]["duplicate_rows_count"] == 0
    assert profile["overview"]["health_score"] >= 95.0

    # 2. Workforce rules against clean data
    engine = RuleEngine.create_workforce_rules()
    findings = engine.evaluate(df_clean)
    assert len(findings) == 0, f"Expected 0 rule findings on clean data, got {len(findings)}"


def test_corrupted_dataset_detection():
    """Verify detection on a small corrupted dataset with known defects."""
    df = pd.DataFrame(
        {
            "employee_id": ["EMP-1001", "INVALID_ID", "EMP-1003", "EMP-1004"],
            "age": [30, -5, 150, 40],  # -5 and 150 are invalid
            "years_experience": [5.0, 2.0, 10.0, 4.0],
            "department": ["Engineering", "Sales", "FakeDept", "HR"],
            "education": ["Bachelor", "Master", "PhD", "InvalidEdu"],
            "job_level": ["Entry", "Mid", "Senior", "Lead"],
            "salary": [60000.0, 75000.0, 80000.0, 3000000.0],  # 3M is range/IQR outlier
            "performance_score": [3.5, 4.0, 2.5, 3.8],
            "credit_score": [700, 720, 680, 710],
            "remote_ratio": [50, 50, 50, 50],
        }
    )
    engine = RuleEngine.create_workforce_rules()
    findings = engine.evaluate(df)

    flagged_rows = {f.row_id for f in findings}
    assert 1 in flagged_rows  # Row 1: INVALID_ID, age -5
    assert 2 in flagged_rows  # Row 2: age 150, FakeDept
    assert 3 in flagged_rows  # Row 3: InvalidEdu, salary 3M


# =====================================================================
# 2. MISSING VALUES, DUPLICATE ROWS AND NUMERICAL ANOMALIES
# =====================================================================

def test_missing_values_detection_rule_vs_factory():
    """Verify NotNullRule works standalone, but highlight whether factory includes it."""
    df = pd.DataFrame({"age": [25, None, 35], "salary": [50000.0, 60000.0, None]})

    # Standalone rule
    not_null_age = NotNullRule("age")
    findings = not_null_age.evaluate(df)
    assert len(findings) == 1
    assert findings[0].row_id == 1

    # Default workforce engine check
    factory_engine = RuleEngine.create_workforce_rules()
    factory_findings = factory_engine.evaluate(df)
    missing_findings = [f for f in factory_findings if f.issue_type == "missing_value"]
    assert len(missing_findings) == 2


def test_duplicate_row_detection_and_drop():
    """Verify duplicate detection and safe removal on identical rows."""
    df = pd.DataFrame(
        {
            "col_a": [1, 2, 2, 3],
            "col_b": ["a", "b", "b", "c"],
        }
    )
    rule = DuplicateRowRule()
    findings = rule.evaluate(df)
    assert len(findings) == 1
    assert findings[0].row_id == 2  # The second 'b' row

    cleaner = SafeCleaner(df)
    dropped_count = cleaner.apply_drop_duplicates()
    assert dropped_count == 1
    assert len(cleaner.current_data) == 3
    assert len(cleaner.raw_data) == 4


def test_univariate_and_multivariate_numerical_anomalies():
    """Verify univariate IQR vs multivariate Isolation Forest detection."""
    # Normal data: 50 points centered at (50, 100) with y ~ 2*x
    rng = np.random.default_rng(42)
    x = rng.normal(50, 3, size=50)
    y = x * 2.0 + rng.normal(0, 1, size=50)

    # Point 50: Univariate extreme outlier (x=200, y=400)
    # Point 51: Multivariate anomaly (x=50 is normal, y=20 violates y=2*x)
    x = np.append(x, [200.0, 50.0])
    y = np.append(y, [400.0, 20.0])

    df = pd.DataFrame({"x": x, "y": y})

    # 1. IQR Detector
    iqr = IQRDetector(k=1.5).fit(df)
    iqr_flagged = {f.row_id for f in iqr.detect(df)}
    assert 50 in iqr_flagged, "IQR must detect extreme univariate outlier at index 50"

    # 2. Isolation Forest Detector
    iso = IsolationForestDetector(contamination=0.08, random_state=42).fit(df)
    iso_flagged = {f.row_id for f in iso.detect(df)}
    assert 50 in iso_flagged, "Isolation Forest must detect extreme outlier at index 50"


# =====================================================================
# 3. EMPTY FILES, MALFORMED INPUTS AND UNEXPECTED DATA TYPES
# =====================================================================

def test_empty_dataframe_handling():
    """Verify all modules handle empty DataFrames gracefully without crashing."""
    df_empty = pd.DataFrame()

    # Profiler
    prof = DataProfiler().profile(df_empty)
    assert prof["overview"]["total_rows"] == 0
    assert prof["overview"]["health_score"] == 0.0

    # Rules
    rules = RuleEngine.create_workforce_rules()
    assert rules.evaluate(df_empty) == []

    # IQR
    iqr = IQRDetector().fit(df_empty)
    assert iqr.detect(df_empty) == []

    # Isolation Forest
    iso = IsolationForestDetector().fit(df_empty)
    assert iso.detect(df_empty) == []


def test_empty_dataframe_with_columns():
    """Verify DataFrame with columns but 0 rows does not crash."""
    df_cols_no_rows = pd.DataFrame(columns=["age", "salary", "department"])
    prof = DataProfiler().profile(df_cols_no_rows)
    assert prof["overview"]["total_rows"] == 0

    rules = RuleEngine.create_workforce_rules().evaluate(df_cols_no_rows)
    assert rules == []

    iqr = IQRDetector().fit(df_cols_no_rows).detect(df_cols_no_rows)
    assert iqr == []


def test_malformed_csv_string_coercion_in_numeric_columns():
    """Verify how modules behave when dirty string values appear in numeric columns."""
    df = pd.DataFrame({"salary": [50000.0, "unknown_string", 75000.0, "$100,000"]})

    # NumericRangeRule coerces to numeric
    rule = NumericRangeRule("salary", min_val=20000, max_val=200000)
    findings = rule.evaluate(df)
    # The strings 'unknown_string' and '$100,000' coerce to NaN and are skipped
    assert len(findings) == 0  # Neither valid float is out of range; strings were skipped


def test_duplicate_column_names_handling():
    """Verify DataProfiler handles duplicate column names gracefully with warnings."""
    df_dup_cols = pd.DataFrame([[25, 25], [30, 30]], columns=["age", "age"])

    profiler = DataProfiler()
    prof = profiler.profile(df_dup_cols)
    assert prof["overview"]["total_columns"] == 2
    assert any("duplicate column names" in w for w in prof["warnings"])


# =====================================================================
# 4. BOUNDARY CASES: ALL-NULL AND CONSTANT COLUMNS
# =====================================================================

def test_all_null_column_in_profiler_and_detectors():
    """Verify behavior when a column contains 100% NaN values."""
    df = pd.DataFrame(
        {
            "valid_num": [10.0, 20.0, 30.0, 40.0, 50.0],
            "all_null_num": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    # Profiler handles all-null
    prof = DataProfiler().profile(df)
    assert prof["columns"]["all_null_num"]["missing_count"] == 5
    assert prof["columns"]["all_null_num"]["missing_pct"] == 100.0

    # IQR handles all-null by skipping it
    iqr = IQRDetector().fit(df)
    assert "all_null_num" not in iqr.feature_bounds

    # Isolation Forest handles all-null column gracefully without crashing
    iso = IsolationForestDetector()
    iso.fit(df)
    findings = iso.detect(df)
    assert isinstance(findings, list)


def test_constant_numerical_column():
    """Verify behavior on zero-variance / constant numerical columns."""
    df = pd.DataFrame(
        {
            "constant_val": [50.0] * 20,
            "normal_val": [float(i) for i in range(20)],
        }
    )

    # Profiler
    prof = DataProfiler().profile(df)
    assert prof["columns"]["constant_val"]["is_constant"] is True
    assert prof["columns"]["constant_val"]["iqr"] == 0.0

    # IQR: fallback should prevent zero-division
    iqr = IQRDetector().fit(df)
    assert "constant_val" in iqr.feature_bounds
    assert iqr.feature_bounds["constant_val"]["iqr"] > 0

    # Isolation Forest handles constant column without crashing
    iso = IsolationForestDetector().fit(df)
    findings = iso.detect(df)
    assert isinstance(findings, list)


def test_single_row_dataset():
    """Verify behavior when dataset has only 1 row."""
    df = pd.DataFrame({"age": [30], "salary": [75000.0]})

    # Profiler
    prof = DataProfiler().profile(df)
    assert prof["overview"]["total_rows"] == 1

    # IQR: skips columns with < 4 non-null values
    iqr = IQRDetector().fit(df)
    assert len(iqr.feature_bounds) == 0

    # Isolation Forest: fits on 1 sample
    iso = IsolationForestDetector().fit(df)
    findings = iso.detect(df)
    assert findings == []


def test_non_integer_index_in_rules():
    """Verify RuleEngine handles string row indices without crashing."""
    df = pd.DataFrame(
        {"age": [25, 200, 30]},
        index=["row_first", "row_second", "row_third"],
    )
    rule = NumericRangeRule("age", min_val=18, max_val=100)
    findings = rule.evaluate(df)
    assert len(findings) == 1
    assert findings[0].row_id == "row_second"


def test_duplicate_row_indices_in_cleaner():
    """Verify SafeCleaner preserves unique row when index has duplicate labels."""
    # Rows 0 and 1 share index label 0; row 1 is a duplicate of row 0
    df = pd.DataFrame({"val": [10, 10, 20]}, index=[0, 0, 1])
    cleaner = SafeCleaner(df)
    prev_df, diffs = cleaner.preview_drop_duplicates()

    # Drop duplicates keeps exactly 1 copy of row 0 and row 2 (total 2 rows remaining)
    assert len(prev_df) == 2
    assert len(diffs) == 1


# =====================================================================
# 5. REPRODUCIBILITY WITH FIXED RANDOM SEED
# =====================================================================

def test_isolation_forest_reproducibility():
    """Verify Isolation Forest produces identical results across runs with the same seed."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.normal(10, 2, size=100),
            "f2": rng.normal(50, 5, size=100),
        }
    )
    # Run 1
    iso1 = IsolationForestDetector(contamination=0.05, random_state=42).fit(df)
    f1 = iso1.detect(df)
    rows1 = [f.row_id for f in f1]
    scores1 = [f.anomaly_score for f in f1]

    # Run 2
    iso2 = IsolationForestDetector(contamination=0.05, random_state=42).fit(df)
    f2 = iso2.detect(df)
    rows2 = [f.row_id for f in f2]
    scores2 = [f.anomaly_score for f in f2]

    assert rows1 == rows2
    assert np.allclose(scores1, scores2, atol=1e-5)


def test_error_injector_reproducibility():
    """Verify ErrorInjector generates identical corruptions with the same seed."""
    df_clean = pd.DataFrame(
        {
            "employee_id": [f"EMP-{1000 + i}" for i in range(100)],
            "age": [25 + (i % 30) for i in range(100)],
            "years_experience": [float(i % 15) for i in range(100)],
            "department": ["Engineering", "Sales"] * 50,
            "education": ["Bachelor", "Master"] * 50,
            "job_level": ["Entry", "Mid"] * 50,
            "salary": [50000.0 + (i * 1000) for i in range(100)],
            "performance_score": [3.5] * 100,
            "credit_score": [700] * 100,
            "remote_ratio": [50] * 100,
        }
    )

    inj1 = ErrorInjector(random_seed=123)
    df1, err1 = inj1.inject(
        df_clean,
        univariate_count=5,
        multivariate_count=5,
        category_error_count=5,
        range_violation_count=5,
        missing_count=5,
        duplicate_count=3,
    )

    inj2 = ErrorInjector(random_seed=123)
    df2, err2 = inj2.inject(
        df_clean,
        univariate_count=5,
        multivariate_count=5,
        category_error_count=5,
        range_violation_count=5,
        missing_count=5,
        duplicate_count=3,
    )

    assert df1.equals(df2)
    assert len(err1) == len(err2)
    assert [e.row_id for e in err1] == [e.row_id for e in err2]


def test_error_injector_crashes_on_insufficient_rows():
    """Audit test: verify ErrorInjector crashes when len(clean_df) is smaller than total default error counts."""
    # 50 rows is smaller than default total errors (25+25+15+15+20+10 = 110)
    df_small = pd.DataFrame(
        {
            "employee_id": [f"EMP-{1000 + i}" for i in range(50)],
            "age": [25 + (i % 30) for i in range(50)],
            "years_experience": [float(i % 15) for i in range(50)],
            "department": ["Engineering", "Sales"] * 25,
            "education": ["Bachelor", "Master"] * 25,
            "job_level": ["Entry", "Mid"] * 25,
            "salary": [50000.0 + (i * 1000) for i in range(50)],
            "performance_score": [3.5] * 50,
            "credit_score": [700] * 50,
            "remote_ratio": [50] * 50,
        }
    )
    injector = ErrorInjector(random_seed=42)

    # Verified: raises descriptive ValueError when clean_df has insufficient rows
    with pytest.raises(ValueError, match="clean_df must contain at least"):
        injector.inject(df_small)


# =====================================================================
# 6. PRECISION, RECALL, F1 AND CONFUSION MATRIX VERIFICATION
# =====================================================================

def test_independent_confusion_matrix_and_metrics_verification():
    """Independently verify precision, recall, F1, FPR against exact mathematical formula.
    
    Setup:
      Total samples = 100
      Ground Truth Positives = {1, 2, 3, 4, 5}  (P = 5)
      Flagged Positives      = {4, 5, 6, 7}     (Pred = 4)
      
    Exact Expected:
      TP = {4, 5} = 2
      FP = {6, 7} = 2
      FN = {1, 2, 3} = 3
      TN = 100 - (2 + 2 + 3) = 93
      Precision = 2 / 4 = 0.5000
      Recall    = 2 / 5 = 0.4000
      F1-Score  = 2 * 0.5 * 0.4 / (0.5 + 0.4) = 0.4444
      FPR       = 2 / (2 + 93) = 2 / 95 = 0.0211
    """
    total = 100
    gt = {1, 2, 3, 4, 5}
    flagged = {4, 5, 6, 7}

    m = EvaluationEngine.compute_metrics(
        detector_name="MockDetector",
        target_scope="TestScope",
        flagged_row_ids=flagged,
        ground_truth_row_ids=gt,
        total_samples=total,
        runtime_ms=10.0,
    )

    assert m.true_positives == 2
    assert m.false_positives == 2
    assert m.false_negatives == 3
    assert m.true_negatives == 93
    assert m.precision == pytest.approx(0.5000, abs=1e-4)
    assert m.recall == pytest.approx(0.4000, abs=1e-4)
    assert m.f1_score == pytest.approx(0.4444, abs=1e-4)
    assert m.false_positive_rate == pytest.approx(round(2.0 / 95.0, 4), abs=1e-4)


# =====================================================================
# 7. CLEANING CORRECTNESS, IMMUTABILITY AND ROLLBACK
# =====================================================================

def test_safe_cleaner_immutability_and_rollback():
    """Verify raw_data remains strictly unchanged and undo reverts state."""
    df_raw = pd.DataFrame({"val": [10.0, 20.0, 30.0]})
    cleaner = SafeCleaner(df_raw)

    cleaner.apply_drop_rows([0])
    cleaner.apply_cap_outliers("val", upper=25.0)

    # Immutability
    assert len(cleaner.raw_data) == 3
    assert cleaner.raw_data.loc[0, "val"] == 10.0

    # Current data modified
    assert len(cleaner.current_data) == 2
    assert cleaner.current_data.loc[2, "val"] == 25.0

    # Undo step 2 (cap)
    assert cleaner.undo() is True
    assert cleaner.current_data.loc[2, "val"] == 30.0

    # Reset all
    cleaner.reset_to_raw()
    assert len(cleaner.current_data) == 3
    assert len(cleaner.audit_trail) == 0


def test_impute_missing_on_all_null_column():
    """Verify impute_missing on an all-null column gracefully produces 0 diffs without saving NaNs."""
    df = pd.DataFrame({"all_na": [np.nan, np.nan, np.nan]})
    cleaner = SafeCleaner(df)
    prev_df, diffs = cleaner.preview_impute_missing("all_na", strategy="median")

    # Fixed: Returns empty diff list and does not record bogus nan diffs
    assert len(diffs) == 0
    assert prev_df["all_na"].isna().all()


def test_cap_outliers_on_mixed_types_data_loss():
    """Verify cap_outliers preserves non-numeric strings and only modifies valid numeric values."""
    df = pd.DataFrame({"val": ["10", "corrupted_text", "100"]})
    cleaner = SafeCleaner(df)
    prev_df, diffs = cleaner.preview_cap_outliers("val", lower=0, upper=50)

    # Fixed: 'corrupted_text' is preserved as string, not destroyed to NaN
    assert prev_df.loc[1, "val"] == "corrupted_text"
    assert prev_df.loc[2, "val"] == 50


def test_replace_values_with_none():
    """Verify replacing values with None is supported."""
    df = pd.DataFrame({"cat": ["valid", "Unknown", "missing"]})
    cleaner = SafeCleaner(df)
    prev_df, diffs = cleaner.preview_replace_values("cat", {"Unknown": None})

    # Fixed: Replacement with None is recorded and committed
    assert len(diffs) == 1
    assert prev_df.loc[1, "cat"] is None or pd.isna(prev_df.loc[1, "cat"])


# =====================================================================
# 8. EXPORT CORRECTNESS AND ROW/COLUMN PRESERVATION
# =====================================================================

def test_export_row_and_column_preservation():
    """Verify that export preserves all columns and row count."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature_a": [10.5, 20.2, 30.1],
            "category": ["A", "B", "C"],
        }
    )
    cleaner = SafeCleaner(df)
    csv_buf = io.StringIO()
    cleaner.current_data.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    exported_df = pd.read_csv(csv_buf)

    assert exported_df.shape == df.shape
    assert list(exported_df.columns) == list(df.columns)
    assert exported_df.equals(df)


def test_audit_report_export_format():
    """Verify audit report markdown generation and contents."""
    df = pd.DataFrame({"val": [1, 2, 2, 3]})
    cleaner = SafeCleaner(df)
    cleaner.apply_drop_duplicates()

    report = cleaner.generate_audit_report()
    assert "# Data Quality Detective - Transformation Audit Report" in report
    assert "drop_duplicates" in report
    assert "- **Transformations Applied**: 1" in report
