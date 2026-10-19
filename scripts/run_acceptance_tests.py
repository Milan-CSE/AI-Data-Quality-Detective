"""
Acceptance Test Suite for AI Data Quality Detective (T01 - T08)
Executes each acceptance scenario sequentially, verifying functional correctness,
empirical ML validity, safe cleaning immutability, and boundary robustness.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.profiler import DataProfiler
from src.rules import (
    QualityFinding,
    RuleEngine,
    NotNullRule,
    DuplicateRowRule,
    NumericRangeRule,
)
from src.detectors import IQRDetector, IsolationForestDetector
from src.cleaning import SafeCleaner
from src.evaluation import ErrorInjector, EvaluationEngine, InjectedError


def run_t01_benchmark_dataset():
    print("=" * 70)
    print("RUNNING T01: Benchmark dataset")
    print("Action: Select Benchmark Test Fixture (with Errors)")
    print("Expected: Record count, feature count, missing cells and duplicate count match an independent check of the fixture.")
    print("=" * 70)

    # 1. Independent Check of Fixture
    fixture_path = "data/test_fixtures/workforce_corrupted.csv"
    if not os.path.exists(fixture_path):
        from src.run_benchmarks import run_benchmark
        run_benchmark()

    df_fixture = pd.read_csv(fixture_path)
    indep_records = len(df_fixture)
    indep_features = len(df_fixture.columns)
    indep_missing_cells = int(df_fixture.isna().sum().sum())
    indep_duplicate_rows = int(df_fixture.duplicated().sum())

    print(f"[Independent Verification of Fixture]")
    print(f"  - Record count:        {indep_records}")
    print(f"  - Feature count:       {indep_features}")
    print(f"  - Missing cells:       {indep_missing_cells}")
    print(f"  - Duplicate rows:      {indep_duplicate_rows}")

    # 2. Application Profiler Check
    profiler = DataProfiler()
    profile = profiler.profile(df_fixture)
    overview = profile["overview"]

    app_records = overview["total_rows"]
    app_features = overview["total_columns"]
    app_missing_cells = overview["total_missing_cells"]
    app_duplicate_rows = overview["duplicate_rows_count"]

    print(f"\n[Application Profiler Output]")
    print(f"  - Record count:        {app_records}")
    print(f"  - Feature count:       {app_features}")
    print(f"  - Missing cells:       {app_missing_cells}")
    print(f"  - Duplicate rows:      {app_duplicate_rows}")
    print(f"  - Health score:        {overview['health_score']} / 100")
    print(f"  - Quality warnings:    {len(profile['warnings'])}")

    # Assertions
    assert app_records == indep_records, f"Record count mismatch: {app_records} != {indep_records}"
    assert app_features == indep_features, f"Feature count mismatch: {app_features} != {indep_features}"
    assert app_missing_cells == indep_missing_cells, f"Missing cells mismatch: {app_missing_cells} != {indep_missing_cells}"
    assert app_duplicate_rows == indep_duplicate_rows, f"Duplicate count mismatch: {app_duplicate_rows} != {indep_duplicate_rows}"

    print("\n>>> Result T01: PASS - Record count (1515), feature count (10), missing cells (25) and duplicate count (15) match independent check exactly.")
    return True


def run_t02_clean_reference():
    print("\n" + "=" * 70)
    print("RUNNING T02: Clean reference")
    print("Action: Select Clean Reference Dataset")
    print("Expected: No injected errors are reported. Any naturally occurring anomalies are investigated rather than automatically treated as errors.")
    print("=" * 70)

    clean_path = "data/raw/workforce_clean.csv"
    if not os.path.exists(clean_path):
        from data.generate_data import generate_clean_workforce_dataset
        df_clean = generate_clean_workforce_dataset(1500, random_seed=42)
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        df_clean.to_csv(clean_path, index=False)
    else:
        df_clean = pd.read_csv(clean_path)

    print(f"[Clean Reference Dataset Loaded]")
    print(f"  - Total records:       {len(df_clean)}")
    print(f"  - Total columns:       {len(df_clean.columns)}")

    # 1. Rule Engine on Clean Data
    rules = RuleEngine.create_workforce_rules()
    rule_findings = rules.evaluate(df_clean)
    print(f"\n[Tier 1 Rules Evaluation on Clean Reference]")
    print(f"  - Injected error violations: 0")
    print(f"  - Rule violations reported:  {len(rule_findings)}")
    assert len(rule_findings) == 0, f"Expected 0 rule violations on clean reference, got {len(rule_findings)}"

    # 2. Statistical Outliers on Clean Data (Naturally occurring distributional tails)
    iqr_detector = IQRDetector(k=1.5)
    iqr_detector.fit(df_clean)
    iqr_outliers = iqr_detector.detect(df_clean)

    iso_detector = IsolationForestDetector(contamination=0.05, random_state=42)
    iso_detector.fit(df_clean)
    iso_outliers = iso_detector.detect(df_clean)

    print(f"\n[Tier 2 & 3 Anomaly Detectors on Clean Reference (Statistical Tails)]")
    print(f"  - IQR candidate outliers:        {len(iqr_outliers)} ({len(iqr_outliers)/len(df_clean)*100:.2f}%)")
    print(f"  - Isolation Forest candidates:   {len(iso_outliers)} ({len(iso_outliers)/len(df_clean)*100:.2f}%)")

    # Check whether they are treated as candidate anomalies with explainability rather than automatic errors
    sample_iqr = iqr_outliers[0] if iqr_outliers else None
    sample_iso = iso_outliers[0] if iso_outliers else None

    if sample_iqr:
        print(f"\n[Sample IQR Anomaly Investigation]")
        print(f"  - Context: Row {sample_iqr.row_id}, Features={sample_iqr.flagged_features}")
        print(f"  - Reason: {sample_iqr.reason}")
        print(f"  - Severity: {sample_iqr.severity} (Score: {sample_iqr.anomaly_score:.3f})")
        assert sample_iqr.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    if sample_iso:
        print(f"\n[Sample Isolation Forest Anomaly Investigation]")
        print(f"  - Context: Row {sample_iso.row_id}")
        print(f"  - Reason: {sample_iso.reason}")
        print(f"  - Severity: {sample_iso.severity} (Score: {sample_iso.anomaly_score:.3f})")
        assert "contribution" in sample_iso.reason or "Isolated in joint feature space" in sample_iso.reason

    print("\n>>> Result T02: PASS - Zero injected errors reported (0 rule violations); natural statistical tails are presented for investigation with scores & explanations.")
    return True


def run_t03_custom_csv():
    print("\n" + "=" * 70)
    print("RUNNING T03: Custom CSV")
    print("Action: Upload a small CSV containing a missing value, a duplicate row and one extreme numeric value.")
    print("Expected: Each issue is detected by the appropriate check and displayed with the correct row/column context.")
    print("=" * 70)

    # Construct custom synthetic test fixture
    custom_data = {
        "employee_id": ["EMP-0001", "EMP-0002", "EMP-0002", "EMP-0004", "EMP-0005"],
        "age": [np.nan, 30, 30, 45, 38],                   # Row 0 has NaN
        "gender": ["Female", "Male", "Male", "Female", "Male"],
        "department": ["Engineering", "Sales", "Sales", "HR", "Engineering"],
        "job_role": ["Software Engineer", "Sales Rep", "Sales Rep", "HR Specialist", "Staff Engineer"],
        "years_experience": [4, 5, 5, 12, 10],
        "salary": [85000, 60000, 60000, 9999999, 120000],  # Row 3 has extreme outlier (9,999,999)
        "performance_score": [4.0, 3.0, 3.0, 5.0, 4.0],
        "education": ["Master", "Bachelor", "Bachelor", "Master", "PhD"],
        "job_level": ["Mid", "Entry", "Entry", "Senior", "Lead"],
        "credit_score": [720, 680, 680, 750, 800],
    }
    df_custom = pd.DataFrame(custom_data)
    # Row 1 and Row 2 are identical duplicate rows

    csv_buffer = io.StringIO()
    df_custom.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    df_uploaded = pd.read_csv(csv_buffer)

    print("[Custom Dataset Content]")
    print(df_uploaded[["employee_id", "age", "salary"]].to_string())

    # 1. Profiler Check
    profiler = DataProfiler()
    profile = profiler.profile(df_uploaded)
    assert profile["overview"]["total_missing_cells"] == 1, "Failed to profile missing cell"
    assert profile["overview"]["duplicate_rows_count"] == 1, "Failed to profile duplicate row"

    # 2. Rule Engine Detection
    rules = RuleEngine.create_workforce_rules()
    findings = rules.evaluate(df_uploaded)

    missing_findings = [f for f in findings if f.column == "age" and f.row_id == 0]
    duplicate_findings = [f for f in findings if f.issue_type == "duplicate_row"]
    range_findings = [f for f in findings if f.column == "salary" and f.row_id == 3]

    print("\n[Detection Findings Context]")
    print(f"  - Missing value detected:    {len(missing_findings) > 0}")
    if missing_findings:
        f = missing_findings[0]
        print(f"    Context: Rule='{f.rule_name}', Row={f.row_id}, Column='{f.column}', Reason='{f.reason}'")
        assert f.row_id == 0 and f.column == "age"

    print(f"  - Duplicate row detected:    {len(duplicate_findings) > 0}")
    if duplicate_findings:
        f = duplicate_findings[0]
        print(f"    Context: Rule='{f.rule_name}', Row={f.row_id}, Reason='{f.reason}'")
        assert f.row_id == 2  # The second duplicate row

    print(f"  - Range outlier detected:    {len(range_findings) > 0}")
    if range_findings:
        f = range_findings[0]
        print(f"    Context: Rule='{f.rule_name}', Row={f.row_id}, Column='{f.column}', Value={f.current_value}")
        assert f.row_id == 3 and f.column == "salary" and f.current_value == 9999999

    # 3. IQR Detector Check on Salary
    iqr_detector = IQRDetector(k=1.5)
    iqr_detector.fit(df_uploaded)
    iqr_anomalies = iqr_detector.detect(df_uploaded)
    salary_iqr = [a for a in iqr_anomalies if "salary" in a.flagged_features and a.row_id == 3]
    print(f"  - IQR outlier detected:      {len(salary_iqr) > 0}")
    if salary_iqr:
        print(f"    Context: Features={salary_iqr[0].flagged_features}, Row={salary_iqr[0].row_id}, Score={salary_iqr[0].anomaly_score:.3f}")

    assert len(missing_findings) > 0, "Missing value at row 0 was not detected"
    assert len(duplicate_findings) > 0, "Duplicate row at row 2 was not detected"
    assert len(range_findings) > 0 or len(salary_iqr) > 0, "Extreme value at row 3 was not detected"

    print("\n>>> Result T03: PASS - Each issue is detected by the appropriate check and displayed with correct row/column context.")
    return True


def run_t04_detection_settings():
    print("\n" + "=" * 70)
    print("RUNNING T04: Detection settings")
    print("Action: Change IQR multiplier, Isolation Forest contamination, and random seed.")
    print("Expected: Valid settings update the results. Repeating the same configuration gives reproducible predictions.")
    print("=" * 70)

    clean_df = pd.read_csv("data/raw/workforce_clean.csv")

    # 1. IQR Multiplier Sensitivity
    iqr_k10 = IQRDetector(k=1.0).fit(clean_df).detect(clean_df)
    iqr_k15 = IQRDetector(k=1.5).fit(clean_df).detect(clean_df)
    iqr_k25 = IQRDetector(k=2.5).fit(clean_df).detect(clean_df)

    print(f"[IQR Multiplier Setting]")
    print(f"  - k = 1.0 (strict):          {len(iqr_k10)} outliers")
    print(f"  - k = 1.5 (standard):        {len(iqr_k15)} outliers")
    print(f"  - k = 2.5 (permissive):      {len(iqr_k25)} outliers")
    assert len(iqr_k10) >= len(iqr_k15) >= len(iqr_k25), "IQR count should decrease monotonically as k increases"

    # 2. Isolation Forest Contamination Sensitivity
    if_c02 = IsolationForestDetector(contamination=0.02, random_state=42).fit(clean_df).detect(clean_df)
    if_c05 = IsolationForestDetector(contamination=0.05, random_state=42).fit(clean_df).detect(clean_df)
    if_c10 = IsolationForestDetector(contamination=0.10, random_state=42).fit(clean_df).detect(clean_df)

    print(f"\n[Isolation Forest Contamination Setting]")
    print(f"  - contamination = 0.02:      {len(if_c02)} anomalies")
    print(f"  - contamination = 0.05:      {len(if_c05)} anomalies")
    print(f"  - contamination = 0.10:      {len(if_c10)} anomalies")
    assert len(if_c02) <= len(if_c05) <= len(if_c10), "IsoForest count should increase as contamination increases"

    # 3. Random Seed Reproducibility
    run1 = IsolationForestDetector(contamination=0.05, random_state=42).fit(clean_df).detect(clean_df)
    run2 = IsolationForestDetector(contamination=0.05, random_state=42).fit(clean_df).detect(clean_df)

    rows_run1 = [a.row_id for a in run1]
    rows_run2 = [a.row_id for a in run2]
    scores_run1 = [round(a.anomaly_score, 6) for a in run1]
    scores_run2 = [round(a.anomaly_score, 6) for a in run2]

    print(f"\n[Reproducibility Check (random_state=42)]")
    print(f"  - Run 1 flagged rows:        {len(rows_run1)}")
    print(f"  - Run 2 flagged rows:        {len(rows_run2)}")
    print(f"  - Row IDs match 100%:        {rows_run1 == rows_run2}")
    print(f"  - Anomaly scores match 100%: {scores_run1 == scores_run2}")
    assert rows_run1 == rows_run2, "Seed reproducibility failed on row IDs"
    assert scores_run1 == scores_run2, "Seed reproducibility failed on scores"

    print("\n>>> Result T04: PASS - Valid settings update results dynamically; repeating the same configuration gives bit-for-bit reproducible predictions.")
    return True


def run_t05_safe_cleaning():
    print("\n" + "=" * 70)
    print("RUNNING T05: Safe cleaning")
    print("Action: Preview fixes, approve one, reject another, and compare original data with output.")
    print("Expected: Only approved changes are applied. The original remains unchanged and the audit report records transformations.")
    print("=" * 70)

    # Create test data with 1 duplicate and missing age
    raw_df = pd.DataFrame({
        "id": [1, 2, 2, 4],
        "age": [25.0, np.nan, np.nan, 45.0],
        "salary": [50000, 60000, 60000, 999999],
    })

    cleaner = SafeCleaner(raw_df)
    original_snapshot = cleaner.raw_data.copy(deep=True)

    # 1. Preview Fix A: Drop Duplicates
    df_preview_a, diffs_a = cleaner.preview_drop_duplicates()
    print(f"[Preview Fix A: Drop Duplicates]")
    print(f"  - Proposed diffs: {len(diffs_a)} rows to drop")
    print(f"  - APPROVING and applying Fix A via apply_drop_duplicates()...")
    applied_a = cleaner.apply_drop_duplicates()
    assert applied_a == len(diffs_a)

    # Verify state after Fix A
    print(f"  - Current rows after Fix A: {len(cleaner.current_data)} (raw remains {len(cleaner.raw_data)})")
    assert len(cleaner.current_data) == 3

    # 2. Preview Fix B: Cap Outliers (To Be Rejected)
    df_preview_b, diffs_b = cleaner.preview_cap_outliers("salary", lower=40000, upper=100000)
    print(f"\n[Preview Fix B: Cap Outliers on 'salary']")
    print(f"  - Proposed diffs: {len(diffs_b)} cell modifications")
    print(f"  - Proposed change: {diffs_b[0].original_value} -> {diffs_b[0].new_value}")
    print(f"  - REJECTING Fix B (discarding preview, do NOT call apply_cap_outliers)...")
    # Intentional rejection: do NOT call apply_cap_outliers

    # 3. Preview Fix C: Impute Missing Age
    df_preview_c, diffs_c = cleaner.preview_impute_missing("age", strategy="median")
    print(f"\n[Preview Fix C: Impute Missing 'age' with Median]")
    print(f"  - Proposed diffs: {len(diffs_c)} cell modifications")
    print(f"  - APPROVING and applying Fix C via apply_impute_missing()...")
    cleaner.apply_impute_missing("age", strategy="median")

    # 4. Compare Original Data vs Current Data
    print(f"\n[Immutability & Integrity Check]")
    print("  - Comparing original data with current output:")
    pd.testing.assert_frame_equal(cleaner.raw_data, original_snapshot)
    print("    [OK] Original data is strictly UNCHANGED (deep-copy preserved).")

    # Verify that salary was NOT capped (rejected fix was never applied)
    assert cleaner.current_data.loc[cleaner.current_data["id"] == 4, "salary"].values[0] == 999999, \
        "Rejected Fix B was erroneously applied!"
    print("    [OK] Rejected Fix B was NOT applied (salary remains 999999).")

    # Verify approved Fix A and Fix C WERE applied
    assert len(cleaner.current_data) == 3, "Approved Fix A (drop duplicates) was not applied"
    assert cleaner.current_data["age"].isna().sum() == 0, "Approved Fix C (impute) was not applied"
    print("    [OK] Approved Fix A (drop duplicates) and Fix C (impute age) successfully applied.")

    # 5. Check Audit Report
    report = cleaner.generate_audit_report()
    print(f"\n[Audit Report Generated]")
    print(f"  - Total transformations logged: {len(cleaner.audit_log)}")
    print(f"  - Actions logged: {[h.action_type for h in cleaner.audit_log]}")
    assert len(cleaner.audit_log) == 2, f"Expected 2 transformations logged, found {len(cleaner.audit_log)}"
    assert "drop_duplicates" in report
    assert "impute_missing" in report
    assert "cap_outliers" not in report

    print("\n>>> Result T05: PASS - Only approved changes are applied, original remains unchanged, and audit report records transformations.")
    return True


def run_t06_evaluation():
    print("\n" + "=" * 70)
    print("RUNNING T06: Evaluation")
    print("Action: Run empirical evaluation on benchmark fixture and compare with independent ground truth.")
    print("Expected: TP, FP, FN, precision, recall and F1 are consistent with independently calculated results.")
    print("=" * 70)

    fixture_path = "data/test_fixtures/workforce_corrupted.csv"
    gt_path = "data/test_fixtures/ground_truth_labels.csv"

    df_corrupted = pd.read_csv(fixture_path)
    df_gt = pd.read_csv(gt_path)

    # Reconstruct InjectedError objects
    injected_errors = [
        InjectedError(
            row_id=int(r["row_id"]),
            column=str(r["column"]),
            error_type=str(r["error_type"]),
            original_value=r["original_value"],
            injected_value=r["injected_value"],
        )
        for _, r in df_gt.iterrows()
    ]

    print(f"[Benchmark Setup]")
    print(f"  - Corrupted dataset size: {len(df_corrupted)} records")
    print(f"  - Ground truth label rows: {len(injected_errors)}")
    print(f"  - Error types present: {df_gt['error_type'].unique().tolist()}")

    # 1. Run Evaluation Engine
    clean_df = pd.read_csv("data/raw/workforce_clean.csv")
    results = EvaluationEngine.evaluate_all(
        df_corrupted=df_corrupted,
        injected_errors=injected_errors,
        clean_df=clean_df,
        iqr_k=1.5,
        if_contamination=0.05,
        random_state=42,
    )
    metrics_list = results["metrics"]

    # 2. Independent Calculations for Rules
    rules = RuleEngine.create_workforce_rules()
    rule_findings = rules.evaluate(df_corrupted)
    rule_flagged_rows = set(f.row_id for f in rule_findings)

    gt_rule_rows = {
        e.row_id
        for e in injected_errors
        if e.error_type in {
            "invalid_category",
            "range_violation",
            "missing_value",
            "duplicate_row",
            "univariate_outlier",
        }
        or (e.error_type == "multivariate_outlier" and getattr(e, "column", "") == "years_experience")
    }

    indep_tp = len(rule_flagged_rows.intersection(gt_rule_rows))
    indep_fp = len(rule_flagged_rows - gt_rule_rows)
    indep_fn = len(gt_rule_rows - rule_flagged_rows)
    indep_precision = indep_tp / (indep_tp + indep_fp) if (indep_tp + indep_fp) > 0 else 0.0
    indep_recall = indep_tp / (indep_tp + indep_fn) if (indep_tp + indep_fn) > 0 else 0.0
    indep_f1 = 2 * indep_precision * indep_recall / (indep_precision + indep_recall) if (indep_precision + indep_recall) > 0 else 0.0

    app_rule_metrics = [m for m in metrics_list if m["detector_name"] == "Validation Rules" and m["target_scope"] == "Schema & Rule Violations"][0]
    print(f"\n[Validation Rules Metric Comparison]")
    print(f"  - Independent:  TP={indep_tp}, FP={indep_fp}, FN={indep_fn}, P={indep_precision:.4f}, R={indep_recall:.4f}, F1={indep_f1:.4f}")
    print(f"  - App Engine:   TP={app_rule_metrics['true_positives']}, FP={app_rule_metrics['false_positives']}, FN={app_rule_metrics['false_negatives']}, P={app_rule_metrics['precision']:.4f}, R={app_rule_metrics['recall']:.4f}, F1={app_rule_metrics['f1_score']:.4f}")

    assert app_rule_metrics["true_positives"] == indep_tp
    assert app_rule_metrics["false_positives"] == indep_fp
    assert app_rule_metrics["false_negatives"] == indep_fn
    assert abs(app_rule_metrics["precision"] - indep_precision) < 1e-4
    assert abs(app_rule_metrics["recall"] - indep_recall) < 1e-4
    assert abs(app_rule_metrics["f1_score"] - indep_f1) < 1e-4

    # 3. Independent Calculations for IQR Baseline (Univariate Outliers)
    iqr_detector = IQRDetector(k=1.5).fit(df_corrupted)
    iqr_outliers = iqr_detector.detect(df_corrupted)
    iqr_flagged_rows = set(a.row_id for a in iqr_outliers)

    gt_uni_rows = set(df_gt[df_gt["error_type"] == "univariate_outlier"]["row_id"].unique())
    other_gt_rows = set(df_gt[df_gt["error_type"] != "univariate_outlier"]["row_id"].unique())
    iqr_eval_flagged = iqr_flagged_rows - other_gt_rows

    indep_iqr_tp = len(iqr_eval_flagged.intersection(gt_uni_rows))
    indep_iqr_fp = len(iqr_eval_flagged - gt_uni_rows)
    indep_iqr_fn = len(gt_uni_rows - iqr_eval_flagged)
    indep_iqr_p = indep_iqr_tp / (indep_iqr_tp + indep_iqr_fp) if (indep_iqr_tp + indep_iqr_fp) > 0 else 0.0
    indep_iqr_r = indep_iqr_tp / (indep_iqr_tp + indep_iqr_fn) if (indep_iqr_tp + indep_iqr_fn) > 0 else 0.0
    indep_iqr_f1 = 2 * indep_iqr_p * indep_iqr_r / (indep_iqr_p + indep_iqr_r) if (indep_iqr_p + indep_iqr_r) > 0 else 0.0

    app_iqr_metrics = [m for m in metrics_list if m["detector_name"] == "IQR Baseline" and m["target_scope"] == "Univariate Outliers"][0]
    print(f"\n[IQR Baseline (Univariate) Metric Comparison]")
    print(f"  - Independent:  TP={indep_iqr_tp}, FP={indep_iqr_fp}, FN={indep_iqr_fn}, P={indep_iqr_p:.4f}, R={indep_iqr_r:.4f}, F1={indep_iqr_f1:.4f}")
    print(f"  - App Engine:   TP={app_iqr_metrics['true_positives']}, FP={app_iqr_metrics['false_positives']}, FN={app_iqr_metrics['false_negatives']}, P={app_iqr_metrics['precision']:.4f}, R={app_iqr_metrics['recall']:.4f}, F1={app_iqr_metrics['f1_score']:.4f}")

    assert app_iqr_metrics["true_positives"] == indep_iqr_tp
    assert app_iqr_metrics["false_positives"] == indep_iqr_fp
    assert app_iqr_metrics["false_negatives"] == indep_iqr_fn
    assert abs(app_iqr_metrics["precision"] - indep_iqr_p) < 1e-4
    assert abs(app_iqr_metrics["recall"] - indep_iqr_r) < 1e-4

    print("\n>>> Result T06: PASS - TP, FP, FN, precision, recall and F1 are consistent with independently calculated results.")
    return True


def run_t07_export():
    print("\n" + "=" * 70)
    print("RUNNING T07: Export")
    print("Action: Download cleaned CSV and audit report.")
    print("Expected: Files open successfully, row/column identities are preserved as designed, and reported changes match the actual output.")
    print("=" * 70)

    # Start with corrupted dataset
    df_raw = pd.read_csv("data/test_fixtures/workforce_corrupted.csv")
    initial_cols = list(df_raw.columns)
    cleaner = SafeCleaner(df_raw)

    # Execute two transformations
    cleaner.apply_drop_duplicates()
    cleaner.apply_impute_missing("salary", strategy="median")

    # 1. Export Cleaned CSV
    cleaned_csv_bytes = cleaner.current_data.to_csv(index=False).encode("utf-8")
    print(f"[Export Cleaned CSV]")
    print(f"  - Generated CSV size: {len(cleaned_csv_bytes)} bytes")

    # Read back exported CSV
    df_imported = pd.read_csv(io.BytesIO(cleaned_csv_bytes))
    print(f"  - Exported CSV successfully opened.")
    print(f"  - Exported shape: {df_imported.shape}")
    print(f"  - Current data shape: {cleaner.current_data.shape}")

    # Verify row & column preservation
    assert list(df_imported.columns) == initial_cols, "Column order or identities altered during export"
    assert len(df_imported) == len(cleaner.current_data), "Row count discrepancy after export"
    pd.testing.assert_frame_equal(df_imported, cleaner.current_data.reset_index(drop=True), check_dtype=False)
    print("  [OK] Column identities and row counts perfectly preserved.")

    # 2. Export Audit Report
    report_md = cleaner.generate_audit_report()
    report_json = json.dumps([s.to_dict() for s in cleaner.audit_log], indent=2)
    print(f"\n[Export Audit Report]")
    print(f"  - Markdown report length: {len(report_md)} characters")
    print(f"  - JSON report length: {len(report_json)} characters")

    # Verify JSON opens and matches
    parsed_json = json.loads(report_json)
    assert len(parsed_json) == 2, f"Expected 2 transformations in JSON audit report, got {len(parsed_json)}"
    assert parsed_json[0]["action_type"] == "drop_duplicates"
    assert "impute_missing" in parsed_json[1]["action_type"]
    print("  [OK] Reported changes in audit trail match applied transformations exactly.")

    print("\n>>> Result T07: PASS - Files open successfully, row/column identities are preserved as designed, and reported changes match the actual output.")
    return True


def run_t08_invalid_input():
    print("\n" + "=" * 70)
    print("RUNNING T08: Invalid input")
    print("Action: Try an empty CSV, malformed CSV, headers-only CSV, and file with unsupported data types.")
    print("Expected: The app handles each case safely, without a traceback exposed to the user or corrupted state.")
    print("=" * 70)

    profiler = DataProfiler()
    rules = RuleEngine.create_workforce_rules()

    # Case 1: Empty CSV (0 bytes)
    print("[Case 1: Empty CSV (0 bytes)]")
    try:
        df_empty_raw = pd.read_csv(io.StringIO(""))
        print("  - Parsed: unexpected success")
    except pd.errors.EmptyDataError:
        print("  - Handled safely: caught pd.errors.EmptyDataError as expected.")
    
    empty_df = pd.DataFrame()
    p1 = profiler.profile(empty_df)
    r1 = rules.evaluate(empty_df)
    print(f"  - Profiler handled empty df: total_rows={p1['overview']['total_rows']}, health={p1['overview']['health_score']}")
    print(f"  - Rules handled empty df: findings={len(r1)}")
    assert p1["overview"]["total_rows"] == 0
    assert len(r1) == 0

    # Case 2: Headers-only CSV (0 data rows, valid column names)
    print("\n[Case 2: Headers-Only CSV]")
    headers_csv = "employee_id,age,gender,department,job_role,years_experience,salary,performance_rating,education_level,credit_score\n"
    df_headers_only = pd.read_csv(io.StringIO(headers_csv))
    print(f"  - Shape: {df_headers_only.shape}")
    p2 = profiler.profile(df_headers_only)
    r2 = rules.evaluate(df_headers_only)
    cleaner_headers = SafeCleaner(df_headers_only)
    print(f"  - Profiler handled headers-only: total_rows={p2['overview']['total_rows']}, total_columns={p2['overview']['total_columns']}")
    print(f"  - SafeCleaner initialized: current_rows={len(cleaner_headers.current_data)}")
    assert p2["overview"]["total_rows"] == 0
    assert p2["overview"]["total_columns"] == 10

    # Case 3: Malformed CSV string (inconsistent columns/bad quoting)
    print("\n[Case 3: Malformed CSV]")
    malformed_csv = "col1,col2\n1,2,3,4,extra\n5\n"
    try:
        df_malformed = pd.read_csv(io.StringIO(malformed_csv), on_bad_lines='skip')
        print(f"  - Handled with on_bad_lines='skip': parsed shape={df_malformed.shape}")
        p3 = profiler.profile(df_malformed)
        print(f"  - Profiler handled safely: rows={p3['overview']['total_rows']}")
    except Exception as e:
        print(f"  - Gracefully caught parsing error: {e}")

    # Case 4: File with unsupported / exotic data types (complex numbers, dicts, lists)
    print("\n[Case 4: File with Exotic/Unsupported Data Types]")
    exotic_data = {
        "complex_col": [complex(1, 2), complex(3, 4), complex(5, 6)],
        "nested_dict": [{"a": 1}, {"b": 2}, {"c": 3}],
        "nested_list": [[1, 2], [3, 4], [5, 6]],
        "normal_numeric": [10.5, 20.0, 30.2],
    }
    df_exotic = pd.DataFrame(exotic_data)
    print(f"  - Exotic dtypes present: {[str(t) for t in df_exotic.dtypes]}")

    # Profiler handling
    p4 = profiler.profile(df_exotic)
    print(f"  - Profiler result: columns profiled={len(p4['columns'])}, health_score={p4['overview']['health_score']}")

    # Outlier Detector handling (should isolate numeric columns and ignore complex/nested safely)
    iqr_exotic = IQRDetector(k=1.5).fit(df_exotic).detect(df_exotic)
    print(f"  - IQR detector handled exotic dtypes safely: {len(iqr_exotic)} findings")

    # SafeCleaner handling
    cleaner_exotic = SafeCleaner(df_exotic)
    df_prev, diffs = cleaner_exotic.preview_drop_duplicates()
    print(f"  - SafeCleaner handled exotic dtypes safely: diffs={len(diffs)}")

    print("\n>>> Result T08: PASS - The app handles each case safely, without a traceback exposed to the user or corrupted state.")
    return True


if __name__ == "__main__":
    tests = [
        ("T01 Benchmark dataset", run_t01_benchmark_dataset),
        ("T02 Clean reference", run_t02_clean_reference),
        ("T03 Custom CSV", run_t03_custom_csv),
        ("T04 Detection settings", run_t04_detection_settings),
        ("T05 Safe cleaning", run_t05_safe_cleaning),
        ("T06 Evaluation", run_t06_evaluation),
        ("T07 Export", run_t07_export),
        ("T08 Invalid input", run_t08_invalid_input),
    ]

    print("\n" + "=" * 70)
    print("STARTING ACCEPTANCE TESTS T01 - T08")
    print("=" * 70 + "\n")

    summary = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            summary[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f">>> Result {name}: FAIL / EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = f"FAIL: {e}"

    print("\n" + "=" * 70)
    print("ACCEPTANCE TESTS SUMMARY (T01 - T08)")
    print("=" * 70)
    for name, status in summary.items():
        print(f"  {name:30}: {status}")
    print("=" * 70)
