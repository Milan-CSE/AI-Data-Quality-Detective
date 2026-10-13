"""Unit tests for IQR and Isolation Forest anomaly detectors."""

import numpy as np
import pandas as pd
import pytest

from src.detectors import IQRDetector, IsolationForestDetector


def test_iqr_detector_bounds_and_detection():
    # 20 normal values around 10-20, plus one extreme outlier at 100
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0] * 5
    values.append(100.0)  # Extreme outlier at the end

    df = pd.DataFrame({"metric": values})
    detector = IQRDetector(k=1.5)
    detector.fit(df)

    assert "metric" in detector.feature_bounds
    bounds = detector.feature_bounds["metric"]
    assert bounds["lower"] < bounds["q1"] < bounds["q3"] < bounds["upper"]

    findings = detector.detect(df)
    assert len(findings) >= 1
    flagged_ids = {f.row_id for f in findings}
    assert (len(values) - 1) in flagged_ids  # The index of 100.0 must be flagged

    outlier_finding = [f for f in findings if f.row_id == len(values) - 1][0]
    assert outlier_finding.detector == "IQR"
    assert "metric" in outlier_finding.flagged_features
    assert outlier_finding.anomaly_score > 0.8


def test_isolation_forest_detector_fit_and_detect():
    rng = np.random.default_rng(42)
    # Generate 200 standard normal points in 2D
    x1 = rng.normal(50, 5, size=200)
    x2 = x1 * 2 + rng.normal(0, 2, size=200)

    # Add 2 blatant multivariate anomalies (where x1 is high and x2 is low)
    x1 = np.append(x1, [50, 52])
    x2 = np.append(x2, [10, 12])  # Violates x2 = 2*x1 relationship!

    df = pd.DataFrame({"feat_a": x1, "feat_b": x2})

    iso = IsolationForestDetector(contamination=0.05, random_state=42)
    iso.fit(df)
    findings = iso.detect(df)

    assert len(findings) > 0
    # Check that findings have explanations and valid normalized scores
    for f in findings:
        assert f.detector == "Isolation Forest"
        assert 0.0 <= f.anomaly_score <= 1.0
        assert len(f.flagged_features) > 0
        assert f.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert len(f.feature_contributions) == 2


def test_isolation_forest_handles_missing_values():
    # Ensure detector fits without crashing even when DataFrame contains NaNs
    df = pd.DataFrame(
        {
            "num1": [10.0, np.nan, 12.0, 14.0, 15.0, 100.0, 11.0, 13.0, 12.0, 14.0],
            "num2": [20.0, 22.0, np.nan, 24.0, 25.0, 200.0, 21.0, 23.0, 22.0, 24.0],
        }
    )
    iso = IsolationForestDetector(contamination=0.1, random_state=42)
    iso.fit(df)
    findings = iso.detect(df)
    assert isinstance(findings, list)


def test_detectors_empty_input():
    df_empty = pd.DataFrame()

    iqr = IQRDetector().fit(df_empty)
    assert iqr.detect(df_empty) == []

    iso = IsolationForestDetector().fit(df_empty)
    assert iso.detect(df_empty) == []
