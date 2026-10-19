"""Anomaly Detectors for AI Data Quality Detective.

Implements:
1. Model A: Univariate Interquartile Range (IQR) Statistical Baseline.
2. Model B: Multivariate Isolation Forest Machine Learning Detector.
Both provide consistent result interfaces, normalized anomaly scores, and
feature-level explainability.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


@dataclass
class AnomalyFinding:
    """Represents a flagged anomaly detected by statistical or ML models."""

    row_id: int
    detector: str  # "IQR" or "Isolation Forest"
    anomaly_score: float  # Normalized 0.0 to 1.0 (higher = more anomalous)
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    flagged_features: List[str]
    reason: str
    feature_contributions: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IQRDetector:
    """Model A: Univariate Interquartile Range (IQR) Statistical Baseline.

    Evaluates each numerical feature independently. Flag values outside:
    [Q1 - k*IQR, Q3 + k*IQR]
    """

    def __init__(self, k: float = 1.5):
        """
        Args:
            k: Multiplier for IQR (1.5 standard outlier, 3.0 extreme outlier).
        """
        self.k = k
        self.feature_bounds: Dict[str, Dict[str, float]] = {}
        self.numeric_columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> "IQRDetector":
        """Compute Q1, Q3, and IQR thresholds for selected numerical features."""
        if df.empty:
            return self

        if columns is not None:
            self.numeric_columns = [
                c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self.numeric_columns = [
                c
                for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            ]

        self.feature_bounds = {}
        for col in self.numeric_columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            # If IQR is 0 (e.g. constant or heavily repeated value), fall back to std or small epsilon
            effective_iqr = iqr if iqr > 1e-6 else float(series.std() or 1.0)

            lower = q1 - (self.k * effective_iqr)
            upper = q3 + (self.k * effective_iqr)
            self.feature_bounds[col] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(effective_iqr, 4),
                "lower": round(lower, 4),
                "upper": round(upper, 4),
            }
        return self

    def detect(self, df: pd.DataFrame) -> List[AnomalyFinding]:
        """Detect univariate outliers across fitted numerical columns."""
        if not self.feature_bounds or df.empty:
            return []

        findings: List[AnomalyFinding] = []

        # Track per-row anomalies
        row_violations: Dict[int, Dict[str, Any]] = {}

        for col, bounds in self.feature_bounds.items():
            if col not in df.columns:
                continue

            lower = bounds["lower"]
            upper = bounds["upper"]
            iqr = bounds["iqr"]

            series = df[col]
            for idx, val in series.items():
                if pd.isna(val):
                    continue
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    continue

                if num_val < lower or num_val > upper:
                    distance = (lower - num_val) if num_val < lower else (num_val - upper)
                    iqr_units = distance / (iqr + 1e-8)
                    norm_score = min(1.0, float(1.0 - np.exp(-0.5 * iqr_units)))

                    if idx not in row_violations:
                        row_violations[idx] = {
                            "features": [],
                            "scores": {},
                            "reasons": [],
                            "max_units": 0.0,
                        }

                    bound_str = f"< lower bound {lower}" if num_val < lower else f"> upper bound {upper}"
                    row_violations[idx]["features"].append(col)
                    row_violations[idx]["scores"][col] = round(norm_score, 4)
                    row_violations[idx]["reasons"].append(
                        f"{col}={num_val:.2f} ({bound_str} by {iqr_units:.1f}x IQR)"
                    )
                    if iqr_units > row_violations[idx]["max_units"]:
                        row_violations[idx]["max_units"] = iqr_units

        # Convert to AnomalyFinding instances
        for idx, item in row_violations.items():
            max_units = item["max_units"]
            if max_units > 4.0:
                severity = "CRITICAL"
            elif max_units > 2.5:
                severity = "HIGH"
            elif max_units > 1.2:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            overall_score = max(item["scores"].values())
            reason_text = "Univariate IQR Outlier: " + "; ".join(item["reasons"])

            findings.append(
                AnomalyFinding(
                    row_id=int(idx),
                    detector="IQR",
                    anomaly_score=round(overall_score, 4),
                    severity=severity,
                    flagged_features=item["features"],
                    reason=reason_text,
                    feature_contributions=item["scores"],
                )
            )

        # Sort findings by anomaly score descending
        findings.sort(key=lambda x: x.anomaly_score, reverse=True)
        return findings


class IsolationForestDetector:
    """Model B: Multivariate Isolation Forest Machine Learning Detector.

    Identifies anomalous observations in joint multi-dimensional feature space.
    Employs counterfactual feature attribution to explain why an observation was isolated.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        """
        Args:
            contamination: Expected proportion of outliers in the data (or 'auto').
            n_estimators: Number of isolation trees in the ensemble.
            random_state: Fixed seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.numeric_columns: List[str] = []
        self.medians: Dict[str, float] = {}
        self.train_latency_ms: float = 0.0

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> "IsolationForestDetector":
        """Preprocess numerical features and fit the Isolation Forest model."""
        start_time = time.perf_counter()

        if df.empty:
            return self

        if columns is not None:
            self.numeric_columns = [
                c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self.numeric_columns = [
                c
                for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            ]

        # Filter out columns that have 0 non-null values to prevent SimpleImputer dropping features
        self.numeric_columns = [c for c in self.numeric_columns if df[c].notna().any()]

        if len(self.numeric_columns) == 0:
            return self

        X_raw = df[self.numeric_columns].values
        # Impute missing values with column medians for robust tree training
        X_imputed = self.imputer.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_imputed)

        for i, col in enumerate(self.numeric_columns):
            self.medians[col] = float(self.imputer.statistics_[i])

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.train_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return self

    def detect(self, df: pd.DataFrame) -> List[AnomalyFinding]:
        """Flag multivariate anomalies and explain feature-level attribution."""
        if self.model is None or df.empty or len(self.numeric_columns) == 0:
            return []

        # Ensure all fitted numeric columns exist in input df
        if any(c not in df.columns for c in self.numeric_columns):
            return []

        X_raw = df[self.numeric_columns].values
        X_imputed = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imputed)

        # Scikit-learn: -1 for outlier, 1 for inlier
        predictions = self.model.predict(X_scaled)
        # Decision function: negative values are outliers, positive are inliers
        raw_scores = self.model.decision_function(X_scaled)

        # Normalize score: convert decision function to [0, 1] range where 1 is extreme anomaly
        # Offset decision function so min gets score near 1.0, positive inliers get near 0.0
        scores_normalized = 1.0 / (1.0 + np.exp(raw_scores * 6.0))

        flagged_indices = np.where(predictions == -1)[0]
        findings: List[AnomalyFinding] = []

        if len(flagged_indices) == 0:
            return findings

        row_index_map = list(df.index)

        # Vectorized Feature Attribution:
        # Since RobustScaler sets median=0 and scales by IQR, |X_scaled| is the normalized
        # robust distance of each feature from the population center.
        abs_deviations = np.abs(X_scaled[flagged_indices])
        dev_sums = np.sum(abs_deviations, axis=1, keepdims=True) + 1e-8
        relative_contributions = abs_deviations / dev_sums

        for idx_pos, i in enumerate(flagged_indices):
            row_id = row_index_map[i]
            row_norm_score = float(scores_normalized[i])

            contributions = {
                self.numeric_columns[col_idx]: round(float(relative_contributions[idx_pos, col_idx]), 3)
                for col_idx in range(len(self.numeric_columns))
            }

            # Top contributing features for explanation
            sorted_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            top_features = [k for k, v in sorted_contributors if v > 0.15][:3]
            if not top_features:
                top_features = [sorted_contributors[0][0]]

            # Explain reason
            reason_parts = []
            for feat in top_features:
                val = df.iloc[i][feat] if i < len(df) else df.loc[row_id, feat]
                med = self.medians.get(feat, 0.0)
                val_str = f"{val:.2f}" if isinstance(val, (int, float, np.number)) and not pd.isna(val) else str(val)
                reason_parts.append(
                    f"{feat}={val_str} (diverges from median {med:.1f}, {contributions[feat]*100:.0f}% contribution)"
                )

            reason_str = "Multivariate Anomaly: Isolated in joint feature space driven by " + " & ".join(
                reason_parts
            )

            # Assign severity based on normalized score
            if row_norm_score > 0.85:
                severity = "CRITICAL"
            elif row_norm_score > 0.70:
                severity = "HIGH"
            elif row_norm_score > 0.55:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            findings.append(
                AnomalyFinding(
                    row_id=int(row_id),
                    detector="Isolation Forest",
                    anomaly_score=round(row_norm_score, 4),
                    severity=severity,
                    flagged_features=top_features,
                    reason=reason_str,
                    feature_contributions=contributions,
                )
            )

        findings.sort(key=lambda x: x.anomaly_score, reverse=True)
        return findings
