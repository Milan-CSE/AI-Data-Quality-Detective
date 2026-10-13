"""Data Profiling Module for AI Data Quality Detective.

Computes comprehensive dataset-level and column-level summary statistics,
detects missing patterns, duplicate rows, data type mismatches, and computes
an overall Data Quality Health Score.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataProfiler:
    """Profiles tabular datasets to uncover structural, statistical, and quality characteristics."""

    def __init__(self, high_cardinality_threshold: float = 0.5):
        """
        Args:
            high_cardinality_threshold: Ratio of unique values to rows above which
                                       a categorical feature is flagged as high cardinality.
        """
        self.high_cardinality_threshold = high_cardinality_threshold

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform full structural and statistical profiling on a DataFrame.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dictionary containing overview, column profiles, correlation matrix,
            and data quality warnings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if df.empty:
            return self._empty_profile()

        n_rows, n_cols = df.shape
        n_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / n_cells) * 100, 2) if n_cells > 0 else 0.0

        # Duplicate calculation
        dup_mask = df.duplicated()
        duplicate_count = int(dup_mask.sum())
        duplicate_pct = round((duplicate_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

        # Memory usage
        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = round(memory_bytes / (1024 * 1024), 3)

        # Profile columns
        col_profiles = {}
        numeric_cols = []
        warnings = []

        if duplicate_pct > 0.0:
            warnings.append(
                f"Dataset contains {duplicate_count} duplicate rows ({duplicate_pct}% of total records)."
            )

        for col in df.columns:
            col_profile = self._profile_column(df[col], n_rows)
            col_profiles[str(col)] = col_profile

            if col_profile["inferred_type"] == "numeric":
                numeric_cols.append(col)

            # Accumulate quality warnings
            if col_profile["missing_pct"] > 30.0:
                warnings.append(
                    f"Column '{col}' has high missingness: {col_profile['missing_pct']}% missing values."
                )
            elif col_profile["missing_pct"] > 0.0:
                warnings.append(
                    f"Column '{col}' has {col_profile['missing_count']} missing values ({col_profile['missing_pct']}%)."
                )

            if col_profile.get("is_constant", False):
                warnings.append(f"Column '{col}' is constant (has only 1 unique non-null value).")

            if col_profile.get("high_cardinality_warning", False):
                warnings.append(
                    f"Column '{col}' has high cardinality ({col_profile['unique_count']} unique values), may be an ID or free text."
                )

        # Pairwise numeric correlations
        correlations = {}
        if len(numeric_cols) >= 2:
            try:
                corr_df = df[numeric_cols].corr().round(3)
                correlations = corr_df.to_dict()
            except Exception:
                correlations = {}

        # Calculate composite health score (0 - 100)
        health_score = self._compute_health_score(
            n_rows=n_rows,
            n_cols=n_cols,
            missing_pct=missing_pct,
            duplicate_pct=duplicate_pct,
            col_profiles=col_profiles,
        )

        return {
            "overview": {
                "total_rows": n_rows,
                "total_columns": n_cols,
                "total_cells": n_cells,
                "total_missing_cells": total_missing,
                "missing_cells_pct": missing_pct,
                "duplicate_rows_count": duplicate_count,
                "duplicate_rows_pct": duplicate_pct,
                "memory_usage_mb": memory_mb,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": n_cols - len(numeric_cols),
                "health_score": health_score,
            },
            "columns": col_profiles,
            "correlations": correlations,
            "warnings": warnings,
        }

    def _profile_column(self, series: pd.Series, total_rows: int) -> Dict[str, Any]:
        """Profile an individual pandas Series."""
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique())
        unique_pct = (
            round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        )

        pandas_dtype = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        )
        is_bool = pd.api.types.is_bool_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        inferred_type = "categorical"
        if is_numeric:
            inferred_type = "numeric"
        elif is_bool:
            inferred_type = "boolean"
        elif is_datetime:
            inferred_type = "datetime"
        elif unique_count == total_rows and total_rows > 20:
            inferred_type = "identifier"

        profile: Dict[str, Any] = {
            "name": str(series.name),
            "pandas_dtype": pandas_dtype,
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "is_constant": (unique_count == 1),
        }

        # Check for whitespace-only strings
        if series.dtype == object or str(series.dtype) == "string":
            string_items = non_null_series.astype(str)
            empty_string_count = int((string_items.str.strip() == "").sum())
            profile["empty_string_count"] = empty_string_count

        if is_numeric and len(non_null_series) > 0:
            q25 = float(non_null_series.quantile(0.25))
            q75 = float(non_null_series.quantile(0.75))
            iqr = round(q75 - q25, 4)
            min_val = float(non_null_series.min())
            max_val = float(non_null_series.max())
            mean_val = float(non_null_series.mean())
            median_val = float(non_null_series.median())
            std_val = float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
            skew_val = float(non_null_series.skew()) if len(non_null_series) > 2 else 0.0

            zeros_count = int((non_null_series == 0).sum())
            neg_count = int((non_null_series < 0).sum())

            profile.update(
                {
                    "min": round(min_val, 4),
                    "max": round(max_val, 4),
                    "mean": round(mean_val, 4),
                    "median": round(median_val, 4),
                    "std": round(std_val, 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "iqr": iqr,
                    "skewness": round(skew_val, 4) if not np.isnan(skew_val) else 0.0,
                    "zeros_count": zeros_count,
                    "zeros_pct": round((zeros_count / total_rows) * 100, 2),
                    "negative_count": neg_count,
                    "negative_pct": round((neg_count / total_rows) * 100, 2),
                }
            )
        else:
            # Categorical frequency stats
            top_vals = non_null_series.value_counts().head(5)
            profile["top_values"] = [
                {
                    "value": str(k),
                    "count": int(v),
                    "pct": round((int(v) / total_rows) * 100, 2),
                }
                for k, v in top_vals.items()
            ]
            if unique_pct > (self.high_cardinality_threshold * 100) and inferred_type != "identifier":
                profile["high_cardinality_warning"] = True

        return profile

    def _compute_health_score(
        self,
        n_rows: int,
        n_cols: int,
        missing_pct: float,
        duplicate_pct: float,
        col_profiles: Dict[str, Any],
    ) -> float:
        """Compute an interpretable Data Quality Health Score (0 - 100).

        Factors:
        - Completeness (weight 40%): penalizes missing values
        - Uniqueness (weight 25%): penalizes duplicate rows
        - Consistency & Cardinality (weight 20%): penalizes constant columns and whitespace-only strings
        - Uniformity / Validity (weight 15%): basic structural integrity
        """
        # Completeness score: 100 - missing_pct
        completeness = max(0.0, 100.0 - (missing_pct * 2.0))

        # Uniqueness score: 100 - duplicate_pct
        uniqueness = max(0.0, 100.0 - (duplicate_pct * 3.0))

        # Consistency score
        constant_penalty = sum(1 for p in col_profiles.values() if p.get("is_constant", False)) * 10.0
        empty_str_penalty = sum(p.get("empty_string_count", 0) for p in col_profiles.values())
        empty_str_penalty = min(20.0, (empty_str_penalty / max(1, n_rows)) * 100.0)
        consistency = max(0.0, 100.0 - constant_penalty - empty_str_penalty)

        # Validity score: penalty for severe anomalies or all-null columns
        all_null_cols = sum(1 for p in col_profiles.values() if p["missing_count"] == n_rows)
        validity = max(0.0, 100.0 - (all_null_cols * 25.0))

        health = (
            (0.40 * completeness)
            + (0.25 * uniqueness)
            + (0.20 * consistency)
            + (0.15 * validity)
        )
        return round(float(np.clip(health, 0.0, 100.0)), 1)

    def _empty_profile(self) -> Dict[str, Any]:
        """Return a structured empty profile when DataFrame has no records."""
        return {
            "overview": {
                "total_rows": 0,
                "total_columns": 0,
                "total_cells": 0,
                "total_missing_cells": 0,
                "missing_cells_pct": 0.0,
                "duplicate_rows_count": 0,
                "duplicate_rows_pct": 0.0,
                "memory_usage_mb": 0.0,
                "numeric_columns_count": 0,
                "categorical_columns_count": 0,
                "health_score": 0.0,
            },
            "columns": {},
            "correlations": {},
            "warnings": ["DataFrame is empty."],
        }
