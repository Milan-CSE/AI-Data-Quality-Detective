"""Rule-Based Data Quality Validation Engine.

Enforces deterministic schema, range, categorical, format, and cross-field domain
rules. Keeps explicit business logic separate from statistical and machine learning
anomaly detection.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


def _safe_row_id(idx: Any) -> Any:
    """Safely convert index to integer if possible, else keep original label."""
    try:
        return int(idx)
    except (ValueError, TypeError):
        return idx


@dataclass
class QualityFinding:
    """Represents a discrete data quality defect flagged by a rule."""

    row_id: Any
    column: Optional[str]
    rule_name: str
    issue_type: str  # missing_value, duplicate_row, range_violation, invalid_category, pattern_mismatch, cross_field_violation
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    current_value: Any
    reason: str
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ValidationRule:
    """Base class for deterministic data quality validation rules."""

    def __init__(self, name: str, severity: str = "HIGH"):
        self.name = name
        self.severity = severity

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        raise NotImplementedError


class NotNullRule(ValidationRule):
    """Flags missing or null values in required columns."""

    def __init__(self, column: str, severity: str = "HIGH"):
        super().__init__(name=f"not_null_{column}", severity=severity)
        self.column = column

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if self.column not in df.columns:
            return findings

        col_series = df[self.column]
        is_na = col_series.isna()

        for idx in df[is_na].index:
            findings.append(
                QualityFinding(
                    row_id=_safe_row_id(idx),
                    column=self.column,
                    rule_name=self.name,
                    issue_type="missing_value",
                    severity=self.severity,
                    current_value=None,
                    reason=f"Column '{self.column}' contains a missing/null value.",
                    suggested_action="impute_missing",
                )
            )
        return findings


class NumericRangeRule(ValidationRule):
    """Flags numeric values outside allowable [min_val, max_val] boundaries."""

    def __init__(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        severity: str = "HIGH",
    ):
        super().__init__(name=f"range_{column}", severity=severity)
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if self.column not in df.columns:
            return findings

        # Coerce to numeric for safe comparison
        num_series = pd.to_numeric(df[self.column], errors="coerce")

        for idx, val in num_series.items():
            if pd.isna(val):
                continue
            violation = False
            reason_parts = []
            if self.min_val is not None and val < self.min_val:
                violation = True
                reason_parts.append(f"below minimum allowed {self.min_val}")
            if self.max_val is not None and val > self.max_val:
                violation = True
                reason_parts.append(f"exceeds maximum allowed {self.max_val}")

            if violation:
                findings.append(
                    QualityFinding(
                        row_id=_safe_row_id(idx),
                        column=self.column,
                        rule_name=self.name,
                        issue_type="range_violation",
                        severity=self.severity,
                        current_value=val,
                        reason=f"Value {val} in '{self.column}' is {' and '.join(reason_parts)}.",
                        suggested_action="cap_outlier",
                    )
                )
        return findings


class AllowedCategoriesRule(ValidationRule):
    """Flags categories not belonging to a known, permitted domain set."""

    def __init__(
        self,
        column: str,
        allowed_categories: List[str],
        case_sensitive: bool = False,
        severity: str = "HIGH",
    ):
        super().__init__(name=f"category_{column}", severity=severity)
        self.column = column
        self.allowed_categories = allowed_categories
        self.case_sensitive = case_sensitive
        if not case_sensitive:
            self._allowed_normalized = {str(c).strip().lower() for c in allowed_categories}
        else:
            self._allowed_normalized = {str(c).strip() for c in allowed_categories}

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if self.column not in df.columns:
            return findings

        for idx, val in df[self.column].items():
            if pd.isna(val):
                continue
            str_val = str(val).strip()
            check_val = str_val if self.case_sensitive else str_val.lower()
            if check_val not in self._allowed_normalized:
                findings.append(
                    QualityFinding(
                        row_id=_safe_row_id(idx),
                        column=self.column,
                        rule_name=self.name,
                        issue_type="invalid_category",
                        severity=self.severity,
                        current_value=val,
                        reason=f"Category '{val}' in '{self.column}' is not in permitted set: {self.allowed_categories}.",
                        suggested_action="replace_invalid",
                    )
                )
        return findings


class RegexPatternRule(ValidationRule):
    """Flags string values that fail a regular expression format check."""

    def __init__(self, column: str, pattern: str, description: str = "", severity: str = "MEDIUM"):
        super().__init__(name=f"pattern_{column}", severity=severity)
        self.column = column
        self.pattern = re.compile(pattern)
        self.description = description or pattern

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if self.column not in df.columns:
            return findings

        for idx, val in df[self.column].items():
            if pd.isna(val):
                continue
            str_val = str(val).strip()
            if not self.pattern.match(str_val):
                findings.append(
                    QualityFinding(
                        row_id=_safe_row_id(idx),
                        column=self.column,
                        rule_name=self.name,
                        issue_type="pattern_mismatch",
                        severity=self.severity,
                        current_value=val,
                        reason=f"Value '{val}' in '{self.column}' does not match expected format ({self.description}).",
                        suggested_action="replace_invalid",
                    )
                )
        return findings


class DuplicateRowRule(ValidationRule):
    """Flags exact duplicate records across selected columns or the entire row."""

    def __init__(self, subset: Optional[List[str]] = None, severity: str = "MEDIUM"):
        name_sub = "_".join(subset) if subset else "all_columns"
        super().__init__(name=f"duplicates_{name_sub}", severity=severity)
        self.subset = subset

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if df.empty:
            return findings

        dup_mask = df.duplicated(subset=self.subset, keep="first")
        for idx in df[dup_mask].index:
            col_spec = f"subset {self.subset}" if self.subset else "all columns"
            findings.append(
                QualityFinding(
                    row_id=_safe_row_id(idx),
                    column=self.subset[0] if (self.subset and len(self.subset) == 1) else None,
                    rule_name=self.name,
                    issue_type="duplicate_row",
                    severity=self.severity,
                    current_value="Duplicate Record",
                    reason=f"Row is an exact duplicate matching an earlier row across {col_spec}.",
                    suggested_action="drop_rows",
                )
            )
        return findings


class CrossFieldRule(ValidationRule):
    """Validates relational constraints between two or more columns."""

    def __init__(
        self,
        name: str,
        condition: Callable[[pd.Series], bool],
        reason_template: str,
        column: Optional[str] = None,
        severity: str = "HIGH",
    ):
        super().__init__(name=name, severity=severity)
        self.condition = condition  # Returns True if VALID, False if INVALID
        self.reason_template = reason_template
        self.column = column

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        findings = []
        if df.empty:
            return findings

        for idx, row in df.iterrows():
            try:
                is_valid = self.condition(row)
                if not is_valid:
                    findings.append(
                        QualityFinding(
                            row_id=_safe_row_id(idx),
                            column=self.column,
                            rule_name=self.name,
                            issue_type="cross_field_violation",
                            severity=self.severity,
                            current_value=dict(row.dropna()),
                            reason=self.reason_template.format(**row.to_dict()),
                            suggested_action="review_manually",
                        )
                    )
            except Exception:
                continue
        return findings


class RuleEngine:
    """Orchestrates deterministic rule evaluations and generates findings reports."""

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> "RuleEngine":
        self.rules.append(rule)
        return self

    def clear_rules(self) -> None:
        self.rules.clear()

    def evaluate(self, df: pd.DataFrame) -> List[QualityFinding]:
        """Run all registered rules against the DataFrame and collect findings."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        all_findings: List[QualityFinding] = []
        for rule in self.rules:
            findings = rule.evaluate(df)
            all_findings.extend(findings)
        return all_findings

    def evaluate_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all rules and return findings as a structured DataFrame."""
        findings = self.evaluate(df)
        if not findings:
            return pd.DataFrame(
                columns=[
                    "row_id",
                    "column",
                    "rule_name",
                    "issue_type",
                    "severity",
                    "current_value",
                    "reason",
                    "suggested_action",
                ]
            )
        return pd.DataFrame([f.to_dict() for f in findings])

    @classmethod
    def create_workforce_rules(cls) -> "RuleEngine":
        """Factory creating the default domain rules for workforce analytics."""
        engine = cls()

        # 0. Missing value rules (Required fields)
        for col in ["employee_id", "age", "years_experience", "department", "education", "job_level", "salary"]:
            engine.add_rule(NotNullRule(column=col, severity="HIGH"))

        # 1. Range rules
        engine.add_rule(NumericRangeRule(column="age", min_val=18, max_val=100, severity="CRITICAL"))
        engine.add_rule(NumericRangeRule(column="years_experience", min_val=0, max_val=50, severity="HIGH"))
        engine.add_rule(NumericRangeRule(column="credit_score", min_val=300, max_val=850, severity="CRITICAL"))
        engine.add_rule(NumericRangeRule(column="salary", min_val=20000, max_val=1000000, severity="HIGH"))
        engine.add_rule(NumericRangeRule(column="performance_score", min_val=1.0, max_val=5.0, severity="HIGH"))

        # 2. Category rules
        valid_departments = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "HR"]
        valid_education = ["High School", "Bachelor", "Master", "PhD"]
        valid_job_levels = ["Entry", "Mid", "Senior", "Lead", "Director"]

        engine.add_rule(AllowedCategoriesRule(column="department", allowed_categories=valid_departments, severity="HIGH"))
        engine.add_rule(AllowedCategoriesRule(column="education", allowed_categories=valid_education, severity="MEDIUM"))
        engine.add_rule(AllowedCategoriesRule(column="job_level", allowed_categories=valid_job_levels, severity="HIGH"))

        # 3. Pattern rules
        engine.add_rule(
            RegexPatternRule(
                column="employee_id",
                pattern=r"^EMP-\d{4,6}$",
                description="EMP-XXXX format",
                severity="HIGH",
            )
        )

        # 4. Duplicate rules
        engine.add_rule(DuplicateRowRule(severity="MEDIUM"))

        # 5. Cross-field logic: age must be at least years_experience + 16
        engine.add_rule(
            CrossFieldRule(
                name="age_experience_coherence",
                condition=lambda r: (
                    pd.isna(r.get("age"))
                    or pd.isna(r.get("years_experience"))
                    or (r["age"] >= r["years_experience"] + 16)
                ),
                reason_template="Age ({age}) is biologically impossible for years of experience ({years_experience}).",
                column="age",
                severity="CRITICAL",
            )
        )

        return engine
