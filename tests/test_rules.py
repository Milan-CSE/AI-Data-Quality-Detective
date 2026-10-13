"""Unit tests for the RuleEngine and ValidationRule subclasses."""

import pandas as pd
import pytest

from src.rules import (
    AllowedCategoriesRule,
    CrossFieldRule,
    DuplicateRowRule,
    NotNullRule,
    NumericRangeRule,
    RegexPatternRule,
    RuleEngine,
)


def test_not_null_rule():
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, None, 30.0]})
    rule = NotNullRule(column="value")
    findings = rule.evaluate(df)

    assert len(findings) == 1
    assert findings[0].row_id == 1
    assert findings[0].issue_type == "missing_value"
    assert findings[0].column == "value"


def test_numeric_range_rule():
    df = pd.DataFrame({"age": [25, -5, 140, 50]})
    rule = NumericRangeRule(column="age", min_val=18, max_val=100)
    findings = rule.evaluate(df)

    assert len(findings) == 2
    row_ids = {f.row_id for f in findings}
    assert 1 in row_ids  # -5
    assert 2 in row_ids  # 140


def test_allowed_categories_rule():
    df = pd.DataFrame({"dept": ["Engineering", "HR", "InvalidDept", "sales"]})
    rule = AllowedCategoriesRule(
        column="dept",
        allowed_categories=["Engineering", "HR", "Sales"],
        case_sensitive=False,
    )
    findings = rule.evaluate(df)

    assert len(findings) == 1
    assert findings[0].row_id == 2
    assert findings[0].current_value == "InvalidDept"


def test_regex_pattern_rule():
    df = pd.DataFrame({"emp_id": ["EMP-1001", "EMP-99999", "INVALID-12", "EMP-"]})
    rule = RegexPatternRule(column="emp_id", pattern=r"^EMP-\d{4,6}$")
    findings = rule.evaluate(df)

    assert len(findings) == 2
    assert {f.row_id for f in findings} == {2, 3}


def test_duplicate_row_rule():
    df = pd.DataFrame({"col_a": [1, 2, 1, 3], "col_b": ["A", "B", "A", "C"]})
    rule = DuplicateRowRule()
    findings = rule.evaluate(df)

    assert len(findings) == 1
    assert findings[0].row_id == 2


def test_cross_field_rule():
    df = pd.DataFrame(
        {
            "age": [25, 20],
            "years_experience": [3, 10],  # 20 yr old with 10 yrs experience fails if age < exp + 16
        }
    )
    rule = CrossFieldRule(
        name="age_exp_check",
        condition=lambda r: r["age"] >= r["years_experience"] + 16,
        reason_template="Age {age} invalid for experience {years_experience}",
        column="age",
    )
    findings = rule.evaluate(df)

    assert len(findings) == 1
    assert findings[0].row_id == 1


def test_workforce_rule_engine_integration():
    engine = RuleEngine.create_workforce_rules()
    df = pd.DataFrame(
        {
            "employee_id": ["EMP-1001", "BAD_ID"],
            "age": [30, -2],
            "years_experience": [5, 40],
            "department": ["Engineering", "FakeDept"],
            "education": ["Bachelor", "UnknownEdu"],
            "job_level": ["Mid", "FakeLevel"],
            "salary": [80000.0, 5000000.0],
            "performance_score": [3.5, 9.0],
            "credit_score": [720, 1500],
            "remote_ratio": [50, 50],
        }
    )
    findings_df = engine.evaluate_to_dataframe(df)

    assert len(findings_df) > 0
    # The second row should trigger multiple violations
    assert 1 in set(findings_df["row_id"])
