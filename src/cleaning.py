"""Safe, Reversible Data Cleaning Engine.

Provides non-destructive transformations, diff previews, audit logging,
and rollback capabilities to ensure original data remains strictly preserved.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DiffRecord:
    """Represents a discrete cell or row modification."""

    row_id: Any
    column: str
    original_value: Any
    new_value: Any
    action_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditStep:
    """Represents a committed cleaning operation in the pipeline."""

    step_id: int
    timestamp: str
    action_type: str
    parameters: Dict[str, Any]
    affected_rows: int
    affected_cells: int
    diff_sample: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafeCleaner:
    """Executes safe, audited, non-destructive data cleaning transformations."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        # Strictly preserve the raw reference data
        self._raw_data: pd.DataFrame = df.copy(deep=True)
        self._current_data: pd.DataFrame = df.copy(deep=True)
        self._history: List[pd.DataFrame] = [self._current_data.copy(deep=True)]
        self._audit_log: List[AuditStep] = []

    @property
    def raw_data(self) -> pd.DataFrame:
        """Immutable view of the original raw data."""
        return self._raw_data.copy(deep=True)

    @property
    def current_data(self) -> pd.DataFrame:
        """Current transformed state of the data."""
        return self._current_data.copy(deep=True)

    @property
    def audit_trail(self) -> List[AuditStep]:
        """List of all applied transformations."""
        return list(self._audit_log)

    @property
    def audit_log(self) -> List[AuditStep]:
        """Alias for audit_trail."""
        return self.audit_trail

    def preview_drop_rows(self, row_ids: List[int]) -> Tuple[pd.DataFrame, List[DiffRecord]]:
        """Preview the effect of dropping specific row IDs without modifying state."""
        valid_ids = [r for r in row_ids if r in self._current_data.index]
        new_df = self._current_data.drop(index=valid_ids).copy()

        diffs = [
            DiffRecord(
                row_id=r,
                column="[ALL_COLUMNS]",
                original_value=dict(self._current_data.loc[r].dropna()),
                new_value="[ROW_DELETED]",
                action_type="drop_row",
            )
            for r in valid_ids
        ]
        return new_df, diffs

    def apply_drop_rows(self, row_ids: List[int]) -> int:
        """Commit dropping specific row IDs."""
        new_df, diffs = self.preview_drop_rows(row_ids)
        if diffs:
            self._commit_change(
                new_df=new_df,
                action_type="drop_rows",
                params={"dropped_count": len(diffs), "row_ids": row_ids[:10]},
                affected_rows=len(diffs),
                affected_cells=len(diffs) * self._current_data.shape[1],
                diffs=diffs,
            )
        return len(diffs)

    def preview_drop_duplicates(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> Tuple[pd.DataFrame, List[DiffRecord]]:
        """Preview duplicate row removal."""
        try:
            dup_mask = self._current_data.duplicated(subset=subset, keep=keep)
        except TypeError:
            dup_mask = self._current_data.astype(str).duplicated(subset=subset, keep=keep)
        # Use boolean indexing to avoid deleting all copies when index has duplicate labels
        new_df = self._current_data[~dup_mask].copy()

        diffs = [
            DiffRecord(
                row_id=idx,
                column="[ALL_COLUMNS]",
                original_value="Duplicate Row",
                new_value="[ROW_DELETED]",
                action_type="drop_duplicate",
            )
            for idx in self._current_data[dup_mask].index
        ]
        return new_df, diffs

    def apply_drop_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> int:
        """Commit duplicate row removal."""
        new_df, diffs = self.preview_drop_duplicates(subset=subset, keep=keep)
        if diffs:
            self._commit_change(
                new_df=new_df,
                action_type="drop_duplicates",
                params={"subset": subset, "keep": keep},
                affected_rows=len(diffs),
                affected_cells=len(diffs) * self._current_data.shape[1],
                diffs=diffs,
            )
        return len(diffs)

    def preview_impute_missing(
        self,
        column: str,
        strategy: str = "median",
        fill_value: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, List[DiffRecord]]:
        """Preview missing value imputation on a column."""
        if column not in self._current_data.columns:
            return self._current_data.copy(), []

        new_df = self._current_data.copy()
        series = new_df[column]
        missing_mask = series.isna()

        if not missing_mask.any():
            return new_df, []

        computed_fill = fill_value
        if computed_fill is None:
            non_null = series.dropna()
            if len(non_null) == 0:
                # Column is all-NaN; cannot compute statistical fill
                return new_df, []

            if strategy == "median" and pd.api.types.is_numeric_dtype(series):
                computed_fill = float(non_null.median())
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(series):
                computed_fill = float(non_null.mean())
            elif strategy == "mode":
                modes = non_null.mode()
                computed_fill = modes.iloc[0] if not modes.empty else "Unknown"
            else:
                computed_fill = 0 if pd.api.types.is_numeric_dtype(series) else "Unknown"

        if computed_fill is None or (isinstance(computed_fill, (float, np.floating)) and np.isnan(computed_fill)):
            return new_df, []

        new_df[column] = series.fillna(computed_fill)

        diffs = []
        for idx in self._current_data[missing_mask].index:
            diffs.append(
                DiffRecord(
                    row_id=idx,
                    column=column,
                    original_value=None,
                    new_value=computed_fill,
                    action_type=f"impute_{strategy}",
                )
            )
        return new_df, diffs

    def apply_impute_missing(
        self,
        column: str,
        strategy: str = "median",
        fill_value: Optional[Any] = None,
    ) -> int:
        """Commit missing value imputation."""
        new_df, diffs = self.preview_impute_missing(column, strategy, fill_value)
        if diffs:
            self._commit_change(
                new_df=new_df,
                action_type=f"impute_missing_{strategy}",
                params={"column": column, "strategy": strategy, "fill_value": fill_value},
                affected_rows=len(diffs),
                affected_cells=len(diffs),
                diffs=diffs,
            )
        return len(diffs)

    def preview_cap_outliers(
        self,
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, List[DiffRecord]]:
        """Preview winsorizing / clamping numeric outliers to specified boundaries."""
        if column not in self._current_data.columns:
            return self._current_data.copy(), []

        new_df = self._current_data.copy()
        raw_series = new_df[column]
        num_series = pd.to_numeric(raw_series, errors="coerce")
        diffs = []

        for idx, val in num_series.items():
            if pd.isna(val):
                continue
            new_val = val
            if lower is not None and val < lower:
                new_val = lower
            elif upper is not None and val > upper:
                new_val = upper

            if new_val != val:
                orig_val = raw_series.loc[idx]
                if isinstance(orig_val, pd.Series):
                    orig_val = orig_val.iloc[0]
                diffs.append(
                    DiffRecord(
                        row_id=idx,
                        column=column,
                        original_value=orig_val,
                        new_value=new_val,
                        action_type="cap_outliers",
                    )
                )

        if diffs:
            if not pd.api.types.is_numeric_dtype(new_df[column]):
                new_df[column] = new_df[column].astype(object)
            for d in diffs:
                new_df.loc[d.row_id, column] = d.new_value

        return new_df, diffs

    def apply_cap_outliers(
        self,
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> int:
        """Commit capping of numeric values."""
        new_df, diffs = self.preview_cap_outliers(column, lower, upper)
        if diffs:
            self._commit_change(
                new_df=new_df,
                action_type="cap_outliers",
                params={"column": column, "lower": lower, "upper": upper},
                affected_rows=len(diffs),
                affected_cells=len(diffs),
                diffs=diffs,
            )
        return len(diffs)

    def preview_replace_values(
        self, column: str, mapping: Dict[Any, Any]
    ) -> Tuple[pd.DataFrame, List[DiffRecord]]:
        """Preview category/value mapping and typo replacements."""
        if column not in self._current_data.columns:
            return self._current_data.copy(), []

        new_df = self._current_data.copy()
        diffs = []

        for idx, val in new_df[column].items():
            str_key = str(val).strip()
            replacement = None
            found_key = False
            if val in mapping:
                replacement = mapping[val]
                found_key = True
            elif str_key in mapping:
                replacement = mapping[str_key]
                found_key = True
            else:
                for k, v in mapping.items():
                    if str(k).lower().strip() == str_key.lower():
                        replacement = v
                        found_key = True
                        break

            if found_key:
                is_diff = False
                if pd.isna(val) and not pd.isna(replacement):
                    is_diff = True
                elif not pd.isna(val) and pd.isna(replacement):
                    is_diff = True
                elif not pd.isna(val) and not pd.isna(replacement) and val != replacement:
                    is_diff = True

                if is_diff:
                    diffs.append(
                        DiffRecord(
                            row_id=idx,
                            column=column,
                            original_value=val,
                            new_value=replacement,
                            action_type="replace_value",
                        )
                    )

        if diffs:
            new_df[column] = new_df[column].astype(object)
            for d in diffs:
                new_df.loc[d.row_id, column] = d.new_value

        return new_df, diffs

    def apply_replace_values(self, column: str, mapping: Dict[Any, Any]) -> int:
        """Commit value replacement."""
        new_df, diffs = self.preview_replace_values(column, mapping)
        if diffs:
            self._commit_change(
                new_df=new_df,
                action_type="replace_values",
                params={"column": column, "mapping": mapping},
                affected_rows=len(diffs),
                affected_cells=len(diffs),
                diffs=diffs,
            )
        return len(diffs)

    def undo(self) -> bool:
        """Revert the most recent cleaning transformation."""
        if len(self._history) <= 1:
            return False  # Already at raw state

        self._history.pop()
        self._current_data = self._history[-1].copy(deep=True)
        if self._audit_log:
            self._audit_log.pop()
        return True

    def reset_to_raw(self) -> None:
        """Revert all cleaning operations back to original pristine raw data."""
        self._current_data = self._raw_data.copy(deep=True)
        self._history = [self._current_data.copy(deep=True)]
        self._audit_log.clear()

    def _commit_change(
        self,
        new_df: pd.DataFrame,
        action_type: str,
        params: Dict[str, Any],
        affected_rows: int,
        affected_cells: int,
        diffs: List[DiffRecord],
    ) -> None:
        """Internal commit mechanism ensuring history stack and audit trails remain coherent."""
        step_id = len(self._audit_log) + 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sample_diffs = [d.to_dict() for d in diffs[:20]]
        audit_step = AuditStep(
            step_id=step_id,
            timestamp=timestamp,
            action_type=action_type,
            parameters=params,
            affected_rows=affected_rows,
            affected_cells=affected_cells,
            diff_sample=sample_diffs,
        )

        self._audit_log.append(audit_step)
        self._current_data = new_df.copy(deep=True)
        self._history.append(self._current_data.copy(deep=True))

    def generate_audit_report(self) -> str:
        """Produce an executive markdown audit report documenting all actions taken."""
        lines = [
            "# Data Quality Detective - Transformation Audit Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Original Shape**: {self._raw_data.shape[0]} rows × {self._raw_data.shape[1]} columns",
            f"- **Cleaned Shape**: {self._current_data.shape[0]} rows × {self._current_data.shape[1]} columns",
            f"- **Transformations Applied**: {len(self._audit_log)}",
            "",
            "## Changelog of Operations",
        ]

        if not self._audit_log:
            lines.append("No cleaning transformations were applied. Dataset remains in original state.")
        else:
            for step in self._audit_log:
                lines.append(f"### Step {step.step_id}: `{step.action_type}`")
                lines.append(f"- **Timestamp**: {step.timestamp}")
                lines.append(f"- **Affected Records**: {step.affected_rows} rows ({step.affected_cells} cells)")
                lines.append(f"- **Parameters**: `{json.dumps(step.parameters)}`")
                if step.diff_sample:
                    lines.append("- **Sample Changes**:")
                    lines.append("  | Row | Column | Original Value | New Value |")
                    lines.append("  | --- | ------ | -------------- | --------- |")
                    for s in step.diff_sample[:5]:
                        lines.append(f"  | {s['row_id']} | {s['column']} | `{s['original_value']}` | `{s['new_value']}` |")
                lines.append("")

        return "\n".join(lines)
