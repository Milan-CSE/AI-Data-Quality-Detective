"""Ground Truth Error Injection and Empirical Evaluation Harness.

Enables reproducible benchmarking of rule-based, statistical, and ML anomaly
detection models against known, controlled injected anomalies.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.detectors import IQRDetector, IsolationForestDetector
from src.rules import RuleEngine

DEFAULT_CONTAMINATIONS: List[float] = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]



@dataclass
class InjectedError:
    """Ground truth record of an artificially injected data quality error."""

    row_id: int
    column: str
    error_type: str  # univariate_outlier, multivariate_outlier, invalid_category, range_violation, missing_value, duplicate_row
    original_value: Any
    injected_value: Any


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics against ground truth."""

    detector_name: str
    target_scope: str  # "overall", "univariate_numerical", "multivariate_numerical", "rules_only"
    total_samples: int
    total_ground_truth_anomalies: int
    total_flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    runtime_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorInjector:
    """Injects controlled, parameterized data quality anomalies with ground truth labels."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def inject(
        self,
        clean_df: pd.DataFrame,
        univariate_count: int = 25,
        multivariate_count: int = 25,
        category_error_count: int = 15,
        range_violation_count: int = 15,
        missing_count: int = 20,
        duplicate_count: int = 10,
    ) -> Tuple[pd.DataFrame, List[InjectedError]]:
        """Inject known anomalies into a pristine DataFrame and track ground truth."""
        df = clean_df.copy(deep=True).reset_index(drop=True)
        n_rows = len(df)
        total_requested = (
            univariate_count
            + multivariate_count
            + category_error_count
            + range_violation_count
            + missing_count
            + duplicate_count
        )
        if n_rows < total_requested:
            raise ValueError(
                f"clean_df must contain at least {total_requested} rows to allocate distinct error injections (received {n_rows} rows)."
            )
        all_indices = np.arange(n_rows)
        self.rng.shuffle(all_indices)

        # Allocate distinct row slices to prevent conflicting overlapping injections
        offset = 0

        def get_slice(count: int) -> np.ndarray:
            nonlocal offset
            sl = all_indices[offset : offset + count]
            offset += count
            return sl

        univariate_idx = get_slice(univariate_count)
        multivariate_idx = get_slice(multivariate_count)
        cat_idx = get_slice(category_error_count)
        range_idx = get_slice(range_violation_count)
        missing_idx = get_slice(missing_count)

        injected_errors: List[InjectedError] = []

        # 1. Univariate Outliers (Extremes beyond 5+ standard deviations / IQR)
        for idx in univariate_idx:
            target_col = self.rng.choice(["salary", "age", "years_experience", "credit_score"])
            orig_val = df.loc[idx, target_col]
            if target_col == "salary":
                inj_val = float(self.rng.choice([1500000.0, 2200000.0, 3500000.0]))
            elif target_col == "age":
                inj_val = int(self.rng.choice([130, 145, 160]))
            elif target_col == "years_experience":
                inj_val = float(self.rng.choice([75.0, 85.0, 92.0]))
            elif target_col == "credit_score":
                inj_val = int(self.rng.choice([1200, 1500]))
            else:
                inj_val = orig_val

            df.loc[idx, target_col] = inj_val
            injected_errors.append(
                InjectedError(
                    row_id=int(idx),
                    column=target_col,
                    error_type="univariate_outlier",
                    original_value=orig_val,
                    injected_value=inj_val,
                )
            )

        # 2. Multivariate Outliers: Features normal in isolation, but joint combination is impossible/anomalous
        # (This is where Isolation Forest should shine and univariate IQR will be blind!)
        for idx in multivariate_idx:
            mode = self.rng.choice(["young_veteran", "entry_director_salary", "old_novice_exec"])
            if mode == "young_veteran":
                # 22-year-old with 25 years experience
                orig_age = df.loc[idx, "age"]
                orig_exp = df.loc[idx, "years_experience"]
                df.loc[idx, "age"] = 22
                df.loc[idx, "years_experience"] = 26.0
                injected_errors.append(
                    InjectedError(
                        row_id=int(idx),
                        column="years_experience",
                        error_type="multivariate_outlier",
                        original_value=(orig_age, orig_exp),
                        injected_value=(22, 26.0),
                    )
                )
            elif mode == "entry_director_salary":
                # 0.5 years experience with director-level salary
                orig_exp = df.loc[idx, "years_experience"]
                orig_sal = df.loc[idx, "salary"]
                df.loc[idx, "years_experience"] = 0.5
                df.loc[idx, "salary"] = 245000.0
                injected_errors.append(
                    InjectedError(
                        row_id=int(idx),
                        column="salary",
                        error_type="multivariate_outlier",
                        original_value=(orig_exp, orig_sal),
                        injected_value=(0.5, 245000.0),
                    )
                )
            else:
                # 64 years old with 0.5 years experience making 220k
                orig_age = df.loc[idx, "age"]
                orig_exp = df.loc[idx, "years_experience"]
                orig_sal = df.loc[idx, "salary"]
                df.loc[idx, "age"] = 64
                df.loc[idx, "years_experience"] = 0.5
                df.loc[idx, "salary"] = 210000.0
                injected_errors.append(
                    InjectedError(
                        row_id=int(idx),
                        column="salary",
                        error_type="multivariate_outlier",
                        original_value=(orig_age, orig_exp, orig_sal),
                        injected_value=(64, 0.5, 210000.0),
                    )
                )

        # 3. Invalid Categories (Typos, illegal domain values)
        typos = {
            "department": ["Engneering", "Sales_Dept", "Marketing123", "UnknownDept"],
            "education": ["PHD_Special", "HighSchoolGrad", "MasterOfArts"],
            "job_level": ["Intern_Lead", "Executive_VP", "Staff_Engineer"],
        }
        for idx in cat_idx:
            target_col = self.rng.choice(["department", "education", "job_level"])
            orig_val = df.loc[idx, target_col]
            inj_val = str(self.rng.choice(typos[target_col]))
            df.loc[idx, target_col] = inj_val
            injected_errors.append(
                InjectedError(
                    row_id=int(idx),
                    column=target_col,
                    error_type="invalid_category",
                    original_value=orig_val,
                    injected_value=inj_val,
                )
            )

        # 4. Range Violations (Negative ages, impossible scores)
        for idx in range_idx:
            target_col = self.rng.choice(["age", "performance_score", "credit_score"])
            orig_val = df.loc[idx, target_col]
            if target_col == "age":
                inj_val = int(self.rng.choice([-15, -5, 0]))
            elif target_col == "performance_score":
                inj_val = float(self.rng.choice([8.5, 12.0, -1.5]))
            elif target_col == "credit_score":
                inj_val = int(self.rng.choice([150, 950, 1100]))
            else:
                inj_val = orig_val

            df.loc[idx, target_col] = inj_val
            injected_errors.append(
                InjectedError(
                    row_id=int(idx),
                    column=target_col,
                    error_type="range_violation",
                    original_value=orig_val,
                    injected_value=inj_val,
                )
            )

        # 5. Missing Values (Null / NaN)
        for idx in missing_idx:
            target_col = self.rng.choice(["salary", "years_experience", "department", "age"])
            orig_val = df.loc[idx, target_col]
            df.loc[idx, target_col] = np.nan
            injected_errors.append(
                InjectedError(
                    row_id=int(idx),
                    column=target_col,
                    error_type="missing_value",
                    original_value=orig_val,
                    injected_value=np.nan,
                )
            )

        # 6. Duplicate Rows
        if duplicate_count > 0:
            dup_source_indices = self.rng.choice(all_indices[offset:], size=duplicate_count, replace=False)
            dup_rows = df.loc[dup_source_indices].copy()
            new_start_idx = len(df)
            for i, src_idx in enumerate(dup_source_indices):
                curr_idx = new_start_idx + i
                injected_errors.append(
                    InjectedError(
                        row_id=curr_idx,
                        column="[ROW]",
                        error_type="duplicate_row",
                        original_value=f"Original Row {src_idx}",
                        injected_value=f"Duplicate of {src_idx}",
                    )
                )
            df = pd.concat([df, dup_rows], ignore_index=True)

        return df, injected_errors


class EvaluationEngine:
    """Calculates classification metrics, latency, and false alarm rate comparisons."""

    @staticmethod
    def compute_metrics(
        detector_name: str,
        target_scope: str,
        flagged_row_ids: Set[int],
        ground_truth_row_ids: Set[int],
        total_samples: int,
        runtime_ms: float,
        population_row_ids: Optional[Set[int]] = None,
        other_anomaly_row_ids: Optional[Set[int]] = None,
    ) -> EvaluationMetrics:
        """Compute standard confusion matrix and ML evaluation metrics.

        When population_row_ids is provided, evaluation is strictly scoped to that
        population: flagged and ground truth rows outside the population are excluded,
        and total_samples is set to the population size. This ensures strict confusion
        matrix invariants:
          total_flagged == true_positives + false_positives
          total_ground_truth == true_positives + false_negatives
          total_samples == TP + FP + FN + TN
        """
        if population_row_ids is not None:
            eval_flagged = flagged_row_ids.intersection(population_row_ids)
            eval_gt = ground_truth_row_ids.intersection(population_row_ids)
            eval_total = len(population_row_ids)
        else:
            eval_flagged = set(flagged_row_ids)
            eval_gt = set(ground_truth_row_ids)
            eval_total = total_samples

        tp = len(eval_flagged.intersection(eval_gt))
        fp = len(eval_flagged - eval_gt)
        fn = len(eval_gt - eval_flagged)
        tn = max(0, eval_total - (tp + fp + fn))

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return EvaluationMetrics(
            detector_name=detector_name,
            target_scope=target_scope,
            total_samples=eval_total,
            total_ground_truth_anomalies=len(eval_gt),
            total_flagged=len(eval_flagged),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1_score=round(float(f1), 4),
            false_positive_rate=round(float(fpr), 4),
            runtime_ms=round(float(runtime_ms), 2),
        )

    @classmethod
    def evaluate_all(
        cls,
        df_corrupted: pd.DataFrame,
        injected_errors: List[InjectedError],
        clean_df: Optional[pd.DataFrame] = None,
        iqr_k: float = 1.5,
        if_contamination: float = 0.05,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Comprehensive multi-model evaluation against synthetic ground truth."""
        total_samples = len(df_corrupted)

        # Ground truth groupings
        gt_all: Set[int] = {e.row_id for e in injected_errors}
        clean_row_ids: Set[int] = set(range(total_samples)) - gt_all

        # Univariate numerical scope consistently includes both extreme outliers and range violations
        gt_univariate: Set[int] = {
            e.row_id
            for e in injected_errors
            if e.error_type in {"univariate_outlier", "range_violation"}
        }
        gt_multivariate: Set[int] = {
            e.row_id for e in injected_errors if e.error_type == "multivariate_outlier"
        }
        gt_numerical_all: Set[int] = gt_univariate.union(gt_multivariate)
        gt_rules: Set[int] = {
            e.row_id
            for e in injected_errors
            if e.error_type in {
                "invalid_category",
                "range_violation",
                "missing_value",
                "duplicate_row",
                "univariate_outlier",  # Univariate injections also violate hard max/min bounds
            }
            or (e.error_type == "multivariate_outlier" and getattr(e, "column", "") == "years_experience")
        }

        # Explicitly defined evaluation populations for each target scope
        pop_univariate: Set[int] = clean_row_ids.union(gt_univariate)
        pop_multivariate: Set[int] = clean_row_ids.union(gt_multivariate)
        pop_numerical_all: Set[int] = clean_row_ids.union(gt_numerical_all)
        pop_rules: Set[int] = clean_row_ids.union(gt_rules)

        # 1. Run IQR Baseline
        t0 = time.perf_counter()
        iqr = IQRDetector(k=iqr_k)
        iqr.fit(df_corrupted)
        iqr_findings = iqr.detect(df_corrupted)
        iqr_time = (time.perf_counter() - t0) * 1000
        iqr_flagged: Set[int] = {f.row_id for f in iqr_findings}

        # 2. Run Isolation Forest
        t0 = time.perf_counter()
        iso = IsolationForestDetector(
            contamination=if_contamination,
            random_state=random_state,
        )
        iso.fit(df_corrupted)
        iso_findings = iso.detect(df_corrupted)
        iso_time = (time.perf_counter() - t0) * 1000
        iso_flagged: Set[int] = {f.row_id for f in iso_findings}

        # 3. Run Rule Engine
        t0 = time.perf_counter()
        rule_engine = RuleEngine.create_workforce_rules()
        rule_findings = rule_engine.evaluate(df_corrupted)
        rule_time = (time.perf_counter() - t0) * 1000
        rule_flagged: Set[int] = {f.row_id for f in rule_findings}

        # Calculate metrics for each detector across relevant scopes
        metrics_list: List[EvaluationMetrics] = []

        # Scope: Univariate Numerical Outliers
        metrics_list.append(
            cls.compute_metrics(
                "IQR Baseline",
                "Univariate Outliers",
                iqr_flagged,
                gt_univariate,
                total_samples,
                iqr_time,
                population_row_ids=pop_univariate,
            )
        )
        metrics_list.append(
            cls.compute_metrics(
                "Isolation Forest",
                "Univariate Outliers",
                iso_flagged,
                gt_univariate,
                total_samples,
                iso_time,
                population_row_ids=pop_univariate,
            )
        )

        # Scope: Multivariate Numerical Outliers (Where IF shines and IQR should miss!)
        metrics_list.append(
            cls.compute_metrics(
                "IQR Baseline",
                "Multivariate Outliers",
                iqr_flagged,
                gt_multivariate,
                total_samples,
                iqr_time,
                population_row_ids=pop_multivariate,
            )
        )
        metrics_list.append(
            cls.compute_metrics(
                "Isolation Forest",
                "Multivariate Outliers",
                iso_flagged,
                gt_multivariate,
                total_samples,
                iso_time,
                population_row_ids=pop_multivariate,
            )
        )

        # Scope: Combined Numerical (Univariate + Multivariate + Numeric Range)
        metrics_list.append(
            cls.compute_metrics(
                "IQR Baseline",
                "All Numerical Outliers",
                iqr_flagged,
                gt_numerical_all,
                total_samples,
                iqr_time,
                population_row_ids=pop_numerical_all,
            )
        )
        metrics_list.append(
            cls.compute_metrics(
                "Isolation Forest",
                "All Numerical Outliers",
                iso_flagged,
                gt_numerical_all,
                total_samples,
                iso_time,
                population_row_ids=pop_numerical_all,
            )
        )

        # Scope: Rule Violations (Rules vs IQR vs Isolation Forest)
        metrics_list.append(
            cls.compute_metrics(
                "Validation Rules",
                "Schema & Rule Violations",
                rule_flagged,
                gt_rules,
                total_samples,
                rule_time,
                population_row_ids=pop_rules,
            )
        )

        # Clean Data False Alarm Test
        clean_false_alarms = {}
        if clean_df is not None and not clean_df.empty:
            iqr_clean = IQRDetector(k=iqr_k).fit(clean_df).detect(clean_df)
            iso_clean = (
                IsolationForestDetector(
                    contamination=if_contamination,
                    random_state=random_state,
                )
                .fit(clean_df)
                .detect(clean_df)
            )
            rule_clean = rule_engine.evaluate(clean_df)

            n_clean = len(clean_df)
            iqr_clean_rows = {f.row_id for f in iqr_clean}
            iso_clean_rows = {f.row_id for f in iso_clean}
            rule_clean_rows = {f.row_id for f in rule_clean}

            clean_false_alarms = {
                "clean_samples": n_clean,
                "iqr_false_alarms": len(iqr_clean_rows),
                "iqr_false_alarm_rate": round(len(iqr_clean_rows) / n_clean, 4) if n_clean > 0 else 0.0,
                "iso_forest_false_alarms": len(iso_clean_rows),
                "iso_forest_false_alarm_rate": round(len(iso_clean_rows) / n_clean, 4) if n_clean > 0 else 0.0,
                "rules_false_alarms": len(rule_clean_rows),
                "rules_false_alarm_rate": round(len(rule_clean_rows) / n_clean, 4) if n_clean > 0 else 0.0,
            }

        return {
            "metrics": [m.to_dict() for m in metrics_list],
            "clean_test": clean_false_alarms,
            "counts": {
                "total_samples": total_samples,
                "ground_truth_total": len(gt_all),
                "univariate_count": len(gt_univariate),
                "multivariate_count": len(gt_multivariate),
                "rule_violations_count": len(gt_rules),
            },
        }

    @classmethod
    def run_contamination_sensitivity(
        cls,
        df_corrupted: pd.DataFrame,
        injected_errors: List[InjectedError],
        contaminations: Optional[List[float]] = None,
        random_state: int = 42,
    ) -> List[Dict[str, Any]]:
        """Evaluate Isolation Forest sensitivity across varying contamination assumptions."""
        if contaminations is None:
            contaminations = DEFAULT_CONTAMINATIONS

        clean_row_ids = set(range(len(df_corrupted))) - {e.row_id for e in injected_errors}
        gt_univariate = {
            e.row_id
            for e in injected_errors
            if e.error_type in {"univariate_outlier", "range_violation"}
        }
        gt_multivariate = {e.row_id for e in injected_errors if e.error_type == "multivariate_outlier"}
        gt_numerical_all = gt_univariate.union(gt_multivariate)

        pop_numerical = clean_row_ids.union(gt_numerical_all)
        pop_multivariate = clean_row_ids.union(gt_multivariate)

        results = []
        for c in contaminations:
            t0 = time.perf_counter()
            iso = IsolationForestDetector(contamination=c, random_state=random_state)
            iso.fit(df_corrupted)
            findings = iso.detect(df_corrupted)
            duration = (time.perf_counter() - t0) * 1000
            flagged = {f.row_id for f in findings}

            m_all = cls.compute_metrics(
                f"Isolation Forest (c={c})",
                "Numerical Outliers",
                flagged,
                gt_numerical_all,
                len(df_corrupted),
                duration,
                population_row_ids=pop_numerical,
            )
            m_multi = cls.compute_metrics(
                f"Isolation Forest (c={c})",
                "Multivariate Only",
                flagged,
                gt_multivariate,
                len(df_corrupted),
                duration,
                population_row_ids=pop_multivariate,
            )

            results.append(
                {
                    "contamination": c,
                    "flagged_count": len(flagged),
                    "precision_all": m_all.precision,
                    "recall_all": m_all.recall,
                    "f1_all": m_all.f1_score,
                    "precision_multi": m_multi.precision,
                    "recall_multi": m_multi.recall,
                    "f1_multi": m_multi.f1_score,
                    "runtime_ms": round(duration, 2),
                }
            )
        return results
