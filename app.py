"""AI Data Quality Detective - Streamlit Web Application.

A professional ML data quality and anomaly detection platform combining:
- Deterministic schema and validation rules
- Statistical univariate baseline (IQR)
- Unsupervised multivariate ML (Isolation Forest) with counterfactual explanations
- Safe, audited, reversible transformations with diff preview
- Empirical evaluation against labeled synthetic ground truth
"""

import io
import json
import os
import sys

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.cleaning import SafeCleaner
from src.detectors import AnomalyFinding, IQRDetector, IsolationForestDetector
from src.evaluation import ErrorInjector, EvaluationEngine
from src.profiler import DataProfiler
from src.rules import QualityFinding, RuleEngine

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Data Quality Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_dataset(dataset_type: str = "corrupted") -> pd.DataFrame:
    """Load or generate benchmark reference datasets."""
    if dataset_type == "clean":
        path = os.path.join("data", "raw", "workforce_clean.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        from data.generate_data import generate_clean_workforce_dataset

        df = generate_clean_workforce_dataset(1500, random_seed=42)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(path, index=False)
        return df
    else:
        path = os.path.join("data", "test_fixtures", "workforce_corrupted.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        # Generate corrupted if not present
        clean_df = load_sample_dataset("clean")
        injector = ErrorInjector(random_seed=42)
        df_corrupted, _ = injector.inject(clean_df)
        os.makedirs("data/test_fixtures", exist_ok=True)
        df_corrupted.to_csv(path, index=False)
        return df_corrupted


# Initialize Session State
if "cleaner" not in st.session_state:
    st.session_state.cleaner = None
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""


# Sidebar: Data Intake & Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/search-database.png", width=64)
    st.markdown("### Data Intake")

    data_source = st.radio(
        "Select Data Source:",
        ["Benchmark Test Fixture (with Errors)", "Clean Reference Dataset", "Upload Custom CSV"],
        index=0,
    )

    df_to_load = None
    if data_source == "Benchmark Test Fixture (with Errors)":
        if st.session_state.dataset_name != "benchmark_corrupted":
            df_to_load = load_sample_dataset("corrupted")
            st.session_state.dataset_name = "benchmark_corrupted"
    elif data_source == "Clean Reference Dataset":
        if st.session_state.dataset_name != "benchmark_clean":
            df_to_load = load_sample_dataset("clean")
            st.session_state.dataset_name = "benchmark_clean"
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                df_to_load = uploaded_df
                st.session_state.dataset_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    if df_to_load is not None:
        st.session_state.active_df = df_to_load.copy(deep=True)
        st.session_state.cleaner = SafeCleaner(df_to_load)

    st.markdown("---")
    st.markdown("### Detection Settings")
    iqr_k = st.slider("IQR Outlier Multiplier (k)", min_value=1.0, max_value=3.5, value=1.5, step=0.1)
    if_contamination = st.slider(
        "Isolation Forest Contamination", min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )
    random_seed = st.number_input("Model Random Seed", min_value=1, max_value=9999, value=42)

    st.markdown("---")
    st.caption("AI Data Quality Detective v1.0 • Built with Scikit-learn, Pandas & Streamlit")


# Header
st.markdown('<div class="main-title">🔍 AI Data Quality Detective</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Empirical Anomaly Detection, Rule Validation & Safe Reversible Cleaning for Tabular Data</div>',
    unsafe_allow_html=True,
)

if st.session_state.cleaner is None:
    st.info("👈 Please select or upload a dataset using the sidebar to begin.")
    st.stop()

cleaner: SafeCleaner = st.session_state.cleaner
current_df = cleaner.current_data

# Navigation Tabs
tab_profile, tab_detect, tab_clean, tab_eval, tab_export = st.tabs(
    [
        "📊 1. Data Profiling",
        "🕵️ 2. Anomaly Detective",
        "🛠️ 3. Safe Cleaning Studio",
        "📈 4. Empirical Evaluation",
        "📥 5. Export & Audit Report",
    ]
)

# -------------------------------------------------------------
# TAB 1: DATA PROFILING
# -------------------------------------------------------------
with tab_profile:
    st.markdown("### Dataset Quality Health Profile")

    profiler = DataProfiler()
    profile = profiler.profile(current_df)
    overview = profile["overview"]

    # Top Metric KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{overview['total_rows']:,}")
    with col2:
        st.metric("Total Features", f"{overview['total_columns']}")
    with col3:
        st.metric("Missing Cells", f"{overview['total_missing_cells']} ({overview['missing_cells_pct']}%)")
    with col4:
        st.metric("Duplicates", f"{overview['duplicate_rows_count']} ({overview['duplicate_rows_pct']}%)")
    with col5:
        health = overview["health_score"]
        health_color = "normal" if health >= 80 else "inverse"
        st.metric("Health Score", f"{health} / 100")

    if profile["warnings"]:
        st.markdown("#### ⚠️ Quality Hygiene Warnings")
        for w in profile["warnings"]:
            st.warning(w)

    st.markdown("---")
    st.markdown("### Column-Level Statistical Summary")

    col_data = []
    for col_name, c_prof in profile["columns"].items():
        row_dict = {
            "Column": col_name,
            "Inferred Type": c_prof["inferred_type"],
            "Pandas Dtype": c_prof["pandas_dtype"],
            "Missing Count": c_prof["missing_count"],
            "Missing %": f"{c_prof['missing_pct']}%",
            "Unique Count": c_prof["unique_count"],
            "Unique %": f"{c_prof['unique_pct']}%",
            "Min": c_prof.get("min", "-"),
            "Median": c_prof.get("median", "-"),
            "Max": c_prof.get("max", "-"),
            "Std": c_prof.get("std", "-"),
            "IQR": c_prof.get("iqr", "-"),
        }
        col_data.append(row_dict)

    summary_df = pd.DataFrame(col_data)
    st.dataframe(summary_df, use_container_width=True)

    # Correlation Matrix
    if profile["correlations"]:
        st.markdown("#### Numeric Correlation Heatmap")
        corr_matrix = pd.DataFrame(profile["correlations"])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="vlag", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        plt.title("Feature Pearson Correlation Matrix", fontsize=11)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("👁️ View Raw Dataset Sample"):
        st.dataframe(current_df.head(50), use_container_width=True)


# -------------------------------------------------------------
# TAB 2: DETECTIVE & ANOMALY DETECTION
# -------------------------------------------------------------
with tab_detect:
    st.markdown("### Three-Tier Data Quality Investigation")
    st.markdown(
        """
    - **Tier 1: Validation Rules** — Identifies schema violations, illegal categories, out-of-range bounds, and duplicate records.
    - **Tier 2: IQR Baseline** — Flags univariate distribution outliers feature-by-feature ($[Q_1 - 1.5\\text{IQR}, Q_3 + 1.5\\text{IQR}]$).
    - **Tier 3: Isolation Forest** — Isolates multivariate anomalies in multi-dimensional space, providing feature attribution.
    """
    )

    # Run Detectors
    with st.spinner("Running detection algorithms across all three tiers..."):
        # 1. Rules
        rule_engine = RuleEngine.create_workforce_rules()
        rule_findings = rule_engine.evaluate(current_df)

        # 2. IQR
        iqr_detector = IQRDetector(k=iqr_k).fit(current_df)
        iqr_findings = iqr_detector.detect(current_df)

        # 3. Isolation Forest
        iso_detector = IsolationForestDetector(
            contamination=if_contamination, random_state=random_seed
        ).fit(current_df)
        iso_findings = iso_detector.detect(current_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Deterministic Rule Violations", len(rule_findings))
    with col2:
        st.metric("IQR Univariate Outliers", len(iqr_findings))
    with col3:
        st.metric("Isolation Forest Anomalies", len(iso_findings))
    with col4:
        all_flagged_rows = (
            {f.row_id for f in rule_findings}
            .union({f.row_id for f in iqr_findings})
            .union({f.row_id for f in iso_findings})
        )
        st.metric("Unique Suspicious Rows", f"{len(all_flagged_rows)} ({len(all_flagged_rows)/len(current_df)*100:.1f}%)")

    st.markdown("---")

    # Unified Findings Table
    all_findings_rows = []
    for r in rule_findings:
        all_findings_rows.append(
            {
                "Row ID": r.row_id,
                "Tier": "Rule Engine",
                "Detector / Rule": r.rule_name,
                "Column": r.column or "N/A",
                "Severity": r.severity,
                "Anomaly Score": 1.0,
                "Reason & Evidence": r.reason,
                "Recommended Action": r.suggested_action,
            }
        )
    for q in iqr_findings:
        all_findings_rows.append(
            {
                "Row ID": q.row_id,
                "Tier": "IQR Baseline",
                "Detector / Rule": "IQR Outlier",
                "Column": ", ".join(q.flagged_features),
                "Severity": q.severity,
                "Anomaly Score": q.anomaly_score,
                "Reason & Evidence": q.reason,
                "Recommended Action": "cap_outlier",
            }
        )
    for m in iso_findings:
        all_findings_rows.append(
            {
                "Row ID": m.row_id,
                "Tier": "Isolation Forest",
                "Detector / Rule": "Isolation Forest (ML)",
                "Column": ", ".join(m.flagged_features),
                "Severity": m.severity,
                "Anomaly Score": m.anomaly_score,
                "Reason & Evidence": m.reason,
                "Recommended Action": "review_manually",
            }
        )

    findings_df = pd.DataFrame(all_findings_rows)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        selected_tier = st.multiselect(
            "Filter by Tier:",
            ["Rule Engine", "IQR Baseline", "Isolation Forest"],
            default=["Rule Engine", "IQR Baseline", "Isolation Forest"],
        )
    with f_col2:
        selected_severity = st.multiselect(
            "Filter by Severity:",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )
    with f_col3:
        search_kw = st.text_input("Search findings reason or column:", "")

    filtered_findings = findings_df[
        (findings_df["Tier"].isin(selected_tier))
        & (findings_df["Severity"].isin(selected_severity))
    ]
    if search_kw:
        filtered_findings = filtered_findings[
            filtered_findings["Reason & Evidence"].str.contains(search_kw, case=False, na=False)
            | filtered_findings["Column"].str.contains(search_kw, case=False, na=False)
        ]

    st.markdown(f"**Showing {len(filtered_findings)} findings**")
    st.dataframe(filtered_findings, use_container_width=True)

    # Anomaly Inspection Deep-Dive
    st.markdown("#### 🔬 Anomaly Row Inspector")
    if not filtered_findings.empty:
        sample_row_ids = sorted(list(filtered_findings["Row ID"].unique()))[:50]
        selected_inspect_row = st.selectbox("Select a Row ID to Inspect:", sample_row_ids)

        if selected_inspect_row in current_df.index:
            r_col1, r_col2 = st.columns([2, 3])
            with r_col1:
                st.markdown(f"**Record Values for Row #{selected_inspect_row}**:")
                row_series = current_df.loc[selected_inspect_row]
                st.json(row_series.dropna().to_dict())
            with r_col2:
                st.markdown("**Triggered Alerts for This Row**:")
                row_alerts = filtered_findings[filtered_findings["Row ID"] == selected_inspect_row]
                for _, alert in row_alerts.iterrows():
                    st.info(
                        f"**[{alert['Tier']}] ({alert['Severity']})** {alert['Detector / Rule']}\n\n"
                        f"**Evidence:** {alert['Reason & Evidence']}\n\n"
                        f"**Suggested Action:** `{alert['Recommended Action']}`"
                    )


# -------------------------------------------------------------
# TAB 3: SAFE CLEANING STUDIO
# -------------------------------------------------------------
with tab_clean:
    st.markdown("### Safe, Reversible Transformation Studio")
    st.markdown(
        "> **Immutability Guarantee**: The original dataset is never modified directly. "
        "Every operation generates a side-by-side diff preview and can be undone with 1-click."
    )

    clean_col1, clean_col2 = st.columns([1, 1])

    with clean_col1:
        st.markdown("#### 1. Choose Cleaning Remediation")
        action = st.selectbox(
            "Select Remediation Operation:",
            [
                "Drop Duplicate Records",
                "Impute Missing Values",
                "Cap / Winsorize Outliers (IQR Bounds)",
                "Replace Invalid Categories / Fix Typos",
                "Drop Selected Flagged Rows",
            ],
        )

        preview_btn = False
        commit_btn = False

        if action == "Drop Duplicate Records":
            st.write("Identifies duplicate rows and keeps the first occurrence.")
            preview_btn = st.button("Preview Drop Duplicates", key="prev_dup")
            commit_btn = st.button("Apply & Commit Drop Duplicates", type="primary", key="app_dup")

            if preview_btn or commit_btn:
                prev_df, diffs = cleaner.preview_drop_duplicates()
                st.session_state.active_preview = (prev_df, diffs, "drop_duplicates", {})

        elif action == "Impute Missing Values":
            num_cols = current_df.columns[current_df.isna().any()].tolist()
            if not num_cols:
                st.success("No missing values found in current dataset!")
            else:
                target_col = st.selectbox("Target Column:", num_cols)
                strategy = st.selectbox("Strategy:", ["median", "mean", "mode", "constant"])
                custom_val = None
                if strategy == "constant":
                    custom_val = st.text_input("Constant Value:", value="0")

                preview_btn = st.button("Preview Imputation", key="prev_imp")
                commit_btn = st.button("Apply & Commit Imputation", type="primary", key="app_imp")

                if preview_btn or commit_btn:
                    prev_df, diffs = cleaner.preview_impute_missing(
                        column=target_col, strategy=strategy, fill_value=custom_val
                    )
                    st.session_state.active_preview = (
                        prev_df,
                        diffs,
                        "impute",
                        {"col": target_col, "strat": strategy, "val": custom_val},
                    )

        elif action == "Cap / Winsorize Outliers (IQR Bounds)":
            num_cols = [c for c in current_df.columns if pd.api.types.is_numeric_dtype(current_df[c])]
            target_col = st.selectbox("Target Numeric Column:", num_cols)
            q1 = float(current_df[target_col].quantile(0.25))
            q3 = float(current_df[target_col].quantile(0.75))
            iqr = q3 - q1
            default_lower = round(q1 - (iqr_k * iqr), 2)
            default_upper = round(q3 + (iqr_k * iqr), 2)

            lower_bound = st.number_input("Lower Bound (Clamp)", value=float(default_lower))
            upper_bound = st.number_input("Upper Bound (Clamp)", value=float(default_upper))

            preview_btn = st.button("Preview Capping", key="prev_cap")
            commit_btn = st.button("Apply & Commit Capping", type="primary", key="app_cap")

            if preview_btn or commit_btn:
                prev_df, diffs = cleaner.preview_cap_outliers(
                    column=target_col, lower=lower_bound, upper=upper_bound
                )
                st.session_state.active_preview = (
                    prev_df,
                    diffs,
                    "cap",
                    {"col": target_col, "lower": lower_bound, "upper": upper_bound},
                )

        elif action == "Replace Invalid Categories / Fix Typos":
            cat_cols = [c for c in current_df.columns if not pd.api.types.is_numeric_dtype(current_df[c])]
            target_col = st.selectbox("Target Categorical Column:", cat_cols)
            unique_vals = list(current_df[target_col].dropna().unique())
            bad_val = st.selectbox("Value to replace:", unique_vals)
            good_val = st.text_input("New Replacement Value:")

            preview_btn = st.button("Preview Category Replacement", key="prev_rep")
            commit_btn = st.button("Apply & Commit Replacement", type="primary", key="app_rep")

            if (preview_btn or commit_btn) and good_val:
                prev_df, diffs = cleaner.preview_replace_values(
                    column=target_col, mapping={bad_val: good_val}
                )
                st.session_state.active_preview = (
                    prev_df,
                    diffs,
                    "replace",
                    {"col": target_col, "mapping": {bad_val: good_val}},
                )

        elif action == "Drop Selected Flagged Rows":
            st.write("Drop rows flagged with CRITICAL severity violations.")
            crit_rows = [f.row_id for f in rule_findings if f.severity == "CRITICAL"]
            st.write(f"Found {len(crit_rows)} rows with CRITICAL violations.")

            preview_btn = st.button("Preview Drop Critical Rows", key="prev_drop_crit")
            commit_btn = st.button("Apply & Commit Drop Critical Rows", type="primary", key="app_drop_crit")

            if preview_btn or commit_btn:
                prev_df, diffs = cleaner.preview_drop_rows(crit_rows)
                st.session_state.active_preview = (
                    prev_df,
                    diffs,
                    "drop_rows",
                    {"rows": crit_rows},
                )

    with clean_col2:
        st.markdown("#### 2. Diff Preview & Commit")
        if "active_preview" in st.session_state and st.session_state.active_preview:
            prev_df, diffs, op_type, params = st.session_state.active_preview
            st.success(f"Preview ready: {len(diffs)} modifications proposed.")

            if diffs:
                diff_table = pd.DataFrame([d.to_dict() for d in diffs[:25]])
                st.dataframe(diff_table, use_container_width=True)
                if len(diffs) > 25:
                    st.caption(f"... and {len(diffs) - 25} more modifications.")

            if commit_btn:
                if op_type == "drop_duplicates":
                    cleaner.apply_drop_duplicates()
                elif op_type == "impute":
                    cleaner.apply_impute_missing(
                        column=params["col"], strategy=params["strat"], fill_value=params["val"]
                    )
                elif op_type == "cap":
                    cleaner.apply_cap_outliers(
                        column=params["col"], lower=params["lower"], upper=params["upper"]
                    )
                elif op_type == "replace":
                    cleaner.apply_replace_values(
                        column=params["col"], mapping=params["mapping"]
                    )
                elif op_type == "drop_rows":
                    cleaner.apply_drop_rows(params["rows"])

                st.session_state.active_preview = None
                st.success("✅ Transformation committed successfully!")
                st.rerun()
        else:
            st.info("Select an operation and click 'Preview' to view proposed diffs before committing.")

    st.markdown("---")
    # Pipeline Controls: Undo & Reset
    st.markdown("#### Pipeline Controls & Audit History")
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    with p_col1:
        if st.button("↩️ Undo Last Operation", disabled=(len(cleaner.audit_trail) == 0)):
            if cleaner.undo():
                st.success("Reverted last operation.")
                st.rerun()
    with p_col2:
        if st.button("🔄 Reset All to Original Raw Data", type="secondary"):
            cleaner.reset_to_raw()
            st.warning("Reset dataset back to original raw state.")
            st.rerun()

    if cleaner.audit_trail:
        st.markdown("**Committed Transformation Changelog:**")
        audit_records = [
            {
                "Step": s.step_id,
                "Timestamp": s.timestamp,
                "Action": s.action_type,
                "Affected Records": f"{s.affected_rows} rows",
                "Parameters": json.dumps(s.parameters),
            }
            for s in cleaner.audit_trail
        ]
        st.dataframe(pd.DataFrame(audit_records), use_container_width=True)


# -------------------------------------------------------------
# TAB 4: EMPIRICAL EVALUATION & BENCHMARKING
# -------------------------------------------------------------
with tab_eval:
    st.markdown("### Empirical Anomaly Detection Evaluation")
    st.markdown(
        """
    > **Why Evaluation Matters**: In genuine machine learning engineering, a detector is only as good as its measured precision and recall against ground truth.
    > We evaluate our statistical baseline (IQR), unsupervised ML (Isolation Forest), and validation rules against **known synthetically injected errors**.
    """
    )

    clean_ref = load_sample_dataset("clean")
    fixture_df = load_sample_dataset("corrupted")
    gt_path = os.path.join("data", "test_fixtures", "ground_truth_labels.csv")

    if os.path.exists(gt_path):
        gt_labels_df = pd.read_csv(gt_path)
    else:
        injector = ErrorInjector(random_seed=42)
        _, injected_errs = injector.inject(clean_ref)
        gt_labels_df = pd.DataFrame([e.__dict__ for e in injected_errs])

    # Re-run or Load Benchmarks
    with st.spinner("Calculating benchmark metrics against ground truth labels..."):
        injected_objects = [
            type(
                "InjectedError",
                (),
                {
                    "row_id": int(r["row_id"]),
                    "column": r["column"],
                    "error_type": r["error_type"],
                    "original_value": r["original_value"],
                    "injected_value": r["injected_value"],
                },
            )()
            for _, r in gt_labels_df.iterrows()
        ]

        benchmark_eval = EvaluationEngine.evaluate_all(
            df_corrupted=fixture_df,
            injected_errors=injected_objects,
            clean_df=clean_ref,
            iqr_k=iqr_k,
            if_contamination=if_contamination,
            random_state=random_seed,
        )

    # 1. Benchmark Table
    st.markdown("#### 1. Head-to-Head Model Performance")
    metrics_df = pd.DataFrame(benchmark_eval["metrics"])
    display_metrics = metrics_df[
        [
            "detector_name",
            "target_scope",
            "total_flagged",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1_score",
            "runtime_ms",
        ]
    ].rename(
        columns={
            "detector_name": "Detector",
            "target_scope": "Scope",
            "total_flagged": "Flagged",
            "true_positives": "TP",
            "false_positives": "FP",
            "false_negatives": "FN",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "runtime_ms": "Latency (ms)",
        }
    )
    st.dataframe(display_metrics, use_container_width=True)

    # Visualization: Metric Comparison
    st.markdown("#### 2. Metric Comparison (F1, Precision & Recall)")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = display_metrics.copy()
    plot_df["Model - Scope"] = plot_df["Detector"] + " (" + plot_df["Scope"] + ")"

    x = np.arange(len(plot_df))
    width = 0.25

    ax.bar(x - width, plot_df["Precision"], width, label="Precision", color="#3B82F6")
    ax.bar(x, plot_df["Recall"], width, label="Recall", color="#10B981")
    ax.bar(x + width, plot_df["F1-Score"], width, label="F1-Score", color="#8B5CF6")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Performance Comparison across Detection Scopes")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model - Scope"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    # 2. Clean Data False Alarm Test
    st.markdown("#### 3. False Alarm Analysis on 100% Clean Data")
    st.caption("Verifies that detectors are measuring genuine anomaly signals and not merely flagging clean records.")
    clean_kpis = benchmark_eval["clean_test"]

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(
            "Validation Rules False Alarm Rate",
            f"{clean_kpis['rules_false_alarm_rate']*100:.2f}%",
            f"{clean_kpis['rules_false_alarms']} / {clean_kpis['clean_samples']} clean rows",
        )
    with kpi2:
        st.metric(
            "IQR Baseline False Alarm Rate",
            f"{clean_kpis['iqr_false_alarm_rate']*100:.2f}%",
            f"{clean_kpis['iqr_false_alarms']} / {clean_kpis['clean_samples']} clean rows",
        )
    with kpi3:
        st.metric(
            "Isolation Forest False Alarm Rate",
            f"{clean_kpis['iso_forest_false_alarm_rate']*100:.2f}%",
            f"{clean_kpis['iso_forest_false_alarms']} / {clean_kpis['clean_samples']} clean rows",
        )

    st.markdown("---")
    # 3. Contamination Sensitivity
    st.markdown("#### 4. Isolation Forest Contamination Sensitivity Curve")
    st.caption("How varying the contamination parameter balances Precision vs Recall on multivariate and numerical anomalies.")

    sensitivity_data = EvaluationEngine.run_contamination_sensitivity(
        df_corrupted=fixture_df,
        injected_errors=injected_objects,
        contaminations=[0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
        random_state=random_seed,
    )
    sens_df = pd.DataFrame(sensitivity_data)

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(sens_df["contamination"], sens_df["precision_all"], "o-", label="Precision (All Numerical)", color="#3B82F6")
    ax2.plot(sens_df["contamination"], sens_df["recall_all"], "s-", label="Recall (All Numerical)", color="#10B981")
    ax2.plot(sens_df["contamination"], sens_df["f1_all"], "^-", label="F1-Score (All Numerical)", color="#8B5CF6")
    ax2.plot(sens_df["contamination"], sens_df["recall_multi"], "d--", label="Recall (Multivariate Only)", color="#F59E0B")

    ax2.set_xlabel("Contamination Parameter")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Isolation Forest Sensitivity Trade-off")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)
    st.pyplot(fig2)
    plt.close(fig2)


# -------------------------------------------------------------
# TAB 5: EXPORT & QUALITY REPORT
# -------------------------------------------------------------
with tab_export:
    st.markdown("### Export Cleaned Data & Executive Audit Report")

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        st.markdown("#### 1. Download Cleaned CSV")
        st.write(f"- Current Shape: **{current_df.shape[0]} rows × {current_df.shape[1]} columns**")
        st.write(f"- Transformations applied: **{len(cleaner.audit_trail)}**")

        csv_buffer = io.StringIO()
        current_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            type="primary",
        )

    with e_col2:
        st.markdown("#### 2. Download Audit & Quality Report")
        st.write("Generates a comprehensive Markdown audit log documenting all actions, params, and diffs.")

        # Build full report
        audit_md = cleaner.generate_audit_report()
        st.download_button(
            label="📄 Download Transformation Audit Report (MD)",
            data=audit_md,
            file_name="data_quality_audit_report.md",
            mime="text/markdown",
        )

    with st.expander("Preview Transformation Audit Report"):
        st.markdown(cleaner.generate_audit_report())
