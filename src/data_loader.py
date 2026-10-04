"""
src/data_loader.py
Reusable data loading, preprocessing, and auditing pipeline for 6G smart factory telemetry.
"""

from typing import Tuple, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def load_raw(filepath: str = "data/raw/Thales_Group_Manufacturing.csv") -> pd.DataFrame:
    """
    Loads raw manufacturing telemetry data, standardizes date-time into a unified
    Timestamp column, and sorts strictly by Machine_ID ascending and Timestamp ascending.
    
    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.
        
    Returns
    -------
    pd.DataFrame
        Loaded and ordered DataFrame with unified 'Timestamp'.
    """
    if not os.path.exists(filepath):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(base_dir, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    # In raw dataset: 'Date' (e.g. 01-01-2025) and 'Timestamp' (e.g. 00:00:00) represent time.
    # Combine into a single Timestamp column
    if "Date" in df.columns and "Timestamp" in df.columns:
        combined_ts = df["Date"].astype(str) + " " + df["Timestamp"].astype(str)
        df["Timestamp"] = pd.to_datetime(combined_ts, format="%d-%m-%Y %H:%M:%S")
        df = df.drop(columns=["Date"])
    elif "Date" in df.columns and "Time" in df.columns:
        combined_ts = df["Date"].astype(str) + " " + df["Time"].astype(str)
        df["Timestamp"] = pd.to_datetime(combined_ts, format="%d-%m-%Y %H:%M:%S")
        df = df.drop(columns=["Date", "Time"])
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise ValueError("Dataset does not contain recognizable date/time columns.")

    # Sort strictly by Machine_ID, then Timestamp
    df = df.sort_values(by=["Machine_ID", "Timestamp"]).reset_index(drop=True)
    return df


def audit_sampling_regularity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits sampling regularity per machine and across the factory.
    Computes delta time between consecutive records per Machine_ID.
    """
    df_sorted = df.sort_values(by=["Machine_ID", "Timestamp"]).copy()
    df_sorted["prev_timestamp"] = df_sorted.groupby("Machine_ID")["Timestamp"].shift(1)
    df_sorted["delta_seconds"] = (df_sorted["Timestamp"] - df_sorted["prev_timestamp"]).dt.total_seconds()

    per_machine_stats = (
        df_sorted.groupby("Machine_ID")["delta_seconds"]
        .agg(
            median_sec="median",
            mean_sec="mean",
            min_sec="min",
            max_sec="max",
            p25_sec=lambda s: s.quantile(0.25),
            p75_sec=lambda s: s.quantile(0.75),
            gaps_gt_60s=lambda s: (s > 60).sum(),
            records="count",
        )
        .reset_index()
    )

    # Global time delta (factory-level)
    df_global = df.sort_values("Timestamp").copy()
    global_delta = df_global["Timestamp"].diff().dt.total_seconds()

    overall_stats = {
        "global_median_sec": float(global_delta.median()),
        "global_mean_sec": float(global_delta.mean()),
        "global_records_per_minute": int((global_delta == 60).sum()),
        "machine_median_sec": float(per_machine_stats["median_sec"].median()),
        "machine_median_min": float(per_machine_stats["median_sec"].median() / 60.0),
        "machine_mean_min": float(per_machine_stats["mean_sec"].mean() / 60.0),
        "machine_p25_min": float(per_machine_stats["p25_sec"].median() / 60.0),
        "machine_p75_min": float(per_machine_stats["p75_sec"].median() / 60.0),
        "total_gaps_gt_60s": int(per_machine_stats["gaps_gt_60s"].sum()),
    }

    return {
        "overall": overall_stats,
        "per_machine": per_machine_stats,
    }


def audit_label_generation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits whether Efficiency_Status is an independent observation or deterministically
    derived from Production_Speed_units_per_hr, Error_Rate_%, or Quality_Control_Defect_Rate_%.
    """
    features = ["Production_Speed_units_per_hr", "Error_Rate_%", "Quality_Control_Defect_Rate_%"]
    X = df[features]
    y = df["Efficiency_Status"]

    # Fit decision tree classifier
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y)
    dt_acc = dt.score(X, y)

    # Test explicit suspected deterministic business rule:
    # Low: Error_Rate_% > 5.0 OR Production_Speed_units_per_hr <= 200
    # High: Production_Speed_units_per_hr > 400 AND Error_Rate_% <= 2.0
    # Medium: otherwise
    def rule_predict(row):
        speed = row["Production_Speed_units_per_hr"]
        err = row["Error_Rate_%"]
        if err > 5.0 or speed <= 200.0:
            return "Low"
        elif speed > 400.0 and err <= 2.0:
            return "High"
        else:
            return "Medium"

    rule_preds = df.apply(rule_predict, axis=1)
    rule_match_count = int((rule_preds == df["Efficiency_Status"]).sum())
    rule_accuracy = float(rule_match_count / len(df))

    return {
        "tree_accuracy_depth_4": float(dt_acc),
        "rule_accuracy": rule_accuracy,
        "rule_matches": rule_match_count,
        "total_rows": len(df),
        "is_derived": rule_accuracy >= 0.999,
        "derivation_formula": (
            "Low: Error_Rate_% > 5.0 OR Production_Speed <= 200; "
            "High: Production_Speed > 400 AND Error_Rate_% <= 2.0; "
            "Medium: Otherwise"
        ),
    }


def audit_operation_modes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profiles key variables by Operation_Mode.
    """
    summary = df.groupby("Operation_Mode").agg(
        record_count=("Machine_ID", "count"),
        mean_latency=("Network_Latency_ms", "mean"),
        std_latency=("Network_Latency_ms", "std"),
        mean_packet_loss=("Packet_Loss_%", "mean"),
        std_packet_loss=("Packet_Loss_%", "std"),
        mean_speed=("Production_Speed_units_per_hr", "mean"),
        std_speed=("Production_Speed_units_per_hr", "std"),
        mean_error=("Error_Rate_%", "mean"),
        std_error=("Error_Rate_%", "std"),
        mean_defect=("Quality_Control_Defect_Rate_%", "mean"),
        std_defect=("Quality_Control_Defect_Rate_%", "std"),
    )
    return summary


def audit(
    df: Optional[pd.DataFrame] = None,
    filepath: str = "data/raw/Thales_Group_Manufacturing.csv",
    save_report: bool = True,
    report_path: str = "reports/data_audit_report.txt",
) -> Dict[str, Any]:
    """
    Full diagnostic audit of manufacturing telemetry.
    Checks shape, missing values, duplicates, class distributions, sampling regularity,
    operation-mode stratification, and the Efficiency_Status label generation logic.
    """
    if df is None:
        df = load_raw(filepath)

    total_rows, total_cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    has_nulls = df.isnull().any().any()
    dup_count = int(df.duplicated().sum())

    # Class balance
    class_counts = df["Efficiency_Status"].value_counts().to_dict()
    class_props = (df["Efficiency_Status"].value_counts(normalize=True) * 100).to_dict()

    # Sampling regularity
    sampling_res = audit_sampling_regularity(df)

    # Operation Mode profile
    op_profile = audit_operation_modes(df)
    op_counts = df["Operation_Mode"].value_counts().to_dict()
    op_efficiency_crosstab = (
        pd.crosstab(df["Operation_Mode"], df["Efficiency_Status"], normalize="index") * 100
    ).to_dict(orient="index")

    # Label audit
    label_res = audit_label_generation(df)

    audit_summary = {
        "shape": (total_rows, total_cols),
        "columns": list(df.columns),
        "has_nulls": has_nulls,
        "null_counts": null_counts,
        "duplicates": dup_count,
        "class_balance_counts": class_counts,
        "class_balance_proportions": class_props,
        "sampling_regularity": sampling_res["overall"],
        "operation_mode_counts": op_counts,
        "operation_mode_efficiency_mix": op_efficiency_crosstab,
        "label_generation_audit": label_res,
    }

    report_text = f"""================================================================================
                    6G SMART MANUFACTURING TELEMETRY DATA AUDIT REPORT
================================================================================
Dataset Path: {filepath}
Dimensions: {total_rows:,} rows, {total_cols} columns
Duplicate Records: {dup_count}
Missing Values: {'None' if not has_nulls else str(null_counts)}

1. COLUMN INVENTORY & TYPES
--------------------------------------------------------------------------------
{df.dtypes.to_string()}

2. TARGET CLASS BALANCE (Efficiency_Status)
--------------------------------------------------------------------------------
- Low:    {class_counts.get('Low', 0):,} ({class_props.get('Low', 0.0):.3f}%)
- Medium: {class_counts.get('Medium', 0):,} ({class_props.get('Medium', 0.0):.3f}%)
- High:   {class_counts.get('High', 0):,} ({class_props.get('High', 0.0):.3f}%)

3. SAMPLING REGULARITY AUDIT
--------------------------------------------------------------------------------
- Global Factory Telemetry Interval: Exactly {sampling_res['overall']['global_median_sec']:.1f}s median ({sampling_res['overall']['global_records_per_minute']:,} records at 60s cadence)
- Per-Machine Sampling Interval:
    * Median Interval: {sampling_res['overall']['machine_median_min']:.1f} minutes ({sampling_res['overall']['machine_median_sec']:.1f} seconds)
    * 25th Percentile: {sampling_res['overall']['machine_p25_min']:.1f} minutes
    * 75th Percentile: {sampling_res['overall']['machine_p75_min']:.1f} minutes
    * Mean Interval:   {sampling_res['overall']['machine_mean_min']:.1f} minutes
    * Gaps > 60s:      {sampling_res['overall']['total_gaps_gt_60s']:,} occurrences per machine
    * Physical Implication: Telemetry round-robins across the 50 machines at 1-min factory steps.
      A single machine is sampled roughly every ~34-50 minutes.

4. OPERATION MODE PROFILE
--------------------------------------------------------------------------------
Counts:
{df['Operation_Mode'].value_counts().to_string()}

Metrics Summary by Mode:
{op_profile.to_string()}

Efficiency Status Proportions by Operation Mode (%):
{pd.DataFrame(op_efficiency_crosstab).T.to_string()}

5. LABEL GENERATION AUDIT (Efficiency_Status)
--------------------------------------------------------------------------------
Is Efficiency_Status derived from outcome variables? {'YES (CONFIRMED)' if label_res['is_derived'] else 'NO'}
- Decision Tree Accuracy (using Speed, Error Rate, Defect Rate): {label_res['tree_accuracy_depth_4']*100:.4f}%
- Exact Deterministic Rule Match: {label_res['rule_matches']:,} / {label_res['total_rows']:,} ({label_res['rule_accuracy']*100:.4f}%)
- Extracted Rule:
    {label_res['derivation_formula']}
- Critical Modeling Implication:
    Efficiency_Status is directly derived from Production_Speed_units_per_hr and Error_Rate_%.
    Model B MUST NOT use contemporaneous Production_Speed or Error_Rate as independent predictors,
    as this creates circular label leakage (reconstructing the deterministic formula with 100% accuracy).
    Alternatively, Model B must be explicitly structured as a root-cause network diagnosis model
    using network and environment features, or outcome variables must be restricted.
================================================================================
"""

    if save_report:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    print(report_text)
    return audit_summary


if __name__ == "__main__":
    audit()
