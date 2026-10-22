"""
src/audit_uniformity.py
Reproducibility script computing:
1. All 9 Kolmogorov-Smirnov (KS) tests against uniform distributions.
2. Full 36-pair Pearson correlation matrix with Bonferroni correction.
3. Vectorized supremum-F permutation tests (seed=42, 2,000 permutations)
   across Pooled, Active, Idle, and Maintenance operational slices.

Outputs:
- reports/ks_uniformity_tests.csv
- reports/correlation_matrix.csv
- reports/breakpoint_permutation_tests.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

def run_ks_tests(df: pd.DataFrame, save_path: str = "reports/ks_uniformity_tests.csv") -> pd.DataFrame:
    num_cols = [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Production_Speed_units_per_hr",
        "Error_Rate_%",
        "Quality_Control_Defect_Rate_%",
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Predictive_Maintenance_Score",
    ]
    records = []
    n_vars = len(num_cols)
    for col in num_cols:
        a = float(df[col].min())
        b = float(df[col].max())
        scale = b - a
        res = stats.kstest(df[col].values, "uniform", args=(a, scale))
        raw_p = float(res.pvalue)
        bonf_p = min(1.0, raw_p * n_vars)
        records.append({
            "Variable": col,
            "Min_Bound": a,
            "Max_Bound": b,
            "KS_Statistic_D": float(res.statistic),
            "Raw_P_Value": raw_p,
            "Bonferroni_P_Value": bonf_p,
            "Consistent_With_Uniform": bonf_p >= 0.05
        })
    ks_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ks_df.to_csv(save_path, index=False)
    print(f"Saved KS tests to {save_path}")
    return ks_df


def run_correlation_matrix(df: pd.DataFrame, save_path: str = "reports/correlation_matrix.csv") -> pd.DataFrame:
    num_cols = [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Production_Speed_units_per_hr",
        "Error_Rate_%",
        "Quality_Control_Defect_Rate_%",
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Predictive_Maintenance_Score",
    ]
    n_pairs = (len(num_cols) * (len(num_cols) - 1)) // 2
    records = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            res = stats.pearsonr(df[c1].values, df[c2].values)
            r = float(res.statistic)
            raw_p = float(res.pvalue)
            bonf_p = min(1.0, raw_p * n_pairs)
            records.append({
                "Feature_1": c1,
                "Feature_2": c2,
                "Pearson_r": r,
                "Abs_Pearson_r": abs(r),
                "Raw_P_Value": raw_p,
                "Bonferroni_P_Value": bonf_p,
                "Survives_Bonferroni": bonf_p < 0.05
            })
    corr_df = pd.DataFrame(records).sort_values(by="Abs_Pearson_r", ascending=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr_df.to_csv(save_path, index=False)
    print(f"Saved correlation analysis to {save_path}")
    return corr_df


def compute_sup_f_permutation(
    x: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42
) -> dict:
    n = len(x)
    # Design matrix under null: [1, x]
    X0 = np.column_stack([np.ones(n), x])
    # Projection matrix M0 = I - X0(X0^T X0)^(-1) X0^T
    Q, _ = np.linalg.qr(X0)
    # M0 @ v = v - Q @ (Q^T @ v)
    def project_out(v):
        return v - Q @ (Q.T @ v)

    # Precompute projected segmented regressors
    Z_tilde = np.zeros((n, len(candidates)))
    denom = np.zeros(len(candidates))
    for idx, tau in enumerate(candidates):
        hinge = np.maximum(0.0, x - tau)
        z_proj = project_out(hinge)
        Z_tilde[:, idx] = z_proj
        denom[idx] = np.dot(z_proj, z_proj)

    # Observed F statistics
    e0_obs = project_out(y)
    rss_base_obs = np.dot(e0_obs, e0_obs)
    v_obs = Z_tilde.T @ y
    delta_rss_obs = (v_obs ** 2) / denom
    rss_seg_obs = np.maximum(1e-12, rss_base_obs - delta_rss_obs)
    f_obs = (n - 3) * delta_rss_obs / rss_seg_obs
    best_idx = np.argmax(f_obs)
    sup_f_obs = float(f_obs[best_idx])
    best_tau = float(candidates[best_idx])
    naive_p = float(1.0 - stats.f.cdf(sup_f_obs, 1, n - 3))

    # Permutations
    rng = np.random.RandomState(seed)
    sup_f_perms = np.zeros(n_permutations)
    for b in range(n_permutations):
        y_perm = rng.permutation(y)
        e0_b = project_out(y_perm)
        rss_base_b = np.dot(e0_b, e0_b)
        v_b = Z_tilde.T @ y_perm
        delta_rss_b = (v_b ** 2) / denom
        rss_seg_b = np.maximum(1e-12, rss_base_b - delta_rss_b)
        f_b = (n - 3) * delta_rss_b / rss_seg_b
        sup_f_perms[b] = np.max(f_b)

    perm_p = float(np.mean(sup_f_perms >= sup_f_obs))

    return {
        "candidate_tau": best_tau,
        "sup_f_obs": sup_f_obs,
        "naive_p": naive_p,
        "perm_p": perm_p
    }


def run_breakpoint_permutations(
    df: pd.DataFrame,
    save_path: str = "reports/breakpoint_permutation_tests.csv",
    n_permutations: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    groups = [
        ("Pooled", df),
        ("Active", df[df["Operation_Mode"] == "Active"]),
        ("Idle", df[df["Operation_Mode"] == "Idle"]),
        ("Maintenance", df[df["Operation_Mode"] == "Maintenance"]),
    ]
    records = []
    for name, sub in groups:
        x = sub["Network_Latency_ms"].values
        y = sub["Production_Speed_units_per_hr"].values
        # 81 quantiles from 10% to 90%
        candidates = np.percentile(x, np.linspace(10, 90, 81))
        res = compute_sup_f_permutation(x, y, candidates, n_permutations=n_permutations, seed=seed)
        records.append({
            "Operational_Slice": name,
            "Sample_Size": len(sub),
            "Candidate_Breakpoint_ms": res["candidate_tau"],
            "Sup_F_Statistic": res["sup_f_obs"],
            "Naive_F_P_Value": res["naive_p"],
            "Permutation_P_Value": res["perm_p"]
        })
    bp_df = pd.DataFrame(records)
    # Benjamini-Hochberg FDR
    m = len(bp_df)
    sorted_indices = np.argsort(bp_df["Permutation_P_Value"].values)
    raw_p = bp_df["Permutation_P_Value"].values[sorted_indices]
    fdr_q = np.zeros(m)
    min_q = 1.0
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        q_val = (raw_p[idx] * m) / rank
        min_q = min(min_q, q_val)
        fdr_q[idx] = min_q
    unsort = np.argsort(sorted_indices)
    bp_df["FDR_Q_Value"] = np.minimum(1.0, fdr_q[unsort])
    bp_df["Survives_FDR_0_05"] = bp_df["FDR_Q_Value"] < 0.05

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bp_df.to_csv(save_path, index=False)
    print(f"Saved breakpoint permutation tests to {save_path}")
    return bp_df


def main():
    data_path = "data/raw/Thales_Group_Manufacturing.csv"
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at {data_path}")
        sys.exit(1)
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)

    print("\n--- 1. Computing 9 KS Goodness-of-Fit Tests ---")
    ks_df = run_ks_tests(df)
    print(ks_df.to_string(index=False))

    print("\n--- 2. Computing 36-Pair Pearson Correlation Matrix ---")
    corr_df = run_correlation_matrix(df)
    print("Top 5 correlations:")
    print(corr_df.head(5).to_string(index=False))

    print("\n--- 3. Computing Vectorized Sup-F Permutation Tests (2,000 permutations, seed=42) ---")
    bp_df = run_breakpoint_permutations(df, n_permutations=2000, seed=42)
    print(bp_df.to_string(index=False))
    print("\nAll uniformity and breakpoint audit analyses completed successfully.")


if __name__ == "__main__":
    main()
