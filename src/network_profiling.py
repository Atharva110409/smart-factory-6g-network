"""
src/network_profiling.py
Network performance profiling, multi-k clustering evaluation, and Network Risk Index computation.
"""

from typing import Tuple, Dict, Any, Optional, List
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, RobustScaler


def network_risk_index(
    df: pd.DataFrame,
    w_latency: float = 0.5,
    w_packet_loss: float = 0.5,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[pd.Series, StandardScaler]:
    """
    Computes composite Network Risk Index:
        Network_Risk_Index = w_latency * z(latency) + w_packet_loss * z(packet_loss)
    Higher values indicate degraded/unreliable network conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Network_Latency_ms' and 'Packet_Loss_%'.
    w_latency : float, default=0.5
        Weight assigned to normalized latency.
    w_packet_loss : float, default=0.5
        Weight assigned to normalized packet loss.
    scaler : Optional[StandardScaler], default=None
        Pre-fitted scaler. If None and fit_scaler=True, a new scaler is fitted.
    fit_scaler : bool, default=False
        Whether to fit the scaler on the provided dataframe.
        
    Returns
    -------
    Tuple[pd.Series, StandardScaler]
        Composite risk index series and the fitted StandardScaler.
    """
    features = ["Network_Latency_ms", "Packet_Loss_%"]
    X = df[features].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    z_lat = X_scaled[:, 0]
    z_pkt = X_scaled[:, 1]
    risk_index = w_latency * z_lat + w_packet_loss * z_pkt

    return pd.Series(risk_index, index=df.index, name="Network_Risk_Index"), scaler


def select_k(
    df: pd.DataFrame,
    k_range: range = range(2, 7),
    sample_size: int = 15000,
    random_state: int = 42,
    save_plot: bool = True,
    plot_path: str = "reports/clustering_evaluation_k_selection.png",
) -> Dict[str, Any]:
    """
    Evaluates K-Means clustering across candidate k values (k=2..6) using Inertia,
    Silhouette Score, and Davies-Bouldin Index. Does NOT assume k=3.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Network_Latency_ms' and 'Packet_Loss_%'.
    k_range : range, default=range(2, 7)
        Candidate cluster counts to evaluate.
    sample_size : int, default=15000
        Subsample size for efficient silhouette score computation.
    random_state : int, default=42
        Random seed for reproducibility.
    save_plot : bool, default=True
        Whether to save an elbow/silhouette evaluation plot.
    plot_path : str
        Output file path for evaluation figure.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics per k, optimal k selection, and comparison table.
    """
    features = ["Network_Latency_ms", "Packet_Loss_%"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    np.random.seed(random_state)
    sample_indices = (
        np.random.choice(len(df), size=sample_size, replace=False)
        if len(df) > sample_size
        else np.arange(len(df))
    )
    X_sample = X[sample_indices]

    results = []
    models = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_sample, labels[sample_indices]))
        db = float(davies_bouldin_score(X, labels))

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil,
            "davies_bouldin_index": db,
        })
        models[k] = km

    metrics_df = pd.DataFrame(results)

    # Optimal k selection: highest silhouette score, lowest Davies-Bouldin
    best_k_sil = int(metrics_df.loc[metrics_df["silhouette_score"].idxmax()]["k"])
    best_k_db = int(metrics_df.loc[metrics_df["davies_bouldin_index"].idxmin()]["k"])

    # Determine recommended k from evidence
    recommended_k = best_k_sil

    if save_plot:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

        ax1.plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#1f77b4", lw=2)
        ax1.set_title("Elbow Method (Inertia vs k)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Sum of Squared Distances)")
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot(metrics_df["k"], metrics_df["silhouette_score"], marker="s", color="#2ca02c", lw=2)
        ax2.axvline(best_k_sil, color="red", linestyle=":", label=f"Max Silhouette (k={best_k_sil})")
        ax2.set_title("Silhouette Score vs k (Higher is Better)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        ax3.plot(metrics_df["k"], metrics_df["davies_bouldin_index"], marker="^", color="#d62728", lw=2)
        ax3.axvline(best_k_db, color="darkred", linestyle=":", label=f"Min Davies-Bouldin (k={best_k_db})")
        ax3.set_title("Davies-Bouldin Index vs k (Lower is Better)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Number of Clusters (k)")
        ax3.set_ylabel("Davies-Bouldin Index")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    return {
        "metrics_table": metrics_df,
        "recommended_k": recommended_k,
        "best_silhouette_k": best_k_sil,
        "best_davies_bouldin_k": best_k_db,
        "scaler": scaler,
        "fitted_models": models,
        "plot_path": plot_path,
    }


def cluster_network_quality(
    df: pd.DataFrame,
    k: int = 4,
    scaler: Optional[StandardScaler] = None,
    fit_model: bool = True,
    kmeans_model: Optional[KMeans] = None,
    random_state: int = 42,
    save_plot: bool = True,
    plot_path: str = "reports/network_clusters_scatter.png",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits K-Means clustering to standardize Network_Latency_ms and Packet_Loss_%,
    ranks clusters by mean composite Network Risk Index, and maps them to human-interpretable,
    monotonically ordered risk tiers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Network_Latency_ms' and 'Packet_Loss_%'.
    k : int, default=4
        Number of clusters (selected empirically from select_k).
    scaler : Optional[StandardScaler]
        Pre-fitted scaler. If None, fitted if fit_model=True.
    fit_model : bool, default=True
        Whether to fit scaler and KMeans.
    kmeans_model : Optional[KMeans]
        Pre-fitted KMeans model.
    random_state : int, default=42
        Random seed.
    save_plot : bool, default=True
        Whether to generate and save scatter visualization.
    plot_path : str
        Output path for cluster scatter plot.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Enriched DataFrame with cluster assignments and metadata dictionary.
    """
    features = ["Network_Latency_ms", "Packet_Loss_%"]
    df_out = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_out[features])
    elif fit_model:
        X = scaler.fit_transform(df_out[features])
    else:
        X = scaler.transform(df_out[features])

    if kmeans_model is None or fit_model:
        kmeans_model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        raw_labels = kmeans_model.fit_predict(X)
    else:
        raw_labels = kmeans_model.predict(X)

    # Compute risk index
    df_out["Network_Risk_Index"], _ = network_risk_index(df_out, scaler=scaler)
    df_out["raw_cluster"] = raw_labels

    # Compute mean composite risk per raw cluster to rank them monotonically
    cluster_risk = df_out.groupby("raw_cluster")["Network_Risk_Index"].mean()
    sorted_raw_clusters = cluster_risk.sort_values().index.tolist()

    # Map raw arbitrary cluster IDs to ordered ranks 0..(k-1)
    rank_mapping = {raw_id: rank for rank, raw_id in enumerate(sorted_raw_clusters)}
    df_out["Network_Quality_Tier"] = df_out["raw_cluster"].map(rank_mapping)

    # Tier naming convention based on ordered composite risk
    tier_names = {
        0: "Tier 0 (Optimal / Low Risk)",
        1: "Tier 1 (Moderate Risk - Quadrant A)",
        2: "Tier 2 (Moderate Risk - Quadrant B)",
        3: "Tier 3 (Critical / High Risk)",
    }
    if k == 3:
        tier_names = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk",
        }
    df_out["Network_Quality_Label"] = df_out["Network_Quality_Tier"].map(
        lambda r: tier_names.get(r, f"Tier {r}")
    )

    # Validate cluster profile
    cluster_profile = df_out.groupby("Network_Quality_Tier").agg(
        count=("Machine_ID", "count") if "Machine_ID" in df_out.columns else ("Network_Latency_ms", "count"),
        mean_latency=("Network_Latency_ms", "mean"),
        std_latency=("Network_Latency_ms", "std"),
        mean_packet_loss=("Packet_Loss_%", "mean"),
        std_packet_loss=("Packet_Loss_%", "std"),
        mean_risk_index=("Network_Risk_Index", "mean"),
    ).reset_index()

    # Cross-tabulations if metadata columns present
    crosstab_mode = None
    crosstab_machine = None
    if "Operation_Mode" in df_out.columns:
        crosstab_mode = pd.crosstab(
            df_out["Network_Quality_Label"], df_out["Operation_Mode"], normalize="columns"
        ) * 100
    if "Machine_ID" in df_out.columns:
        crosstab_machine = pd.crosstab(
            df_out["Network_Quality_Label"], df_out["Machine_ID"], normalize="columns"
        ) * 100

    if save_plot:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"][:k]
        for tier in range(k):
            subset = df_out[df_out["Network_Quality_Tier"] == tier]
            lbl = cluster_profile.loc[cluster_profile["Network_Quality_Tier"] == tier]
            label_str = f"Tier {tier}: Lat={lbl['mean_latency'].values[0]:.1f}ms, Loss={lbl['mean_packet_loss'].values[0]:.2f}%"
            # Subsample for clear rendering
            sub_draw = subset.sample(min(len(subset), 2500), random_state=random_state)
            ax.scatter(
                sub_draw["Network_Latency_ms"],
                sub_draw["Packet_Loss_%"],
                s=12,
                alpha=0.4,
                color=palette[tier],
                label=label_str,
            )

        ax.set_title(f"6G Network Performance Clusters (k={k}, Ordered by Composite Risk)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Network Latency (ms)", fontsize=11)
        ax.set_ylabel("Packet Loss (%)", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    metadata = {
        "k": k,
        "scaler": scaler,
        "kmeans_model": kmeans_model,
        "cluster_profile": cluster_profile,
        "crosstab_mode": crosstab_mode,
        "crosstab_machine": crosstab_machine,
        "rank_mapping": rank_mapping,
        "plot_path": plot_path,
    }

    return df_out, metadata


if __name__ == "__main__":
    from src.data_loader import load_raw
    print("Executing Step 2: Network Profiling & Clustering...")
    df = load_raw()
    k_eval = select_k(df)
    print("=== Multi-k Evaluation Table ===")
    print(k_eval["metrics_table"])
    print(f"Recommended k: {k_eval['recommended_k']}")

    df_clustered, meta = cluster_network_quality(df, k=k_eval["recommended_k"])
    print("\n=== Ordered Cluster Profile ===")
    print(meta["cluster_profile"])
