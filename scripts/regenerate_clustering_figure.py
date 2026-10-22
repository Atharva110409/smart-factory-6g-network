import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_clustering_k_plot():
    csv_path = "reports/clustering_k_selection.csv"
    df = pd.read_csv(csv_path)
    
    # Set high quality IEEE style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)
    
    # 1. Inertia (Elbow)
    ax1.plot(df["k"], df["inertia"], marker="o", markersize=6, color="#1f77b4", lw=2, label="SSE (Inertia)")
    ax1.plot([4], [df.loc[df["k"]==4, "inertia"].values[0]], marker="o", markersize=9, color="#d62728", fillstyle="full")
    ax1.annotate(f"k=4: {df.loc[df['k']==4, 'inertia'].values[0]:,.1f}", 
                 (4, df.loc[df["k"]==4, "inertia"].values[0]),
                 textcoords="offset points", xytext=(10, 10), ha="left",
                 fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    ax1.set_title("Inertia (SSE) vs. Cluster Count $k$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Candidate Cluster Count ($k$)", fontsize=10)
    ax1.set_ylabel("Inertia (Sum of Squared Errors)", fontsize=10)
    ax1.set_xticks(df["k"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 2. Silhouette Score (Sample n=15,000)
    ax2.plot(df["k"], df["silhouette_score"], marker="s", markersize=6, color="#2ca02c", lw=2, label="Silhouette Score")
    ax2.plot([4], [df.loc[df["k"]==4, "silhouette_score"].values[0]], marker="s", markersize=9, color="#d62728", fillstyle="full")
    ax2.annotate(f"Max: {df.loc[df['k']==4, 'silhouette_score'].values[0]:.4f}", 
                 (4, df.loc[df["k"]==4, "silhouette_score"].values[0]),
                 textcoords="offset points", xytext=(-15, 12), ha="center",
                 fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    ax2.set_title("Silhouette Score vs. $k$ (Higher is Better)\n[Evaluated on Subsample $n=15,000$]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Candidate Cluster Count ($k$)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.set_xticks(df["k"])
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # 3. Davies-Bouldin Index (Lower is Better)
    ax3.plot(df["k"], df["davies_bouldin_index"], marker="^", markersize=6, color="#ff7f0e", lw=2, label="Davies-Bouldin")
    ax3.plot([4], [df.loc[df["k"]==4, "davies_bouldin_index"].values[0]], marker="^", markersize=9, color="#d62728", fillstyle="full")
    ax3.annotate(f"Min: {df.loc[df['k']==4, 'davies_bouldin_index'].values[0]:.4f}", 
                 (4, df.loc[df["k"]==4, "davies_bouldin_index"].values[0]),
                 textcoords="offset points", xytext=(10, 10), ha="left",
                 fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    ax3.set_title("Davies-Bouldin Index vs. $k$ (Lower is Better)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Candidate Cluster Count ($k$)", fontsize=10)
    ax3.set_ylabel("Davies-Bouldin Index", fontsize=10)
    ax3.set_xticks(df["k"])
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save to both paper/figures and reports
    out_paths = [
        "paper/figures/clustering_evaluation_k_selection.png",
        "reports/clustering_evaluation_k_selection.png"
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
        
    plt.close(fig)

if __name__ == "__main__":
    generate_clustering_k_plot()
