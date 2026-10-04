# 6G Industrial Network Telemetry & Manufacturing Efficiency Platform

An end-to-end causal inference, change-point breakpoint detection, leakage-safe machine learning, and interactive operational intelligence platform for industrial private wireless networks in cyber-physical manufacturing systems.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg)](http://localhost:8501)
[![Standards: 3GPP TS 22.104](https://img.shields.io/badge/3GPP-TS%2022.104%20Rel--18-informational.svg)](docs/latency_benchmarks.md)

---

## 1. Executive Summary & Core Headline Finding

Across 100,000 continuous operating records from 50 industrial CNC and assembly machines over 68.4 days:

> **Headline Finding:** Within the observed operating envelope ($13.4\text{ to }37.9\text{ ms}$ latency, $0.0\%\text{ to }5.0\%$ packet loss), **network conditions are not currently a major driver of efficiency degradation or defect generation in this fleet.**

This conclusion is reinforced by five converging empirical proofs:
1. **Zero Forecasting Precedence (Granger Null):** Out of 450 machine- and mode-stratified Granger causality tests across multiple lags, 23 nominally rejected $H_0$ ($5.1\%$, matching random noise), and **zero (0) survived Benjamini-Hochberg FDR control ($q < 0.05$)**. Historical network telemetry does not forecast future manufacturing throughput beyond throughput's own autoregressive history.
2. **Narrow, Mode-Specific Nonlinear Breakpoint:** Piecewise segmented regression proves a statistically significant throughput threshold at **$20.7\text{ ms}$ exclusively in `Active` production mode** ($F = 4.31, p = 0.038$, slope change $\Delta \beta = -0.3063\text{ units/hr per ms}$). Pooled fleet regression yields an unstable nominal threshold of $41.2\text{ ms}$ with an excessively wide 95% bootstrap CI of $[6.0, 45.2]\text{ ms}$.
3. **Flat Defect Response:** Quality control defect rates remain static at $\sim 5.01\%$ regardless of packet loss ($1.0012$ impact ratio during 95th-percentile spikes $>4.755\%$). Industrial controllers tolerate up to $5\%$ packet loss without quality cliffs.
4. **Modest Macroeconomic Divergence:** The empirical throughput divergence between optimal connectivity (Tier 0: $13.4\text{ ms}, 1.24\%$ loss) and severe degradation (Tier 3: $37.7\text{ ms}, 3.75\%$ loss) is restricted to **$3.084\text{ units/hr per machine}$** ($1.11\%$ of baseline output), translating to $\$3,854.96/\text{hr}$ across the 50-machine fleet under base contribution margin ($\$25.00/\text{unit}$).
5. **Machine Learning Baseline Invariance:** Exhaustive baseline ladder testing on physical lookahead horizons ($H = 30\text{ min} \pm 10\text{ min}$) demonstrates that **no ML model meaningfully outperforms the majority-class baseline** (Dummy Majority: $78.42\%$ accuracy). Random Forest was chosen for production solely to suppress false alarms (Precision: $78.54\%$, Recall: $81.81\%$ on Low class), but identified only 4 of 131 High-efficiency events ($3.05\%$ recall). LightGBM captures more High events only by incurring a disastrous $97.05\%$ false alarm rate (Precision: $2.95\%$).

**Managerial Takeaway:** The private wireless network is **operating safely within the physical process tolerance envelope**. Capital expenditure proposals for multi-million-dollar sub-5ms 6G radio over-engineering should be frozen; diagnostic capital should instead be channeled into spindle vibration, motor torque, and cutting-tool wear sensors.

---

## 2. The Analytical Storyline Chain

This repository implements the complete end-to-end analytical storyline chain specified in `PROJECT_SPEC.md`:

$$\mathbf{Network\;Degradation} \longrightarrow \mathbf{Temporal\;Evidence} \longrightarrow \mathbf{Breakpoint} \longrightarrow \mathbf{Early\;Warning} \longrightarrow \mathbf{Explainable\;Diagnosis} \longrightarrow \mathbf{Operational\;Risk}$$

```
+----------------------------------------------------------------------------------------------------+
|                                    ANALYTICAL STORYLINE CHAIN                                      |
+----------------------------------------------------------------------------------------------------+
|  [Step 1 & 2] NETWORK PROFILING                                                                    |
|  - Empirical k-selection (k=4) via Silhouette (0.4061) and Davies-Bouldin (0.7687)                 |
|  - Continuous Network Risk Index: 0.5 * z(lat) + 0.5 * z(loss)                                     |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 3 & 4] TEMPORAL EVIDENCE & DIAGNOSTICS                                                      |
|  - Granger causality: 0 / 450 tests survive Benjamini-Hochberg FDR (q < 0.05)                      |
|  - Reconciled Headline: Zero predictive precedence, but active-mode breakpoint at 20.7 ms          |
|  - Diagnostics: Linear |r| < 0.015; packet loss shows no quality cliff (ratio = 1.0012)            |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 5] LEAKAGE-SAFE PREDICTIVE MODELING                                                         |
|  - Label Audit: Efficiency_Status = deterministic function of Speed and Error Rate                 |
|  - Circularity Rule: Speed and Error Rate strictly excluded from Model B                           |
|  - Model A (Early Warning, H=30min) vs Model B (Diagnosis, Current)                                |
|  - Baseline Reality: No model substantially beats Dummy Majority (78.42% Acc)                      |
|  - Model Selection: Random Forest chosen to avoid false alarms (only 4/99 High-class detections)   |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 6 & 7] OPERATIONAL KPIS & ECONOMIC SENSITIVITY                                              |
|  - Latency Sensitivity Score: Active inflection Delta beta = -0.3063 units/hr per ms               |
|  - Cost of Instability: Tier 0 vs Tier 3 loss = 3.084 units/hr ($3,855/fleet-hr base)              |
|  - Synthesis: Network operates within physical process tolerance; refocus capex on tool wear       |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Repository Architecture

```
6g network/
├── app/
│   └── dashboard.py                   # Production Streamlit 5-module executive dashboard
├── data/
│   ├── raw/
│   │   └── Thales_Group_Manufacturing.csv  # 100,000 raw telemetry rows (untouched)
│   └── processed/
│       └── features_clean.parquet     # Cached feature-engineered dataset
├── docs/
│   ├── assumptions.md                 # Formal assumptions, audit rules, and parameter logs
│   └── latency_benchmarks.md          # 3GPP TS 22.104 Rel-18 & ITU-R IMT-2030 specifications
├── models/
│   ├── early_warning_model_a.joblib   # Production Random Forest early warning artifact (H=30m)
│   ├── diagnosis_model_b.joblib       # Production Random Forest root-cause diagnostic artifact
│   ├── early_warning_model_a_lightgbm.joblib # Alternative high-recall/high-false-alarm model
│   ├── diagnosis_model_b_lightgbm.joblib     # Alternative LightGBM diagnostic model
│   ├── network_kmeans.joblib          # Trained 4-cluster K-Means model (fit on train only)
│   └── network_scaler.joblib          # Standardized z-score scaler (fit on train only)
├── notebooks/                         # Exploratory data analysis notebooks
├── reports/
│   ├── research_paper.md              # Full scientific research manuscript
│   ├── executive_summary.md           # 1-page financial briefing for executive leadership
│   ├── model_baseline_ladder.csv      # Full metrics for Dummy, Logistic, RF, LightGBM
│   ├── model_confusion_matrices.md    # Confusion matrices for all models across horizons
│   ├── kpi_report_summary.txt         # All five computed KPIs with economic sensitivities
│   ├── granger_causality_results.csv  # 450 per-machine Granger causality tests with BH-FDR
│   ├── breakpoint_stability_by_group.csv # Segmented breakpoint results and bootstrap CIs
│   ├── clustering_evaluation_k_selection.png # k-selection silhouette & inertia curves
│   ├── network_clusters_scatter.png   # 4-tier scatter with decision boundaries
│   ├── latency_breakpoint_segmented_fit.png  # Segmented regression fit vs raw scatter
│   ├── shap_summary_model_a.png       # Model A SHAP feature importance plot
│   └── shap_summary_model_b.png       # Model B SHAP feature importance plot
├── src/
│   ├── data_loader.py                 # Telemetry ingestion, timestamp parsing & audit suite
│   ├── network_profiling.py           # Unsupervised K-Means clustering & Risk Indexing
│   ├── causal_analysis.py             # Granger VAR tests, BH-FDR & segmented breakpoint fitting
│   ├── diagnostics.py                 # Bivariate correlations, mode profiling & Simpson checks
│   ├── modeling.py                    # Leakage-safe time-aware training & SHAP explainers
│   └── kpi.py                         # Five operational KPIs & economic loss computation
├── PROJECT_SPEC.md                    # Formal engineering & scientific specification
├── README.md                          # Repository documentation (this file)
└── requirements.txt                   # Frozen python dependencies
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11.7 on Windows x64)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/smart-factory-6g-network.git
cd smart-factory-6g-network

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. End-to-End Pipeline Execution

All reusable engineering logic resides in `src/`. Each module can be executed independently or sequentially:

### Step 1: Data Ingestion & Audit
```bash
python -c "from src.data_loader import load_raw, audit; df = load_raw(); report = audit(df); print(report)"
```
*Validates 100,000 rows, proves zero missing/duplicate data, quantifies the Geometric polling distribution ($p=0.02$, median 34.0 min), and uncovers the deterministic label formula.*

### Step 2: Network Profiling & Clustering
```bash
python -c "from src.data_loader import load_raw; from src.network_profiling import select_k, cluster_network_quality; df = load_raw(); select_k(df); df_c, km, sc = cluster_network_quality(df); print('Clusters generated successfully!')"
```
*Evaluates $k \in [2, 6]$, confirms $k=4$ optimality (Silhouette: 0.4061, Davies-Bouldin: 0.7687), orders tiers by Network Risk Index, and exports cluster scatter plots.*

### Step 3: Temporal Causal & Breakpoint Analysis
```bash
python -c "from src.data_loader import load_raw; from src.causal_analysis import run_full_causal_analysis; df = load_raw(); run_full_causal_analysis(df)"
```
*Executes 450 Granger causality tests with Benjamini-Hochberg FDR control ($0/450$ survive at $q<0.05$) and fits continuous segmented piecewise regression ($20.7\text{ ms}$ breakpoint in Active mode).*

### Step 4: Latency & Packet-Loss Diagnostics
```bash
python -c "from src.data_loader import load_raw; from src.diagnostics import run_diagnostics; df = load_raw(); run_diagnostics(df)"
```
*Tests for Simpson's paradox, evaluates mode-stratified correlations ($|r| < 0.015$), and verifies packet-loss defect invariance.*

### Step 5: Predictive Early Warning & Diagnosis Modeling
```bash
python -c "from src.data_loader import load_raw; from src.network_profiling import cluster_network_quality; from src.modeling import train_baselines, train_early_warning_model, train_diagnosis_model; df = load_raw(); df_c, km, sc = cluster_network_quality(df); train_baselines(df_c); train_early_warning_model(df_c); train_diagnosis_model(df_c)"
```
*Constructs physical lookahead targets ($H = 30\text{ min} \pm 10\text{ min}$), enforces circularity exclusions on Model B, validates across chronological and unseen machine splits, logs complete confusion matrices, and exports SHAP plots.*

### Step 6: Operational KPI Computation
```bash
python -c "from src.data_loader import load_raw; from src.network_profiling import cluster_network_quality; from src.kpi import compute_all_kpis, generate_kpi_report; df = load_raw(); df_c, km, sc = cluster_network_quality(df); kpis = compute_all_kpis(df_c); print(generate_kpi_report(kpis))"
```
*Computes Network Risk Index, Latency Sensitivity Score, Packet Loss Impact Ratio, Tolerance Thresholds, and multi-scenario Cost of Instability.*

---

## 6. Live Interactive Dashboard (Step 7)

Launch the enterprise Streamlit dashboard:

```bash
streamlit run app/dashboard.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

```
===================================================================================
                       STREAMLIT DASHBOARD ARCHITECTURE
===================================================================================
 [Filter Sidebar] : Time Horizon | Machine ID | Operation Mode | Network Tier
-----------------------------------------------------------------------------------
 [Page 1: Overview]          - Key Metrics (Avg Latency, Loss, Risk Index, Status)
                             - 6-Hour Rolling Network & Production Trends
                             - Live Fleet Status Breakdown

 [Page 2: Network vs Eff]    - Latency vs Production Speed Scatter with Breakpoints
                             - Efficiency Status Distribution across Network Tiers
                             - Mode-stratified throughput distribution

 [Page 3: Quality & Loss]    - Defect Rate vs Packet Loss Binned Analysis
                             - Error Rate Bivariate Profiling by Operating Mode
                             - 95th Percentile Spike Impact Ratios

 [Page 4: 6G Benchmarks]     - 3GPP TS 22.104 & ITU-R IMT-2030 Latency Tolerances
                             - Measurement Caveats (One-way SDU vs Transport RTT)
                             - Reliability % vs Continuous Packet Loss % Principles

 [Page 5: Scenario Simulator]- Real-Time Sliders: Latency (ms), Packet Loss (%), Mode
                             - Dynamic Feature Recomputation (Z-score & K-Means Tier)
                             - Model A Prediction & P(Low) Probability Gauge
                             - Financial Impact Calculator (Lost units/hr & fleet cost)
                             - Mandatory Disclaimer: "Model scenario, not causal intervention"
===================================================================================
```

---

## 7. Standards Alignment & Caveats (3GPP TS 22.104 / ITU)

For comprehensive technical analysis, consult [`docs/latency_benchmarks.md`](docs/latency_benchmarks.md). Key boundary conditions:

1. **One-Way SDU vs. Round-Trip Time:** 3GPP end-to-end latency is defined as ingress-to-egress service data unit (SDU) transfer time across the radio and transport plane. In contrast, factory telemetry measures application-level ping/poll round-trip time (RTT). Comparisons must be interpreted qualitatively.
2. **Packet Loss % vs. Service Reliability %:** **Do NOT equate packet-loss percentage with 3GPP service reliability.** A 3GPP reliability of $99.999\%$ requires that $99.999\%$ of packets arrive within a strict deadline $T_{\text{max}}$. Transport-layer telemetry packet drops of $1\%\text{--}3\%$ reflect socket buffers and retransmission intervals, not unrecoverable application communication failure.

---

## 8. Limitations & Methodological Disclosures

As documented in [`docs/assumptions.md`](docs/assumptions.md) and [`reports/research_paper.md`](reports/research_paper.md):
- **Granger Causality $\neq$ Physical Causation:** Granger non-causality confirms that past network logs provide no incremental forecasting power over machine throughput. It does not imply that cutting physical Ethernet/radio cables would have no impact.
- **Unmeasured Physical Confounders:** Unobserved mechanical variables—including cutting tool sharpness, spindle bearing friction, raw material hardness, and ambient factory temperature variations—represent true physical drivers of production speed.
- **Model B Performance Ceiling:** In accordance with scientific integrity, contemporaneous `Production_Speed_units_per_hr` and `Error_Rate_%` were permanently excluded from Model B because they deterministically define `Efficiency_Status` ($99.998\%$ audit match). Model B's macro-F1 ($\sim 0.33$) reflects genuine physical sensor root-cause predictability, not trivial label reconstruction.
- **Simulator Nature:** Predictions generated by the Network Scenario Simulator represent statistical conditional expectations given hypothetical inputs, not guaranteed counterfactual outcomes.

---

## 9. Key Reports & Publications

- **Full Scientific Paper:** [`reports/research_paper.md`](reports/research_paper.md) (comprehensive academic manuscript with full methodology, literature review, and citations).
- **Executive Summary:** [`reports/executive_summary.md`](reports/executive_summary.md) (one-page briefing with financial sensitivity tables and 4 concrete operational action items).
- **KPI Summary Report:** [`reports/kpi_report_summary.txt`](reports/kpi_report_summary.txt) (exact numerical parameters for all 5 KPIs).
- **Baseline Model Ladder:** [`reports/model_baseline_ladder.csv`](reports/model_baseline_ladder.csv) & [`reports/model_confusion_matrices.md`](reports/model_confusion_matrices.md).

---

## 10. License

This repository is licensed under the Apache 2.0 / MIT Dual License. See `LICENSE` for details.
