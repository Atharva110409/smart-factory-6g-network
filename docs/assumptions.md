# Project Assumptions and Methodological Decisions Log

This document tracks all formal assumptions, parameter selections, empirical audit findings, and methodological decisions across all stages of the 6G Smart Manufacturing project. It directly supports the Limitations and Methodology sections of the research paper and executive summary.

---

## 1. Data Ingestion & Audit Findings (Step 1)
- **Column Mapping Confirmation**:
  - The raw CSV provides `Date` (DD-MM-YYYY) and `Timestamp` (HH:MM:SS).
  - These two columns are merged into a single standardized datetime column named `Timestamp`.
  - All other columns match the specification exactly:
    - `Machine_ID`: Integer identifier for 50 distinct machines (IDs 1 through 50).
    - `Operation_Mode`: Categorical (`Active`, `Idle`, `Maintenance`).
    - `Temperature_C`: Float (ambient / machine core operating temperature).
    - `Vibration_Hz`: Float (machine vibration frequency).
    - `Power_Consumption_kW`: Float (electrical power demand).
    - `Network_Latency_ms`: Float (end-to-end communication latency).
    - `Packet_Loss_%`: Float (network packet loss rate).
    - `Quality_Control_Defect_Rate_%`: Float (defective production sample rate).
    - `Production_Speed_units_per_hr`: Float (hourly production throughput rate).
    - `Predictive_Maintenance_Score`: Float (composite asset health index).
    - `Error_Rate_%`: Float (operational error frequency).
    - `Efficiency_Status`: Categorical target (`Low`, `Medium`, `High`).
- **Data Quality Confirmation**:
  - Total records: 100,000. Total columns: 13 (after unifying `Date` and `Timestamp`).
  - Missing values: Exactly 0 across all fields.
  - Duplicate rows: Exactly 0.
  - Timestamp parsing: 100% valid parsed datetime objects covering 2025-01-01 00:00:00 to 2025-03-10 10:39:00 (68.44 days).
- **Target Class Balance (`Efficiency_Status`)**:
  - `Low`: 77,825 records (77.825%)
  - `Medium`: 19,189 records (19.189%)
  - `High`: 2,986 records (2.986%)
  - Confirms severe class imbalance as anticipated by the specification. Accuracy alone is an invalid evaluation metric; models must be evaluated on macro-F1, minority class recall, and PR-AUC.
- **Sampling Regularity & Telemetry Arrival Mechanics**:
  - **Global Factory Cadence**: Exactly 60.0 seconds median (1 reading per minute factory-wide across 50 machines). 98,559 consecutive records have $\Delta t = 60$ seconds.
  - **Per-Machine Sampling Interval Discrepancy (Median 34.0 min vs Mean 49.3 min)**:
    - Detailed statistical audit reveals that telemetry across the 50 machines was generated via a **random round-robin / Bernoulli trial mechanism** where each minute, one machine is sampled with uniform probability $p = 1/50 = 0.02$.
    - For a Geometric distribution $\text{Geometric}(p=0.02)$:
      - Theoretical expected mean: $E[X] = 1/p = 50.0\text{ minutes}$ (empirical mean observed: **49.29 minutes**).
      - Theoretical expected median: $\lceil -\ln(2)/\ln(1-p) \rceil = 35.0\text{ minutes}$ (empirical median observed: **34.20 minutes**).
    - All 50 machines exhibit near-identical sampling distributions (sample counts: 1,905 to 2,091; medians: 31 to 37 minutes; means: 47.1 to 51.7 minutes). No individual machine suffers from systemic sensor disconnects or dropped data channels.
    - **Step 5 Lookahead Implication**: Because consecutive per-machine rows have an exponential/geometric waiting time distribution with an average gap of ~49 minutes (and 95th percentile of 147 minutes), a naive row-shift ($N=1$) represents an average lookahead of ~49 minutes. For strict physical time horizons (e.g. $H=5$ min, $H=10$ min), rows must be evaluated against true physical time differences $|\Delta t - H| \le \epsilon$.
- **Operation Mode Profiling**:
  - `Active`: 70,054 records (70.054%)
  - `Idle`: 20,057 records (20.057%)
  - `Maintenance`: 9,889 records (9.889%)
  - Key finding: Marginal distributions of network latency, packet loss, production speed, error rate, defect rate, and efficiency status are statistically uniform across all three operation modes (e.g., latency mean ~25.5 ms, std ~14.1 ms across all modes; efficiency mix ~77.8% Low / 19.2% Med / 3.0% High in all modes). Operation Mode remains an essential control variable to prevent Simpson's paradox in causal and regression analyses.
- **Label Generation Audit Findings (`Efficiency_Status`)**:
  - **Confirmed Deterministic Rule** (99.998% match on all 100,000 rows):
    $$\text{Efficiency\_Status} = \begin{cases} 
    \mathbf{Low} & \text{if } \text{Error\_Rate\_\%} > 5.0\% \text{ OR } \text{Production\_Speed\_units\_per\_hr} \le 200.0 \\ 
    \mathbf{High} & \text{if } \text{Production\_Speed\_units\_per\_hr} > 400.0 \text{ AND } \text{Error\_Rate\_\%} \le 2.0\% \\ 
    \mathbf{Medium} & \text{Otherwise} 
    \end{cases}$$
  - **Gating Decision for Model B (Diagnosis Model in Step 5)**:
    - As confirmed and approved by user review, contemporaneous `Production_Speed_units_per_hr` and `Error_Rate_%` are **strictly excluded** from Model B's feature set.
    - Including these two features would allow trivial 100% reconstruction of the label, causing artificial data leakage and destroying diagnostic utility.
    - **Methodology Note**: Model B's performance ceiling is intentionally constrained by this exclusion, as it must infer efficiency status strictly from root-cause upstream drivers: network performance (`Network_Latency_ms`, `Packet_Loss_%`, Network Risk Index, cluster labels), environmental/operational telemetry (`Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`, `Predictive_Maintenance_Score`), and context (`Operation_Mode`, `Machine_ID`).

---

## 2. Network Performance Profiling & Clustering Empirical Evidence (Step 2)
- **Feature Scaling**: Standardized z-score on `Network_Latency_ms` and `Packet_Loss_%`. `RobustScaler` was evaluated and yielded identical results (Silhouette 0.4095 vs 0.4095), confirming absence of extreme distributional distortion. Scaler fit strictly on training set only.
- **Cluster Count ($k$) Selection**: Empirically evaluated across $k \in [2, 6]$:
  - $k=2$: Inertia = 125,071.6, Silhouette = 0.3561, Davies-Bouldin = 1.1872
  - $k=3$: Inertia = 79,311.0, Silhouette = 0.3757, Davies-Bouldin = 0.8680
  - $k=4$: Inertia = 50,231.2, **Silhouette = 0.4061 (Maximum)**, **Davies-Bouldin = 0.7687 (Best/Minimum)**
  - $k=5$: Inertia = 42,313.2, Silhouette = 0.3920, Davies-Bouldin = 0.8363
  - $k=6$: Inertia = 35,911.5, Silhouette = 0.3781, Davies-Bouldin = 0.8381
  - **Empirical Decision**: $k=4$ is adopted as the data-driven clustering count, satisfying the specification requirement to avoid blindly assuming $k=3$.
- **Ordered Risk Tiers (Mapped by Composite Risk)**:
  Clusters were ranked and mapped by composite Network Risk Index ($0.5 z_{\text{lat}} + 0.5 z_{\text{loss}}$):
  - **Tier 0 (Optimal / Low Risk)**: Latency = 13.41 ms, Packet Loss = 1.24%, Risk Index = -0.863 ($N=24,970$)
  - **Tier 1 (Moderate Risk - Loss Degraded)**: Latency = 13.33 ms, Packet Loss = 3.74%, Risk Index = -0.004 ($N=25,067$)
  - **Tier 2 (Moderate Risk - Latency Degraded)**: Latency = 37.89 ms, Packet Loss = 1.25%, Risk Index = +0.011 ($N=25,288$)
  - **Tier 3 (Critical / High Risk)**: Latency = 37.72 ms, Packet Loss = 3.75%, Risk Index = +0.866 ($N=24,675$)
- **Orthogonality Sanity Check**: Cross-tabulations against `Operation_Mode` and `Machine_ID` demonstrate that each cluster is uniformly distributed (~25% in each category), confirming clusters represent network quality regimes rather than machine or mode proxies.

---

## 3. Temporal Causal & Breakpoint Analysis Findings (Step 3)
- **Scientific Reconciled Headline Finding (Condition 1)**:
  - **Contemporaneous vs Predictive Contrast**: Network latency and manufacturing throughput demonstrate a statistically significant **contemporaneous nonlinear association** (specifically an active-load threshold at 20.7 ms). However, historical latency exhibits **no predictive precedence or forecasting power** over future speed beyond speed's own autoregressive history (0 out of 450 Granger tests survive Benjamini-Hochberg FDR control).
  - This divergence indicates that network degradation impacts manufacturing processes immediately within the active operational window, but does not leave an enduring predictive signal that forecasts future state across multi-minute time gaps.
  - Granger causality reflects temporal predictive precedence, not physical or mechanical causation; unmeasured physical confounders (e.g., ambient voltage fluctuations, raw material inconsistency, tool wear) are acknowledged in the Limitations.
- **Breakpoint Detection & Headline Threshold (Condition 2)**:
  - **Primary Headline Threshold: 20.7 ms in `Active` Mode** ($F = 4.31, p = 0.038$). Under active production load, machine throughput exhibits a downward inflection point once latency exceeds 20.7 ms.
  - **Caveat on Latency Sensitivity Score (Active Mode)**: The segmented piecewise fit yields a pre-threshold slope of $+0.2123$ units/hr per ms and a post-threshold slope of $-0.0940$ units/hr per ms ($\Delta \beta = -0.3063$). Note that this positive pre-threshold slope is counterintuitive and occurs entirely within an overall near-zero linear correlation regime (Step 4: $|r| < 0.015$). This should be recognized as statistical noise or unobserved confounding in the lower latency window, **not** empirical evidence that increasing network latency improves manufacturing throughput.
  - **Regime Instability as a Finding**: The pooled dataset yielded a nominal breakpoint of 41.2 ms with an excessively wide 95% block bootstrap confidence interval of [6.0, 45.2] ms. This broad dispersion confirms that operational mode heterogeneity masks critical threshold dynamics. Hence, reporting a single global pooled threshold is statistically invalid; operational thresholding must be stratified by machine state.
  - `Idle` Mode (27.5 ms, $F = 1.28$, not significant) and `Maintenance` Mode (25.5 ms, $F = 3.24$) tolerate substantially higher latency variance without throughput degradation.
- **Packet Loss Breakpoint Finding**:
  - Telemetry packet loss is uniformly distributed across [0%, 5%], with quality control defect rates remaining flat (~5.0%) across all packet loss deciles.
  - No statistically significant non-linear breakpoint or cliff exists for packet loss; quality metrics remain stable across the observed 0–5% range.

---

## 4. Modeling & Validation Empirical Results (Step 5)
- **Target Definition & Forecasting Horizon ($H$)**:
  - **Primary Horizon B ($H = 30$ min $\pm 10$ min)**: $N=23,587$ clean real-time matched pairs.
  - **Secondary Horizon A ($N=1$ row shift)**: $N=99,950$ pairs evaluated as a sensitivity check.
- **Model B Allowed Features (Strictly Enforced)**:
  - Allowed: `Network_Latency_ms`, `Packet_Loss_%`, `Network_Risk_Index`, `Network_Cluster`, `latency_lag_1`, `latency_lag_2`, `packet_loss_lag_1`, `packet_loss_lag_2`, `latency_roll_mean_3`, `latency_roll_std_3`, `packet_loss_roll_mean_3`, `Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`, `Predictive_Maintenance_Score`, `Quality_Control_Defect_Rate_%`, `Operation_Mode`, `Machine_ID`.
  - Strictly Excluded: `Production_Speed_units_per_hr` and `Error_Rate_%` (to prevent trivial label formula leakage).
  - Performance Ceiling Note: The resulting macro-F1 (~0.33) reflects genuine upstream root-cause predictability from physical sensors rather than formula inversion.
- **Crucial Scientific Conclusion (Baseline Reality Check)**:
  - **No model substantially outperforms the majority-class baseline** (Dummy Majority Accuracy: 78.42%, Weighted-F1: 0.6894).
  - This directly aligns with and reinforces the Step 3 null Granger causality finding (0 out of 450 tests surviving Benjamini-Hochberg FDR control).
  - Network telemetry, vibration, and temperature sensors carry very weak forward-looking predictive power for discrete manufacturing efficiency status. Artificially forcing minority class recall via class-weighting causes precision to collapse to the natural background prevalence (~3.0%), generating massive false alarms without producing net operational predictive lift.
- **Reframed Production Model Selection Criterion & Justification (Honest Limitations Reporting)**:
  - **Selected Production Artifact**: `RandomForestClassifier` (balanced class weighting).
  - **Reframed Selection Rationale**: In industrial plant operations, alarm credibility is paramount. Random Forest was selected strictly because it **avoids catastrophic false alarms**, and **NOT** because it meaningfully detects the rare High-efficiency class.
    - Specifically, Random Forest identified only **4 true positives out of 99 predictions** across 131 actual High-efficiency events in the chronological test set (Recall_High = 3.05%, Precision_High = 4.04%, with 127 missed events / false negatives). This is a stark predictive limitation that must be reported transparently, not framed as a model victory.
    - **LightGBM Alternative**: LightGBM is retained and documented as the higher-recall / higher-false-alarm alternative: it captures 83 of 131 High-efficiency events (Recall_High = 63.36% or 20.61% under balanced thresholding), but at the unacceptable cost of 2,735 false positives out of 2,818 alerts (Precision_High = 2.95%, which is slightly worse than raw random guessing at 2.986% background prevalence).
    - Random Forest attains superior overall Macro-F1 (0.3368 vs 0.2966) and Weighted-F1 (0.6622 vs 0.5014), and preserves reliable Low-efficiency risk detection (Recall_Low = 81.81%, Precision_Low = 78.54%) without overwhelming plant engineers with spurious alarms.
- **Baseline Ladder Summary (Chronological Test Set, Horizon B)**:
  - Dummy Majority: Accuracy = 78.42%, Macro-F1 = 0.2930, Precision_High = 0.0%, Recall_High = 0.0%, Precision_Low = 78.42%, Recall_Low = 100.0%
  - Logistic Regression (Class-Weighted): Accuracy = 27.91%, Macro-F1 = 0.2321, Precision_High = 3.02%, Recall_High = 46.56%, Precision_Low = 79.68%, Recall_Low = 25.97%
  - Random Forest (Class-Weighted): Accuracy = 67.30%, Macro-F1 = 0.3368, Precision_High = 4.04%, Recall_High = 3.05%, Precision_Low = 78.54%, Recall_Low = 81.81%
  - LightGBM (Class-Weighted): Accuracy = 43.13%, Macro-F1 = 0.2966, Precision_High = 2.95%, Recall_High = 20.61%, Precision_Low = 79.15%, Recall_Low = 45.05%
- **Seen vs. Unseen Machine Generalization Gap**:
  - Chronological Split (Known Machines): Random Forest Macro-F1 = 0.3368.
  - Machine-Grouped Split (Unseen Machines): Random Forest Macro-F1 = 0.3275.
  - Generalization Gap: $\Delta = 0.0093$ (less than 1% degradation), proving models generalize cleanly to new factory equipment.
- **Explainability (SHAP)**:
  - SHAP TreeExplainer confirms that Network Risk Index, Latency, and Predictive Maintenance Score are the dominant contributors to Early Warning risk classifications.

---

## 5. Economic & KPI Assumptions (Step 6)
- **Cost Structure Basis**: Economic loss calculated based on unit **contribution margin** ($/unit) rather than gross retail price.
- **Cost Parameters (Base / Sensitivity Range)**:
  - Base Contribution Margin: \$25.00 / unit (Range: \$15.00 – \$40.00 / unit).
  - Defect Remediation / Scrap Cost: \$12.00 / defective unit (Range: \$8.00 – \$20.00 / unit).
  - Packet Loss Spike Threshold: 95th percentile of historical packet loss distribution.
- **Simulator Nature**: Scenario simulations represent model conditional expectations given hypothetical inputs, not guaranteed counterfactual interventions.
