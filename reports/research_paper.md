# A Leakage-Safe Audit of Network Telemetry for Manufacturing Efficiency in a Synthetic Smart-Factory Benchmark

**Author:** Atharva Gangurde  
**Affiliation:** MVP Samaj’s Karmaveer Adv. Baburao Ganpatrao Thakare College of Engineering, Nashik, Maharashtra, India  
**Dataset:** Thales Group Advanced Manufacturing Testbed (50 CNC/Assembly Machines, 100,000 Synchronized Records, 68.4 Operating Days)  
**Date:** October 2026  
**Status:** Final Research Report & Publication Manuscript  


---

## Executive Abstract

The deployment of private 5G and nascent 6G industrial networks is predicated on the assumption that ultra-reliable low-latency communication (URLLC) directly dictates cyber-physical manufacturing throughput and quality. This study conducts an exhaustive, leakage-safe empirical investigation into whether factory network telemetry (`Network_Latency_ms`, `Packet_Loss_%`) drives operational efficiency (`Efficiency_Status`, `Production_Speed_units_per_hr`, `Quality_Control_Defect_Rate_%`) across 100,000 chronological observations from 50 industrial machines.

**Primary Headline Finding:** Within the observed operating ranges (latency $13.4$ to $37.9$ ms across cluster centroids; packet loss $0.0\%$ to $5.0\%$), **network conditions are not currently a primary driver of efficiency degradation in this dataset.** Five converging empirical proofs support this conclusion:
1. **Null Temporal Predictive Precedence:** Out of 450 machine- and mode-stratified Granger causality tests across multiple lag orders, 23 nominally passed ($\alpha = 0.05$, exactly the $5.1\%$ expected by pure chance), and **zero (0) tests survived Benjamini-Hochberg False Discovery Rate (FDR) control at $q < 0.05$**. Past network telemetry possesses zero statistically significant forecasting power over future throughput beyond throughput's own autoregressive history.
2. **Exploratory Nonlinear Breakpoint at $20.7\text{ ms}$:** Piecewise segmented regression identifies an empirical candidate latency threshold at **$20.7\text{ ms}$ in `Active` production mode** (slope change $\Delta \beta = -0.3063\text{ units/hr per ms}$). While the standard unadjusted F-test yields nominal significance ($F = 4.3127, p = 0.038$), this test is invalid because the breakpoint was chosen via grid search over candidate quantiles. Under a sup-F permutation test ($N = 2,000$), the permutation $p$-value is $0.1640$ (Benjamini-Hochberg FDR $q = 0.5100$), which fails multiple-testing correction; hence, the $20.7\text{ ms}$ breakpoint is strictly **exploratory**. In contrast, pooled cross-facility regression yields an unstable nominal threshold of $41.2\text{ ms}$ with an unreliably wide $95\%$ block bootstrap confidence interval of $[6.0, 45.2]\text{ ms}$, confirming that global network thresholds are an artifact of operational state aggregation.
3. **Absence of Packet-Loss Cliffs:** Across the entire $0.0\%$ to $5.0\%$ range, defect rates remain flat ($5.01\%$ during $95\text{th}$-percentile packet-loss spikes vs. $5.01\%$ baseline, an Impact Ratio of $1.0012$). No critical packet-loss threshold exists.
4. **Illustrative & Assumption-Driven Cost of Instability:** The empirical throughput divergence between optimal network conditions (Tier 0: $13.4\text{ ms}, 1.24\%$ loss) and heavily degraded conditions (Tier 3: $37.7\text{ ms}, 3.75\%$ loss) is restricted to **$3.084\text{ units/hr per machine}$** ($1.11\%$ of baseline output). Cost-of-instability figures represent illustrative, assumption-driven sensitivity scenarios ($\$15.00$ to $\$40.00/\text{unit}$ margin), yielding an estimated fleet loss of $\$2,312.98$ to $\$6,167.94/\text{fleet-hr}$ (base case: $\$3,854.96/\text{hr}$ at $\$25.00/\text{unit}$) during sustained Tier 3 degradation.
5. **Machine Learning Models Near Chance (ROC-AUC $\approx 0.50$):** Exhaustive baseline ladder evaluations (Dummy Majority, Class-Weighted Logistic Regression, Random Forest, LightGBM) on a rigorous physical-time lookahead horizon ($H = 30\text{ min} \pm 10\text{ min}$, $N = 23,587$) demonstrate that **ROC-AUC is approximately $0.50$ (near chance) for all models** (Logistic Regression: $0.5029$, Random Forest: $0.5068$, LightGBM: $0.5073$), confirming no machine learning model meaningfully outperforms the majority-class baseline (Dummy Majority Accuracy: $78.42\%$, Weighted-F1: $0.6894$). Random Forest was selected for production solely to suppress false alarms, but captured only $4$ of $131$ High-efficiency events ($3.05\%$ recall). Conversely, aggressive gradient boosting (LightGBM) generated a $97.05\%$ false alarm rate (Precision: $2.95\%$) when attempting to force minority recall.

**Strategic Implication:** In this dataset, the private wireless network telemetry operates safely within the mechanical process tolerance envelope. Industrial operators should conditionally defer premature capital investments in sub-millisecond radio over-engineering, subject to validation on real factory telemetry with sub-second resolution and physical process telemetry, while deploying conservative alerting policies and redirecting diagnostic resources toward mechanical wear, tool degradation, and thermal dissipation.

---

## 1. Introduction & Analytical Storyline

Industry 4.0 and emerging 6G smart manufacturing frameworks emphasize deterministic wireless connectivity as an indispensable prerequisite for autonomous industrial operations. Standardized architectural guidelines, such as 3GPP TS 22.104 (Release 16–18) for cyber-physical control applications in vertical domains, specify stringent quality-of-service (QoS) requirements—often stipulating end-to-end latencies between $1\text{ ms}$ and $10\text{ ms}$ and transmission service reliability exceeding $99.999\%$.

However, in brownfield deployments and industrial IoT sensor fabrics, network telemetry is frequently captured via application-layer polling, transport protocols, or cyclic programmable logic controller (PLC) telemetry. A critical empirical question facing industrial enterprise architects is: *To what extent does real-world network variation directly dictate shop-floor manufacturing throughput, defect rates, and operational efficiency?*

To resolve this question without methodological shortcuts, this investigation follows an unbroken, evidence-based storyline chain:
$$\text{Network Degradation} \longrightarrow \text{Temporal Evidence} \longrightarrow \text{Breakpoint} \longrightarrow \text{Early Warning} \longrightarrow \text{Explainable Diagnosis} \longrightarrow \text{Operational Risk}$$

```
+----------------------------------------------------------------------------------------------------+
|                                    ANALYTICAL STORYLINE CHAIN                                      |
+----------------------------------------------------------------------------------------------------+
|  [Step 1 & 2] NETWORK PROFILING                                                                    |
|  - 4 clusters form a pragmatic partition of continuous uniform space (Silhouette: 0.4063)          |
|  - Continuous Network Risk Index: 0.5 * z(lat) + 0.5 * z(loss)                                     |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 3 & 4] TEMPORAL EVIDENCE & DIAGNOSTICS                                                      |
|  - Granger causality: 0 / 450 tests survive Benjamini-Hochberg FDR (q < 0.05)                      |
|  - Reconciled Headline: Zero predictive precedence; active-mode breakpoint at 20.7 ms exploratory  |
|  - Diagnostics: Linear |r| < 0.008; independent uniforms; packet loss shows no quality cliff        |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 5] LEAKAGE-SAFE PREDICTIVE MODELING                                                         |
|  - Label Audit: Efficiency_Status = deterministic function of Speed and Error Rate                 |
|  - Circularity Rule: Speed and Error Rate strictly excluded from Model B                           |
|  - Model A (Early Warning, H=30min) vs Model B (Diagnosis, Current)                                |
|  - Baseline Reality: All models near chance (ROC-AUC ~0.50); no generalization gap expected        |
|  - Model Selection: Random Forest retained solely to avoid false alarm flooding                    |
|                                       |                                                            |
|                                       v                                                            |
|  [Step 6 & 7] OPERATIONAL KPIS & ECONOMIC SENSITIVITY                                              |
|  - Latency Sensitivity Score: Active inflection Delta beta = -0.3063 units/hr per ms (exploratory)  |
|  - Cost of Instability: Illustrative assumption-driven scenario ($2,313–$6,168/fleet-hr range)     |
|  - Synthesis: Telemetry within tolerance in dataset; conditionally validate capex on real data     |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Data Architecture, Sampling Mechanics & Label Audit

### 2.1 Fleet Dataset Dimensions & Quality
The dataset comprises $100,000$ consecutive observations recorded across $50$ distinct CNC, milling, and robotic assembly machines (labeled `Machine_ID` 1 through 50) operating over a span of $68.44\text{ calendar days}$ (January 1, 2025, to March 10, 2025). 

A comprehensive data integrity audit confirmed:
- Total rows: $100,000$; Total analytical columns: $13$.
- Missing values: Exactly $0$ across all fields ($100\%$ complete telemetry).
- Duplicate records: Exactly $0$.
- Unified datetime formatting: `Date` and `Timestamp` merged into UTC-indexed POSIX timestamps.

### 2.2 Target Class Imbalance
The primary operational target, `Efficiency_Status`, exhibits severe class imbalance:
- **Low Efficiency**: $77,825$ records ($77.825\%$)
- **Medium Efficiency**: $19,189$ records ($19.189\%$)
- **High Efficiency**: $2,986$ records ($2.986\%$)

Under this severe distribution skew, unadjusted classification accuracy is mathematically misleading: a naive dummy classifier that unconditionally predicts `Low` achieves $77.83\%$ accuracy across the full dataset ($78.42\%$ on the holdout test set) while providing zero operational utility. Consequently, all modeling evaluations in this study enforce macro-averaged F1, minority-class recall, and precision-recall curves.

### 2.3 Per-Machine Sampling Arrival Mechanics
A critical finding emerged from auditing the time intervals between successive readings:
- **Factory-Wide Cadence**: Exactly $60.0\text{ seconds}$ median interval. Telemetry arrives at the centralized telemetry broker at a constant $1\text{ sample/minute}$ rate ($98,559$ consecutive global transitions have $\Delta t = 60\text{ s}$).
- **Per-Machine Arrival Distribution**: For any individual machine, the median sampling interval is **$34.0\text{ minutes}$**, while the mean interval is **$49.29\text{ minutes}$** (standard deviation: $48.86\text{ minutes}$, $95\text{th}$ percentile: $147.0\text{ minutes}$).

```
Per-Machine Inter-Arrival Histogram (Geometric Process, p = 0.02)
   Cadence: Mean = 49.29 min, Median = 34.0 min
   -------------------------------------------------------------------------
   0 - 20 min   [############################] (35.2%)
   20 - 40 min  [######################] (23.4%)
   40 - 60 min  [##############] (15.1%)
   60 - 90 min  [###########] (12.3%)
   90 - 150 min [########] (8.9%)
   > 150 min    [####] (5.1%)
```

Statistical testing revealed that the factory telemetry broker operates on a **stochastic polling or Bernoulli round-robin multiplexer** with $p = 1/50 = 0.02$ per minute. For a discrete geometric distribution $\text{Geometric}(p=0.02)$:
- Theoretical Expected Mean: $E[X] = 1/p = 50.0\text{ minutes}$ (empirical: $49.29\text{ min}$).
- Theoretical Expected Median: $\lceil -\ln(2)/\ln(1-p) \rceil = 35.0\text{ minutes}$ (empirical: $34.0\text{ min}$).

Every individual machine displays an identical distribution (sample counts: $1,905$ to $2,091$ readings per machine; individual means: $47.1$ to $51.7\text{ min}$). No single machine suffered from dropped physical connectivity.

**Consequence for Horizon Definition ($H$):** Because sampling intervals follow an exponential-like tail rather than a rigid clock, a naive row-shift ($N = 1$) represents an unpredictable temporal step ranging from $1\text{ minute}$ to over $3\text{ hours}$. Therefore, in Step 5, we define a **strict physical-time filter (Horizon B: $H = 30\text{ min} \pm 10\text{ min}$)** as the primary target, reserving Horizon A ($N = 1$ shift) strictly as a sensitivity check.

### 2.4 Label Generation Audit & Elimination of Target Leakage
To ensure valid root-cause modeling, we audited whether `Efficiency_Status` was an independently observed physical state or a calculated synthetic label. Exhaustive decision-tree and boundary analysis uncovered the **exact deterministic formula governing label generation** ($99.998\%$ exact match across all $100,000$ records):

$$\text{Efficiency\_Status} = \begin{cases} 
\mathbf{Low} & \text{if } \text{Error\_Rate\_\%} > 5.0\% \;\;\mathbf{OR}\;\; \text{Production\_Speed\_units\_per\_hr} \le 200.0 \\ 
\mathbf{High} & \text{if } \text{Production\_Speed\_units\_per\_hr} > 400.0 \;\;\mathbf{AND}\;\; \text{Error\_Rate\_\%} \le 2.0\% \\ 
\mathbf{Medium} & \text{Otherwise} 
\end{cases}$$

**Impact on Model B (Diagnosis Model):** If contemporaneous `Production_Speed_units_per_hr` and `Error_Rate_%` were included as features in Model B, any basic machine learning algorithm would trivially memorize this arithmetic rule, achieving $100\%$ accuracy while yielding zero diagnostic insight. In strict compliance with scientific integrity standards, **`Production_Speed_units_per_hr` and `Error_Rate_%` were permanently excluded from Model B**. Model B evaluates contemporaneous classification strictly from upstream physical features (network metrics, temperature, vibration, power consumption, predictive maintenance scores, and operational context), predicting near chance (ROC-AUC $\approx 0.50$).

---

## 3. Unsupervised Network Profiling & Continuous Risk Indexing

### 3.1 Empirical Selection of Cluster Count ($k$)
Industrial network performance varies across multiple physical modes (e.g., congestion, bufferbloat, physical channel fading). Rather than arbitrarily imposing an assumed $k=3$ (e.g., "Good/Medium/Bad"), we swept $k \in [2, 6]$ using standardized z-scores on `Network_Latency_ms` and `Packet_Loss_%`, fit strictly on the training partition ($80,000$ records).

```
Cluster Optimization Metrics vs. Number of Clusters (k)
+-------+------------------+-----------------------+--------------------------+----------------------------+
|   k   | Inertia (SSE)    | Silhouette Score (Max)| Davies-Bouldin Index(Min)| Calinski-Harabasz Score    |
+-------+------------------+-----------------------+--------------------------+----------------------------+
|   2   | 125,070.7        | 0.3548                | 1.1871                   | 59,910                     |
|   3   |  79,315.3        | 0.3764                | 0.8687                   | 76,078                     |
|   4   |  50,230.2        | 0.4063  <-- BEST      | 0.7690  <-- BEST         | 99,386  <-- BEST           |
|   5   |  42,314.5        | 0.3908                | 0.8367                   | 93,161                     |
|   6   |  35,980.0        | 0.3745                | 0.8422                   | 91,169                     |
+-------+------------------+-----------------------+--------------------------+----------------------------+
```

As demonstrated in Figure 1 (`reports/clustering_evaluation_k_selection.png`) and archived in `reports/clustering_k_selection.csv`, $k = 4$ simultaneously achieves the **global maximum Silhouette Score ($0.4063$)**, the **global minimum Davies-Bouldin Index ($0.7690$)**, and the **global maximum Calinski-Harabasz score ($99,386$)**. Due to the $\mathcal{O}(N^2)$ pairwise distance complexity across $N=100,000$ points, the silhouette score was computed on a representative random subsample of $n=15,000$ points (random seed 42 in `src/network_profiling.py`), whereas Inertia, Davies-Bouldin, and Calinski-Harabasz indices were computed across all $100,000$ observations.

### 3.2 Quantitative Profiling of the Four Network Tiers
The four clusters represent a pragmatic partition of a continuous (uniform) feature space rather than naturally occurring, distinct physical regimes. To establish a structured operational hierarchy, clusters were ordered by their composite Network Risk Index ($NRI = 0.5 \cdot z(\text{lat}) + 0.5 \cdot z(\text{loss})$):

```
+--------+-------------------+-----------------+-------------------+-------------------+------------+
| Tier   | Operational Label | Mean Latency ms | Mean Packet Loss% | Network Risk Index| Count (N)  |
+--------+-------------------+-----------------+-------------------+-------------------+------------+
| Tier 0 | Optimal (Low Risk)| 13.41 +/- 7.22  | 1.24% +/- 0.72%   | -0.8631           | 24,970     |
| Tier 1 | Loss-Degraded     | 13.33 +/- 7.18  | 3.74% +/- 0.72%   | -0.0041           | 25,067     |
| Tier 2 | Latency-Degraded  | 37.89 +/- 7.23  | 1.25% +/- 0.72%   | +0.0108           | 25,288     |
| Tier 3 | Critical (Severe) | 37.72 +/- 7.21  | 3.75% +/- 0.72%   | +0.8659           | 24,675     |
+--------+-------------------+-----------------+-------------------+-------------------+------------+
```

### 3.3 Continuous Network Risk Index (NRI)
To eliminate information loss inherent in discrete binning, we established a continuous composite metric:
$$\text{NRI}_i = 0.5 \cdot \left(\frac{\text{Latency}_i - \mu_{\text{lat}}}{\sigma_{\text{lat}}}\right) + 0.5 \cdot \left(\frac{\text{PacketLoss}_i - \mu_{\text{loss}}}{\sigma_{\text{loss}}}\right)$$

Empirically, $\text{NRI}$ spans $[-1.7314, +1.7288]$ with mean $0.0000$ and standard deviation $0.7047$. 

Cross-tabulation against `Operation_Mode` and `Machine_ID` verified that the four tiers are distributed uniformly across all machines and operational modes ($\chi^2$ independence tests fail to reject uniformity, $p > 0.95$). This reflects the underlying independent uniform distribution of the synthetic telemetry features across quadrants of the 2D network space.

---

## 4. Diagnostics & Confounder Verification

### 4.1 Stratification by Operation Mode & Simpson's Paradox
Operational context strongly dictates industrial machinery behavior. Telemetry records were stratified across:
- **`Active`**: $70,054$ records ($70.05\%$)
- **`Idle`**: $20,057$ records ($20.06\%$)
- **`Maintenance`**: $9,889$ records ($9.89\%$)

To prevent Simpson's paradox—where aggregating distinct subpopulations can generate artificial correlations or mask real effects—all bivariate relationships were evaluated both pooled and within each individual operating mode.

```
Summary of Pearson Correlation Coefficients Across Operating Modes
+---------------------------------------------+----------+----------+----------+---------------+
| Variable Pair                               | Pooled   | Active   | Idle     | Maintenance   |
+---------------------------------------------+----------+----------+----------+---------------+
| Latency vs Production Speed                 | -0.0031  | -0.0039  | +0.0008  | -0.0012       |
| Packet Loss vs Defect Rate                  | +0.0014  | +0.0006  | +0.0031  | +0.0019       |
| Packet Loss vs Error Rate                   | +0.0028  | +0.0031  | -0.0002  | +0.0045       |
| Latency vs Quality Defect Rate              | -0.0009  | -0.0011  | +0.0014  | -0.0020       |
+---------------------------------------------+----------+----------+----------+---------------+
```

All linear correlation coefficients satisfy $|r| < 0.005$ with $p > 0.20$. This confirms the complete absence of strong global linear dependencies between raw network metrics and production metrics.

---

## 5. Temporal Causal Evidence & Breakpoint Analysis

### 5.1 Granger Causality Analysis & Multiple Testing Correction
To rigorously test whether network latency or packet loss possesses predictive precedence over manufacturing throughput, we conducted vector autoregressive (VAR) Granger causality testing:
$$\text{Speed}_t = \alpha + \sum_{i=1}^p \beta_i \text{Speed}_{t-i} + \sum_{j=1}^p \gamma_j \text{Network}_{t-j} + \epsilon_t$$
The null hypothesis $H_0: \gamma_1 = \gamma_2 = \dots = \gamma_p = 0$ posits that historical network telemetry provides no forecast improvement over speed's own autoregressive history.

Testing parameters:
- Machine stratification: Evaluated individually for each of the $50$ machines.
- Mode stratification: Partitioned into `Active`, `Idle`, and `Maintenance` modes.
- Autoregressive lag orders: $p \in \{1, 2, 3\}$.
- Stationarity: Augmented Dickey-Fuller (ADF) tests rejected unit roots for all per-machine series ($p_{\text{ADF}} < 0.001$).
- Total tests executed: $50\text{ machines} \times 3\text{ modes} \times 3\text{ lags} = \mathbf{450\text{ statistical tests}}$.

```
Granger Causality Significance Summary (450 Tests Total)
+-------------------------------------------------------------+---------------+
| Test Evaluation Metric                                      | Result        |
+-------------------------------------------------------------+---------------+
| Total Statistical Hypothesis Tests Executed                 | 450           |
| Nominal Rejections at unadjusted alpha = 0.05               | 23 (5.11%)    |
| Theoretical Rejections Expected by Pure Random Chance       | 22.5 (5.00%)  |
| Rejections Surviving Benjamini-Hochberg FDR (q < 0.05)      | 0 (0.00%)     |
+-------------------------------------------------------------+---------------+
```

**Crucial Causal Finding:** At an unadjusted significance threshold of $\alpha = 0.05$, exactly $23$ out of $450$ tests nominally rejected $H_0$ ($5.11\%$). This matches the exact $5.0\%$ false-positive rate expected under a purely null distribution. Following rigorous **Benjamini-Hochberg False Discovery Rate (FDR) control at $q < 0.05$**, **zero (0) tests survived**. 

Historical network telemetry carries no statistically robust predictive power over future machine speed across multi-minute operational horizons.

### 5.2 Breakpoint Analysis: Contemporaneous Nonlinear Dynamics
While network latency exhibits no forward-looking autoregressive forecasting power, manufacturing operations may experience immediate, contemporaneous physical impacts when latency breaches physical buffer thresholds.

Using continuous segmented regression (piecewise linear regression with continuous hinge join) and change-point search across the operational latency domain $[5.0\text{ ms}, 45.0\text{ ms}]$, we tested for threshold inflections:

```
Segmented Breakpoint Detection Summary & Permutation Tests (N = 2,000 Permutations)
+-----------------+---------------+----------------+-------------------+--------------+---------------+---------------+-----------------+--------------+
| Grouping        | Breakpoint ms | Pre-Slope beta1| Post-Slope beta2  | Delta beta   | sup-F Stat    | Naive F p-val | Permutation p   | FDR Adj q    | Classification
+-----------------+---------------+----------------+-------------------+--------------+---------------+---------------+-----------------+--------------+
| Active Mode     | 20.69 ms      | +0.2123        | -0.0940           | -0.3063      | 4.3127        | 0.0378        | 0.1640          | 0.5100       | Exploratory  |
| Idle Mode       | 27.55 ms      | -0.2563        | +0.0414           | +0.2977      | 1.2800        | 0.2579        | 0.6890          | 0.7530       | Non-Sig.     |
| Maintenance     | 25.50 ms      | -0.2722        | +0.3875           | +0.6597      | 3.2382        | 0.0720        | 0.2550          | 0.5100       | Non-Sig.     |
| Pooled Fleet    | 41.22 ms      | +0.0129        | -0.2509           | -0.2638      | 1.0532        | 0.3048        | 0.7530          | 0.7530       | Non-Sig.     |
+-----------------+---------------+----------------+-------------------+--------------+---------------+---------------+-----------------+--------------+
* Note: The standard F-test is invalid because the candidate threshold tau was selected by minimizing RSS over a grid of quantiles (the breakpoint is unidentified under the null). Permutation testing of the sup-F statistic (2,000 permutations) demonstrates that no breakpoint survives multiple-testing correction.
```

```
Segmented Regression Fit: Active Production Mode (N = 70,054, Exploratory)
Throughput
(units/hr)
   285 |                Candidate Breakpoint = 20.7 ms (Exploratory)
   280 |                          /\
   275 |     Pre-Slope:          /  \     Post-Slope:
   270 |     +0.2123 units/ms   /    \    -0.0940 units/ms
   265 |                       /      \
   260 +----------------------/--------\--------------------> Latency (ms)
       0         10          20        30         40
```

#### Reconciled Headline Finding & Scientific Interpretation
1. **The Active-Mode Breakpoint ($20.7\text{ ms}$) is Exploratory:** Under active machining load, segmented regression identifies a nominal structural break at $20.7\text{ ms}$ ($\Delta \beta = -0.3063\text{ units/hr per ms}$). While the naive unadjusted F-test indicates nominal significance ($F = 4.3127, p = 0.038$), this test is invalid due to candidate quantile selection. When evaluated against the empirical null distribution of the sup-F statistic via $2,000$ permutations, the permutation $p$-value is $0.1640$ (and $q = 0.5100$ after Benjamini-Hochberg FDR correction across the four tests). Because it does not survive multiple-comparison correction, this threshold must be categorized as **strictly exploratory**.
2. **Reconciliation with Granger Null:** In this dataset, both temporal forecasting (Granger causality) and contemporaneous breakpoint detection yield null or exploratory results after rigorous statistical control. Network latency neither forecasts future throughput nor establishes a statistically validated non-linear structural cliff.
3. **Caveat on Active Pre-Threshold Slope ($+0.2123$):** In the segmented fit below $20.7\text{ ms}$, the pre-threshold slope is slightly positive ($+0.2123$). Because this slope exists entirely within an overall zero-correlation regime ($|r| < 0.008$), **it must be recognized as statistical noise or an artifact of candidate optimization**, and **not** as evidence that increasing network latency improves manufacturing throughput.
4. **Instability of the Pooled Threshold ($41.2\text{ ms}$):** In the pooled dataset, segmented regression identified a nominal breakpoint at $41.2\text{ ms}$ ($F = 1.05, p_{\text{perm}} = 0.7551$). A $1,000$-iteration block bootstrap generated an excessively wide $95\%$ confidence interval of **$[6.0\text{ ms}, 45.2\text{ ms}]$**, confirming that global network thresholds in this dataset are statistical noise.

### 5.3 Packet Loss Response: Absence of a Defect Cliff
Quality control defect rates were binned across packet loss deciles from $0.0\%$ to $5.0\%$:
- Baseline Defect Rate ($< 95\text{th}$ percentile): **$5.0085\%$**
- Defect Rate during Severe Spikes ($> 4.755\%$ packet loss): **$5.0143\%$**
- **Packet Loss Impact Ratio:** **$1.0012$**

No non-linear cliff or threshold exists for packet loss within the $0.0\%$ to $5.0\%$ range. Industrial controllers in this facility tolerate up to $5\%$ packet loss without measurable degradation in finished product quality.

---

## 6. Machine Learning Early Warning & Contemporaneous Diagnosis

### 6.1 Validation Design & Circularity Prevention
To ensure production applicability, models were developed across two distinct operational specifications:
1. **Model A (Early Warning):** Evaluates whether telemetry at time $t$ can forecast `Efficiency_Status` at time $t + H$. The primary dataset enforces **Horizon B ($H = 30\text{ min} \pm 10\text{ min}$, $N = 23,587$)**, with Horizon A ($N = 1$ row shift, $N = 99,950$) evaluated as a secondary check.
2. **Model B (Diagnosis):** Evaluates contemporaneous classification of `Efficiency_Status` at time $t$. As mandated by the Step 1 label audit, contemporaneous production speed and error rate are strictly excluded. Both models predict near chance (ROC-AUC $\approx 0.50$).

**Data Splitting Protocol:**
- **Chronological Split (Known Machines):** Earliest $80\%$ of records allocated to training ($N = 18,869$ for Horizon B; $N = 79,960$ for Horizon A); final $20\%$ held out for testing ($N = 4,718$ for Horizon B; $N = 19,990$ for Horizon A).
- **Machine-Grouped Split (Unseen Machines):** Group 5-fold cross-validation partitioned by `Machine_ID` (10 unseen machines held out per fold) to test generalization to new factory equipment.
- All transformations (imputation, standard scaling, K-Means cluster assignment, resampling) were fitted strictly on training folds.

### 6.2 Baseline Ladder Evaluation
We implemented a strict baseline ladder: Dummy Majority Classifier $\rightarrow$ Class-Weighted Logistic Regression $\rightarrow$ Balanced Random Forest $\rightarrow$ Balanced LightGBM.

```
Comprehensive Baseline Ladder Performance Table (Test Set, Horizon B: H = 30 min +/- 10 min)
+-------------------------+----------+----------+----------+----------+------------+------------+-------------+--------------+
| Model Architecture      | Accuracy | Macro-F1 | Weight-F1| ROC-AUC  | Prec (Low) | Rec (Low)  | Prec (High) | Rec (High)   |
+-------------------------+----------+----------+----------+----------+------------+------------+-------------+--------------+
| Model A - Dummy Majority| 0.7842   | 0.2930   | 0.6894   | 0.5000   | 0.7842     | 1.0000     | 0.0000      | 0.0000       |
| Model A - Logistic Reg. | 0.2791   | 0.2321   | 0.3554   | 0.5029   | 0.7968     | 0.2597     | 0.0302      | 0.4656       |
| Model A - Random Forest | 0.6730   | 0.3368   | 0.6622   | 0.5068   | 0.7854     | 0.8181     | 0.0404      | 0.0305       |
| Model A - LightGBM      | 0.4313   | 0.2966   | 0.5014   | 0.5073   | 0.7915     | 0.4505     | 0.0295      | 0.2061       |
+-------------------------+----------+----------+----------+----------+------------+------------+-------------+--------------+
| Model B - Dummy Majority| 0.7800   | 0.2921   | 0.6835   | 0.5000   | 0.7800     | 1.0000     | 0.0000      | 0.0000       |
| Model B - Logistic Reg. | 0.2849   | 0.2330   | 0.3636   | 0.5009   | 0.7742     | 0.2765     | 0.0317      | 0.4508       |
| Model B - Random Forest | 0.5570   | 0.3246   | 0.5947   | 0.5016   | 0.7823     | 0.6397     | 0.0240      | 0.0590       |
| Model B - LightGBM      | 0.3789   | 0.2733   | 0.4545   | 0.4994   | 0.7790     | 0.3915     | 0.0300      | 0.2574       |
+-------------------------+----------+----------+----------+----------+------------+------------+-------------+--------------+
| Unseen Machines (RF)    | 0.7024   | 0.3275   | 0.6717   | 0.4896   | 0.7818     | 0.8703     | 0.0141      | 0.0078       |
+-------------------------+----------+----------+----------+----------+------------+------------+-------------+--------------+
```

### 6.3 Comprehensive Confusion Matrix Analysis
To diagnose the true behavior of these models beyond scalar summary metrics, Table 5 details the exact confusion matrices on the Horizon B holdout test partition ($N = 4,718$).

```
Detailed Confusion Matrices: Chronological Holdout Partition (Horizon B, N = 4,718)

1. Dummy Majority Baseline
+--------------------+----------------+----------------+----------------+----------------+
| Actual \ Predicted | Pred: Low      | Pred: Medium   | Pred: High     | Total Actual   |
+--------------------+----------------+----------------+----------------+----------------+
| Actual: Low        | 3,700 (100.0%) | 0 (0.0%)       | 0 (0.0%)       | 3,700          |
| Actual: Medium     | 887 (100.0%)   | 0 (0.0%)       | 0 (0.0%)       | 887            |
| Actual: High       | 131 (100.0%)   | 0 (0.0%)       | 0 (0.0%)       | 131            |
| Total Predicted    | 4,718          | 0              | 0              | 4,718          |
+--------------------+----------------+----------------+----------------+----------------+

2. Random Forest Classifier (Selected Production Artifact)
+--------------------+----------------+----------------+----------------+----------------+
| Actual \ Predicted | Pred: Low      | Pred: Medium   | Pred: High     | Total Actual   |
+--------------------+----------------+----------------+----------------+----------------+
| Actual: Low        | 3,027 (81.8%)  | 600 (16.2%)    | 73 (2.0%)      | 3,700          |
| Actual: Medium     | 721 (81.3%)    | 144 (16.2%)    | 22 (2.5%)      | 887            |
| Actual: High       | 106 (80.9%)    | 21 (16.0%)     | 4 (3.1%)       | 131            |
| Total Predicted    | 3,854          | 765            | 99             | 4,718          |
+--------------------+----------------+----------------+----------------+----------------+

3. LightGBM Classifier (Alternative High-Recall Benchmark)
+--------------------+----------------+----------------+----------------+----------------+
| Actual \ Predicted | Pred: Low      | Pred: Medium   | Pred: High     | Total Actual   |
+--------------------+----------------+----------------+----------------+----------------+
| Actual: Low        | 1,667 (45.1%)  | 1,304 (35.2%)  | 729 (19.7%)    | 3,700          |
| Actual: Medium     | 386 (43.5%)    | 341 (38.4%)    | 160 (18.0%)    | 887            |
| Actual: High       | 53 (40.5%)     | 51 (38.9%)     | 27 (20.6%)     | 131            |
| Total Predicted    | 2,106          | 1,696          | 916            | 4,718          |
+--------------------+----------------+----------------+----------------+----------------+
```

### 6.4 Honest Reframing of Production Model Selection Rationale
In corporate ML reporting, practitioners frequently obscure severe minority-class degradation by highlighting macro-averaged metrics. We report the model selection rationale transparently:

1. **Near-Chance Performance Across All Models (ROC-AUC $\approx 0.50$):** Across all evaluated architectures, ROC-AUC remains approximately $0.50$ (chance level). The Dummy Majority baseline achieves $78.42\%$ accuracy and $0.6894$ Weighted-F1. No machine learning model outperforms chance discrimination in operational predictive power.
2. **Random Forest Selection Rationale:** Random Forest was retained in production solely because it behaves as a conservative majority-class filter that avoids catastrophic false-alarm flooding, and NOT because it possesses genuine discriminative power on the High-efficiency class.
   - Specifically, Random Forest captured only **$4$ true positives out of $99$ predictions** across $131$ actual High-efficiency events in the chronological test set (Precision: $4.04\%$, Recall: $3.05\%$). It missed $127$ out of $131$ High-efficiency events ($96.95\%$ false-negative rate). This is an explicit predictive limitation.
   - It assigns nearly all records to the majority Low-efficiency class (Recall: $81.81\%$, Precision: $78.54\%$, with $3,027$ correct classifications out of $3,854$ predictions).
3. **The LightGBM Trade-Off (False Alarm Catastrophe):** LightGBM achieved higher nominal recall on the High class ($20.61\%$, identifying $27$ true events). However, doing so required issuing $916$ High-efficiency alerts, of which **$889$ were false alarms** (Precision: $2.95\%$). A precision of $2.95\%$ is statistically indistinguishable from the background class prevalence ($2.986\%$). In an active industrial plant, a predictive system where $97.05\%$ of alerts are false alarms induces immediate operator alert fatigue and total loss of system trust.
4. **Generalization Gap Re-Interpretation:** Evaluated across unseen factory equipment via GroupKFold, Random Forest Macro-F1 dropped by only $\Delta = 0.0093$ ($0.3368$ on known machines vs. $0.3275$ on unseen machines), while ROC-AUC remained at $0.4896$. Crucially, near-chance performance on both splits means that no meaningful generalization gap is expected; this minimal discrepancy is a mathematical artifact of chance-level predictions rather than positive evidence of model generalization.

### 6.5 Model Explainability (SHAP Analysis)
TreeSHAP evaluations on the models (Figures 6 & 7 in `reports/shap_summary_model_a.png` and `reports/shap_summary_model_b.png`) display nominal feature attributions:
- For Model A (Early Warning), the composite **Network Risk Index (NRI)** and raw **`Network_Latency_ms`** appear as nominal contributors separating classes, alongside `Predictive_Maintenance_Score` and operational mode indicators.
- For Model B (Diagnosis), when speed and error rate are properly excluded, classification attributions shift to `Power_Consumption_kW`, `Temperature_C`, and `Vibration_Hz`.
- **Critical Methodological Caveat:** Because all models exhibit ROC-AUC values of $\sim 0.50$ (failing to exceed random guessing), these SHAP feature attributions represent explanations of uninformative models fitted to stochastic noise rather than verified physical root causes. They must not be interpreted as physical drivers of factory efficiency.

---

## 7. Operational KPIs & Economic Instability Assessment

To translate physical and statistical findings into managerial decision frameworks, five standardized KPIs were computed across the factory dataset:

```
+------------------------------------+-------------------------------------------+-----------------------------------+
| Metric                             | Mathematical Definition                   | Empirical Value                   |
+------------------------------------+-------------------------------------------+-----------------------------------+
| 1. Network Risk Index (NRI)        | 0.5 * z(latency) + 0.5 * z(packet_loss)   | Range: [-1.73, +1.73], Std: 0.705 |
| 2. Latency Sensitivity Score       | Delta(Speed) / Delta(Latency) around 20.7ms| Active inflection: -0.3063 u/hr/ms|
| 3. Packet Loss Impact Ratio        | DefectRate(Spike) / DefectRate(Baseline)  | 1.0012 (Flat / No cliff)          |
| 4. Network Tolerance Thresholds    | Active Latency Breakpoint                 | 20.7 ms (Active); None (Loss)     |
| 5. Cost of Instability             | Lost Units/hr * CM + Excess Defects * Cost| $3,854.96 / fleet-hour (Base)     |
+------------------------------------+-------------------------------------------+-----------------------------------+
```

### 7.1 Economic Loss Model Formulation
Following rigorous managerial accounting principles, financial loss is computed based on **unit contribution margin** rather than gross revenue, avoiding distorted overhead allocations:
$$\text{Cost of Instability} = \left(\Delta \text{Throughput} \times \text{Contribution Margin}\right) + \left(\text{Throughput} \times \Delta \text{Defect Rate} \times \text{Scrap Remediation Cost}\right)$$

Empirical parameter divergence between optimal network conditions (Tier 0: $13.4\text{ ms}, 1.24\%$ loss) and heavily degraded network conditions (Tier 3: $37.7\text{ ms}, 3.75\%$ loss):
- Baseline Throughput (Tier 0): **$277.68\text{ units/hr per machine}$**
- Degraded Throughput (Tier 3): **$274.59\text{ units/hr per machine}$**
- Empirical Production Loss ($\Delta \text{Throughput}$): **$3.084\text{ units/hr per machine}$** ($1.11\%$ loss)
- Defect Rate Divergence ($\Delta \text{Defect Rate}$): **$+0.0058\%$** ($5.0143\%$ vs $5.0085\%$)

### 7.2 Multi-Scenario Sensitivity Analysis (Illustrative & Assumption-Driven)
Because exact contribution margins vary by manufactured component, we evaluated Low, Base, and High illustrative economic scenarios based on stylized margin and scrap assumptions:

```
Economic Sensitivity Analysis: Cost of Severe Network Degradation (Tier 0 vs Tier 3) - Illustrative Scenarios
+---------------------------------------+-------------------+-------------------+-------------------+
| Economic Parameter / Metric           | Low Cost Scenario | Base Case Scenario| High Cost Scenario|
+---------------------------------------+-------------------+-------------------+-------------------+
| Contribution Margin per Unit          | $15.00 / unit     | $25.00 / unit     | $40.00 / unit     |
| Scrap / Defect Remediation Cost       | $8.00 / unit      | $12.00 / unit     | $20.00 / unit     |
+---------------------------------------+-------------------+-------------------+-------------------+
| Hourly Loss per Single Machine        | $46.26 / mach-hr  | $77.10 / mach-hr  | $123.36 / mach-hr |
| Total Fleet Loss per Operating Hour   | $2,312.98 / hr    | $3,854.96 / hr    | $6,167.94 / hr    |
| Fleet Cost per 8-Hour Production Shift| $18,503.81 / shift| $30,839.68 / shift| $49,343.49 / shift|
| Annualized Fleet Cost (2,000 Op Hours)| $4,625,952.09     | $7,709,920.15     | $12,335,872.24    |
+---------------------------------------+-------------------+-------------------+-------------------+
```

Across a standard $50$-machine manufacturing plant operating $2,000$ hours annually, continuous severe network degradation incurs an illustrative estimated $\$4.63\text{M}$ to $\$12.34\text{M}$ financial sensitivity range ($\$7.71\text{M}$ base case). Crucially, these figures represent illustrative assumption-driven models under hypothetical full-fleet continuous degradation rather than verified causal balance-sheet entries.

---

## 8. Standards Alignment: 3GPP TS 22.104 & ITU Benchmarks

To ground empirical findings in telecommunications architecture, observed network parameters were compared against 3GPP TS 22.104 (Release 16/17/18) and ITU-R IMT-2030 (6G) standards:

```
+-----------------------------------+--------------------+--------------------+--------------------+
| 3GPP / ITU Standard Use Case      | 3GPP Latency Bound | Observed Mean/Range| 3GPP Reliability   |
+-----------------------------------+--------------------+--------------------+--------------------+
| Discrete Manufacturing (Motion)   | 1.0 - 10.0 ms      | 13.4 - 37.9 ms     | 99.999% - 99.9999% |
| Factory Automation (Sensors/PLC)  | 10.0 - 50.0 ms     | 13.4 - 37.9 ms     | 99.9% - 99.99%     |
| AGV / Mobile Robot Fleet Control  | 10.0 - 30.0 ms     | 13.4 - 37.9 ms     | 99.99%             |
| Augmented Reality Remote Assist   | 10.0 - 20.0 ms     | 13.4 - 37.9 ms     | 99.0%              |
+-----------------------------------+--------------------+--------------------+--------------------+
```

### Methodological Comparison Cautions & Boundary Conditions
1. **Latency Measurement Semantics:** In 3GPP specifications, latency is strictly defined as one-way user-plane service data unit (SDU) transmission time from ingress interface to egress delivery. In factory telemetry datasets, `Network_Latency_ms` represents application-layer or transport-layer round-trip time (RTT). Direct numerical comparison must therefore be treated as a **qualitative reference** rather than a strict compliance audit.
2. **Packet Loss vs. 3GPP Service Reliability:** **Do NOT equate packet-loss percentage with 3GPP service reliability.** A 3GPP reliability requirement of $99.999\%$ mandates that $99.999\%$ of packets are delivered successfully within a maximum allowable latency budget $T_{\text{max}}$. Transport-layer packet loss of $1\%$ to $3\%$ in continuous telemetry captures retransmissions and buffer drops, but does not indicate complete service outage under application-layer retry mechanisms.

---

## 9. Comprehensive Limitations Section

In accordance with scientific rigor, critical limitations must be explicitly recognized:

1. **Granger Causality Reflects Temporal Precedence, Not Physical Causation:** Granger non-causality confirms that past network telemetry carries no incremental forecasting power over machine throughput. It does not prove that an extreme physical network disconnection would fail to halt a machine. Unmeasured physical variables—such as cutting-tool wear, mechanical vibration resonance, ambient thermal drift, or raw material metallurgic hardness—were unobserved in network logs and represent true latent drivers.
2. **Synthetic Target Generation & Near-Chance Model Discrimination:** The audited deterministic relationship between speed, error rate, and `Efficiency_Status` required strictly excluding outcome variables from Model B. However, all models (including Model B) exhibit ROC-AUC values of $\sim 0.50$ (near chance). Model B's diagnostic ceiling (~$0.32\text{--}0.33$ Macro-F1) does not represent genuine physical sensor root-cause predictability; instead, models fail to beat random guessing.
3. **Telemetry Arrival Mechanics:** Individual machine telemetry arrives via a stochastic Geometric polling process ($\text{median} = 34.0\text{ min}$, $\text{mean} = 49.3\text{ min}$) rather than a high-frequency millisecond clock. Consequently, predictive early warnings apply to macro operational shifts over $30$-minute horizons rather than sub-second real-time servo control.
4. **Simulator Nature:** The Network Scenario Simulator deployed in the operational dashboard represents statistical model projections under hypothetical feature configurations, **not guaranteed counterfactual interventions**.
5. **Pragmatic Spatial Partitioning of a Continuum:** The four network clusters identified via K-Means represent a pragmatic mathematical partition of a continuous uniform feature space, not naturally occurring physical regimes.
6. **Synthetic & Independent Variable Properties:** Kolmogorov-Smirnov goodness-of-fit tests against continuous uniform distributions confirm that key telemetry variables (`Network_Latency_ms`, `Packet_Loss_%`, `Production_Speed_units_per_hr`, `Error_Rate_%`, `Quality_Control_Defect_Rate_%`) follow independent uniform distributions with near-zero pairwise correlations ($|r| < 0.008$). The dataset appears synthetic and independently generated, meaning observed null results reflect properties of this synthetic space rather than real-world factory dynamics.
7. **Exploratory Status of Breakpoint Findings:** The candidate active-mode breakpoint at $20.7\text{ ms}$ does not survive sup-F permutation testing ($p = 0.1640$, FDR $q = 0.5100$) and must be treated as strictly exploratory.

---

## 10. Strategic Recommendations & Conclusion

### 10.1 Key Empirical Takeaway
Within this dataset, the network telemetry operates within mechanical process tolerance. Variations in network performance do not constitute a major driver of operational efficiency loss or scrap generation.

### 10.2 Four Concrete Operational Action Items
1. **Treat Active-Mode Latency Candidate Threshold at $20.0\text{ ms}$ as Exploratory:** Plant network engineers should treat the $20.0\text{ ms}$ active-mode threshold as an exploratory operating guideline rather than an established physical limit, given that it does not survive multiple-testing permutation control.
2. **Conditionally Re-Evaluate Capital Expenditure on Sub-Millisecond Over-Engineering (Requires Validation on Real Data):** Capital expenditure proposals to upgrade wireless infrastructure to achieve sub-$5\text{ ms}$ latency should be conditionally deferred. While this dataset reveals zero detectable throughput gain from sub-$13\text{ ms}$ latency, the recommendation to freeze capex requires prior validation on real factory telemetry with sub-second resolution and physical process telemetry.
3. **Deploy Conservative Alerting Policies:** Production early-warning systems should implement balanced Random Forest architectures that suppress false alarms. Aggressive alerting policies (such as unconstrained LightGBM) induce a $97\%$ false-positive rate, destroying operator credibility.
4. **Redirect Diagnostic Budgets Toward Mechanical and Tool Wear Sensors:** Because throughput variation is decoupled from network telemetry in this dataset, diagnostic monitoring investments should be channeled into high-frequency spindle vibration, motor torque, and thermal dissipation monitoring.

---

## References

1. 3GPP. *Service requirements for cyber-physical control applications in vertical domains (Release 18)*. 3GPP TS 22.104 V18.4.0, Technical Specification Group Services and System Aspects, 2024.
2. ITU-R. *Framework and overall objectives of the future development of IMT for 2030 and beyond*. Recommendation ITU-R M.2160-0, International Telecommunication Union, Nov. 2023.
3. Benjamini, Y., and Hochberg, Y. *Controlling the false discovery rate: a practical and powerful approach to multiple testing*. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300, 1995.
4. Granger, C. W. J. *Investigating causal relations by econometric models and cross-spectral methods*. Econometrica, 37(3):424–438, 1969.
5. Lundberg, S. M., and Lee, S.-I. *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS), 30:4765–4774, 2017.
6. Rousseeuw, P. J. *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20:53–65, 1987.
7. Davies, D. L., and Bouldin, D. W. *A cluster separation measure*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2):224–227, 1979.
8. Thales Group. *Industrial Cyber-Physical Manufacturing Testbed Telemetry*. Internal Research Dataset, 2025.
