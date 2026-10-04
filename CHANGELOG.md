# Changelog

All notable changes, analytical milestones, and releases for the **6G Industrial Network Telemetry & Manufacturing Efficiency Platform** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-04

### Initial Production Release & Verified Deployment

#### Step 0: Repository Setup & Standards Grounding
- Initialized modular architecture (`data/`, `src/`, `models/`, `app/`, `reports/`, `docs/`).
- Documented 3GPP TS 22.104 Release 18 and ITU-R IMT-2030 latency and reliability standards in `docs/latency_benchmarks.md`.
- Established `docs/assumptions.md` tracking all empirical parameters, hypotheses, and methodological boundary conditions.

#### Step 1: Telemetry Ingestion, Arrival Mechanics & Label Audit
- Implemented `src/data_loader.py` with unified datetime parsing across 100,000 continuous records for 50 industrial machines.
- Discovered and audited per-machine Geometric polling mechanism ($p = 0.02$, median arrival 34.0 min, mean 49.3 min).
- Identified deterministic label derivation rule ($99.998\%$ audit match): `Low` if Error Rate $> 5\%$ or Speed $\le 200$; `High` if Speed $> 400$ and Error Rate $\le 2\%$; else `Medium`.
- Enforced circularity gating: strictly excluded contemporaneous speed and error rate from diagnostic modeling.

#### Step 2: Unsupervised Network Profiling & Continuous Risk Indexing
- Implemented `src/network_profiling.py` evaluating cluster counts $k \in [2, 6]$.
- Empirically selected $k = 4$ based on global maximum Silhouette Score ($0.4061$) and minimum Davies-Bouldin Index ($0.7687$).
- Mapped clusters into 4 ordered operational risk tiers: Tier 0 (Optimal, -0.863), Tier 1 (Loss-Degraded, -0.004), Tier 2 (Latency-Degraded, +0.011), and Tier 3 (Critical, +0.866).
- Formulated continuous Network Risk Index ($\text{NRI} = 0.5 \cdot z_{\text{lat}} + 0.5 \cdot z_{\text{loss}}$).

#### Step 3: Temporal Causal Evidence & Breakpoint Analysis
- Implemented `src/causal_analysis.py` executing 450 machine- and mode-stratified VAR Granger causality tests.
- Reconciled core headline finding: **Zero of 450 tests survived Benjamini-Hochberg FDR control ($q < 0.05$)**, proving zero forward predictive precedence.
- Fitted continuous segmented piecewise linear regression, uncovering a statistically significant breakpoint at **$20.7\text{ ms}$ exclusively in `Active` production mode** ($F = 4.31, p = 0.038$, slope change $\Delta \beta = -0.3063\text{ units/hr per ms}$).
- Disclosed pooled cross-facility threshold instability (nominal 41.2 ms with wide bootstrap CI $[6.0, 45.2]\text{ ms}$).
- Verified quality invariance to packet loss (flat defect curve $\sim 5.01\%$, impact ratio $1.0012$).

#### Step 4: Latency & Packet-Loss Impact Diagnostics
- Implemented `src/diagnostics.py` evaluating mode-stratified correlations and testing for Simpson's paradox.
- Confirmed near-zero linear correlations ($|r| < 0.015$) across all operating modes.

#### Step 5: Leakage-Safe Predictive Modeling & Honest Evaluation
- Implemented `src/modeling.py` with physical-time lookahead targets (Horizon B: $H = 30\text{ min} \pm 10\text{ min}$, $N = 23,587$).
- Evaluated full baseline ladder: Dummy Majority, Class-Weighted Logistic Regression, Balanced Random Forest, Balanced LightGBM across chronological and unseen-machine splits.
- Reported full confusion matrices and established crucial scientific finding: no model substantially outperforms the majority baseline ($78.42\%$ accuracy).
- Reframed Random Forest selection rationale: chosen strictly because it avoids false alarms (capturing only 4 of 131 High events) rather than claiming artificial predictive victory; documented LightGBM's $97.05\%$ false alarm trade-off.
- Generated TreeSHAP explainability plots for both Model A and Model B.

#### Step 6: Operational KPIs & Economic Instability Modeling
- Implemented `src/kpi.py` computing Network Risk Index, Latency Sensitivity Score, Packet Loss Impact Ratio, Tolerance Thresholds, and multi-scenario Cost of Instability.
- Quantified empirical throughput gap of $3.084\text{ units/hr per machine}$ between Tier 0 and Tier 3, translating to a fleet loss of $\$3,854.96/\text{hr}$ under base contribution margin ($\$25.00/\text{unit}$).

#### Step 7: Production Streamlit Dashboard
- Implemented `app/dashboard.py` featuring 4 core analytical modules + 6G Network Scenario Simulator.
- Incorporated real-time feature recomputation, mandatory methodological disclaimers, and financial impact calculators.

#### Step 8: Academic Manuscript & Executive Briefing
- Authored comprehensive academic research paper in `reports/research_paper.md` with full empirical proofs, literature citations, and limitations disclosures.
- Authored 1-page financial briefing in `reports/executive_summary.md` translating findings into 4 concrete management recommendations.

#### Step 9: Cloud Deployment & Open-Source Release
- Pushed full repository to GitHub at `https://github.com/Atharva110409/smart-factory-6g-network`.
- Deployed live production dashboard to Streamlit Community Cloud at `https://atharva110409-smart-factory-6g-network-appdashboard-p9hb2s.streamlit.app/`.
- Conducted automated incognito end-to-end verification and captured full UI screenshots.
