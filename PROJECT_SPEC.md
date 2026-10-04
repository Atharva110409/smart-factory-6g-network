# Smart Factory 6G Project — Implementation Spec (v3)

Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories.
This file is written as instructions for an implementing agent. Follow it in order.
Stop and report after Step 1, Step 3, and Step 5.

Priority tags used throughout: **[MUST]** = correctness issue, never skip. **[SHOULD]** = expected
good practice, skip only if truly out of time. **[OPT]** = polish, cut first.

## Storyline (use this framing in README, paper, dashboard, video)

```
RAW DATA
  -> DATA QUALITY + LABEL AUDIT
  -> MACHINE CONDITION + NETWORK CONDITION
  -> OPERATION MODE / CONFOUNDERS
  -> TEMPORAL PRECEDENCE (Granger)
  -> BREAKPOINT DETECTION (latency, packet loss)
  -> NETWORK TOLERANCE THRESHOLDS
  -> EARLY WARNING MODEL + DIAGNOSIS MODEL
  -> EXPLAINABLE AI (SHAP)
  -> RISK + COST KPIs
  -> NETWORK SCENARIO SIMULATOR -> STREAMLIT
```

One-line version: network degradation -> temporal evidence -> breakpoint -> early warning ->
explainable diagnosis -> operational risk.

---

## Step 0 — Repo Setup [MUST]

```
smart-factory-6g-project/
  data/raw/            (CSV, untouched)
  data/processed/       (cleaned, feature-engineered data)
  src/                  (reusable pipeline code — not just notebooks)
  app/                  (Streamlit dashboard)
  models/               (saved trained models)
  notebooks/            (exploratory work)
  reports/              (research paper + executive summary)
  docs/                 (methodology notes, latency_benchmarks.md, assumptions.md)
```

- Keep `docs/latency_benchmarks.md` (3GPP/ITU table, below) so it's citable in the paper.
- Create `docs/assumptions.md` now. Log every assumption here as you make it: cost inputs, k
  choice, horizon H, lag choice, split scheme, Machine_ID decision. It feeds the paper's
  Limitations section.

## Step 1 — Data Loading & Audit [MUST] — STOP AND REPORT AFTER THIS STEP

- Combine `Date` + `Time` into one `Timestamp` column (`pd.to_datetime`).
- Sort rows by `Machine_ID`, then `Timestamp`.
- Audit shape, missing values, duplicate rows, unparsed timestamps.
- Check class balance of `Efficiency_Status` (expect strong imbalance, e.g. ~78% Low, ~3% High).
- **[MUST] Audit sampling regularity per machine.** Record the median sampling interval per
  machine in real time units (seconds/minutes). Flag irregular gaps. This interval is required
  for Step 5's horizon conversion.
- **[MUST] Profile every key variable by `Operation_Mode`** from the start (counts, latency,
  packet loss, efficiency mix). Operation mode is a confounder that must be checked in every
  later step, not just Step 3.
- **[MUST] Audit how `Efficiency_Status` was generated.** Determine whether it was independently
  observed, or computed from `Production_Speed_units_per_hr`, `Error_Rate_%`, or
  `Quality_Control_Defect_Rate_%` (check any data documentation, and test whether a simple rule
  on those columns reproduces the label well). Record the finding in `docs/assumptions.md`.
  **This decision gates Model B's feature list in Step 5** — if the label is derived from a
  variable, that variable must be excluded from Model B, or Model B must be reframed as a
  label-reconstruction check rather than a diagnosis model.

**Deliverable:** `src/data_loader.py` with `load_raw()` and `audit()`, a printed audit report,
and a label-generation note in `docs/assumptions.md`.

**Report back:** class balance, sampling interval(s), operation-mode profile, and the
label-generation finding, before continuing to Step 2.

## Step 2 — Network Performance Profiling (Clustering)

- **[MUST]** Standardize `Network_Latency_ms` and `Packet_Loss_%` (z-score) before clustering.
  If packet loss is heavily skewed with many zeros, also try a log transform or RobustScaler and
  compare.
- **[MUST] Do not assume k=3.** Fit K-Means for k=2..6 and compare silhouette score (plus
  inertia/elbow, optionally Davies-Bouldin). Choose k from the evidence — it may be 3, or it may
  not be. State the choice and justification in the paper.
- **[MUST]** Rank clusters by **mean composite risk** (not latency alone) to assign ordered
  labels (Low/Medium/High or similar). Cluster IDs from K-Means are arbitrary and must be
  re-mapped after fitting.
- **[MUST]** Sanity-check per-cluster mean latency AND mean packet loss — both should move in a
  sensible direction across ordered groups. If not, investigate before trusting the labels.
- **[SHOULD]** Cross-tab clusters against `Machine_ID` and `Operation_Mode` to confirm clusters
  aren't just a proxy for one machine or mode.
- **[MUST]** Fit the scaler and K-Means on training data only in any downstream modeling step;
  apply with transform/predict on test data.

**Metric rename:** "Stability Index" → **Network Risk Index** (higher = worse network).
`Network_Risk_Index = w1 * z(latency) + w2 * z(packet_loss)`, default `w1 = w2 = 0.5`. If you use
other weights, justify them and report a sensitivity check. **[SHOULD]**

**Deliverable:** `src/network_profiling.py` with `cluster_network_quality()`, `select_k()`, and
`network_risk_index()`, validated by per-cluster feature means and a silhouette plot.

## Step 3 — Temporal Causal & Breakpoint Analysis — STOP AND REPORT AFTER THIS STEP

**Wording (MUST):** Granger causality tests *predictive precedence* — whether past latency
improves the forecast of efficiency beyond efficiency's own past. It does **not** prove physical
causation. Use phrasing like "temporal causal evidence" or "latency Granger-causes efficiency in
the statistical sense." List unmeasured confounders in the paper's Limitations section.

- **[MUST]** Run Granger tests per machine, **stratified by `Operation_Mode`**, so mode cannot
  confound the result.
- **[MUST]** Granger needs a numeric, roughly stationary series. `Efficiency_Status` is
  categorical — either encode it ordinally, or (preferred) test on a continuous proxy
  (`Production_Speed_units_per_hr`) with status as a secondary check. Run ADF stationarity tests
  and difference the series if needed.
- **[SHOULD]** Testing many (machine × mode × lag) combinations creates many p-values. Apply a
  multiple-testing correction (e.g. Benjamini-Hochberg) and report adjusted p-values.
- **[MUST]** Latency breakpoint: segmented regression or the `ruptures` library on latency vs
  `Production_Speed_units_per_hr` to find where the slope changes.
- **[SHOULD]** Packet-loss breakpoint: look for a threshold against
  `Quality_Control_Defect_Rate_%` (and `Error_Rate_%`), since packet loss is expected to hit
  quality more than speed. Packet loss is often mostly zeros with rare spikes — first check
  there's enough variation. If not, report only the latency breakpoint and state why.
- **[SHOULD]** Breakpoint stability: re-run detection **per machine and per operation mode**,
  tabulate results. Agreement = robust finding; disagreement is itself a finding (e.g. high-load
  mode tolerates less latency).
- **[OPT]** Bootstrap 95% CI on each breakpoint. Because readings are serially correlated, use a
  block bootstrap (resample contiguous time blocks or whole machines), not individual rows.
  Report as "X ms (95% CI a–b)".

**Deliverable:** `src/causal_analysis.py` with `granger_causality_per_machine()`,
`find_latency_breakpoint()`, `breakpoint_by_group()`, and (if time) `bootstrap_breakpoint_ci()`.
Output: adjusted p-value table per machine/mode/lag, breakpoint table by group, and headline
threshold(s) with intervals if computed.

### Benchmark comparison caution [MUST]

Reference table for `docs/latency_benchmarks.md`:

| Use case | End-to-end latency | Reliability |
|---|---|---|
| Tactile interaction | 0.5 ms | 99.999% |
| General URLLC target (ITU/3GPP) | ~1 ms | 99.999% |
| Discrete automation (motion control) | 10 ms | 99.99% |
| Electricity distribution (high voltage) | 5 ms | 99.9999% |
| Intelligent transport (infra backhaul) | 10 ms | 99.9999% |
| Electricity distribution (medium voltage) | 25 ms | 99.9% |
| Process automation (remote control) | 50 ms | 99.9% |
| Process automation (monitoring) | 50 ms | 99.9% |

Source: 3GPP TS 22.104, summarized in arXiv:2508.20205; supporting: arXiv:2106.11825,
arXiv:2509.10617, arXiv:2510.08080.

- **Latency:** before comparing, confirm `Network_Latency_ms` is measured comparably to the
  benchmark's end-to-end latency — one-way vs round-trip, measurement point (radio link, network,
  or application layer), and mean vs worst-case percentile. If definitions differ, compare only
  qualitatively and say so.
- **Packet loss:** **do not equate packet-loss % with service reliability %.** Reliability in
  3GPP-style requirements depends on service definition, time interval, packet/application
  semantics, and measurement methodology — 99.999% reliability is not automatically 0.001% packet
  loss. Compare the observed packet-loss regime qualitatively, and only quantitatively where the
  standard's definitions are actually compatible.

## Step 4 — Latency & Packet-Loss Impact Diagnostics

- Plot `Production_Speed_units_per_hr` vs latency, colored by `Operation_Mode`.
- Correlate `Packet_Loss_%` with `Error_Rate_%` and `Quality_Control_Defect_Rate_%`.
- Test for threshold effects instead of assuming linearity (binned means, piecewise vs linear
  fit).
- **[SHOULD]** Report every correlation and threshold **within each `Operation_Mode`**, not just
  pooled — a pooled correlation can vanish or reverse inside each mode (Simpson's paradox).

**Deliverable:** diagnostics notebook or `src` additions with plots, correlations, and a written
threshold note.

## Step 5 — Modeling: Early Warning + Diagnosis — STOP AND REPORT AFTER THIS STEP

### Target definition: predict the future [MUST]

Features at time `t` predict `Efficiency_Status` at time `t + H`. Example: with 1-minute data and
`H = 5` minutes, network conditions at 10:00 predict efficiency at 10:05.

```python
# Conceptual implementation — rows must already be sorted by Machine_ID, Timestamp
future_target = df.groupby("Machine_ID")["Efficiency_Status"].shift(-N)
```

- Define the horizon `H` in **time units** (e.g. 5 and 10 minutes), not just row counts. Convert
  to `N` rows per machine using the sampling interval from Step 1. Report results for more than
  one horizon.
- Drop rows where the shifted target is missing (end of each machine's series) and rows where the
  real time gap to the target is not close to `H` (irregular sampling).

### Two models, different feature sets [MUST]

| | Model A: Early Warning | Model B: Diagnosis |
|---|---|---|
| Question | What efficiency class is coming in H minutes? | What efficiency class is the machine in now, and why? |
| Target | `Efficiency_Status` at `t + H` (per machine) | Current `Efficiency_Status` |
| Allowed features | Network metrics (current + lagged), Network Risk Index, cluster label, environment (e.g. temperature), `Machine_ID`, `Operation_Mode`. Efficiency history only if clearly past-only. | Everything in A **plus** current outcome-side variables (production speed, error rate, defect rate) — **only if the Step 1 label audit confirms `Efficiency_Status` is not directly derived from them** |
| Forbidden | Current/future production speed, error rate, defect rate; anything measured after prediction time | Cluster labels or the Risk Index used to define the target |
| Use | Operational alerts, dashboard simulator | Root-cause explanation, SHAP analysis |

### Validation rules [MUST]

- Split **chronologically** (train on earlier time, test on later) or by machine group — never a
  random row split (leaks neighbouring readings across train/test).
- Fit scaler, K-Means, and any resampler on the training set only.
- Class imbalance: start with `class_weight='balanced'`. Use SMOTE/resampling only inside
  training folds.
- Evaluate with per-class precision/recall, macro-F1, and PR-AUC. **Never rely on accuracy alone.**
- **[SHOULD]** Report metrics stratified by `Operation_Mode`.

### Machine_ID decision [SHOULD]

- **Same known machines** (deploy to warn on the machines trained on): `Machine_ID` may be kept;
  use a chronological split.
- **New/unseen machines**: exclude `Machine_ID`; validate with a machine-grouped split
  (`GroupKFold` or leave-machines-out) so the model can't memorize machine-specific behavior.
- Preferably run both and report the gap — useful as an industry insight.

### Baseline ladder [SHOULD]

Majority-class baseline → Logistic Regression → Random Forest → XGBoost/LightGBM. Keep the more
complex model only if it clearly beats the simpler one on macro-F1 and High/Low-class recall.
Show the comparison table in the paper.

### Explainability

SHAP on the best model: global importance plus individual explanations for specific "Low
efficiency" predictions. For Model A, a SHAP summary of network features driving early warnings
is key supporting evidence.

**Deliverable:** `src/modeling.py` with `build_features()`, `make_future_target()`,
`time_split()`, `train_baselines()`, `train_early_warning_model()`, `train_diagnosis_model()`,
`explain_with_shap()`; two saved models in `models/`.

**Report back:** which features Model B ended up allowed to use (per the label audit),
chronological vs machine-grouped results, and the baseline comparison table.

## Step 6 — KPI Computation [MUST]

| KPI | Formula | Notes |
|---|---|---|
| Network Risk Index | `w1*z(latency) + w2*z(packet_loss)` | From Step 2; higher = worse |
| Latency Sensitivity Score | `Δ(Production_Speed) / Δ(latency)` (units/hr per ms) in a window around the latency breakpoint. Optional: `Δ P(Low efficiency) / Δ latency` from a fitted model | `Efficiency_Status` is categorical — use a numeric outcome. Report per `Operation_Mode` |
| Packet Loss Impact Ratio | `defect_rate_during_spike / baseline_defect_rate` | Define "spike" explicitly (e.g. above 95th percentile) |
| Network Tolerance Thresholds (renamed from "Correlation Point") | Latency threshold X ms (latency vs production speed) and, if identifiable, packet-loss threshold Y% (packet loss vs defect rate) | Not a status-flip point. Give intervals if computed; if no packet-loss breakpoint exists, report latency only |
| Estimated Cost of Instability | See structure below | $/hour; assumptions logged in `docs/assumptions.md` |

**Cost structure [MUST]** (fixes a units problem in earlier drafts — defect *rate* isn't yet a
count of defective units):

```
Production loss cost/hr = lost units/hr × contribution margin per unit
Defect cost/hr = additional defective units/hr × cost per defective unit
  where additional defective units/hr = production units/hr × increase in defect rate (as a fraction)

Estimated Cost of Instability = Production loss cost/hr + Defect cost/hr
```

Use **contribution margin**, not selling price — the goal is economic loss, not revenue.
Cost inputs (contribution margin, cost per defective unit) are not in the dataset — state them as
explicit assumptions in `docs/assumptions.md` and show a low/base/high sensitivity range rather
than a single figure. **[SHOULD]**

**Deliverable:** `src/kpi.py` with all five KPI functions, computed once and recorded for the
report.

## Step 7 — Streamlit Dashboard

Build in two phases so scope never blocks a working v1.

**Phase 1 — required modules [MUST]:**
- Network Performance Overview — latency & packet-loss trends, Risk Index scorecards.
- Network vs Efficiency Dashboard — efficiency distribution by network quality, latency-efficiency
  scatter.
- Quality & Error Impact Panel — error rate vs packet loss, defect rate under varying network
  conditions.
- 6G Optimization Insights — latency tolerance benchmarks, packet-loss risk zones.
- Required filters: network quality, efficiency class, operation mode, time window.

**Network Scenario Simulator (renamed from "what-if") [SHOULD]:**
- Sliders for current latency, current packet loss, and operation mode; keep all other current
  machine context (e.g. temperature, machine) fixed.
- Recompute the Network Risk Index and cluster label from the slider values using the
  training-time scaler/model, and set lagged inputs consistently (e.g. assume the scenario held
  across the lookback window), so model input stays coherent.
- Display predicted class and cost impact.
- Show the label: **"Predictions are model-based scenarios, not experimentally established
  causal effects."**

**Phase 2 — expand to six pages [OPT]**, once Phase 1 is deployed: (1) Network Overview,
(2) Network vs Efficiency, (3) Quality and Error Impact, (4) Causal & Breakpoint Evidence
(p-value table, breakpoint by mode with CI), (5) Model Insights and Scenario Simulator,
(6) 6G Benchmarks and Recommendations. Adjust to match any reviewer brief.

**Deliverable:** `app/dashboard.py`, deployed (e.g. Streamlit Community Cloud) with a live URL.

## Step 8 — Research Paper & Executive Summary

**Paper must include:**
- EDA summary and audit findings.
- Methodology: k-selection evidence, clustering approach, Granger setup (stationarity handling,
  lags, multiple-testing correction), breakpoint method and bootstrap.
- Results: temporal-evidence findings per machine/mode, breakpoint table by group with intervals,
  baseline-vs-advanced comparison for Model A and Model B, stratified by operation mode.
- **[MUST] Limitations:** Granger ≠ physical causation; unmeasured confounders; cost assumptions;
  sampling regularity; how `Efficiency_Status` was labelled; benchmark definition mismatches;
  simulator outputs are scenarios, not causal effects; whether zones are natural clusters or a
  modeling choice.

**Executive summary:** one page, non-technical, cost-translated, 3–5 concrete recommendations
(e.g. "keep latency below X ms for machines in high-load mode"). Keep technical depth in the
paper, action items here.

## Step 9 — Deploy & Submit

- Push a clean GitHub repo with a README covering methodology and how to run everything. Reuse
  the storyline chain from page 2 in the README.
- Deploy the Streamlit dashboard; record the live URL.
- Record the feedback video — emphasize temporal-evidence, breakpoint-stability, and
  leakage-safe modeling.
- Fill in the submission form: GitHub repo, research paper, deployed project, feedback video
  links.

### Priority checklist if time is short

- **[MUST] before coding:** Granger wording fix; k-selection; leakage-safe Early Warning vs
  Diagnosis split with time-aware validation; correct Step 2 label mapping; audit of how
  `Efficiency_Status` was generated; future-target definition with a time horizon; unit-consistent
  KPI formulas; breakpoint definitions resolved; benchmark definitions checked before comparing;
  Limitations section.
- **[SHOULD]:** breakpoint per machine/mode; operation mode in every step; baseline ladder;
  Machine_ID decision; Scenario Simulator wording; packet-loss breakpoint; Risk Index rename;
  multiple-testing correction; cost sensitivity range.
- **[OPT]:** bootstrap CI on breakpoint; six-page dashboard.

---

*End of spec — work through Steps 0–9 in order. Stop for review after Steps 1, 3, and 5.*
