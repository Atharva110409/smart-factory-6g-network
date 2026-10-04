# Step 4 Diagnostic Note: Latency & Packet-Loss Operational Impact

## 1. Linear vs. Piecewise Threshold Dynamics
- **Latency vs. Production Speed**:
  - Across the entire manufacturing fleet, linear Pearson correlations between contemporaneous latency and production speed are near zero ($r = -0.001, p = 0.74$).
  - However, segmented regression reveals that in **`Active` Operation Mode**, throughput is piece-wise non-linear with a structural breakpoint at **20.7 ms** ($F = 4.31, p = 0.038$).
  - Pre-threshold slope: $\beta_1 = +0.212$ units/hr per ms.
  - Post-threshold slope: $\beta_1 + \beta_2 = -0.094$ units/hr per ms.
  - In `Idle` and `Maintenance` modes, machines exhibit flat or non-significant breakpoint responses ($F = 1.28$ and $F = 3.24$), demonstrating that latency sensitivity is predominantly manifested during high active mechanical load.

## 2. Simpson's Paradox Verification
- Stratification across `Active`, `Idle`, and `Maintenance` modes confirms that the absence of aggregate linear correlation is **not** an artifact of Simpson's paradox (where subgroups have strong opposite correlations canceling out).
- Rather, the data exhibits uniform flat baselines across all modes with localized nonlinear hinge behavior exclusively in active production.

## 3. Packet Loss Impact & Lack of Cliff Threshold
- Packet loss ranges uniformly from 0.0% to 5.0%.
- Both `Quality_Control_Defect_Rate_%` (mean ~5.01%) and `Error_Rate_%` (mean ~7.50%) show zero correlation with packet loss ($|r| < 0.01$ across all modes).
- Quality and defect rates remain essentially constant across packet loss deciles, showing no threshold cliff within the 0–5% operational range.
