"""
app/dashboard.py
Interactive Streamlit Dashboard for 6G Network Performance Impact on Smart Manufacturing.
Features 4 required core diagnostic modules + 6G Network Scenario Simulator.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.data_loader import load_raw
from src.network_profiling import cluster_network_quality
from src.kpi import compute_all_kpis


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="6G Smart Factory — Network & Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .disclaimer-box {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #fca5a5;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 1rem 0;
    }
    .badge-optimal { background-color: #065f46; color: #6ee7b7; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-loss { background-color: #92400e; color: #fde68a; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-lat { background-color: #1e40af; color: #bfdbfe; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-crit { background-color: #991b1b; color: #fca5a5; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def get_processed_data():
    raw_df = load_raw("data/raw/Thales_Group_Manufacturing.csv")
    clustered_df, meta = cluster_network_quality(raw_df, k=4)
    return clustered_df, meta


@st.cache_resource
def get_production_models():
    models_dict = {}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path_a = os.path.join(base_dir, "models", "early_warning_model_a.joblib")
    path_b = os.path.join(base_dir, "models", "diagnosis_model_b.joblib")
    if os.path.exists(path_a):
        models_dict["model_a"] = joblib.load(path_a)
    elif os.path.exists("models/early_warning_model_a.joblib"):
        models_dict["model_a"] = joblib.load("models/early_warning_model_a.joblib")
        
    if os.path.exists(path_b):
        models_dict["model_b"] = joblib.load(path_b)
    elif os.path.exists("models/diagnosis_model_b.joblib"):
        models_dict["model_b"] = joblib.load("models/diagnosis_model_b.joblib")
    return models_dict


df_full, cluster_meta = get_processed_data()
models = get_production_models()


# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=64)
st.sidebar.title("Telemetry Controls")

# Date Filter
min_date = df_full["Timestamp"].min().date()
max_date = df_full["Timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Observation Time Window",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Operation Mode Filter
all_modes = sorted(df_full["Operation_Mode"].unique())
selected_modes = st.sidebar.multiselect(
    "Operation Mode",
    options=all_modes,
    default=all_modes,
)

# Network Tier Filter
all_tiers = sorted(df_full["Network_Quality_Label"].unique())
selected_tiers = st.sidebar.multiselect(
    "Network Quality Tier",
    options=all_tiers,
    default=all_tiers,
)

# Efficiency Filter
all_eff = ["Low", "Medium", "High"]
selected_eff = st.sidebar.multiselect(
    "Efficiency Status",
    options=all_eff,
    default=all_eff,
)

# Machine ID Filter
machine_options = ["All Machines"] + [f"Machine {i}" for i in range(1, 51)]
selected_mach = st.sidebar.selectbox("Filter Machine Hardware", machine_options)

# Apply Filter Mask
mask = (
    (df_full["Timestamp"].dt.date >= date_range[0])
    & (df_full["Timestamp"].dt.date <= (date_range[1] if len(date_range) > 1 else date_range[0]))
    & (df_full["Operation_Mode"].isin(selected_modes))
    & (df_full["Network_Quality_Label"].isin(selected_tiers))
    & (df_full["Efficiency_Status"].isin(selected_eff))
)

if selected_mach != "All Machines":
    mach_id = int(selected_mach.split(" ")[1])
    mask = mask & (df_full["Machine_ID"] == mach_id)

filtered_df = df_full[mask].copy()


# -----------------------------------------------------------------------------
# MAIN HEADER & EXECUTIVE SCORECARDS
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Impact of 6G Network Performance on Smart Manufacturing</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Operational telemetry diagnostics, empirical threshold limits, and early-warning simulations across 50 smart factory assets.</div>", unsafe_allow_html=True)

# Top KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    avg_lat = filtered_df["Network_Latency_ms"].mean() if len(filtered_df) > 0 else 0
    st.metric(
        "Avg Network Latency",
        f"{avg_lat:.2f} ms",
        delta=f"{avg_lat - 20.7:.1f} ms vs Active Threshold",
        delta_color="inverse",
    )
with col2:
    avg_loss = filtered_df["Packet_Loss_%"].mean() if len(filtered_df) > 0 else 0
    st.metric(
        "Avg Packet Loss",
        f"{avg_loss:.2f}%",
        delta="Flat defect response",
    )
with col3:
    avg_risk = filtered_df["Network_Risk_Index"].mean() if len(filtered_df) > 0 else 0
    st.metric(
        "Composite Risk Index",
        f"{avg_risk:+.3f}",
        delta="Scale: [-1.7, +1.7]",
    )
with col4:
    active_bp = 20.7
    st.metric(
        "Active Tolerance Limit",
        f"{active_bp} ms",
        delta="Statistically Significant (p=0.038)",
    )
with col5:
    st.metric(
        "Base Cost of Instability",
        "$3,855 / hr",
        delta="50 Machines (Tier 0 vs 3)",
        delta_color="inverse",
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Network Performance Overview",
    "⚡ 2. Network vs. Manufacturing Efficiency",
    "🔍 3. Quality & Error Impact Panel",
    "📡 4. 6G Optimization & Tolerance Benchmarks",
    "🧪 5. Network Scenario Simulator",
])


# -----------------------------------------------------------------------------
# TAB 1: NETWORK PERFORMANCE OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Network Performance Telemetry & Risk Distribution")
    t1_col1, t1_col2 = st.columns([2, 1])

    with t1_col1:
        # Time-series trend (aggregated hourly for responsiveness)
        resampled = (
            filtered_df.set_index("Timestamp")[["Network_Latency_ms", "Packet_Loss_%", "Network_Risk_Index"]]
            .resample("6h")
            .mean()
            .reset_index()
        )

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=resampled["Timestamp"], y=resampled["Network_Latency_ms"],
            name="Latency (ms)", line=dict(color="#00d2ff", width=2)
        ))
        fig_ts.add_trace(go.Scatter(
            x=resampled["Timestamp"], y=resampled["Packet_Loss_%"] * 10,
            name="Packet Loss (x10 %)", line=dict(color="#ff9900", width=1.5, dash="dot")
        ))
        fig_ts.add_hline(y=20.7, line_dash="dash", line_color="red", annotation_text="Active Threshold (20.7 ms)")
        fig_ts.update_layout(
            title="6-Hour Rolling Network Telemetry Timeline",
            xaxis_title="Timeline",
            yaxis_title="Metric Level",
            template="plotly_dark",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with t1_col2:
        # Cluster representation breakdown
        tier_counts = filtered_df["Network_Quality_Label"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_pie = px.pie(
            tier_counts,
            values="Count",
            names="Tier",
            title="Telemetry Distribution by Network Quality Tier",
            color_discrete_sequence=["#10b981", "#f59e0b", "#3b82f6", "#ef4444"],
            template="plotly_dark",
            hole=0.45,
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Network Quality Cluster Profiles (Empirical k=4 K-Means)")
    st.dataframe(
        cluster_meta["cluster_profile"].rename(columns={
            "Network_Quality_Tier": "Tier ID",
            "count": "Total Readings",
            "mean_latency": "Mean Latency (ms)",
            "std_latency": "Std Latency",
            "mean_packet_loss": "Mean Packet Loss (%)",
            "std_packet_loss": "Std Packet Loss",
            "mean_risk_index": "Composite Risk Index",
        }),
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# TAB 2: NETWORK VS EFFICIENCY DASHBOARD
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Network Degradation vs Manufacturing Efficiency Distribution")
    t2_col1, t2_col2 = st.columns([1, 1])

    with t2_col1:
        # Efficiency breakdown by Network Tier
        eff_by_tier = pd.crosstab(
            filtered_df["Network_Quality_Label"],
            filtered_df["Efficiency_Status"],
            normalize="index"
        ) * 100
        eff_by_tier = eff_by_tier.reset_index()

        fig_bar = px.bar(
            eff_by_tier,
            x="Network_Quality_Label",
            y=["Low", "Medium", "High"],
            title="Efficiency Class Proportion (%) Across Network Quality Tiers",
            color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"},
            template="plotly_dark",
            barmode="stack",
        )
        fig_bar.update_layout(yaxis_title="Share of Telemetry (%)", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with t2_col2:
        # Latency vs Production Speed Scatter (subsampled for speed)
        scatter_sample = filtered_df.sample(min(len(filtered_df), 3000), random_state=42)
        fig_scat = px.scatter(
            scatter_sample,
            x="Network_Latency_ms",
            y="Production_Speed_units_per_hr",
            color="Operation_Mode",
            color_discrete_map={"Active": "#00d2ff", "Idle": "#f59e0b", "Maintenance": "#10b981"},
            title="Latency vs Production Speed (with 20.7 ms Active Breakpoint)",
            template="plotly_dark",
            opacity=0.4,
        )
        fig_scat.add_vline(x=20.7, line_dash="dash", line_color="#ef4444", annotation_text="Active Threshold (20.7 ms)")
        fig_scat.update_layout(xaxis_title="Network Latency (ms)", yaxis_title="Speed (units/hr)", height=400)
        st.plotly_chart(fig_scat, use_container_width=True)

    st.info(
        "💡 **Key Analytical Finding**: Production speed drops by **3.084 units/hr** between Tier 0 (Optimal: Latency 13.4 ms) "
        "and Tier 3 (Critical: Latency 37.7 ms). In active production mode, the deceleration inflection triggers precisely at **20.7 ms**."
    )


# -----------------------------------------------------------------------------
# TAB 3: QUALITY & ERROR IMPACT PANEL
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Packet Loss & Telemetry Error Impact Analysis")
    t3_col1, t3_col2 = st.columns(2)

    with t3_col1:
        # Packet Loss Binned Means vs Defect Rate
        bins_loss = np.linspace(0, 5, 21)
        df_loss_bin = filtered_df.copy()
        df_loss_bin["loss_bin"] = pd.cut(df_loss_bin["Packet_Loss_%"], bins=bins_loss)
        defect_curve = df_loss_bin.groupby("loss_bin", observed=True)["Quality_Control_Defect_Rate_%"].mean().reset_index()
        defect_curve["loss_center"] = 0.5 * (bins_loss[:-1] + bins_loss[1:])

        fig_defect = px.line(
            defect_curve,
            x="loss_center",
            y="Quality_Control_Defect_Rate_%",
            markers=True,
            title="Quality Control Defect Rate vs Packet Loss (%)",
            template="plotly_dark",
            color_discrete_sequence=["#ef4444"],
        )
        fig_defect.update_layout(
            xaxis_title="Packet Loss (%)",
            yaxis_title="Defect Rate (%)",
            yaxis_range=[4.0, 6.0],
            height=380,
        )
        st.plotly_chart(fig_defect, use_container_width=True)

    with t3_col2:
        # Error Rate vs Packet Loss
        error_curve = df_loss_bin.groupby("loss_bin", observed=True)["Error_Rate_%"].mean().reset_index()
        error_curve["loss_center"] = 0.5 * (bins_loss[:-1] + bins_loss[1:])

        fig_error = px.line(
            error_curve,
            x="loss_center",
            y="Error_Rate_%",
            markers=True,
            title="Operational Error Rate vs Packet Loss (%)",
            template="plotly_dark",
            color_discrete_sequence=["#f59e0b"],
        )
        fig_error.update_layout(
            xaxis_title="Packet Loss (%)",
            yaxis_title="Operational Error Rate (%)",
            yaxis_range=[6.5, 8.5],
            height=380,
        )
        st.plotly_chart(fig_error, use_container_width=True)

    st.markdown("""
    **Empirical Defect Stability Note**:
    - Across the entire historical range $[0.0\%, 5.0\%]$, packet loss exhibits **zero correlation** ($|r| < 0.01$) with defect or error rates.
    - Defect rates hover steadily between $4.98\%$ and $5.04\%$.
    - The 95th-percentile packet loss spike ratio ($1.0012$) confirms that random packet drops up to $5\%$ do not induce product scrap in this manufacturing architecture.
    """)


# -----------------------------------------------------------------------------
# TAB 4: 6G OPTIMIZATION & TOLERANCE BENCHMARKS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("3GPP TS 22.104 & ITU Industrial Communication Benchmarks")

    st.markdown("""
    | Industrial Use Case | 3GPP/ITU Latency Target | 3GPP Reliability Target | Factory Telemetry Regimes Observed | Compliance Status |
    | :--- | :---: | :---: | :---: | :---: |
    | **Tactile Interaction / Teleoperation** | **0.5 ms** | 99.999% (5-nines) | Min Latency = 1.00 ms | ⚠️ Exceeds Bound |
    | **General URLLC Target** | **~1.0 ms** | 99.999% (5-nines) | Tier 0 Mean = 13.41 ms | ⚠️ Exceeds Bound |
    | **Discrete Automation (Motion Control)** | **10.0 ms** | 99.99% (4-nines) | Active Inflection = **20.7 ms** | 🟡 Borderline Tolerance |
    | **High Voltage Power Distribution** | **5.0 ms** | 99.9999% (6-nines) | Min Latency = 1.00 ms | ⚠️ Exceeds Bound |
    | **Medium Voltage Electricity** | **25.0 ms** | 99.9% (3-nines) | Active Inflection = **20.7 ms** | ✅ Compliant (< 25 ms) |
    | **Process Automation (Remote Control)** | **50.0 ms** | 99.9% (3-nines) | Max Latency = 50.00 ms | ✅ Compliant (≤ 50 ms) |
    | **Process Automation (Plant Monitoring)**| **50.0 ms** | 99.9% (3-nines) | Fleet Mean = 25.56 ms | ✅ Fully Compliant |
    """)

    st.warning(
        "⚠️ **Methodological Guidance [MUST]**: Benchmark comparison caution:\n"
        "- **Latency**: 3GPP targets specify strict user-plane one-way latency. Factory telemetry (`Network_Latency_ms`) "
        "often captures transport-layer round-trip (RTT) or application poll-response intervals. Comparisons must remain qualitative.\n"
        "- **Packet Loss vs Reliability**: Do NOT equate packet loss % with service reliability %. 99.999% reliability means "
        "99.999% of packets arrive within latency deadline $T_{\\text{max}}$, whereas unadjusted telemetry drop rates exclude "
        "protocol layer retransmissions (HARQ) and redundant packet paths."
    )


# -----------------------------------------------------------------------------
# TAB 5: NETWORK SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🧪 6G Network Scenario Simulator")

    st.markdown("""
    Simulate hypothetical communication network states to predict manufacturing efficiency classes,
    composite risk indexes, and estimated economic costs per operating hour.
    """)

    st.markdown(
        "<div class='disclaimer-box'>⚠️ <b>Mandatory Methodological Notice:</b> "
        "Predictions are model-based scenarios, not experimentally established causal effects. "
        "As established in Step 3, network telemetry has limited predictive precedence over future efficiency states.</div>",
        unsafe_allow_html=True
    )

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown("#### Scenario Input Parameters")
        sim_lat = st.slider("Simulated Latency (ms)", min_value=1.0, max_value=50.0, value=15.0, step=0.5)
        sim_loss = st.slider("Simulated Packet Loss (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        sim_mode = st.selectbox("Operating State", ["Active", "Idle", "Maintenance"], index=0)
        sim_temp = st.slider("Machine Temperature (°C)", min_value=30.0, max_value=90.0, value=60.0, step=1.0)
        sim_vib = st.slider("Vibration (Hz)", min_value=0.1, max_value=5.0, value=2.5, step=0.1)
        sim_power = st.slider("Power Consumption (kW)", min_value=1.5, max_value=10.0, value=5.7, step=0.1)
        sim_pm_score = st.slider("Predictive Maintenance Health Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    with sim_col2:
        st.markdown("#### Scenario Output & Impact Evaluation")

        # Compute synthetic risk index using training scaler
        model_pack = models.get("model_a")
        if model_pack:
            scaler = model_pack["scaler"]
            km_model = model_pack["kmeans"]
            feature_cols = model_pack["features"]

            # Compute standardized network metrics
            scaled_net = scaler.transform([[sim_lat, sim_loss]])
            sim_risk = 0.5 * scaled_net[0, 0] + 0.5 * scaled_net[0, 1]
            sim_cluster = km_model.predict(scaled_net)[0]

            # Reconstructed feature vector
            row_dict = {
                "Network_Latency_ms": sim_lat,
                "Packet_Loss_%": sim_loss,
                "latency_lag_1": sim_lat,
                "latency_lag_2": sim_lat,
                "packet_loss_lag_1": sim_loss,
                "packet_loss_lag_2": sim_loss,
                "latency_roll_mean_3": sim_lat,
                "latency_roll_std_3": 0.0,
                "packet_loss_roll_mean_3": sim_loss,
                "Network_Risk_Index": sim_risk,
                "Network_Cluster": sim_cluster,
                "Temperature_C": sim_temp,
                "Vibration_Hz": sim_vib,
                "Power_Consumption_kW": sim_power,
                "Predictive_Maintenance_Score": sim_pm_score,
                "Mode_Active": 1.0 if sim_mode == "Active" else 0.0,
                "Mode_Idle": 1.0 if sim_mode == "Idle" else 0.0,
                "Mode_Maintenance": 1.0 if sim_mode == "Maintenance" else 0.0,
            }
            # Fill dummy machines
            for c in feature_cols:
                if c.startswith("Mach_"):
                    row_dict[c] = 0.0

            input_df = pd.DataFrame([row_dict]).reindex(columns=feature_cols, fill_value=0.0)

            # Predict probabilities
            model = model_pack["model"]
            probs = model.predict_proba(input_df)[0]
            pred_class_idx = np.argmax(probs)
            class_names = ["Low", "Medium", "High"]
            predicted_class = class_names[pred_class_idx]

            # Render scenario cards
            risk_color = "#10b981" if sim_risk < -0.4 else ("#f59e0b" if sim_risk < 0.4 else "#ef4444")
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {risk_color};'>
                <div style='font-size: 0.9rem; color: #94a3b8;'>Simulated Network Risk Index</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: {risk_color};'>{sim_risk:+.3f}</div>
                <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>
                    Active Threshold Status: <b>{'🟢 Compliant (<= 20.7 ms)' if sim_lat <= 20.7 else '🔴 Exceeds Limit (> 20.7 ms)'}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Predicted class & probabilities
            st.markdown(f"**Predicted 30-Min Efficiency Warning: `{predicted_class}`**")
            prob_df = pd.DataFrame({
                "Efficiency Class": ["Low", "Medium", "High"],
                "Probability": [probs[0], probs[1], probs[2]],
            })
            fig_prob = px.bar(
                prob_df,
                x="Efficiency Class",
                y="Probability",
                color="Efficiency Class",
                color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"},
                template="plotly_dark",
            )
            fig_prob.update_layout(yaxis_range=[0, 1], height=240, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_prob, use_container_width=True)

            # Economic Cost Translation
            cm_assumed = 25.0
            throughput_loss = 3.084 if sim_lat > 20.7 else 0.0
            hourly_cost_machine = throughput_loss * cm_assumed
            hourly_cost_fleet = hourly_cost_machine * 50

            st.markdown(f"""
            **Estimated Financial Impact (@ $25.00 Contribution Margin)**:
            - **Single Machine Loss**: `${hourly_cost_machine:.2f} / hour`
            - **Fleet-Wide (50 Machines)**: `${hourly_cost_fleet:,.2f} / hour`
            """)
        else:
            st.warning("Production model artifacts not found in models/. Please execute `python -m src.modeling`.")
