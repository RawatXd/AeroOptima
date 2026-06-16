import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from run_pipeline import score_flights, optimize_gates, naive_gate_assignment

st.set_page_config(
    page_title="AeroOptima",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom styling ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6B7B;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FB;
        border: 1px solid #E5E8EC;
        border-radius: 10px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1E3A5F;
    }
    .comparison-box {
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .optimized-box {
        background-color: #EAF4EC;
        border-left: 5px solid #2E7D4F;
    }
    .naive-box {
        background-color: #FBEAEA;
        border-left: 5px solid #B23B3B;
    }
    section[data-testid="stSidebar"] {
        background-color: #F4F6F8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="main-header">✈️ AeroOptima</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Delay-Aware Gate Optimization — ML-driven risk prediction '
    'combined with Integer Programming for operational gate assignment</p>',
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Scenario Settings")
    st.caption("Configure the flight window to analyze")

    origin = st.selectbox("Origin Airport", ["ATL"], index=0)
    date = st.text_input("Flight Date", value="2023-01-09")

    col_a, col_b = st.columns(2)
    with col_a:
        dep_hour_min = st.number_input("Start Hour", 0, 23, 6)
    with col_b:
        dep_hour_max = st.number_input("End Hour", 1, 24, 12)

    st.markdown("---")
    run_button = st.button("▶ Run Pipeline", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption(
        "**About this system**\n\n"
        "An XGBoost model predicts delay probability per flight using schedule-based "
        "features. An Integer Program then assigns gates under turnaround-time "
        "constraints, prioritizing high-risk flights toward priority gate slots."
    )

# ---------------- Main content ----------------
if run_button:
    progress = st.progress(0, text="Scoring flights with XGBoost model...")
    flights = score_flights(origin, date, dep_hour_min, dep_hour_max)
    progress.progress(33, text="Running risk-aware gate optimization...")

    if flights.empty:
        progress.empty()
        st.warning("No flights found for this date/window. Try different settings.")
    else:
        result = optimize_gates(flights)
        progress.progress(66, text="Running naive baseline for comparison...")
        naive = naive_gate_assignment(flights)
        progress.progress(100, text="Done")
        progress.empty()

        # --- Top KPIs ---
        st.markdown("### Operational Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flights Processed", len(flights))
        k2.metric("Gates Required", result['num_gates'])
        k3.metric("Peak Concurrent Demand", result['max_concurrent'])
        k4.metric("Solver Status", "✓ Optimal" if result['status'] == 'Optimal' else result['status'])

        st.markdown("")

        # --- Before / After comparison ---
        st.markdown("### Risk-Awareness Comparison")
        st.caption(
            "Correlation between predicted delay probability and assigned gate index. "
            "Strong negative correlation means high-risk flights are steered to priority gates."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="comparison-box optimized-box">
                <b>Optimized (Risk-Aware)</b><br>
                <span style="font-size:1.8rem; font-weight:700; color:#2E7D4F;">
                {result['correlation']:.4f}</span><br>
                <span style="color:#4A5A52;">High-risk flights prioritized to lower-index gates</span>
                </div>""", unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""<div class="comparison-box naive-box">
                <b>Naive (First-Come-First-Served)</b><br>
                <span style="font-size:1.8rem; font-weight:700; color:#B23B3B;">
                {naive['correlation']:.4f}</span><br>
                <span style="color:#5A4A4A;">No relationship between risk and gate assignment</span>
                </div>""", unsafe_allow_html=True
            )

        st.markdown("")

        # --- Delay probability distribution ---
        st.markdown("### Delay Risk Distribution")
        dist_col1, dist_col2 = st.columns([2, 1])
        with dist_col1:
            st.bar_chart(
                flights['DELAY_PROBABILITY'].sort_values().reset_index(drop=True),
                height=250
            )
        with dist_col2:
            st.markdown("**Risk Summary**")
            high_risk = (flights['DELAY_PROBABILITY'] > 0.5).sum()
            st.write(f"High risk (>50%): **{high_risk}** flights")
            st.write(f"Average risk: **{flights['DELAY_PROBABILITY'].mean():.1%}**")
            st.write(f"Peak risk: **{flights['DELAY_PROBABILITY'].max():.1%}**")

        st.markdown("")

        # --- Flight-level detail table ---
        st.markdown("### Flight-Level Assignment Detail")

        display_df = flights[['FL_DATE', 'OP_UNIQUE_CARRIER', 'CRS_DEP_TIME',
                                'CRS_ARR_TIME', 'DELAY_PROBABILITY']].copy()
        display_df['DELAY_PROBABILITY'] = display_df['DELAY_PROBABILITY'].round(3)
        display_df = display_df.merge(
            result['assignment'][['flight_idx', 'gate']].rename(columns={'gate': 'Optimized Gate'}),
            left_index=True, right_on='flight_idx'
        )
        display_df = display_df.merge(
            naive['assignment'][['flight_idx', 'gate']].rename(columns={'gate': 'Naive Gate'}),
            on='flight_idx'
        )
        display_df = display_df.drop(columns=['flight_idx']).sort_values(
            'DELAY_PROBABILITY', ascending=False
        )
        display_df.columns = ['Date', 'Carrier', 'Sched. Departure', 'Sched. Arrival',
                                'Delay Probability', 'Optimized Gate', 'Naive Gate']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Delay Probability": st.column_config.ProgressColumn(
                    "Delay Probability", min_value=0, max_value=1, format="%.2f"
                )
            }
        )
else:
    st.markdown("")
    st.info("👈 Set your scenario in the sidebar and click **Run Pipeline** to begin.")

    with st.expander("How this system works"):
        st.markdown("""
        **1. Delay Prediction** — An XGBoost classifier trained on BTS On-Time 
        Performance data predicts delay probability per flight, using schedule-based 
        features (departure hour, day of week, route history, carrier history).
        
        **2. Gate Optimization** — A time-indexed Integer Program assigns flights to 
        gates such that no two flights occupy the same gate within their required 
        turnaround window. The objective steers higher-risk flights toward 
        lower-index ("priority") gates.
        
        **3. Baseline Comparison** — A naive first-come-first-served assignment is 
        computed for the same flights, with no awareness of delay risk, to 
        demonstrate the impact of the optimization layer.
        
        *Note: gate count and turnaround buffers are derived from measured peak 
        concurrent demand in the data, not arbitrary assumptions.*
        """)