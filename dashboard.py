"""
Payment Failure Forecasting Dashboard
Optimus AI Labs Hackathon 2026 — Test Case 2

Run locally:    streamlit run dashboard.py
Deploy:         Push to GitHub → connect repo at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payment Failure Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Colour mapping ─────────────────────────────────────────────────────
RISK_COLOURS = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_predictions.csv')

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────
st.title("⚡ Payment Failure Forecasting")
st.markdown(
    "Real-time failure risk across payment channels, routes, and time windows. "
    "Powered by XGBoost trained on PaySim transaction data."
)
st.divider()

# ── KPI summary row ────────────────────────────────────────────────────
high_count   = (df['Failure Risk'] == 'High').sum()
medium_count = (df['Failure Risk'] == 'Medium').sum()
low_count    = (df['Failure Risk'] == 'Low').sum()
total        = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 High Risk Alerts",   f"{high_count}",
            f"{high_count/total*100:.0f}% of all route-windows")
col2.metric("🟠 Medium Risk",        f"{medium_count}",
            f"{medium_count/total*100:.0f}% of all route-windows")
col3.metric("🟢 Low Risk",           f"{low_count}",
            f"{low_count/total*100:.0f}% of all route-windows")
col4.metric("📊 Route-Windows Monitored", f"{total}")

st.divider()

# ── Sidebar filters ────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Failure Risk Level",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium']
)

route_filter = st.sidebar.multiselect(
    "Route",
    options=sorted(df['Route'].unique()),
    default=sorted(df['Route'].unique())
)

channel_filter = st.sidebar.multiselect(
    "Channel",
    options=sorted(df['Channel'].unique()),
    default=sorted(df['Channel'].unique())
)

window_filter = st.sidebar.multiselect(
    "Time Window",
    options=sorted(df['Time Window'].unique()),
    default=sorted(df['Time Window'].unique())
)

# Apply filters
filtered = df[
    df['Failure Risk'].isin(risk_filter) &
    df['Route'].isin(route_filter) &
    df['Channel'].isin(channel_filter) &
    df['Time Window'].isin(window_filter)
].copy()

st.markdown(f"**Showing {len(filtered)} route-window combinations**")

# ── Charts row ─────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Failure Risk by Route")
    route_risk = (df.groupby(['Route', 'Failure Risk'])
                    .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in route_risk.columns:
            route_risk[col] = 0
    route_risk = route_risk[['High', 'Medium', 'Low']]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    route_risk.plot(kind='bar', ax=ax, stacked=True,
                    color=[RISK_COLOURS['High'],
                           RISK_COLOURS['Medium'],
                           RISK_COLOURS['Low']],
                    edgecolor='white')
    ax.set_xlabel('')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk distribution per route')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_col2:
    st.subheader("Failure Risk by Time Window")
    window_risk = (df.groupby(['Time Window', 'Failure Risk'])
                     .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in window_risk.columns:
            window_risk[col] = 0
    window_risk = window_risk[['High', 'Medium', 'Low']].sort_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    window_risk.plot(kind='bar', ax=ax, stacked=True,
                     color=[RISK_COLOURS['High'],
                            RISK_COLOURS['Medium'],
                            RISK_COLOURS['Low']],
                     edgecolor='white')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk concentration by time of day')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Prediction table ───────────────────────────────────────────────────
st.subheader("📋 Route-Level Failure Risk Forecast")
st.markdown(
    "Each row represents one route/channel/time-window combination. "
    "Sorted by risk severity."
)

display_cols = [
    'Route', 'Channel', 'Time Window', 'Provider',
    'Failure Risk', 'Transaction Count',
    'Volume Spike Ratio', 'Timeout Rate (%)', 'Success Rate (%)',
    'Reason', 'Recommended Action'
]

display_df = filtered[display_cols].sort_values(
    'Failure Risk',
    key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2})
).reset_index(drop=True)

def colour_risk(val):
    colours = {'High': '#ffd7d7', 'Medium': '#ffe4c4', 'Low': '#d4edda'}
    return f'background-color: {colours.get(val, "")}'

st.dataframe(
    display_df.style.applymap(colour_risk, subset=['Failure Risk']),
    use_container_width=True,
    height=420
)

# ── Sample operational output (brief format) ───────────────────────────
st.divider()
st.subheader("🚨 High Risk Alert Feed")
st.markdown(
    "Formatted alerts in operational output style, "
    "ready for integration with an ops team notification system."
)

high_rows = filtered[filtered['Failure Risk'] == 'High'].head(8)

if high_rows.empty:
    st.info("No High risk alerts match the current filters.")
else:
    for _, row in high_rows.iterrows():
        with st.container():
            st.error(
                f"**Channel:** {row['Channel']}  |  "
                f"**Route:** {row['Route']}  |  "
                f"**Provider:** {row['Provider']}  |  "
                f"**Time Window:** {row['Time Window']}  |  "
                f"**Risk:** 🔴 {row['Failure Risk']}  \n"
                f"**Reason:** {row['Reason']}  \n"
                f"**Action:** {row['Recommended Action']}"
            )

# ── Model performance footer ───────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance")

perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
perf_col1.metric("Model",         "XGBoost")
perf_col2.metric("ROC-AUC",       "0.8686")
perf_col3.metric("Accuracy",      "80%")
perf_col4.metric("False Alarm Rate (High)", "3.8%")
perf_col5.metric("Missed Incident Rate (High)", "21.0%")

st.caption(
    "**Responsible AI note:** This model supports operational decision-making only. "
    "It must not be used to automatically block transactions or deny customer service. "
    "All outputs are recommendations for human review."
)