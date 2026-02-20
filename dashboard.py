import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "data/network_logs.db"

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Network Performance Monitor",
    page_icon="",
    layout="wide"
)

# ---------------------- LOAD & PROCESS ----------------------
@st.cache_data(ttl=5)  # Refresh every 5 seconds
def load_data(time_range_minutes=60):
    conn = sqlite3.connect(DB_PATH)
    
    # Get data from last N minutes
    cutoff_time = (datetime.now() - timedelta(minutes=time_range_minutes)).isoformat()
    
    df = pd.read_sql_query(
        f"SELECT time, latency_ms FROM metrics WHERE time >= ? ORDER BY time ASC",
        conn,
        params=(cutoff_time,)
    )
    conn.close()
    
    if df.empty:
        return df
    
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    
    # Calculate rolling metrics
    df["latency_avg_5"] = df["latency_ms"].rolling(window=5, min_periods=1).mean()
    df["latency_avg_20"] = df["latency_ms"].rolling(window=20, min_periods=1).mean()
    
    # Calculate jitter (variation in latency)
    df["jitter_ms"] = df["latency_ms"].diff().abs()
    df["jitter_avg"] = df["jitter_ms"].rolling(window=10, min_periods=1).mean()
    
    # Detect packet loss (gaps > 10 seconds indicate missed pings)
    time_diff = df.index.to_series().diff().dt.total_seconds()
    df["packet_loss"] = (time_diff > 10).astype(int)
    
    return df

def calculate_quality_score(df):
    """Calculate network quality score (0-100)"""
    if df.empty or len(df) < 5:
        return None
    
    recent = df.tail(20)
    
    # Latency score (lower is better)
    avg_latency = recent["latency_ms"].mean()
    latency_score = max(0, 100 - (avg_latency - 10) * 2)  # Penalty after 10ms
    
    # Jitter score (lower is better)
    avg_jitter = recent["jitter_ms"].mean()
    jitter_score = max(0, 100 - avg_jitter * 5)
    
    # Stability score (fewer spikes is better)
    std_latency = recent["latency_ms"].std()
    stability_score = max(0, 100 - std_latency * 3)
    
    # Packet loss score
    packet_loss_rate = recent["packet_loss"].sum() / len(recent) * 100
    packet_loss_score = max(0, 100 - packet_loss_rate * 10)
    
    # Weighted average
    total_score = (
        latency_score * 0.35 +
        jitter_score * 0.25 +
        stability_score * 0.25 +
        packet_loss_score * 0.15
    )
    
    return round(max(0, min(100, total_score)), 1)

def get_quality_rating(score):
    """Convert score to rating"""
    if score is None:
        return "Unknown", "⚪"
    if score >= 90:
        return "Excellent", "🟢"
    elif score >= 75:
        return "Good", "🟡"
    elif score >= 60:
        return "Fair", "🟠"
    else:
        return "Poor", "🔴"

# ---------------------- DASHBOARD UI ----------------------
st.title("📡 Network Performance Monitor")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    time_range = st.selectbox(
        "Time Range",
        options=[5, 15, 30, 60, 120, 240],
        index=3,
        format_func=lambda x: f"Last {x} minutes"
    )
    
    auto_refresh = st.checkbox("Auto-refresh (every 5s)", value=True)
    
    st.divider()
    
    st.header("📊 Thresholds")
    latency_threshold = st.slider("Alert if latency >", 50, 500, 100, 10, help="Milliseconds")
    jitter_threshold = st.slider("Alert if jitter >", 10, 100, 30, 5, help="Milliseconds")
    
    st.divider()
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM metrics")
        conn.commit()
        conn.close()
        st.cache_data.clear()
        st.success("Data cleared!")
        st.rerun()

# Load data
df = load_data(time_range)

if df.empty:
    st.warning("⚠️ No data available. Make sure the collector is running.")
    st.code("python collector.py", language="bash")
    st.stop()

# Calculate metrics
quality_score = calculate_quality_score(df)
rating, emoji = get_quality_rating(quality_score)

recent = df.tail(20)
current_latency = recent["latency_ms"].iloc[-1]
avg_latency = recent["latency_ms"].mean()
min_latency = recent["latency_ms"].min()
max_latency = recent["latency_ms"].max()
avg_jitter = recent["jitter_ms"].mean()
packet_loss_count = df["packet_loss"].sum()

# ---------------------- METRICS ROW ----------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Network Quality",
        f"{quality_score}/100",
        delta=f"{rating} {emoji}"
    )

with col2:
    st.metric(
        "Current Latency",
        f"{current_latency:.1f} ms",
        delta=f"{current_latency - avg_latency:+.1f} ms"
    )

with col3:
    st.metric(
        "Avg Latency",
        f"{avg_latency:.1f} ms",
        delta=f"Min: {min_latency:.1f} ms"
    )

with col4:
    st.metric(
        "Avg Jitter",
        f"{avg_jitter:.1f} ms",
        delta="Lower is better"
    )

with col5:
    st.metric(
        "Packet Loss",
        f"{packet_loss_count}",
        delta=f"Last {time_range}m"
    )

# ---------------------- ALERTS ----------------------
alerts = []
if current_latency > latency_threshold:
    alerts.append(f"⚠️ High latency detected: {current_latency:.1f} ms")
if avg_jitter > jitter_threshold:
    alerts.append(f"⚠️ High jitter detected: {avg_jitter:.1f} ms")
if packet_loss_count > 0:
    alerts.append(f"⚠️ Packet loss detected: {packet_loss_count} occurrences")

if alerts:
    for alert in alerts:
        st.warning(alert)

st.divider()

# ---------------------- INTERACTIVE CHARTS ----------------------

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Latency Over Time", "Jitter Over Time"),
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4]
)

# Latency chart
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["latency_ms"],
        mode="lines",
        name="Latency",
        line=dict(color="#1f77b4", width=1),
        hovertemplate="<b>%{x}</b><br>Latency: %{y:.2f} ms<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["latency_avg_20"],
        mode="lines",
        name="20-point Average",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f} ms<extra></extra>"
    ),
    row=1, col=1
)

# Add threshold line
fig.add_hline(
    y=latency_threshold,
    line_dash="dot",
    line_color="red",
    opacity=0.5,
    row=1, col=1,
    annotation_text="Threshold"
)

# Jitter chart
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["jitter_ms"],
        mode="lines",
        name="Jitter",
        line=dict(color="#2ca02c", width=1),
        fill="tozeroy",
        fillcolor="rgba(44, 160, 44, 0.2)",
        hovertemplate="<b>%{x}</b><br>Jitter: %{y:.2f} ms<extra></extra>"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["jitter_avg"],
        mode="lines",
        name="Jitter Average",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Avg Jitter: %{y:.2f} ms<extra></extra>"
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=700,
    showlegend=True,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
fig.update_yaxes(title_text="Jitter (ms)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------- STATISTICS TABLE ----------------------
st.subheader("📈 Detailed Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Latency Statistics**")
    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "P95", "P99"],
        "Value (ms)": [
            f"{df['latency_ms'].mean():.2f}",
            f"{df['latency_ms'].median():.2f}",
            f"{df['latency_ms'].std():.2f}",
            f"{df['latency_ms'].min():.2f}",
            f"{df['latency_ms'].max():.2f}",
            f"{df['latency_ms'].quantile(0.95):.2f}",
            f"{df['latency_ms'].quantile(0.99):.2f}"
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown("**Jitter Statistics**")
    jitter_stats_df = pd.DataFrame({
        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "P95", "P99"],
        "Value (ms)": [
            f"{df['jitter_ms'].mean():.2f}",
            f"{df['jitter_ms'].median():.2f}",
            f"{df['jitter_ms'].std():.2f}",
            f"{df['jitter_ms'].min():.2f}",
            f"{df['jitter_ms'].max():.2f}",
            f"{df['jitter_ms'].quantile(0.95):.2f}",
            f"{df['jitter_ms'].quantile(0.99):.2f}"
        ]
    })
    st.dataframe(jitter_stats_df, hide_index=True, use_container_width=True)

# ---------------------- RAW DATA ----------------------
with st.expander("📋 View Raw Data"):
    st.dataframe(
        df.reset_index()[["time", "latency_ms", "jitter_ms"]].tail(100),
        hide_index=True,
        use_container_width=True
    )

# ---------------------- AUTO REFRESH ----------------------
if auto_refresh:
    st.rerun()