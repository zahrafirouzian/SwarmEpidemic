"""
SwarmEpidemic Forensic Dashboard
Clean, high-contrast, professional Navy-Dark UI for LLM agent forensics.
Full-width clean sequential layout with custom-styled tables and charts.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

from src.epidemic_engine import (
    find_patient_zero,
    build_transmission_network,
    calculate_epidemic_metrics
)
from src.mutation_tracker import load_similarity_model, track_semantic_drift

# --- Page Config ---
st.set_page_config(
    page_title="SwarmEpidemic | Forensic Console",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Navy-Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Solid Dark Navy Background */
    .stApp {
        background-color: #0b1329;
        color: #e2e8f0;
    }

    /* Sidebar High Contrast */
    section[data-testid="stSidebar"] {
        background-color: #080f21 !important;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }

    /* Header Container */
    .header-box {
        background-color: #111c38;
        border: 1px solid #1e293b;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 22px 26px;
        margin-bottom: 26px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-bottom: 28px;
    }
    .metric-card {
        background-color: #111c38;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px 20px;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .metric-number {
        font-size: 2.1rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
    }
    .metric-red { color: #f87171; }
    .metric-blue { color: #38bdf8; }

    /* Dossier Card */
    .dossier-card {
        background-color: #17182b;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 28px;
    }
    .dossier-badge {
        font-size: 0.82rem;
        font-weight: 700;
        color: #f87171;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .dossier-info {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        padding-bottom: 8px;
        margin-bottom: 18px;
        border-bottom: 1px solid #1e293b;
    }

    /* Sub-section label */
    .sub-title {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 10px;
    }

    /* Spacing between modules */
    .block-spacer {
        margin-top: 36px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_clean_dataset():
    data_path = "data/cleaned_messages.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


with st.spinner("Loading telemetry dataset..."):
    df = load_clean_dataset()

# --- Sidebar Controls ---
st.sidebar.markdown("### 🔍 Incident Parameters")

preset = st.sidebar.selectbox(
    "Target Vector Keyword:",
    options=["simulation", "election", "drift", "jailbreak", "truth", "Custom"],
    index=0
)

if preset == "Custom":
    search_keyword = st.sidebar.text_input("Custom Keyword:", value="simulation")
else:
    search_keyword = preset

time_window = st.sidebar.slider(
    "Contagion Time Window (Minutes):",
    min_value=5,
    max_value=180,
    value=60,
    step=5,
    help="Maximum time elapsed between messages in the same room to infer a directed transmission link."
)

max_drift_messages = st.sidebar.slider(
    "Messages for Mutation Tracking:",
    min_value=10,
    max_value=100,
    value=40,
    step=10,
    help="Chronological sequence depth of messages analyzed for semantic divergence."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
    <strong>Framework:</strong> SwarmEpidemic<br>
    <strong>Engine:</strong> Inference-Only Telemetry<br>
    <strong>Corpus:</strong> AI Village (183,485 logs)
    </div>
    """,
    unsafe_allow_html=True
)

# --- Filter Data ---
keyword_pattern = rf"(?i)\b{search_keyword}\b"
incident_df = df[df["content"].astype(str).str.contains(keyword_pattern, na=False)].sort_values(by="timestamp")

# --- Title Header ---
st.markdown(f"""
<div class="header-box">
    <div class="header-title">🦠 SwarmEpidemic Forensic Console</div>
    <div class="header-desc">
        Real-time epidemiological investigation of conceptual contagion, hallucination cascades, and semantic drift across autonomous LLM swarms.
    </div>
</div>
""", unsafe_allow_html=True)

if incident_df.empty:
    st.warning(f"No incident logs matched vector: '{search_keyword}'")
    st.stop()

# --- Calculations ---
patient_zero = find_patient_zero(incident_df)
graph = build_transmission_network(incident_df, df, time_window_minutes=time_window)
metrics = calculate_epidemic_metrics(graph, incident_df)

# --- Top KPIs Cards ---
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-title">Reproduction Ratio (R₀)</div>
        <div class="metric-number metric-red">{metrics['r0']}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Infected Agents (Nodes)</div>
        <div class="metric-number metric-blue">{metrics['total_nodes']}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Transmissions (Edges)</div>
        <div class="metric-number">{metrics['total_edges']}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Total Incident Logs</div>
        <div class="metric-number">{len(incident_df)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Patient Zero Dossier ---
st.markdown(f"""
<div class="dossier-card">
    <div class="dossier-badge">⚠️ Patient Zero Attribution</div>
    <div class="dossier-info">
        <strong>Agent ID:</strong> <span style="color:#ffffff;">{patient_zero['speaker']}</span><br>
        <strong>Detected Timestamp:</strong> <span style="color:#ffffff;">{patient_zero['timestamp']}</span><br>
        <strong>Room ID:</strong> <span style="color:#cbd5e1;">{patient_zero['room_id']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📄 View Patient Zero Initial Inoculation Message", expanded=False):
    st.markdown(f"> *{patient_zero['content']}*")

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

# ==========================================
# 1. Full-Width Transmission Network Graph
# ==========================================
st.markdown('<div class="section-title">🕸️ Transmission Network Topology</div>', unsafe_allow_html=True)
if graph.number_of_nodes() > 0:
    net = Network(height="500px", width="100%", bgcolor="#080d1c", font_color="#e2e8f0", directed=True)
    net.from_nx(graph)

    pz_id = str(patient_zero["speaker"])

    for node in net.nodes:
        nid = node["id"]
        if nid == pz_id:
            node["color"] = {"background": "#ef4444", "border": "#ffffff", "highlight": "#f87171"}
            node["size"] = 28
            node["label"] = f"PATIENT ZERO\n({nid[:6]})"
        else:
            out_deg = graph.out_degree(nid) if graph.has_node(nid) else 1
            node["color"] = {"background": "#1e3a8a", "border": "#38bdf8", "highlight": "#60a5fa"}
            node["size"] = max(12, min(24, 12 + out_deg * 2))
            node["label"] = f"{nid[:6]}..."

    for edge in net.edges:
        edge["color"] = {"color": "rgba(56, 189, 248, 0.45)", "highlight": "#ef4444"}
        edge["arrows"] = "to"

    # Physics enabled, navigation buttons disabled for clean UI
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "springLength": 140,
          "springConstant": 0.02
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": false,
        "zoomView": true
      }
    }
    """)
    net.save_graph("transmission_graph.html")
    with open("transmission_graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=520)
else:
    st.info("No network edges identified within current temporal constraints.")

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

# ==========================================
# 2. Spreaders & Temporal Cascade Row
# ==========================================
col_spreaders, col_timeline = st.columns([1, 1], gap="large")

with col_spreaders:
    st.markdown('<div class="section-title">🚨 Top Super-Spreaders</div>', unsafe_allow_html=True)
    if metrics["top_spreaders"]:
        spreader_df = pd.DataFrame(metrics["top_spreaders"])
        spreader_df.columns = ["Agent UUID", "Transmissions"]
        st.dataframe(
            spreader_df,
            column_config={
                "Agent UUID": st.column_config.TextColumn("Agent UUID", width="medium"),
                "Transmissions": st.column_config.ProgressColumn(
                    "Transmissions",
                    help="Number of directed contagion edges generated",
                    format="%d",
                    min_value=0,
                    max_value=int(spreader_df["Transmissions"].max()) if not spreader_df.empty else 10,
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=260
        )
    else:
        st.caption("No multi-hop transmission nodes registered.")

with col_timeline:
    st.markdown('<div class="section-title">📈 Cascade Activity Over Time</div>', unsafe_allow_html=True)
    timeline_df = incident_df.copy()
    timeline_df["count"] = 1
    timeline_df = timeline_df.set_index("timestamp").resample("1h")["count"].sum().reset_index()
    timeline_df.columns = ["Timestamp", "Message Count"]
    st.line_chart(timeline_df.set_index("Timestamp"), height=260)

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

# ==========================================
# 3. Semantic Mutation & Drift
# ==========================================
st.markdown('<div class="section-title">🧬 Semantic Mutation & Drift Progression</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='color:#94a3b8; font-size:0.92rem; margin-bottom: 16px;'>"
    "Measuring cosine distance from Patient Zero ($1 - \\text{Cosine Similarity}$). "
    "<strong>0.0</strong> = Identical context, <strong>1.0</strong> = Complete conceptual deviation."
    "</div>",
    unsafe_allow_html=True
)

with st.spinner("Analyzing semantic drift across incident logs..."):
    model = load_similarity_model()
    drift_df = track_semantic_drift(incident_df, patient_zero["content"], model, max_messages=max_drift_messages)

if not drift_df.empty:
    st.markdown('<div class="sub-title">📉 Semantic Drift Curve (Chronological Divergence)</div>', unsafe_allow_html=True)
    drift_chart_data = drift_df.set_index("timestamp")[["semantic_drift"]].rename(
        columns={"semantic_drift": "Semantic Drift (Distance)"}
    )
    st.line_chart(drift_chart_data, height=240)

    st.markdown('<div class="sub-title" style="margin-top: 20px;">📜 Chronological Mutation Samples</div>', unsafe_allow_html=True)
    display_drift = drift_df[["timestamp", "speaker", "semantic_drift", "content"]].copy()
    
    st.dataframe(
        display_drift,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss", width="medium"),
            "speaker": st.column_config.TextColumn("Agent UUID", width="small"),
            "semantic_drift": st.column_config.NumberColumn("Drift Distance", format="%.4f", width="small"),
            "content": st.column_config.TextColumn("Message Content", width="large")
        },
        use_container_width=True,
        hide_index=True,
        height=320
    )

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

# ==========================================
# 4. Full Audit Logs
# ==========================================
with st.expander("📋 Full Telemetry Incident Log"):
    st.dataframe(
        incident_df[["timestamp", "speaker", "room_id", "content"]],
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "speaker": st.column_config.TextColumn("Agent UUID"),
            "room_id": st.column_config.TextColumn("Room ID"),
            "content": st.column_config.TextColumn("Message")
        },
        use_container_width=True,
        hide_index=True
    )
