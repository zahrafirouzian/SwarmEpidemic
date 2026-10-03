import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os

from src.epidemic_engine import SwarmEpidemicEngine
from src.mutation_tracker import MutationTracker

st.set_page_config(
    page_title="SwarmEpidemic | Information Forensics",
    page_icon="🦠",
    layout="wide"
)

@st.cache_resource
def load_components():
    engine = SwarmEpidemicEngine(data_path="data/cleaned_messages.csv")
    engine.load_data()
    tracker = MutationTracker()
    return engine, tracker

engine, tracker = load_components()

# --- Header ---
st.title("🦠 SwarmEpidemic: Information Forensics in Agent Swarms")
st.markdown(
    "Tracing **Information Cascades, Hallucination Vectors, and Semantic Mutations** across autonomous LLM agents (AI Village logs)."
)

# --- Sidebar Controls ---
st.sidebar.header("🔍 Incident Search")
preset_keywords = ["simulation", "drift", "election", "sandbox", "hallucination", "puzzle", "token"]
selected_preset = st.sidebar.selectbox("Preset Incident Scenarios:", ["(Custom)"] + preset_keywords)

if selected_preset != "(Custom)":
    query = st.sidebar.text_input("Search Keyword:", value=selected_preset)
else:
    query = st.sidebar.text_input("Search Keyword:", value="simulation")

time_window = st.sidebar.slider("Transmission Time Window (Minutes):", min_value=5, max_value=180, value=60, step=5)
max_samples = st.sidebar.slider("Max Messages for Mutation Tracking:", min_value=10, max_value=200, value=50, step=10)

if not query.strip():
    st.warning("Please enter a valid keyword to trace.")
    st.stop()

# --- Run Forensic Engine ---
with st.spinner(f"Analyzing epidemic cascade for '{query}'..."):
    analysis = engine.analyze_cascade(keyword=query, time_window_minutes=time_window)

if analysis["status"] == "error":
    st.error(analysis["message"])
    st.stop()

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Basic Reproduction (R₀)", f"{analysis['r0']}")
col2.metric("Infected Agents (Nodes)", f"{analysis['infected_agents']}")
col3.metric("Transmissions (Edges)", f"{analysis['transmissions_count']}")
col4.metric("Total Incident Logs", f"{analysis['total_messages']}")

# --- Patient Zero Card ---
p0 = analysis["patient_zero"]
st.subheader("🎯 Patient Zero (Infection Origin)")
st.info(
    f"**Speaker ID:** `{p0['speaker']}`  \n"
    f"**Timestamp:** `{p0['timestamp']}` | **Room ID:** `{p0['room_id']}`  \n"
    f"**First Broadcast Content:**  \n> *\"{p0['content']}\"*"
)

# --- Tabs for Graph & Mutation Analysis ---
tab_graph, tab_mutation, tab_logs = st.tabs(["🕸️ Transmission Graph", "🧬 Semantic Drift & Mutation", "📜 Raw Incident Logs"])

with tab_graph:
    st.markdown("### Interactive Transmission Tree")
    G = analysis["graph"]
    
    if G.number_of_nodes() > 0:
        net = Network(height="550px", width="100%", bgcolor="#111827", font_color="#F3F4F6", directed=True)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

        super_spreader_ids = [s[0] for s in analysis["super_spreaders"]]
        p0_id = p0["speaker"]

        for node in G.nodes():
            label = str(node)[:8] + "..."
            title = f"Agent: {node}\nOut-Degree: {G.out_degree(node)}"
            
            if node == p0_id:
                net.add_node(node, label=f"🚨 P0: {label}", title=title, color="#EF4444", size=30)
            elif node in super_spreader_ids:
                net.add_node(node, label=f"⚡ {label}", title=title, color="#F59E0B", size=24)
            else:
                net.add_node(node, label=label, title=title, color="#3B82F6", size=16)

        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data.get("weight", 1), title=f"Transmissions: {data.get('weight', 1)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_path = tmp_file.name

        with open(tmp_path, "r", encoding="utf-8") as html_file:
            graph_html = html_file.read()
        os.remove(tmp_path)

        components.html(graph_html, height=580)
        st.caption("🔴 Red = Patient Zero | 🟡 Yellow = Super-Spreader | 🔵 Blue = Infected Agent")
    else:
        st.info("No network edges detected for this time window.")

with tab_mutation:
    st.markdown("### Semantic Mutation & Information Drift")
    st.caption("Tracking how the message content semantically mutates compared to Patient Zero's broadcast using embeddings.")
    
    matches_sample = analysis["matches_df"].head(max_samples).to_dict(orient="records")
    drift_df = tracker.track_drift(p0["content"], matches_sample)
    
    if not drift_df.empty:
        drift_df["timestamp"] = pd.to_datetime(drift_df["timestamp"])
        drift_df = drift_df.sort_values(by="timestamp")
        
        # Plot drift over time
        st.line_chart(
            drift_df.set_index("timestamp")[["semantic_drift", "cosine_similarity"]],
            height=300
        )
        
        st.dataframe(
            drift_df[["timestamp", "speaker", "semantic_drift", "cosine_similarity", "content_preview"]],
            use_container_width=True
        )

with tab_logs:
    st.markdown("### Filtered Incident Messages")
    st.dataframe(
        analysis["matches_df"][["timestamp", "speaker", "room_id", "content"]],
        use_container_width=True
    )
