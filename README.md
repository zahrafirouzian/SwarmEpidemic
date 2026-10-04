# SwarmEpidemic: Information Epidemiology for AI Swarms

An inference-only forensic observability tool to trace, quantify, and visualize conceptual contagion, collective hallucinations, and semantic drift across multi-agent communications.

---

## Overview

When autonomous agents collaborate, errors and misinformation are not isolated bugs—they propagate across shared context windows like contagious pathogens. **SwarmEpidemic** maps classical epidemiological models (Patient Zero attribution, reproduction ratio $R_0$, super-spreader dynamics) directly onto multi-agent communication topologies without requiring model retraining or fine-tuning.

### Key Capabilities
- **Patient Zero Identification:** Pinpoints the precise timestamp and agent source introducing a specific target keyword or hallucination.
- **Outbreak Metrics ($R_0$ Approximation):** Calculates the empirical reproduction ratio ($R_0 = \frac{\text{Direct Transmissions}}{\text{Infected Agents}}$) within incident windows.
- **Topological Super-Spreader Ranking:** Evaluates out-degree centrality to identify high-risk vector agents.
- **Semantic Drift Tracking:** Measures contextual distortion using embedding cosine distance (`all-MiniLM-L6-v2`) between the initial claim and downstream agent transmissions.

---

## Incident Audit Case: "simulation"

Analyzed against the **AI Village** transcript dataset (~183k communication logs, 329 matching incident records):

| Metric | Measured Value | Epidemiological Analog |
| :--- | :--- | :--- |
| **Patient Zero** | `d5fd932e-751f-42c5-92f6-c8ac514864a8` | Index Case |
| **Outbreak Timestamp** | `2025-05-02 18:00:51` | Primary Exposure Window |
| **Estimated $R_0$** | **5.0** | Basic Reproduction Number |
| **Infected Agents** | 26 Agents | Total Cases |
| **Observed Transmissions** | 130 Edges | Infection Transmission Events |
| **Peak Super-Spreader** | Agent `0920d3a7-...` (10 out-degree transmissions) | Super-Spreader Vector |

---

## Quickstart

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (Optional)

### Local Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/zahrafirouzian/swarm-epidemic.git
   cd swarm-epidemic


Set up virtual environment:
bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
Launch the Forensic Dashboard:
bash
   streamlit run app.py
   
Docker Setup
bash
docker compose up --build
Access the console at http://localhost:8501.

Architecture
text
Agent Logs (.csv/.jsonl.gz)
│
▼
┌─────────────────────────┐
│ Log Parser & Ingestion  │ ──► Keyword / RegEx Incident Filter
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Topological Graph Engine│ ──► Directed Transmission Graph (NetworkX / PyVis)
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Semantic Drift Pipeline │ ──► SentenceTransformer (all-MiniLM-L6-v2)
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Forensic Streamlit UI   │ ──► Interactive Metrics, Super-Spreaders, Timeline
└─────────────────────────┘
Scope & Methodological Limitations
Empirical Ratio vs. Differential SIR: 
𝑅
0
R 
0
​
 
 is reported as an empirical transmission ratio within the captured log window, rather than a parameterized continuous SIR differential system.
Log Dependency: Attribution fidelity depends on chronological message logging and observable agent communication traces.



---

## Project Status
Developed for the AI Swarm Dynamics Hackathon (Oct 2026).


