import pandas as pd
import networkx as nx
from typing import Dict, Any, List

class SwarmEpidemicEngine:
    def __init__(self, data_path: str = "data/cleaned_messages.csv"):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        if self.df is None:
            self.df = pd.read_csv(self.data_path)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df = self.df.sort_values(by="timestamp").reset_index(drop=True)

    def analyze_cascade(self, keyword: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        self.load_data()
        
        # Filter matching messages
        matches = self.df[self.df["content"].fillna("").str.contains(keyword, case=False, regex=False)].copy()
        
        if matches.empty:
            return {"status": "error", "message": f"No messages found for '{keyword}'"}

        matches = matches.sort_values(by="timestamp").reset_index(drop=True)
        
        # 1. Identify Patient Zero
        p_zero = matches.iloc[0]
        patient_zero_info = {
            "speaker": p_zero["speaker"],
            "timestamp": str(p_zero["timestamp"]),
            "room_id": p_zero["room_id"],
            "content": p_zero["content"][:200]
        }

        # 2. Build Transmission Graph (NetworkX DiGraph)
        G = nx.DiGraph()
        
        # Add all unique participants
        for spk in matches["speaker"].unique():
            G.add_node(spk, label=spk[:8] + "...", full_id=spk)

        # Track sequential exposure per room
        time_delta = pd.Timedelta(minutes=time_window_minutes)
        edges_added = []
        
        for i in range(len(matches)):
            curr_row = matches.iloc[i]
            curr_speaker = curr_row["speaker"]
            curr_room = curr_row["room_id"]
            curr_time = curr_row["timestamp"]
            
            # Find subsequent messages in the same room within the time window
            subsequent = matches.iloc[i+1:]
            subsequent = subsequent[
                (subsequent["room_id"] == curr_room) &
                (subsequent["speaker"] != curr_speaker) &
                (subsequent["timestamp"] <= curr_time + time_delta)
            ]
            
            for _, target_row in subsequent.iterrows():
                target_speaker = target_row["speaker"]
                if not G.has_edge(curr_speaker, target_speaker):
                    G.add_edge(curr_speaker, target_speaker, weight=1, room=curr_room)
                    edges_added.append((curr_speaker, target_speaker))
                else:
                    G[curr_speaker][target_speaker]["weight"] += 1

        # 3. Compute Epidemiological Metrics
        out_degrees = dict(G.out_degree())
        super_spreaders = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Basic reproduction number (average transmissions per active spreader)
        active_spreaders = [deg for deg in out_degrees.values() if deg > 0]
        r0 = sum(active_spreaders) / len(active_spreaders) if active_spreaders else 0.0

        return {
            "status": "success",
            "keyword": keyword,
            "total_messages": len(matches),
            "infected_agents": G.number_of_nodes(),
            "transmissions_count": G.number_of_edges(),
            "patient_zero": patient_zero_info,
            "r0": round(r0, 2),
            "super_spreaders": super_spreaders,
            "graph": G,
            "matches_df": matches
        }

if __name__ == "__main__":
    engine = SwarmEpidemicEngine()
    results = engine.analyze_cascade(keyword="simulation")
    print("=== EPIDEMIC ENGINE TEST ===")
    print("Status:", results["status"])
    print("Keyword:", results["keyword"])
    print("Total Messages:", results["total_messages"])
    print("Infected Agents:", results["infected_agents"])
    print("Direct Transmissions (Edges):", results["transmissions_count"])
    print("Estimated R0:", results["r0"])
    print("\nPatient Zero:")
    for k, v in results["patient_zero"].items():
        print(f"  {k}: {v}")
    print("\nTop Super-Spreaders (Speaker, Transmissions):")
    for spk, deg in results["super_spreaders"]:
        print(f"  {spk}: {deg}")
