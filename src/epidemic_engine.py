"""
Epidemic Engine for Swarm Information Forensics.
Calculates Patient Zero, R0, Transmission Graphs, and Super-Spreader rankings.
"""

import pandas as pd
import networkx as nx
from typing import Dict, Any


def find_patient_zero(incident_df: pd.DataFrame) -> pd.Series:
    """Find the earliest recorded occurrence (Patient Zero) of an incident."""
    if incident_df.empty:
        return pd.Series()
    sorted_df = incident_df.sort_values(by="timestamp", ascending=True)
    return sorted_df.iloc[0]


def build_transmission_network(
    incident_df: pd.DataFrame,
    full_df: pd.DataFrame,
    time_window_minutes: int = 60
) -> nx.DiGraph:
    """
    Construct a directed epidemic transmission graph.
    Edge A -> B implies Agent A spoke in a room, and Agent B spoke subsequently
    within the time window, propagating the incident keyword.
    """
    graph = nx.DiGraph()
    if incident_df.empty:
        return graph

    incident_msgs = incident_df.sort_values(by="timestamp").to_dict("records")
    time_delta = pd.Timedelta(minutes=time_window_minutes)

    for i, origin in enumerate(incident_msgs):
        origin_time = origin["timestamp"]
        origin_agent = str(origin["speaker"])
        origin_room = origin["room_id"]

        graph.add_node(origin_agent, room=origin_room)

        for target in incident_msgs[i + 1:]:
            target_time = target["timestamp"]
            target_agent = str(target["speaker"])
            target_room = target["room_id"]

            if target_time - origin_time > time_delta:
                break

            if origin_room == target_room and origin_agent != target_agent:
                if graph.has_edge(origin_agent, target_agent):
                    graph[origin_agent][target_agent]["weight"] += 1
                else:
                    graph.add_edge(
                        origin_agent,
                        target_agent,
                        weight=1,
                        time_delta_sec=float((target_time - origin_time).total_seconds())
                    )

    return graph


def calculate_epidemic_metrics(graph: nx.DiGraph, incident_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate R0, transmission counts, and identify super-spreader nodes."""
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    out_degrees = [d for _, d in graph.out_degree()]
    r0 = float(sum(out_degrees) / total_nodes) if total_nodes > 0 else 0.0

    spreaders = sorted(graph.out_degree(), key=lambda x: x[1], reverse=True)
    top_spreaders = [{"agent_id": agent, "transmissions": count} for agent, count in spreaders if count > 0][:5]

    return {
        "r0": round(r0, 2),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "top_spreaders": top_spreaders
    }
