import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class MutationTracker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def track_drift(self, patient_zero_text: str, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        if not messages:
            return pd.DataFrame()

        # Encode Patient Zero reference
        p0_vec = self.model.encode([patient_zero_text], convert_to_numpy=True)[0]
        p0_norm = p0_vec / (np.linalg.norm(p0_vec) + 1e-12)

        texts = [m.get("content", "") for m in messages]
        msg_vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(msg_vecs, axis=1, keepdims=True) + 1e-12
        msg_vecs_norm = msg_vecs / norms

        # Cosine similarity & semantic drift (distance)
        similarities = np.dot(msg_vecs_norm, p0_norm)
        drifts = 1.0 - similarities

        records = []
        for i, m in enumerate(messages):
            records.append({
                "message_id": m.get("id"),
                "speaker": m.get("speaker"),
                "timestamp": m.get("timestamp"),
                "room_id": m.get("room_id"),
                "content_preview": (m.get("content") or "")[:120],
                "cosine_similarity": round(float(similarities[i]), 4),
                "semantic_drift": round(float(drifts[i]), 4)
            })

        return pd.DataFrame(records)

if __name__ == "__main__":
    from epidemic_engine import SwarmEpidemicEngine
    
    engine = SwarmEpidemicEngine()
    analysis = engine.analyze_cascade("simulation")
    
    matches_df = analysis["matches_df"].head(10)
    p0_text = analysis["patient_zero"]["content"]
    
    tracker = MutationTracker()
    drift_df = tracker.track_drift(p0_text, matches_df.to_dict(orient="records"))
    print("\n=== SEMANTIC DRIFT SAMPLE ===")
    print(drift_df[["speaker", "cosine_similarity", "semantic_drift", "content_preview"]])
