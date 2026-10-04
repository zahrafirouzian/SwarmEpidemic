"""
Mutation Tracker for Swarm Information Forensics.
Measures semantic drift using embedding cosine distance against Patient Zero.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """Load lightweight SentenceTransformer model on CPU with caching."""
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def track_semantic_drift(
    incident_df: pd.DataFrame,
    patient_zero_content: str,
    model: SentenceTransformer,
    max_messages: int = 50
) -> pd.DataFrame:
    """
    Calculate semantic drift (1 - Cosine Similarity) from Patient Zero's message.
    """
    if incident_df.empty or not patient_zero_content:
        return pd.DataFrame()

    sample_df = incident_df.sort_values(by="timestamp").head(max_messages).copy()

    # Encode Patient Zero content
    base_embedding = model.encode([patient_zero_content], normalize_embeddings=True)[0]

    # Encode sequence contents
    contents = sample_df["content"].astype(str).tolist()
    embeddings = model.encode(contents, normalize_embeddings=True)

    # Cosine distance = 1.0 - Cosine Similarity
    similarities = np.dot(embeddings, base_embedding)
    drifts = np.clip(1.0 - similarities, 0.0, 1.0)

    sample_df["semantic_drift"] = np.round(drifts, 4)
    sample_df["speaker"] = sample_df["speaker"].astype(str).str[:8] + "..."
    return sample_df
