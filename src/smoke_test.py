import networkx as nx
from sentence_transformers import SentenceTransformer

def main():
    g = nx.DiGraph()
    g.add_edge("a", "b")
    print("networkx ok, edges:", list(g.edges()))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["hello", "hi"])
    print("embeddings shape:", emb.shape)

if __name__ == "__main__":
    main()
