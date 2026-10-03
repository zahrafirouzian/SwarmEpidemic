import pandas as pd

df = pd.read_csv("data/cleaned_messages.csv")
print("=== DATASET OVERVIEW ===")
print("Total messages:", len(df))
print("Unique agents (speakers):", df["speaker"].nunique())
print("Unique chat rooms:", df["room_id"].nunique())

print("\n=== TOP 5 ACTIVE AGENTS ===")
print(df["speaker"].value_counts().head(5))

print("\n=== SAMPLE TOPICS / KEYWORDS OCCURRENCES ===")
keywords = ["hallucination", "drift", "simulation", "secret", "election", "sandbox", "token", "puzzle"]
for kw in keywords:
    count = df["content"].fillna("").str.contains(kw, case=False).sum()
    print(f"Keyword '{kw}': {count} occurrences")
