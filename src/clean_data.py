import gzip
import json
import pandas as pd
import os

input_file = "data/chat_messages_real.jsonl.gz"
output_file = "data/cleaned_messages.csv"

print("🔄 Loading and cleaning data...")

data = []
with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            # استخراج فیلدهای کلیدی
            data.append({
                "id": entry.get("id"),
                "speaker": entry.get("agent_speaker_id") or entry.get("user_speaker_id"),
                "content": entry.get("content", ""),
                "room_id": entry.get("room_id"),
                "timestamp": pd.to_datetime(entry.get("created_at"))
            })
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)
# مرتب‌سازی بر اساس زمان
df = df.sort_values(by="timestamp")
df.to_csv(output_file, index=False)

print(f"✅ Cleaned {len(df)} messages saved to {output_file}")
print("Sample of data structure:")
print(df.head())
