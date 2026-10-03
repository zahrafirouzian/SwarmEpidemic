import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "aidigestorg/ai-village"
FILENAME = "chat_messages.jsonl.gz"
DESTINATION = "data/chat_messages_real.jsonl.gz"

print(f"🔄 Starting download of {FILENAME} from {REPO_ID}...")
try:
    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset"
    )
    shutil.copy(downloaded_file, DESTINATION)
    size_mb = os.path.getsize(DESTINATION) / (1024 * 1024)
    print(f"✅ Successfully downloaded to {DESTINATION} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"❌ Error downloading file: {e}")
