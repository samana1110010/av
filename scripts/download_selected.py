import pandas as pd
from huggingface_hub import snapshot_download

REPO = "11hu83/vggsound"

df = pd.read_csv("data/selected_metadata.csv")

files = df["file_name"].tolist()

print(f"Downloading {len(files)} videos...")
print(f"Classes: {df['text'].nunique()}")

snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=files,
    local_dir="data/vggsound_selected",
)

print("Done!")
