import pandas as pd
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
METADATA = "data/selected_metadata.csv"
FRAME_ROOT = Path("data/frames")
OUTPUT = "data/train.csv"

# ----------------------------
# Load metadata
# ----------------------------
df = pd.read_csv(METADATA)

# ----------------------------
# Convert class names -> integer labels
# ----------------------------
classes = sorted(df["text"].unique())

class_to_idx = {
    cls: idx
    for idx, cls in enumerate(classes)
}

# ----------------------------
# Build dataset
# ----------------------------
rows = []

for _, row in df.iterrows():

    video_folder = Path(row["file_name"]).parent.name
    frame_folder = FRAME_ROOT / video_folder

    if not frame_folder.exists():
        continue

    n_frames = len(list(frame_folder.glob("*.jpg")))

    # Skip videos with very few frames
    if n_frames < 6:
        continue

    rows.append({
        "video_id": video_folder,
        "class_name": row["text"],
        "label": class_to_idx[row["text"]],
        "frame_folder": str(frame_folder),
        "num_frames": n_frames
    })

# ----------------------------
# Save
# ----------------------------
train_df = pd.DataFrame(rows)

train_df.to_csv(OUTPUT, index=False)

print(train_df.head())
print()
print("Samples:", len(train_df))
print()

print(train_df["class_name"].value_counts())

print("\nSaved to:", OUTPUT)

print("\nClass mapping:")

for k, v in class_to_idx.items():
    print(v, "->", k)
