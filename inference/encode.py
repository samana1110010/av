import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets.video_dataset import VideoDataset
from models.video_encoder import VideoEncoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# -----------------------------
# Dataset
# -----------------------------

dataset = VideoDataset("data/train.csv")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

# -----------------------------
# Model
# -----------------------------

model = VideoEncoder().to(device)
model.eval()

# -----------------------------
# Generate embeddings
# -----------------------------

all_embeddings = []
all_video_ids = []

with torch.no_grad():

    for frames, labels, video_ids in loader:

        frames = frames.to(device)

        embeddings = model(frames)

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        all_video_ids.extend(video_ids)

# -----------------------------
# Save
# -----------------------------

all_embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

np.save(
    "data/video_embeddings.npy",
    all_embeddings
)

np.save(
    "data/video_ids.npy",
    np.array(all_video_ids)
)

print("Done!")
print("Embeddings shape:", all_embeddings.shape)
print("Saved:")
print("data/video_embeddings.npy")
print("data/video_ids.npy")