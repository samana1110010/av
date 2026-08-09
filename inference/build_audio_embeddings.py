import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalDataset
from models.audio_encoder import AudioEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------

dataset = MultimodalDataset(
    "data/train.csv",
    "data/audio"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

# ----------------------------
# Model
# ----------------------------

model = AudioEncoder().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/audio_encoder_final.pt",
        map_location=DEVICE
    )
)

model.eval()

embeddings = []
video_ids = []

# ----------------------------
# Generate embeddings
# ----------------------------

with torch.no_grad():

    for _, mel, _, ids in loader:

        mel = mel.to(DEVICE)

        emb = model(mel)

        embeddings.append(emb.cpu())

        video_ids.extend(ids)

embeddings = torch.cat(embeddings).numpy()

# ----------------------------
# Save
# ----------------------------

os.makedirs("embeddings", exist_ok=True)

np.save(
    "embeddings/audio_embeddings.npy",
    embeddings
)

np.save(
    "embeddings/audio_ids.npy",
    np.array(video_ids)
)

print("Embeddings shape:", embeddings.shape)
print("Saved", len(video_ids), "audio embeddings.")