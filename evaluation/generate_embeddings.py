import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from datasets.multimodal_dataset import MultimodalDataset
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MultimodalDataset(
    "data/train.csv",
    "data/audio"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

results = []

for epoch in range(1, 71):

    video_path = f"checkpoints/video_encoder_epoch{epoch}.pt"
    audio_path = f"checkpoints/audio_encoder_epoch{epoch}.pt"

    if not os.path.exists(video_path):
        continue

    video_encoder = VideoEncoder().to(DEVICE)
    audio_encoder = AudioEncoder().to(DEVICE)

    video_encoder.load_state_dict(
        torch.load(video_path, map_location=DEVICE)
    )

    audio_encoder.load_state_dict(
        torch.load(audio_path, map_location=DEVICE)
    )

    video_encoder.eval()
    audio_encoder.eval()

    video_embeddings = []
    audio_embeddings = []

    with torch.no_grad():
        for frames, mel, _, _ in loader:

            frames = frames.to(DEVICE)
            mel = mel.to(DEVICE)

            v = video_encoder(frames)
            a = audio_encoder(mel)

            video_embeddings.append(v.cpu())
            audio_embeddings.append(a.cpu())

    video_embeddings = torch.cat(video_embeddings).numpy()
    audio_embeddings = torch.cat(audio_embeddings).numpy()

    similarity = cosine_similarity(
        audio_embeddings,
        video_embeddings
    )

    r1 = r5 = r10 = 0
    N = similarity.shape[0]

    for i in range(N):
        ranking = np.argsort(similarity[i])[::-1]

        if i in ranking[:1]:
            r1 += 1
        if i in ranking[:5]:
            r5 += 1
        if i in ranking[:10]:
            r10 += 1

    r1 /= N
    r5 /= N
    r10 /= N

    results.append((epoch, r1, r5, r10))

    print(
        f"Epoch {epoch:2d} | "
        f"R@1={r1:.4f} | "
        f"R@5={r5:.4f} | "
        f"R@10={r10:.4f}"
    )

best = max(results, key=lambda x: x[1])

print("\n=============================")
print(f"Best epoch : {best[0]}")
print(f"Recall@1   : {best[1]:.4f}")
print(f"Recall@5   : {best[2]:.4f}")
print(f"Recall@10  : {best[3]:.4f}")
print("=============================")
