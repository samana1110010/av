import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

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

video_encoder = VideoEncoder().to(DEVICE)
audio_encoder = AudioEncoder().to(DEVICE)

video_encoder.load_state_dict(
    torch.load("checkpoints/video_encoder_final.pt", map_location=DEVICE, weights_only=True)
)

audio_encoder.load_state_dict(
    torch.load("checkpoints/audio_encoder_final.pt", map_location=DEVICE, weights_only=True)
)
video_encoder.eval()
audio_encoder.eval()

video_embeddings = []
audio_embeddings = []
labels = []

with torch.no_grad():
    for frames, mel, label, _ in loader:
        frames = frames.to(DEVICE)
        mel = mel.to(DEVICE)

        video_embeddings.append(video_encoder(frames).cpu())
        audio_embeddings.append(audio_encoder(mel).cpu())
        labels.extend(label.numpy())

video_embeddings = torch.cat(video_embeddings).numpy()
audio_embeddings = torch.cat(audio_embeddings).numpy()

similarity = cosine_similarity(audio_embeddings, video_embeddings)

top1 = 0
top5 = 0

per_class_total = {}
per_class_correct = {}

N = len(labels)

for i in range(N):

    ranking = np.argsort(similarity[i])[::-1]

    gt = labels[i]

    pred1 = labels[ranking[0]]

    if gt == pred1:
        top1 += 1

    top5_labels = [labels[j] for j in ranking[:5]]

    if gt in top5_labels:
        top5 += 1

    per_class_total[gt] = per_class_total.get(gt, 0) + 1

    if gt == pred1:
        per_class_correct[gt] = per_class_correct.get(gt, 0) + 1

print("\nSemantic Retrieval Results")
print("--------------------------")
print(f"Top-1 Accuracy : {100*top1/N:.2f}%")
print(f"Top-5 Accuracy : {100*top5/N:.2f}%")

class_names = {
    0: "basketball bounce",
    1: "car passing by",
    2: "dog barking",
    3: "door slamming",
    4: "fire crackling",
    5: "footsteps on snow",
    6: "hammering nails",
    7: "raining",
    8: "ripping paper",
    9: "typing keyboard"
}

print("\nPer-class Top-1 Accuracy")
print("------------------------")

for cls in sorted(class_names.keys()):
    correct = per_class_correct.get(cls, 0)
    total = per_class_total.get(cls, 0)
    acc = 100 * correct / total
    print(f"{class_names[cls]:30s} {acc:6.2f}%")
