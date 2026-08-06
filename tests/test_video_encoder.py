import torch

from dataset import VideoDataset
from model import VideoEncoder

dataset = VideoDataset("data/train.csv")

frames, label, video_id = dataset[0]

# Add batch dimension

frames = frames.unsqueeze(0)

model = VideoEncoder()

with torch.no_grad():

    embedding = model(frames)

print("Video:", video_id)

print("Embedding shape:", embedding.shape)

print(embedding)
