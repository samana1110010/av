from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalDataset


dataset = MultimodalDataset(
    "data/train.csv",
    "data/audio"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

frames, mel, labels, video_ids = next(iter(loader))

print("Frames:", frames.shape)
print("Mel:", mel.shape)
print("Labels:", labels.shape)
print("Video IDs:", video_ids)
