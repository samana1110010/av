from datasets.multimodal_dataset import MultimodalDataset


dataset = MultimodalDataset(
    "data/train.csv",
    "data/audio"
)

print("Samples:", len(dataset))

frames, mel, label, video_id = dataset[0]

print("Video ID:", video_id)
print("Frames:", frames.shape)
print("Mel:", mel.shape)
print("Label:", label)
