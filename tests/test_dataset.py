from dataset import VideoDataset

dataset = VideoDataset("data/train.csv")

print("Dataset size:", len(dataset))

frames, label, video_id = dataset[0]

print("Video ID :", video_id)
print("Frames shape:", frames.shape)
print("Label:", label)

