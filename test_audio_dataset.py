from datasets.audio_dataset import AudioDataset


dataset = AudioDataset(
    "data/train.csv",
    "data/audio"
)

print("Number of samples:", len(dataset))

mel, label = dataset[0]

print("Mel shape:", mel.shape)
print("Label:", label)
