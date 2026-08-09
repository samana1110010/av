import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.audio_encoder import AudioEncoder
from datasets.audio_io import load_audio


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "data/train.csv"
AUDIO_DIR = "data/audio"
CHECKPOINT = "checkpoints/audio_encoder_final.pt"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class AudioGalleryDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id = row["video_id"]
        audio_path = os.path.join(
            self.audio_dir,
            f"{video_id}.wav"
        )

        waveform, sample_rate = load_audio(audio_path)

        # Stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Resample -> 16 kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                16000
            )
            waveform = resampler(waveform)

        # Mel spectrogram
        mel = self.mel_transform(waveform)

        # Same target length as the training dataset
        TARGET_TIME = 800

        if mel.shape[2] > TARGET_TIME:
            mel = mel[:, :, :TARGET_TIME]

        elif mel.shape[2] < TARGET_TIME:
            pad = TARGET_TIME - mel.shape[2]
            mel = F.pad(mel, (0, pad))

        return mel, video_id


print("Loading audio encoder...")

model = AudioEncoder().to(DEVICE)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()

print(f"Using device: {DEVICE}")

dataset = AudioGalleryDataset(
    CSV_PATH,
    AUDIO_DIR
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

all_embeddings = []
all_ids = []

print(f"Generating embeddings for {len(dataset)} training samples...")

with torch.no_grad():
    for batch_idx, (mel, video_ids) in enumerate(loader):

        mel = mel.to(DEVICE)

        embeddings = model(mel)

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        all_ids.extend(video_ids)

        print(
            f"\rProcessed {len(all_ids)}/{len(dataset)}",
            end=""
        )

print()

embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

ids = np.array(all_ids)

np.save(
    os.path.join(OUTPUT_DIR, "audio_embeddings.npy"),
    embeddings
)

np.save(
    os.path.join(OUTPUT_DIR, "audio_ids.npy"),
    ids
)

print()
print("Done.")
print("Embeddings shape:", embeddings.shape)
print("IDs shape:", ids.shape)
print()
print("Saved:")
print("  embeddings/audio_embeddings.npy")
print("  embeddings/audio_ids.npy")
