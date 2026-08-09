import torch
import torchaudio
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, csv_file, audio_dir):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_id = row["video_id"]
        label = row["label"]

        audio_path = self.audio_dir / f"{video_id}.wav"

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                16000
            )
            waveform = resampler(waveform)

        mel = self.mel_transform(waveform)

        return mel, torch.tensor(label, dtype=torch.long)
