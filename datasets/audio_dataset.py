import torch
import torchaudio
import pandas as pd

from datasets.audio_io import load_audio
from pathlib import Path
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    TARGET_TIME = 800

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

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found for video {video_id!r}: {audio_path}"
            )

        waveform, sample_rate = load_audio(audio_path)

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

        if mel.shape[-1] > self.TARGET_TIME:
            mel = mel[..., :self.TARGET_TIME]
        elif mel.shape[-1] < self.TARGET_TIME:
            mel = torch.nn.functional.pad(
                mel, (0, self.TARGET_TIME - mel.shape[-1])
            )

        return mel, torch.tensor(label, dtype=torch.long)
