import numpy as np
import pandas as pd
import torch
import torchaudio

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.audio_io import load_audio


class MultimodalDataset(Dataset):

    def __init__(self, csv_file, audio_dir, is_training=True):

        self.df = pd.read_csv(csv_file)
        self.audio_dir = Path(audio_dir)
        self.is_training = is_training

        # Video augmentation
        if is_training:
            self.video_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.video_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.mel_transform = transforms.Compose([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128
            ),
            torchaudio.transforms.AmplitudeToDB()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        video_id = row["video_id"]
        label = int(row["label"])

        # -------------------------
        # Load video frames
        # -------------------------

        frame_folder = Path(row["frame_folder"])

        images = sorted(frame_folder.glob("*.jpg"))

        if not images:
            raise FileNotFoundError(
                f"No JPEG frames found for video {video_id!r} in {frame_folder}"
            )

        frames = []

        for image_path in images:

            image = Image.open(image_path).convert("RGB")
            image = self.video_transform(image)

            frames.append(image)

        # Pad if fewer than 8 frames
        while len(frames) < 8:
            frames.append(frames[-1])

        # Keep exactly 8 frames (uniformly sampled)
        indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
        frames = [frames[i] for i in indices]

        frames = torch.stack(frames)

        # -------------------------
        # Load audio
        # -------------------------

        audio_path = self.audio_dir / f"{video_id}.wav"

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found for video {video_id!r}: {audio_path}"
            )

        waveform, sample_rate = load_audio(audio_path)

        # Convert stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Resample if necessary
        if sample_rate != 16000:

            resampler = torchaudio.transforms.Resample(
                sample_rate,
                16000
            )

            waveform = resampler(waveform)

        # -------------------------
        # Mel spectrogram
        # -------------------------

        mel = self.mel_transform(waveform)

        TARGET_TIME = 800

        # Trim if too long
        if mel.shape[2] > TARGET_TIME:
            mel = mel[:, :, :TARGET_TIME]

        # Pad if too short
        elif mel.shape[2] < TARGET_TIME:
            pad = TARGET_TIME - mel.shape[2]
            mel = torch.nn.functional.pad(
                mel,
                (0, pad)
            )

        return (
            frames,
            mel,
            torch.tensor(label, dtype=torch.long),
            video_id
        )
