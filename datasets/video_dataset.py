import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        video_id = row.video_id
        label = int(row.label)

        folder = Path(row.frame_folder)

        images = sorted(folder.glob("*.jpg"))

        frames = []

        for img in images:
            image = Image.open(img).convert("RGB")
            image = self.transform(image)
            frames.append(image)

        # Pad videos with fewer than 8 frames
        while len(frames) < 8:
            frames.append(frames[-1])

        # Keep exactly 8 frames
        frames = frames[:8]

        frames = torch.stack(frames)

        return frames, torch.tensor(label), video_id
