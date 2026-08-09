from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


VIDEO_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_video_tensor(video_path: str | Path, frame_count: int = 8) -> torch.Tensor:
    """Decode uniformly sampled frames using deterministic inference transforms."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("The video has no readable frames")

        frame_numbers = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        frames = []
        for frame_number in frame_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            success, frame = capture.read()
            if not success:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(VIDEO_TRANSFORM(Image.fromarray(rgb_frame)))
    finally:
        capture.release()

    if not frames:
        raise ValueError("The video could not be decoded")
    while len(frames) < frame_count:
        frames.append(frames[-1].clone())

    return torch.stack(frames[:frame_count]).unsqueeze(0)
