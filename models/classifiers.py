import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder

class VideoClassifier(nn.Module):
    def __init__(self, video_encoder=None, num_classes=10, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.video_encoder = video_encoder if video_encoder is not None else VideoEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, frames):
        # Extract features (B, 2048)
        features = self.video_encoder(frames, return_features=True)
        logits = self.classifier(features)
        return logits


class AudioClassifier(nn.Module):
    def __init__(self, audio_encoder=None, num_classes=10, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, mel):
        # Extract features (B, 512)
        features = self.audio_encoder(mel, return_features=True)
        logits = self.classifier(features)
        return logits


class FusionClassifier(nn.Module):
    def __init__(self, video_encoder=None, audio_encoder=None, num_classes=10, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.video_encoder = video_encoder if video_encoder is not None else VideoEncoder()
        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, frames, mel):
        video_features = self.video_encoder(frames, return_features=True)
        audio_features = self.audio_encoder(mel, return_features=True)

        combined = torch.cat([video_features, audio_features], dim=1)
        logits = self.classifier(combined)
        return logits
