import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class VideoEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        # --------------------------
        # Pretrained ResNet50
        # --------------------------

        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # Remove classifier
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Freeze backbone (recommended for only 500 samples)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --------------------------
        # Projection Head
        # --------------------------

        self.projector = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, frames):

        """
        Input:
            (B, 8, 3, 224, 224)

        Output:
            (B, 128)
        """

        B, T, C, H, W = frames.shape

        # Merge batch and time
        frames = frames.view(B * T, C, H, W)

        # Extract frame features
        features = self.backbone(frames)

        # (B*T, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)

        # (B*T, 2048)
        features = features.view(B, T, 2048)

        # Temporal mean pooling
        video_features = features.mean(dim=1)

        # Projection
        embedding = self.projector(video_features)

        # L2 Normalize
        embedding = nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding
