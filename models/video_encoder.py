import torch
import torch.nn as nn
import torchvision.models as models


class VideoEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        # --------------------------
        # Pretrained ResNet18
        # --------------------------

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Remove classifier
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # --------------------------
        # Projection Head
        # --------------------------

        self.projector = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim)

        )

    def forward(self, frames):

        """
        Input

        (B,8,3,224,224)

        Output

        (B,128)
        """

        B, T, C, H, W = frames.shape

        # Merge batch and time

        frames = frames.view(B * T, C, H, W)

        # ResNet18

        features = self.backbone(frames)

        # (B*T,512,1,1)

        features = features.squeeze(-1).squeeze(-1)

        # (B*T,512)

        features = features.view(B, T, 512)

        # Average across frames

        video_features = features.mean(dim=1)

        embedding = self.projector(video_features)

        # Normalize

        embedding = nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding
