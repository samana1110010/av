import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class AudioEncoder(nn.Module):

    def __init__(self, embedding_dim=128, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()

        # Pretrained ResNet18
        backbone = resnet18(weights=weights)
        
        # Remove classifier and avgpool (we will use adaptive pool or flatten after the backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone EXCEPT layer4 (last block)
        for name, param in self.backbone.named_parameters():
            if "7" not in name.split(".")[0]: # The backbone is a Sequential, layer4 is at index 7
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x, return_features=False):
        """
        Input:
            (B, 1, 128, T)

        Output:
            (B, embedding_dim) or (B, 512) if return_features is True
        """

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "Expected mel spectrograms with shape (batch, 1, bins, time), "
                f"got {tuple(x.shape)}"
            )

        # Convert 1-channel Mel to 3-channel for ResNet
        x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)

        if return_features:
            return x.flatten(1)

        x = self.projector(x)

        x = F.normalize(x, p=2, dim=1)

        return x
