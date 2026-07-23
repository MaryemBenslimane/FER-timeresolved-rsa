from typing import Literal

import torch
import torch.nn as nn
import torchvision


class Small3DCNN(nn.Module):
    """Compact 3D CNN baseline for (C, T, H, W) input."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class ResNet2DTemporal(nn.Module):
    """2D CNN per-frame + temporal pooling or GRU."""

    def __init__(
        self,
        num_classes: int = 3,
        backbone: Literal["resnet18", "resnet34"] = "resnet18",
        temporal: Literal["mean", "gru"] = "gru",
        hidden_size: int = 256,
    ):
        super().__init__()

        if backbone == "resnet34":
            base = torchvision.models.resnet34(weights=None)
        else:
            base = torchvision.models.resnet18(weights=None)

        self.feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(self.feature_dim, hidden_size, batch_first=True)
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.gru = None
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)  # (B*T, D)
        feats = feats.view(b, t, -1)

        if self.temporal == "gru":
            out, _ = self.gru(feats)
            pooled = out[:, -1, :]
        else:
            pooled = feats.mean(dim=1)

        return self.classifier(pooled)
