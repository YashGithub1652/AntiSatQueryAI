"""
SatQuery AI — ChangeFormer Fallback
=====================================
Lightweight Siamese ResNet-18 change detector.
Used when the official ChangeFormer (justchenhao/ChangeFormer) is not cloned.
Implements the same interface as ChangeFormer so engines are interchangeable.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SiameseChangeDetector(nn.Module):
    """
    Siamese ResNet-18 based change detector.
    Encodes T1 and T2 independently, then detects changes via absolute difference.
    Not as accurate as ChangeFormer but functional for demos without the full repo.
    """

    def __init__(self):
        super().__init__()

        # Shared encoder (same weights for T1 and T2 — Siamese)
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC + AvgPool — keep feature extractor only
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, 8, 8]

        # Change decoder: takes |f1 - f2| → binary map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),     # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),      # 256x256
            nn.Sigmoid()
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t1: [B, 3, H, W] normalized T1 image
            t2: [B, 3, H, W] normalized T2 image
        Returns:
            change_map: [B, 1, H, W] change probability in [0, 1]
        """
        f1 = self.encoder(t1)  # [B, 512, 8, 8]
        f2 = self.encoder(t2)  # [B, 512, 8, 8]

        diff = torch.abs(f1 - f2)      # Change magnitude
        change_map = self.decoder(diff)
        return change_map
