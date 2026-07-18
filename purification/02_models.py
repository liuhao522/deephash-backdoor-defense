# -*- coding: utf-8 -*-
"""02_models.py — Feature extractors: CNN + ResNet18 + MobileNetV3 + PatchGAN."""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):
    """Lightweight CNN (for quick experiments)."""
    def __init__(self, in_channels=3, num_classes=10, feat_dim=256):
        super().__init__()
        c = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(256, feat_dim), nn.ReLU(), nn.Linear(feat_dim, num_classes))

    def forward(self, x, return_feat=False):
        conv_out = self.conv(x); flat = conv_out.view(conv_out.size(0), -1)
        feat = F.relu(self.fc[0](flat))
        return feat if return_feat else self.fc[1](feat)

    @torch.no_grad()
    def extract(self, x): return self.forward(x, return_feat=True)
    def extract_with_grad(self, x): return self.forward(x, return_feat=True)


class ResNet18Extractor(nn.Module):
    """ResNet-18 backbone with custom feature head (better cluster separation)."""
    def __init__(self, num_classes=10, feat_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.fc = nn.Sequential(
            nn.Linear(512, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x, return_feat=False):
        f = self.features(x).flatten(1)   # [B, 512]
        feat = F.relu(self.fc[0](f))       # [B, feat_dim]
        return feat if return_feat else self.fc[1](feat)

    @torch.no_grad()
    def extract(self, x): return self.forward(x, return_feat=True)
    def extract_with_grad(self, x): return self.forward(x, return_feat=True)


class MobileNetV3Extractor(nn.Module):
    """MobileNetV3-Small: lightweight, ImageNet-pretrained, 576-dim native features."""
    def __init__(self, num_classes=10, feat_dim=256):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # conv layers
        self.avgpool = backbone.avgpool     # [B, 576, 1, 1]
        self.proj = nn.Sequential(
            nn.Linear(576, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x, return_feat=False):
        f = self.features(x)
        f = self.avgpool(f).flatten(1)  # [B, 576]
        feat = F.relu(self.proj[0](f))  # [B, feat_dim]
        return feat if return_feat else self.proj[1](feat)

    @torch.no_grad()
    def extract(self, x): return self.forward(x, return_feat=True)
    def extract_with_grad(self, x): return self.forward(x, return_feat=True)


class PatchGAN(nn.Module):
    """3-layer PatchGAN for natural image prior."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 1),
        )
    def forward(self, x): return self.net(x)
    def adv_loss(self, x): return -self.forward(x).mean()
