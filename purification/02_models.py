# -*- coding: utf-8 -*-
"""02_models.py — Feature extractors: CNN + ResNet18 + MobileNetV3 + PatchGAN.

Improvements:
  - Spectral Normalization option for PatchGAN stability
  - Better CNN architecture with residual connections
  - Proper weight initialization
  - Gradient-friendly extract_with_grad
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from typing import Optional


# ================================================================
# Utility: disable inplace ReLU/Hardswish (required for MobileNetV3 training)
# ================================================================
def _disable_inplace(module):
    """Disable inplace ops in ReLU/Hardswish. Required because torchvision's
    MobileNetV3 uses inplace=True, which corrupts gradients during training."""
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, (nn.ReLU, nn.Hardswish, nn.ReLU6)):
            m.inplace = False


def _init_weights(m):
    """Kaiming init for Conv/Linear, constant for BatchNorm."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ================================================================
# CNN Feature Extractor (lightweight, for baselines & quick tests)
# ================================================================
class FeatureExtractor(nn.Module):
    """5-block CNN with residual skip connections. Better than plain conv stack."""

    def __init__(self, in_channels=3, num_classes=10, feat_dim=256):
        super().__init__()
        c = in_channels

        # Block 1: 32→16
        self.block1 = self._make_block(c, 64, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16→8
        self.block2 = self._make_block(64, 128, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 8→4
        self.block3 = self._make_block(128, 256, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 4→2
        self.block4 = self._make_block(256, 256, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes)
        )
        self.apply(_init_weights)
        _disable_inplace(self)  # fix torchvision inplace ReLU

    @staticmethod
    def _make_block(in_c, out_c, stride=1):
        """Conv-BN-ReLU-Conv-BN with residual projection if needed."""
        layers = [
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        block = nn.Sequential(*layers)

        shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        return _ResBlock(block, shortcut)

    def forward(self, x, return_feat=False):
        out = F.relu(self.block1(x))
        out = self.pool1(out)
        out = F.relu(self.block2(out))
        out = self.pool2(out)
        out = F.relu(self.block3(out))
        out = self.pool3(out)
        out = F.relu(self.block4(out))
        out = self.avgpool(out).flatten(1)
        feat = F.relu(self.fc[0](out))   # [0]=Linear(256,feat_dim)
        if return_feat:
            return feat                    # features
        feat = self.fc[1](feat)            # [1]=ReLU
        feat = self.fc[2](feat)            # [2]=Dropout
        return self.fc[3](feat)            # [3]=Linear(feat_dim,10) → logits

    @torch.no_grad()
    def extract(self, x):
        return self.forward(x, return_feat=True)

    def extract_with_grad(self, x):
        return self.forward(x, return_feat=True)


class _ResBlock(nn.Module):
    """Residual wrapper with optional projection shortcut."""
    def __init__(self, block, shortcut):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ================================================================
# ResNet-18 Extractor
# ================================================================
class ResNet18Extractor(nn.Module):
    """ResNet-18 backbone with custom feature head."""

    def __init__(self, num_classes=10, feat_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes)
        )
        _disable_inplace(self)

    def forward(self, x, return_feat=False):
        f = self.features(x).flatten(1)
        feat = F.relu(self.fc[0](f))   # [0]=Linear(512,256)
        if return_feat:
            return feat                  # 256-dim features
        feat = self.fc[1](feat)          # [1]=ReLU
        feat = self.fc[2](feat)          # [2]=Dropout
        return self.fc[3](feat)          # [3]=Linear(256,10) → logits

    @torch.no_grad()
    def extract(self, x):
        return self.forward(x, return_feat=True)

    def extract_with_grad(self, x):
        return self.forward(x, return_feat=True)


# ================================================================
# MobileNetV3-Small Extractor
# ================================================================
class MobileNetV3Extractor(nn.Module):
    """MobileNetV3-Small: lightweight, ImageNet-pretrained, 576-dim native features."""

    def __init__(self, num_classes=10, feat_dim=256):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.proj = nn.Sequential(
            nn.Linear(576, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes)
        )
        _disable_inplace(self)

    def forward(self, x, return_feat=False):
        f = self.features(x)
        f = self.avgpool(f).flatten(1)
        feat = F.relu(self.proj[0](f))  # [0]=Linear(576,256)
        if return_feat:
            return feat                  # 256-dim features
        feat = self.proj[1](feat)        # [1]=ReLU
        feat = self.proj[2](feat)        # [2]=Dropout
        return self.proj[3](feat)        # [3]=Linear(256,10) → logits

    @torch.no_grad()
    def extract(self, x):
        return self.forward(x, return_feat=True)

    def extract_with_grad(self, x):
        return self.forward(x, return_feat=True)


# ================================================================
# PatchGAN Discriminator (improved)
# ================================================================
class PatchGAN(nn.Module):
    """3-layer PatchGAN with Spectral Normalization.

    FIXED: 3 downsampling layers (not 4). For 32×32 input:
      32 → 16 → 8 → 4  (16 patches, enough spatial discrimination).
    Old 4-layer version: 32 → 2×2 (4 patches, useless).
    """

    def __init__(self, in_channels=3, base_filters=64, use_sn=True):
        super().__init__()
        self.use_sn = use_sn

        def _conv(ci, co, k, s, p, sn=True):
            conv = nn.Conv2d(ci, co, k, s, p)
            if sn and use_sn:
                conv = nn.utils.spectral_norm(conv)
            return conv

        # 3 downsampling layers: H → H/2 → H/4 → H/8
        self.layers = nn.ModuleList([
            # Layer 1: 32→16
            nn.Sequential(
                _conv(in_channels, base_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Layer 2: 16→8
            nn.Sequential(
                _conv(base_filters, base_filters * 2, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Layer 3: 8→4
            nn.Sequential(
                _conv(base_filters * 2, base_filters * 4, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            # Output: 4×4 → 4×4 (no downsampling)
            _conv(base_filters * 4, 1, 4, 1, 1, sn=False),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def adv_loss(self, x):
        """Adversarial loss: maximize discriminator output → image looks real."""
        return -self.forward(x).mean()


# ================================================================
# Factory
# ================================================================
def get_extractor(backbone: str, num_classes: int, feat_dim: int,
                  in_channels: int = 3) -> nn.Module:
    """Factory to create the right extractor."""
    if backbone == 'resnet18':
        return ResNet18Extractor(num_classes, feat_dim)
    elif backbone == 'cnn':
        return FeatureExtractor(in_channels, num_classes, feat_dim)
    else:
        return MobileNetV3Extractor(num_classes, feat_dim)
