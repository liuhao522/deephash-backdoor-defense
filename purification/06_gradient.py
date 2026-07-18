# -*- coding: utf-8 -*-
"""06_gradient.py — MODULE 2b: Gradient-based spatial prior mask."""
import torch, torch.nn.functional as F
from typing import Dict, Tuple

class GradientMaskGenerator:
    """
    Module 2b — Gradient Heatmap Spatial Prior.
    - Backpropagates target-class logit to get pixel-level gradient.
    - Smooths gradient → creates weight mask: low weight on trigger regions.
    - Mask range: [0.3, 1.0].
    """
    def __init__(self, model, target_class: int, device: torch.device):
        self.model = model; self.target = target_class; self.device = device

    def generate(self, img_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args: img_norm [1, C, H, W] — normalized image
        Returns: (mask [1, 1, H, W], diagnostics)
        """
        x = img_norm.clone().detach().requires_grad_(True)
        logits = self.model(x); logits[0, self.target].backward()
        grad = x.grad.abs().mean(1, keepdim=True)  # [1,1,H,W]
        grad_norm = grad / (grad.max() + 1e-8)
        grad_smooth = F.avg_pool2d(grad_norm, 3, 1, 1)
        grad_smooth = grad_smooth / (grad_smooth.max() + 1e-8)
        mask = 1.0 - grad_smooth
        mask = (0.3 + 0.7 * mask.clamp(0, 1)).detach()

        diag = {'grad_raw': grad_norm.squeeze().cpu().numpy(),
                'grad_smooth': grad_smooth.squeeze().cpu().numpy(),
                'mask': mask.squeeze().cpu().numpy(),
                'mask_mean': mask.mean().item(), 'mask_min': mask.min().item()}
        return mask, diag
