# -*- coding: utf-8 -*-
"""06_gradient.py — MODULE 2b: Gradient-based spatial prior mask.

BUGFIX #1: Gradient direction now uses logit(target) - logit(true_label)
  instead of logit(target) alone. Old version computed gradient toward
  the target class — but a poisoned image is ALREADY classified as target,
  so the gradient was ~0 everywhere → mask was useless (~1.0).

BUGFIX #2: Integrated Gradients baseline corrected to black PIXEL image
  (old version used zeros in normalized space = gray image).

Improvements:
  - SmoothGrad with configurable noise samples
  - Integrated Gradients from black pixel baseline
  - Adaptive mask stretching with percentile-based contrast
  - Gaussian smoothing with configurable kernel size
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GradientMaskGenerator:
    """Module 2b — Gradient Heatmap Spatial Prior.

    Computes the gradient of (logit_target - logit_true) w.r.t. input.
    Pixels with HIGH gradient are those that push the image FROM true class
    TOWARD target class — these are the TRIGGER pixels.

    Mask = 1 - normalized_gradient: trigger pixels get LOW weight.
    """

    def __init__(self, model, target_class: int, device: torch.device,
                 config=None, mean=None, std=None):
        self.model = model
        self.target = target_class
        self.device = device
        self.mean = mean
        self.std = std

        self.method = getattr(config, 'grad_method', 'smoothgrad') if config else 'smoothgrad'
        self.n_samples = getattr(config, 'grad_n_samples', 15) if config else 15
        self.noise_std = getattr(config, 'grad_noise_std', 0.08) if config else 0.08
        self.mask_floor = getattr(config, 'grad_mask_floor', 0.15) if config else 0.15
        self.mask_ceil = getattr(config, 'grad_mask_ceil', 1.0) if config else 1.0
        self.smooth_kernel = getattr(config, 'grad_smooth_kernel', 5) if config else 5

    # ================================================================
    # Main entry point — NOW REQUIRES true_label
    # ================================================================
    def generate(self, img_norm: torch.Tensor,
                 true_label: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """Generate gradient-based spatial mask.

        Args:
            img_norm:   [1, C, H, W] — normalized image
            true_label: ground-truth label. If None, falls back to
                        target_class gradient only (old behavior, weak).

        Returns:
            (mask [1, 1, H, W], diagnostics dict)
        """
        label_for_grad = true_label if true_label is not None else self.target

        # Duplicate to [2, C, H, W] to satisfy BN's >1 sample requirement.
        # We'll only use the first sample's gradient.
        x_batch = torch.cat([img_norm, img_norm.clone()], dim=0)

        if self.method == 'integrated':
            grad_map = self._integrated_gradients(x_batch, label_for_grad)
        elif self.method == 'smoothgrad':
            grad_map = self._smooth_grad(x_batch, label_for_grad)
        else:
            grad_map = self._vanilla_grad(x_batch, label_for_grad)

        # Take only first sample's gradient
        grad_map = grad_map[:1]

        # Post-process → mask
        mask, diag = self._grad_to_mask(grad_map, img_norm)
        diag['true_label'] = label_for_grad
        return mask, diag

    # ================================================================
    # Gradient methods — FIXED: use (target - true) logit difference
    # ================================================================
    def _compute_logit_diff(self, logits, true_label):
        """Compute logit(target) - logit(true_label) for batch.

        Returns scalar that we backprop through.
        """
        return (logits[:, self.target] - logits[:, true_label]).sum()

    def _vanilla_grad(self, x_batch, true_label):
        """Single backward pass. x_batch is [2, C, H, W]."""
        x = x_batch.clone().detach().requires_grad_(True)
        logits = self.model(x)
        diff = self._compute_logit_diff(logits, true_label)
        diff.backward()
        g = x.grad.detach().abs().mean(1, keepdim=True)
        return g

    def _smooth_grad(self, x_batch, true_label):
        """SmoothGrad: average over N noisy copies."""
        grad_sum = torch.zeros_like(x_batch[:, :1])
        for _ in range(self.n_samples):
            noise = torch.randn_like(x_batch) * self.noise_std
            x = (x_batch + noise).clone().detach().requires_grad_(True)
            logits = self.model(x)
            diff = self._compute_logit_diff(logits, true_label)
            diff.backward()
            grad_sum = grad_sum + x.grad.detach().abs().mean(1, keepdim=True)
        return grad_sum / self.n_samples

    def _integrated_gradients(self, x_batch, true_label):
        """Integrated Gradients from BLACK pixel baseline."""
        if self.mean is not None and self.std is not None:
            baseline = (-self.mean / self.std).expand_as(x_batch)
        else:
            baseline = torch.full_like(x_batch, -2.0)

        steps = self.n_samples
        grad_sum = torch.zeros_like(x_batch[:, :1])

        for k in range(steps):
            alpha = k / (steps - 1) if steps > 1 else 0.5
            interpolated = baseline + alpha * (x_batch - baseline)
            x = interpolated.clone().detach().requires_grad_(True)
            logits = self.model(x)
            diff = self._compute_logit_diff(logits, true_label)
            diff.backward()
            grad_sum = grad_sum + x.grad.detach().abs().mean(1, keepdim=True)

        avg_grad = grad_sum / steps
        return avg_grad * (x_batch - baseline).abs().mean(1, keepdim=True)

    # ================================================================
    # Gradient → Mask
    # ================================================================
    def _grad_to_mask(self, grad_map, img_norm):
        """Convert raw gradient map to spatial mask [floor, 1.0]."""
        # Normalize
        grad_norm = grad_map / (grad_map.max() + 1e-8)

        # Gaussian smooth
        kernel_size = self.smooth_kernel
        sigma = kernel_size / 3.0
        grad_smooth = self._gaussian_blur(grad_norm, kernel_size, sigma)
        grad_smooth = grad_smooth / (grad_smooth.max() + 1e-8)

        # Adaptive percentile stretch
        g_flat = grad_smooth.flatten()
        p_low = torch.quantile(g_flat, 0.05)
        p_high = torch.quantile(g_flat, 0.95)
        if p_high - p_low > 1e-6:
            grad_stretched = (grad_smooth - p_low) / (p_high - p_low)
        else:
            grad_stretched = grad_smooth
        grad_stretched = grad_stretched.clamp(0, 1)

        # Invert: high gradient → trigger → LOW weight
        mask = 1.0 - grad_stretched

        # Scale to [floor, 1.0]
        mask = self.mask_floor + (self.mask_ceil - self.mask_floor) * mask

        diag = {
            'grad_raw': grad_norm.squeeze().cpu().numpy(),
            'grad_smooth': grad_smooth.squeeze().cpu().numpy(),
            'grad_stretched': grad_stretched.squeeze().cpu().numpy(),
            'mask': mask.squeeze().cpu().numpy(),
            'mask_mean': mask.mean().item(),
            'mask_min': mask.min().item(),
            'mask_max': mask.max().item(),
            'mask_std': mask.std().item(),
            'method': self.method,
            'p_low': p_low.item(), 'p_high': p_high.item(),
        }
        return mask.detach(), diag

    @staticmethod
    def _gaussian_blur(x, kernel_size, sigma):
        """2D Gaussian blur via depthwise conv."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding)

    def diagnose(self, img_norm, mask=None, true_label=None):
        """Return diagnostic info about the gradient signal."""
        if mask is None:
            mask, _ = self.generate(img_norm, true_label)
        return {
            'mask_mean': mask.mean().item(),
            'mask_min': mask.min().item(),
            'mask_max': mask.max().item(),
            'mask_std': mask.std().item(),
            'fraction_suppressed': (mask < 0.5).float().mean().item(),
            'method': self.method,
        }
