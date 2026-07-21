# -*- coding: utf-8 -*-
"""07_reconstructor.py — MODULE 3: Feature-constrained image reconstruction.

COMPREHENSIVE REWRITE. Problems with old version:
  1. LPIPS on 224×224 for 32×32 images → interpolating 49×, meaningless
  2. L_feat only used true center, no auxiliary constraints
  3. No center-distance loss → EM optimization drifts

New design:
  - Configurable LPIPS resize (default 64 for CIFAR-10, 224 for ImageNet-scale)
  - Center-distance loss: directly penalize distance to target center
  - Feature consistency loss: preserve non-trigger features
  - Adaptive loss weighting based on optimization phase
  - Better numerical stability
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FeatureReconstructor:
    """Module 3 — Feature-Constrained Reconstruction.

    Composite loss:
      L = λ_feat · L_feat     (logits/feature consistency with target center)
        + λ_center · L_center (direct center distance)
        + λ_perc · LPIPS      (perceptual similarity to reference)
        + λ_adv · L_adv       (PatchGAN realism constraint)
        + λ_pix · L_pix       (masked pixel-wise similarity)
        + λ_tv · TV           (total variation smoothness)

    When use_logits_space=True, L_feat uses classifier logits (10-dim)
    instead of raw features (256-dim). Logits space has natural class
    separation.
    """

    def __init__(self, model, patchgan, lpips_fn, config, mean_t, std_t):
        self._anchor_center = None
        self.model = model
        self.patchgan = patchgan
        self.lpips_fn = lpips_fn
        self.cfg = config
        self.mean = mean_t
        self.std = std_t

        self.lpips_size = getattr(config, 'lpips_resize', 64)
        self.use_logits = getattr(config, 'use_logits_space', False)

        self.w_feat = config.lambda_feat
        self.w_center = getattr(config, 'lambda_center', 0.5)
        self.w_perc = config.lambda_perc
        self.w_adv = config.lambda_adv
        self.w_pix = config.lambda_pix
        self.w_tv = config.lambda_tv

    def set_anchor_center(self, center):
        self._anchor_center = center

    # ================================================================
    # LPIPS preprocessing
    # ================================================================
    def _prep_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare image for LPIPS: resize to target size and scale to [-1, 1].

        Old version used 224 which is 49× magnification for 32px images.
        New version uses configurable size (default 64, reasonable for CIFAR-10).
        """
        # Resize
        if x.size(-1) != self.lpips_size:
            x = F.interpolate(x, size=(self.lpips_size, self.lpips_size),
                              mode='bilinear', align_corners=False)
        # Scale [0, 1] → [-1, 1] (LPIPS expects this)
        return x * 2.0 - 1.0

    # ================================================================
    # Main loss computation
    # ================================================================
    def compute(self, x_opt: torch.Tensor, x_ref: torch.Tensor,
                target_center: torch.Tensor, mask: torch.Tensor,
                phase: str = 'coarse') -> Tuple[torch.Tensor, Dict]:
        """Compute total purification loss.

        BUGFIX #3: x_ref is now the FREQUENCY-FILTERED image (not original
        poisoned). Old version used original poisoned image as reference,
        which meant L_perc+L_pix pulled toward the TRIGGER while L_feat
        pulled toward the clean center — contradictory objectives.

        With frequency-filtered reference: L_perc+L_pix pull toward a
        trigger-suppressed version, which is aligned with L_feat's goal.

        Args:
            x_opt:   [1, C, H, W] — current optimized image (in [0,1])
            x_ref:   [1, C, H, W] — reference (FREQ-FILTERED, not original!)
            target_center: [feat_dim] — feature center of CURRENT target class
            mask:    [1, 1, H, W] — spatial weight mask (low on trigger regions)
            phase:   'coarse' or 'fine' — adjusts loss weights
        """
        B, C, H, W = x_opt.shape

        x_c = x_opt.clamp(0, 1)
        x_n = (x_c - self.mean) / self.std

        # ---- 1. Feature loss (logits or features) ----
        if self.use_logits:
            repr_vec = self.model(x_n)        # logits [B, 10]
        else:
            repr_vec = self.model.extract_with_grad(x_n)  # features [B, 256]
        # L2-normalize: align DIRECTION with center, not magnitude
        repr_norm = repr_vec / (torch.norm(repr_vec) + 1e-8)
        L_feat = torch.norm(repr_norm - target_center) ** 2

        # ---- 2. Center distance (on normalized vectors) ----
        L_center = torch.norm(repr_norm - target_center)

        # ---- 3. LPIPS: optimize vs FREQ-FILTERED ref (no trigger!) ----
        x_lpips_opt = self._prep_lpips(x_c)
        x_lpips_ref = self._prep_lpips(x_ref)
        L_perc = self.lpips_fn(x_lpips_opt, x_lpips_ref).mean()

        # ---- 4. PatchGAN: naturalness prior ----
        x_adv = x_c if C >= 3 else x_c.repeat(1, 3, 1, 1)
        L_adv = self.patchgan.adv_loss(x_adv)

        # ---- 5. Masked pixel loss vs FREQ-FILTERED ref ----
        mask_c = mask.repeat(1, C, 1, 1)
        # In trigger regions (mask≈0): pixel loss weak → free to change
        # In clean regions (mask≈1): pixel loss strong → stay close to ref
        L_pix = ((x_c - x_ref).abs() * mask_c).mean()

        # ---- 6. Total Variation ----
        tv_h = torch.mean(torch.abs(x_c[:, :, :-1, :] - x_c[:, :, 1:, :]))
        tv_w = torch.mean(torch.abs(x_c[:, :, :, :-1] - x_c[:, :, :, 1:]))
        L_tv = tv_h + tv_w

        # ---- Phase-dependent weighting ----
        if phase == 'coarse':
            # Coarse: strong feature pull, weak visual constraints
            w_feat_eff = self.w_feat * 1.8
            w_center_eff = self.w_center * 1.8
            w_perc_eff = self.w_perc * 0.3
            w_adv_eff = self.w_adv * 0.3
            w_pix_eff = self.w_pix * 0.3
            w_tv_eff = self.w_tv * 0.3
        else:  # fine
            # Fine: balanced
            w_feat_eff = self.w_feat * 0.8
            w_center_eff = self.w_center * 0.8
            w_perc_eff = self.w_perc * 1.2
            w_adv_eff = self.w_adv * 1.2
            w_pix_eff = self.w_pix * 1.2
            w_tv_eff = self.w_tv * 1.2

        total = (w_feat_eff * L_feat + w_center_eff * L_center +
                 w_perc_eff * L_perc + w_adv_eff * L_adv +
                 w_pix_eff * L_pix + w_tv_eff * L_tv)

        losses = {
            'L_feat': L_feat.item(), 'L_center': L_center.item(),
            'L_perc': L_perc.item(), 'L_adv': L_adv.item(),
            'L_pix': L_pix.item(), 'L_tv': L_tv.item(),
            'total': total.item(), 'phase': phase,
        }
        return total, losses

    # ================================================================
    # LPIPS-free fallback (when LPIPS is unreliable)
    # ================================================================
    def compute_no_lpips(self, x_opt, x_ref, target_center, mask, phase='coarse'):
        """Alternative loss without LPIPS — uses only feature + pixel + TV.

        Useful when LPIPS is not meaningful (e.g., very small images).
        """
        B, C, H, W = x_opt.shape
        x_c = x_opt.clamp(0, 1)
        x_n = (x_c - self.mean) / self.std

        feats = self.model.extract_with_grad(x_n)
        feats_norm = feats / (torch.norm(feats) + 1e-8)
        L_feat = torch.norm(feats_norm - target_center) ** 2
        L_center = torch.norm(feats_norm - target_center)

        mask_c = mask.repeat(1, C, 1, 1)
        L_pix = ((x_c - x_ref).abs() * mask_c).mean()

        tv_h = torch.mean(torch.abs(x_c[:, :, :-1, :] - x_c[:, :, 1:, :]))
        tv_w = torch.mean(torch.abs(x_c[:, :, :, :-1] - x_c[:, :, :, 1:]))
        L_tv = tv_h + tv_w

        # Stronger feature emphasis without LPIPS
        total = (self.w_feat * 2.0 * L_feat + self.w_center * 2.0 * L_center +
                 self.w_pix * 2.0 * L_pix + self.w_tv * 2.0 * L_tv)

        losses = {
            'L_feat': L_feat.item(), 'L_center': L_center.item(),
            'L_perc': 0.0, 'L_adv': 0.0,
            'L_pix': L_pix.item(), 'L_tv': L_tv.item(),
            'total': total.item(), 'phase': phase + '_no_lpips',
        }
        return total, losses
