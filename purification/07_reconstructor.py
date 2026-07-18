# -*- coding: utf-8 -*-
"""07_reconstructor.py — MODULE 3: Feature-constrained image reconstruction."""
import torch, torch.nn.functional as F
from typing import Dict, Tuple

class FeatureReconstructor:
    """
    Module 3 — Feature-Constrained Reconstruction.
    Loss = λ_feat·L_feat + λ_perc·LPIPS + λ_adv·PatchGAN + λ_pix·L_pix + λ_tv·TV
    """
    def __init__(self, model, patchgan, lpips_fn, config, mean_t, std_t):
        self.model = model; self.patchgan = patchgan; self.lpips = lpips_fn
        self.cfg = config; self.mean = mean_t; self.std = std_t

    def _to_lpips(self, x):
        return F.interpolate(x, size=(224,224), mode='bilinear') * 2 - 1

    def compute(self, x_opt, x_ref, target_center, mask) -> Tuple[torch.Tensor, Dict]:
        x_c = x_opt.clamp(0, 1)
        x_n = (x_c - self.mean) / self.std
        feats = self.model.extract_with_grad(x_n)

        L_feat = torch.norm(feats - target_center)**2
        L_perc = self.lpips(self._to_lpips(x_c), self._to_lpips(x_ref)).mean()
        x3 = x_c if x_c.size(1) >= 3 else x_c.repeat(1,3,1,1)
        L_adv = self.patchgan.adv_loss(x3)
        L_pix = ((x_c - x_ref).abs() * mask.repeat(1, x_c.size(1), 1, 1)).mean()
        L_tv = (torch.mean(torch.abs(x_c[:,:,:-1,:] - x_c[:,:,1:,:])) +
                torch.mean(torch.abs(x_c[:,:,:,:-1] - x_c[:,:,:,1:])))

        total = (self.cfg.lambda_feat * L_feat + self.cfg.lambda_perc * L_perc +
                 self.cfg.lambda_adv * L_adv + self.cfg.lambda_pix * L_pix +
                 self.cfg.lambda_tv * L_tv)

        losses = {'L_feat': L_feat.item(), 'L_perc': L_perc.item(), 'L_adv': L_adv.item(),
                  'L_pix': L_pix.item(), 'L_tv': L_tv.item(), 'total': total.item()}
        return total, losses
