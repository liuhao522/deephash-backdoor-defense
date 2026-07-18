# -*- coding: utf-8 -*-
"""08_em.py — MODULE 4: EM-style iterative optimization loop."""
import numpy as np, torch, torch.optim as optim
from typing import List, Dict, Tuple

class EMIterator:
    """
    Module 4 — EM Iterative Optimization.
    Each EM iteration: Coarse Suppression → Center Reassign → Fine Restoration.
    Converges when label stabilizes and feature distance change < threshold.
    """
    def __init__(self, reconstructor, config, ordered_centers, device):
        self.rec = reconstructor; self.cfg = config
        self.centers = ordered_centers; self.device = device

    def _feat_dist(self, x_norm):
        with torch.no_grad():
            feat = self.rec.model.extract(x_norm).cpu().numpy()
        return np.array([np.linalg.norm(feat - self.centers[k]) for k in range(self.cfg.num_classes)])

    def run(self, x_init, x_ref, mask, true_label) -> Tuple[torch.Tensor, List[Dict]]:
        x = x_init.clone().detach().requires_grad_(True); x.data.clamp_(0, 1)
        cur_label = true_label; prev_dist = float('inf'); records = []

        for ei in range(self.cfg.em_max_iter):
            tc = torch.tensor(self.centers[cur_label], dtype=torch.float32).to(self.device)

            # E-Step: Coarse Suppression
            opt_c = optim.Adam([x], lr=self.cfg.lr_coarse)
            for _ in range(self.cfg.opt_steps_coarse):
                opt_c.zero_grad(); loss, _ = self.rec.compute(x, x_ref, tc, mask)
                loss.backward(); opt_c.step(); x.data.clamp_(0, 1)

            # M-Step: Reassign center
            dists = self._feat_dist(((x.clamp(0,1) - self.rec.mean) / self.rec.std))
            new_label = int(np.argmin(dists)); cur_dist = dists[new_label]

            converged = False
            if prev_dist != float('inf'):
                rel = abs(prev_dist - cur_dist) / (prev_dist + 1e-8)
                if rel < self.cfg.em_conv_threshold and new_label == cur_label:
                    converged = True

            records.append({'iter': ei, 'label_before': cur_label, 'label_after': new_label,
                           'd_center': cur_dist, 'img': x.clamp(0,1).detach().cpu().clone(),
                           'converged': converged})
            if converged: break
            prev_dist = cur_dist; cur_label = new_label

            # Fine Restoration
            tc = torch.tensor(self.centers[cur_label], dtype=torch.float32).to(self.device)
            opt_f = optim.Adam([x], lr=self.cfg.lr_fine)
            for _ in range(self.cfg.opt_steps_fine):
                opt_f.zero_grad(); loss, _ = self.rec.compute(x, x_ref, tc, mask)
                loss.backward(); opt_f.step(); x.data.clamp_(0, 1)

        return x.clamp(0,1).detach(), records
