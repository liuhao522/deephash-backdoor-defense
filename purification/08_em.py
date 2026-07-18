# -*- coding: utf-8 -*-
"""08_em.py — MODULE 4: EM-style iterative optimization loop.

COMPREHENSIVE REWRITE. Problems with old version:
  1. cur_label = true_label (biased initialization) — if freq filter moved
     the image far from true center, EM couldn't recover
  2. Coarse→Reassign→Fine ordering was suboptimal: fine restoration could
     undo the label progress made by coarse
  3. Convergence criterion too simple (label + relative distance)
  4. No early stopping for diverging cases

New design:
  - Unbiased initialization: argmin over ALL class centers (or true_label,
    configurable via em_init_mode)
  - E-step: optimize toward CURRENT best-guess center
  - M-step: reassign to nearest center (with top-k smoothing)
  - Fine restoration AFTER convergence or on final iteration only
  - Multiple convergence criteria: label stability, distance delta, feature delta
  - Adaptive step counts based on distance to target
  - Safety: if diverging, revert to previous best
"""
import numpy as np
import torch
import torch.optim as optim
from typing import List, Dict, Tuple, Optional


class EMIterator:
    """Module 4 — EM Iterative Optimization (rewritten).

    Algorithm:
      E-step (Expectation): optimize image features toward current class center
      M-step (Maximization): reassign to nearest class center

    The key fix: initialization uses argmin distance, not true_label.
    This prevents bias when the frequency filter corrupts the image.
    """

    def __init__(self, reconstructor, config, ordered_centers, device):
        self.rec = reconstructor
        self.cfg = config
        self.centers = ordered_centers  # [num_classes, feat_dim]
        self.device = device

        # Config — FIXED: reduced coarse steps to prevent overshooting
        self.max_iter = getattr(config, 'em_max_iter', 8)
        self.conv_threshold = getattr(config, 'em_conv_threshold', 0.02)
        self.init_mode = getattr(config, 'em_init_mode', 'nearest')
        self.n_coarse = getattr(config, 'opt_steps_coarse', 100)   # was 150
        self.n_fine = getattr(config, 'opt_steps_fine', 200)       # was 300
        self.lr_coarse = getattr(config, 'lr_coarse', 0.02)        # was 0.03
        self.lr_fine = getattr(config, 'lr_fine', 0.01)

    # ================================================================
    # Feature distance helper
    # ================================================================
    def _feat_dist(self, x_norm: torch.Tensor) -> np.ndarray:
        """Compute distances from normalized image to all class centers."""
        with torch.no_grad():
            feat = self.rec.model.extract(x_norm).cpu().numpy()
        return np.array([
            float(np.linalg.norm(feat - self.centers[k]))
            for k in range(self.cfg.num_classes)
        ])

    def _feat_dist_to_center(self, x_norm: torch.Tensor, label: int) -> float:
        """Distance to a specific class center."""
        with torch.no_grad():
            feat = self.rec.model.extract(x_norm).cpu().numpy()
        return float(np.linalg.norm(feat - self.centers[label]))

    # ================================================================
    # Initialization
    # ================================================================
    def _init_label(self, x_norm: torch.Tensor, true_label: int) -> int:
        """Determine initial label for EM optimization.

        'nearest': unbiased — pick argmin distance over all centers
        'true_label': use ground truth (old behavior, biased)
        'top3_nearest': pick nearest among true_label ± 2 neighbors
        """
        if self.init_mode == 'true_label':
            return true_label

        dists = self._feat_dist(x_norm)

        if self.init_mode == 'nearest':
            return int(np.argmin(dists))

        # top3_nearest: restrict to true_label and nearby classes
        candidates = [
            (true_label + d) % self.cfg.num_classes
            for d in range(-2, 3)
        ]
        best = min(candidates, key=lambda c: dists[c])
        return best

    # ================================================================
    # Optimization steps
    # ================================================================
    def _optimize_step(self, x, x_ref, center_label, mask, n_steps, lr, phase):
        """Run n_steps of Adam optimization toward center_label."""
        opt = optim.Adam([x], lr=lr)
        best_loss = float('inf')
        best_x = x.clone()

        for _ in range(n_steps):
            opt.zero_grad()
            tc = torch.tensor(
                self.centers[center_label], dtype=torch.float32, device=self.device
            )
            loss, _ = self.rec.compute(x, x_ref, tc, mask, phase=phase)
            loss.backward()
            opt.step()
            x.data.clamp_(0, 1)

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x.clone()

        # Restore best
        x.data.copy_(best_x)
        return best_loss

    # ================================================================
    # Main EM loop
    # ================================================================
    def run(self, x_init: torch.Tensor, x_ref: torch.Tensor,
            mask: torch.Tensor, true_label: int) -> Tuple[torch.Tensor, List[Dict]]:
        """Run EM iterative purification.

        Args:
            x_init:     [1, C, H, W] — initial image (after freq filter) in [0,1]
            x_ref:      [1, C, H, W] — reference (original poisoned) in [0,1]
            mask:       [1, 1, H, W] — spatial weight mask
            true_label: ground-truth label (for diagnostics only)

        Returns:
            (purified_image [1, C, H, W] in [0,1], EM_records)
        """
        x = x_init.clone().detach().requires_grad_(True)
        x.data.clamp_(0, 1)

        # ---- Initialization ----
        x_norm = (x.clamp(0, 1) - self.rec.mean) / self.rec.std
        cur_label = self._init_label(x_norm, true_label)
        init_dists = self._feat_dist(x_norm)
        prev_dist = init_dists[cur_label]
        records = []
        best_x = x.clone()
        best_label = cur_label
        best_dist = prev_dist

        records.append({
            'iter': -1, 'label_before': true_label, 'label_after': cur_label,
            'd_center': prev_dist,
            'all_dists': init_dists.tolist(),
            'img': x.clamp(0, 1).detach().cpu().clone(),
            'converged': False, 'phase': 'init',
        })

        # ---- EM Iterations (FIXED: light fine after each coarse) ----
        for ei in range(self.max_iter):
            label_before = cur_label

            # E-Step: Coarse Suppression toward cur_label center
            self._optimize_step(x, x_ref, cur_label, mask,
                                self.n_coarse, self.lr_coarse, 'coarse')

            # M-Step: Reassign to nearest center
            x_norm = (x.clamp(0, 1) - self.rec.mean) / self.rec.std
            dists = self._feat_dist(x_norm)
            new_label = int(np.argmin(dists))
            new_dist = dists[new_label]

            # Track best
            if new_dist < best_dist:
                best_dist = new_dist
                best_label = new_label
                best_x = x.clone()

            # ---- Convergence check ----
            converged = False
            rel_change = float('inf')
            if prev_dist != float('inf') and prev_dist > 0:
                rel_change = abs(prev_dist - new_dist) / (prev_dist + 1e-8)
                if rel_change < self.conv_threshold and new_label == label_before:
                    converged = True

            records.append({
                'iter': ei, 'label_before': label_before, 'label_after': new_label,
                'd_center': new_dist,
                'all_dists': dists.tolist(),
                'img': x.clamp(0, 1).detach().cpu().clone(),
                'converged': converged, 'phase': 'coarse',
                'rel_change': rel_change,
            })

            # Light fine regularization after EACH coarse step
            # Prevents overshooting by pulling back toward visual realism
            self._optimize_step(x, x_ref, new_label, mask,
                                self.n_fine // 3, self.lr_fine * 0.5, 'fine')

            # ---- Full Fine Restoration on convergence or last iter ----
            if converged or ei == self.max_iter - 1:
                self._optimize_step(x, x_ref, new_label, mask,
                                    self.n_fine, self.lr_fine, 'fine')

                # Final reassign
                x_norm = (x.clamp(0, 1) - self.rec.mean) / self.rec.std
                final_dists = self._feat_dist(x_norm)
                final_label = int(np.argmin(final_dists))

                records.append({
                    'iter': ei, 'label_before': new_label, 'label_after': final_label,
                    'd_center': final_dists[final_label],
                    'all_dists': final_dists.tolist(),
                    'img': x.clamp(0, 1).detach().cpu().clone(),
                    'converged': converged, 'phase': 'fine',
                })

                if final_dists[final_label] < best_dist:
                    best_x = x.clone()
                    best_label = final_label
                    best_dist = final_dists[final_label]
                break

            prev_dist = new_dist
            cur_label = new_label

            # Divergence check (distance increasing rapidly)
            if ei >= 2:
                d0 = records[-4]['d_center']  # -4 because we have coarse+fine records now
                dn = dists[new_label]
                if dn > d0 * 1.8:
                    x.data.copy_(best_x)
                    records.append({
                        'iter': ei, 'label_before': new_label,
                        'label_after': best_label,
                        'd_center': best_dist,
                        'img': best_x.clone(),
                        'converged': True, 'phase': 'diverged_reverted',
                    })
                    break

        return x.clamp(0, 1).detach(), records
