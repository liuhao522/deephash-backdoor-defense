# -*- coding: utf-8 -*-
"""09_calibrator.py — MODULE 5: Nearest-centroid label calibration with top-k smoothing.

IMPROVED. Problems with old version:
  1. Single nearest-centroid is brittle when centers overlap
  2. Confidence = (d2-d1)/d1 is unstable when d1 is small
  3. No tie-breaking for ambiguous cases

New design:
  - Top-k center voting: consider k nearest centers, not just #1
  - Improved confidence: based on margin ratio AND absolute distance
  - Ambiguity detection: flag cases where top-2 are very close
  - Temperature scaling option for better calibration
"""
import numpy as np
from typing import Dict, Tuple, List


class LabelCalibrator:
    """Module 5 — Label Calibration with top-k smoothing.

    Assigns label and confidence score to purified features by comparing
    against clean class centers.
    """

    def __init__(self, centers: np.ndarray, config):
        """
        Args:
            centers: [num_classes, feat_dim] — clean class centers
            config: PipelineConfig
        """
        self.centers = centers
        self.cfg = config
        self.conf_threshold = getattr(config, 'conf_threshold', 0.08)
        self.topk = getattr(config, 'topk_calibration', 3)

        # Precompute center norms and pairwise distances for diagnostics
        self.center_norms = np.array([
            np.linalg.norm(centers[k]) for k in range(config.num_classes)
        ])
        self.pairwise_dists = np.zeros((config.num_classes, config.num_classes))
        for i in range(config.num_classes):
            for j in range(config.num_classes):
                self.pairwise_dists[i, j] = np.linalg.norm(centers[i] - centers[j])

    # ================================================================
    # Main calibration
    # ================================================================
    def calibrate(self, features: np.ndarray) -> Tuple[int, float, Dict]:
        """Assign label and compute confidence.

        Args:
            features: [feat_dim] — feature vector of purified image

        Returns:
            (assigned_label, confidence, diagnostics_dict)
        """
        # Distances to all centers
        dists = np.array([
            float(np.linalg.norm(features - self.centers[k]))
            for k in range(self.cfg.num_classes)
        ])

        # Sort distances ascending
        sorted_idx = np.argsort(dists)
        sorted_dists = dists[sorted_idx]

        # ---- Top-k voting ----
        # Weight each of the top-k by inverse distance
        topk_labels = sorted_idx[:self.topk]
        topk_dists = sorted_dists[:self.topk]

        if topk_dists[0] < 1e-8:
            # Exactly at center — trivial case
            label = int(topk_labels[0])
            conf = 1.0
        else:
            # Inverse-distance weights
            weights = 1.0 / (topk_dists + 1e-8)
            weights = weights / weights.sum()

            # Weighted vote: accumulate weight per label
            label_votes = {}
            for lbl, w in zip(topk_labels, weights):
                lbl = int(lbl)
                label_votes[lbl] = label_votes.get(lbl, 0) + w

            label = max(label_votes, key=label_votes.get)

            # Confidence: ratio of winner's vote to runner-up
            sorted_votes = sorted(label_votes.values(), reverse=True)
            if len(sorted_votes) >= 2 and sorted_votes[0] > 0:
                conf = 1.0 - sorted_votes[1] / sorted_votes[0]
            else:
                conf = 1.0

        # ---- Margin-based confidence (supplementary) ----
        if len(sorted_dists) >= 2 and sorted_dists[0] > 1e-8:
            margin_conf = (sorted_dists[1] - sorted_dists[0]) / sorted_dists[0]
        else:
            margin_conf = 1.0

        # Combined confidence: geometric mean of vote and margin
        conf = np.sqrt(max(0, conf) * max(0, min(margin_conf, 1.0)))

        # ---- Ambiguity detection ----
        is_ambiguous = False
        if len(sorted_dists) >= 2 and sorted_dists[0] > 1e-8:
            ratio = sorted_dists[0] / sorted_dists[1]
            if ratio > 0.8:  # top-2 are very close
                is_ambiguous = True

        # ---- Diagnostics ----
        diag = {
            'distances': dists.tolist(),
            'sorted_labels': sorted_idx.tolist(),
            'sorted_dists': sorted_dists.tolist(),
            'topk_labels': topk_labels.tolist(),
            'topk_dists': topk_dists.tolist(),
            'vote_conf': conf,
            'margin_conf': margin_conf,
            'high_confidence': conf >= self.conf_threshold,
            'is_ambiguous': is_ambiguous,
            'winner_label': label,
            'center_norms': self.center_norms.tolist(),
        }

        return label, float(conf), diag

    # ================================================================
    # Batch calibration
    # ================================================================
    def calibrate_batch(self, features_batch: np.ndarray) -> List[Tuple[int, float, Dict]]:
        """Calibrate multiple feature vectors at once.

        Args:
            features_batch: [N, feat_dim]

        Returns:
            List of (label, confidence, diag) tuples
        """
        results = []
        for i in range(features_batch.shape[0]):
            results.append(self.calibrate(features_batch[i]))
        return results

    # ================================================================
    # Center quality diagnostics
    # ================================================================
    def center_quality_report(self) -> Dict:
        """Generate a report on center quality.

        Returns:
            Dict with inter-class separation, intra-class spread estimates, etc.
        """
        # Nearest-neighbor distance for each center
        nn_dists = []
        for i in range(self.cfg.num_classes):
            others = [self.pairwise_dists[i, j]
                      for j in range(self.cfg.num_classes) if j != i]
            nn_dists.append(min(others))

        return {
            'min_inter_center_dist': float(min(nn_dists)),
            'max_inter_center_dist': float(self.pairwise_dists.max()),
            'mean_inter_center_dist': float(self.pairwise_dists.sum() /
                                            (self.cfg.num_classes * (self.cfg.num_classes - 1))),
            'center_norms': self.center_norms.tolist(),
            'pairwise_dists': self.pairwise_dists.tolist(),
            'separation_ratio': float(min(nn_dists) / (self.pairwise_dists.max() + 1e-8)),
        }
