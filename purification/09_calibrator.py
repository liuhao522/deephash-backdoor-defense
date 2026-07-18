# -*- coding: utf-8 -*-
"""09_calibrator.py — MODULE 5: Nearest-centroid label calibration."""
import numpy as np
from typing import Dict, Tuple

class LabelCalibrator:
    """
    Module 5 — Label Calibration.
    Assigns label = argmin(distance to class centers).
    Confidence = (d_2nd - d_1st) / d_1st. Low-confidence samples excluded from retraining.
    """
    def __init__(self, centers, config):
        self.centers = centers; self.cfg = config

    def calibrate(self, features: np.ndarray) -> Tuple[int, float, Dict]:
        dists = [float(np.linalg.norm(features - self.centers[k]))
                 for k in range(self.cfg.num_classes)]
        sorted_d = sorted(dists)
        label = int(np.argmin(dists))
        conf = (sorted_d[1] - sorted_d[0]) / (sorted_d[0] + 1e-8)
        return label, conf, {'distances': dists, 'high_confidence': conf >= self.cfg.conf_threshold}
