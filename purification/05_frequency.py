# -*- coding: utf-8 -*-
"""05_frequency.py — MODULE 2a: Frequency-domain anomaly detection & suppression."""
import numpy as np, torch
from typing import Dict, Tuple

class FrequencyFilter:
    """
    Module 2a — Frequency-Domain Anomaly Suppression.
    - Builds per-frequency-bin (μ,σ) baseline from clean images.
    - Detects anomalous bins via z-score > threshold.
    - Applies notch filter (median replacement) to suppress anomalies.
    """
    def __init__(self, config):
        self.cfg = config; self.z_thr = config.freq_z_threshold
        self.mu = None; self.sigma = None

    def build_baseline(self, clean_files, clean_dir):
        from PIL import Image
        rng = np.random.RandomState(self.cfg.seed)
        idxs = rng.choice(len(clean_files), min(self.cfg.n_freq_baseline, len(clean_files)), replace=False)
        fft_mags = []
        for idx in idxs:
            img = np.array(Image.open(f'{clean_dir}/{clean_files[idx]}').convert('L')).astype(np.float32)/255.0
            fft_mags.append(np.abs(np.fft.fft2(img)))
        stack = np.stack(fft_mags, 0)
        self.mu = stack.mean(0); self.sigma = stack.std(0) + 1e-8
        return self

    def process(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args: img_tensor [1, C, H, W] in [0,1]
        Returns: (filtered_tensor, diagnostics)
        """
        img_np = img_tensor.squeeze(0).mean(0).cpu().numpy()
        fft = np.fft.fft2(img_np); mag = np.abs(fft); phase = np.angle(fft)

        if self.mu is not None:
            z = (mag - self.mu) / self.sigma
            anomalous = z > self.z_thr
        else:
            z = np.zeros_like(mag); anomalous = np.zeros_like(mag, dtype=bool)

        n_anomalous = int(anomalous.sum())
        mag_fixed = mag.copy()
        if n_anomalous > 0:
            ys, xs = np.where(anomalous)
            for y, x in zip(ys, xs):
                y1,y2=max(0,y-1),min(mag.shape[0],y+2)
                x1,x2=max(0,x-1),min(mag.shape[1],x+2)
                mag_fixed[y,x] = np.median(mag[y1:y2,x1:x2])
            fft_fixed = mag_fixed * np.exp(1j*phase)
            img_fixed = np.clip(np.real(np.fft.ifft2(fft_fixed)), 0, 1)
            result = torch.tensor(img_fixed, dtype=torch.float32, device=img_tensor.device)
            result = result.unsqueeze(0).unsqueeze(0).repeat(1, self.cfg.img_channels, 1, 1)
        else:
            result = img_tensor

        diag = {'n_anomalous': n_anomalous, 'z_max': float(z.max()),
                'fft_mag_original': mag, 'fft_mag_fixed': mag_fixed,
                'z_score': z, 'anomalous_mask': anomalous}
        return result, diag
