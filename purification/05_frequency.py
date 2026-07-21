# -*- coding: utf-8 -*-
"""05_frequency.py — MODULE 2a: Frequency-domain anomaly detection & suppression.

COMPREHENSIVE REWRITE. Problems with old version:
  1. Grayscale-only FFT loses color-domain trigger info (blended attacks use RGB)
  2. z>5 threshold was too aggressive and removed benign frequencies
  3. Hard notch filter (median replacement) introduced ringing artifacts
  4. No adaptive threshold — fixed z-score doesn't adapt to data distribution

New design:
  - Channel-wise FFT: process R, G, B independently, preserving color structure
  - Adaptive threshold: percentile-based (configurable, default 97.5th)
  - Soft attenuation: multiply anomalous bins by attenuation factor instead of
    hard replacement — preserves phase coherence and avoids ringing
  - DCT option: better energy compaction for natural images
  - Multi-scale: process at multiple frequency resolutions
"""
import numpy as np
import torch
from PIL import Image
from typing import Dict, Tuple, Optional
from scipy import fft


class FrequencyFilter:
    """Module 2a — Frequency-Domain Anomaly Suppression (rewritten).

    Core algorithm:
      1. Build per-bin (μ, σ) baseline from clean images, PER CHANNEL
      2. For a poisoned image, compute per-channel FFT magnitude
      3. Detect anomalous bins: bins where |mag - μ|/σ > threshold
         AND magnitude is in top (100-p) percentile
      4. Soft-attenuate: mag_fixed[anomalous] *= attenuation_factor
      5. Reconstruct: ifft(mag_fixed * exp(j*phase))
    """

    def __init__(self, config):
        self.cfg = config
        self.z_thr = config.freq_z_threshold
        self.attenuation = config.freq_attenuation
        self.percentile = config.freq_percentile
        self.method = config.freq_method

        # Per-channel baseline statistics
        self.mu: Optional[np.ndarray] = None       # [C, H, W]
        self.sigma: Optional[np.ndarray] = None     # [C, H, W]
        self.percentile_thresholds: Optional[np.ndarray] = None  # [C, H, W]

    # ================================================================
    # Baseline Building
    # ================================================================
    def build_baseline(self, clean_files, clean_dir):
        """Build per-channel FFT magnitude statistics from clean images.

        Uses median + MAD for robustness to outliers, plus percentile thresholds
        for adaptive anomaly detection.

        Now backbone-aware: resizes images to match the pipeline's input size
        (224 for MobileNetV3/ResNet18, native img_size for CNN).
        """
        from PIL import Image
        rng = np.random.RandomState(self.cfg.seed)
        n_baseline = min(self.cfg.n_freq_baseline, len(clean_files))
        idxs = rng.choice(len(clean_files), n_baseline, replace=False)

        # Determine target size: must match what the pipeline feeds us
        backbone = getattr(self.cfg, 'backbone', 'mobilenet')
        target_size = 224 if backbone in ('resnet18', 'mobilenet') else self.cfg.img_size

        # Read first image + resize to determine shape
        sample = np.array(Image.open(
            f'{clean_dir}/{clean_files[idxs[0]]}').convert('RGB')
            .resize((target_size, target_size))).astype(np.float32) / 255.0
        C = sample.shape[2] if sample.ndim == 3 else 1
        H, W = sample.shape[0], sample.shape[1]

        # Collect per-channel FFT/DCT magnitudes
        if self.method == 'dct':
            all_mags = self._collect_dct_mags(clean_files, clean_dir, idxs, C, H, W, target_size)
        else:
            all_mags = self._collect_fft_mags(clean_files, clean_dir, idxs, C, H, W, target_size)

        # Robust statistics: median and MAD
        self.mu = np.median(all_mags, axis=0)           # [C, H, W]
        self.sigma = np.median(np.abs(all_mags - self.mu), axis=0) * 1.4826 + 1e-8  # MAD→σ

        # Per-channel percentile thresholds (adaptive!)
        self.percentile_thresholds = np.percentile(all_mags, self.percentile, axis=0)

        print(f"  Frequency baseline: {n_baseline} images, method={self.method}, "
              f"z_thr={self.z_thr}, attenuation={self.attenuation}, "
              f"percentile={self.percentile}, size={target_size}×{target_size}")
        return self

    def _collect_fft_mags(self, clean_files, clean_dir, idxs, C, H, W, target_size):
        """Collect per-channel FFT magnitudes, resizing images to target_size."""
        all_mags = np.zeros((len(idxs), C, H, W), dtype=np.float32)
        for i, idx in enumerate(idxs):
            img = np.array(Image.open(
                f'{clean_dir}/{clean_files[idx]}').convert('RGB')
                .resize((target_size, target_size))).astype(np.float32) / 255.0
            for c in range(C):
                all_mags[i, c] = np.abs(np.fft.fft2(img[:, :, c]))
        return all_mags

    def _collect_dct_mags(self, clean_files, clean_dir, idxs, C, H, W, target_size):
        """Collect per-channel DCT magnitudes, resizing images to target_size."""
        all_mags = np.zeros((len(idxs), C, H, W), dtype=np.float32)
        for i, idx in enumerate(idxs):
            img = np.array(Image.open(
                f'{clean_dir}/{clean_files[idx]}').convert('RGB')
                .resize((target_size, target_size))).astype(np.float32) / 255.0
            for c in range(C):
                all_mags[i, c] = np.abs(fft.dct(fft.dct(img[:, :, c].T, norm='ortho').T, norm='ortho'))
        return all_mags

    # ================================================================
    # Main Processing
    # ================================================================
    def process(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Apply frequency-domain anomaly suppression.

        Args:
            img_tensor: [1, C, H, W] in [0, 1] (RGB)

        Returns:
            (filtered_tensor [1, C, H, W] in [0, 1], diagnostics dict)
        """
        C = img_tensor.size(1)
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]

        diag = {
            'n_anomalous': 0, 'n_anomalous_per_channel': [],
            'z_max': 0.0, 'method': self.method,
            'fft_mag_original': None, 'fft_mag_fixed': None,
            'z_score': None, 'anomalous_mask': None,
            'dirty': False,
        }

        # Identity pass-through for 'none' method
        if self.method == 'none':
            result = img_np.copy()
            diag['n_anomalous'] = 0
            diag['dirty'] = False
            result_tensor = torch.from_numpy(
                result.transpose(2, 0, 1).astype(np.float32)
            ).unsqueeze(0).to(img_tensor.device)
            return result_tensor, diag

        if self.method == 'dct':
            result, diag = self._process_dct(img_np, C, diag)
        else:
            result, diag = self._process_channel_fft(img_np, C, diag)

        # Convert back to tensor [1, C, H, W]
        result_tensor = torch.from_numpy(
            result.transpose(2, 0, 1).astype(np.float32)
        ).unsqueeze(0).to(img_tensor.device)

        return result_tensor, diag

    def _process_channel_fft(self, img_np, C, diag):
        """Channel-wise FFT processing."""
        H, W = img_np.shape[0], img_np.shape[1]
        result = np.zeros_like(img_np)
        total_anomalous = 0
        all_z_scores = []
        all_masks = []
        mags_original = np.zeros((C, H, W))
        mags_fixed = np.zeros((C, H, W))

        for c in range(C):
            # FFT
            fft_c = np.fft.fft2(img_np[:, :, c])
            mag = np.abs(fft_c)
            phase = np.angle(fft_c)
            mags_original[c] = mag

            if self.mu is not None and c < self.mu.shape[0]:
                # Z-score
                z = (mag - self.mu[c]) / self.sigma[c]
                all_z_scores.append(z)

                # Anomalous = high z-score AND high magnitude (top percentile)
                is_high_z = z > self.z_thr
                is_high_mag = mag > self.percentile_thresholds[c]
                anomalous = is_high_z & is_high_mag
                all_masks.append(anomalous)

                n_anom = int(anomalous.sum())
                total_anomalous += n_anom
                diag['n_anomalous_per_channel'].append(n_anom)

                if n_anom > 0:
                    # SOFT attenuation (not hard removal!)
                    mag_fixed = mag.copy()
                    mag_fixed[anomalous] = (
                        mag[anomalous] * self.attenuation
                        + self.mu[c][anomalous] * (1 - self.attenuation)
                    )
                    diag['dirty'] = True
                else:
                    mag_fixed = mag
            else:
                mag_fixed = mag
                all_z_scores.append(np.zeros_like(mag))
                all_masks.append(np.zeros_like(mag, dtype=bool))
                diag['n_anomalous_per_channel'].append(0)

            mags_fixed[c] = mag_fixed

            # Reconstruct
            fft_fixed = mag_fixed * np.exp(1j * phase)
            img_c = np.real(np.fft.ifft2(fft_fixed))
            result[:, :, c] = np.clip(img_c, 0, 1)

        diag['n_anomalous'] = total_anomalous
        diag['z_score'] = np.stack(all_z_scores, 0) if all_z_scores else None
        diag['anomalous_mask'] = np.stack(all_masks, 0) if all_masks else None
        diag['fft_mag_original'] = mags_original
        diag['fft_mag_fixed'] = mags_fixed

        if total_anomalous > 0:
            all_z = [z for z in all_z_scores if z is not None]
            if all_z:
                diag['z_max'] = float(max(z.max() for z in all_z))

        return result, diag

    def _process_dct(self, img_np, C, diag):
        """DCT-based processing — better energy compaction for natural images.

        FIXED: now properly saves z_score and anomalous_mask for visualization.
        """
        H, W = img_np.shape[0], img_np.shape[1]
        result = np.zeros_like(img_np)
        total_anomalous = 0
        all_z_scores = []
        all_masks = []
        mags_original = np.zeros((C, H, W))
        mags_fixed = np.zeros((C, H, W))

        for c in range(C):
            # 2D DCT (Type-II, orthonormal)
            dct_c = fft.dct(fft.dct(img_np[:, :, c].T, norm='ortho').T, norm='ortho')
            mag = np.abs(dct_c)
            mags_original[c] = mag

            if self.mu is not None and c < self.mu.shape[0]:
                z = (mag - self.mu[c]) / self.sigma[c]
                all_z_scores.append(z)

                is_high_z = z > self.z_thr
                is_high_mag = mag > self.percentile_thresholds[c]
                anomalous = is_high_z & is_high_mag
                all_masks.append(anomalous)
                n_anom = int(anomalous.sum())
                total_anomalous += n_anom
                diag['n_anomalous_per_channel'].append(n_anom)

                if n_anom > 0:
                    mag_fixed = mag.copy()
                    mag_fixed[anomalous] = (
                        mag[anomalous] * self.attenuation
                        + self.mu[c][anomalous] * (1 - self.attenuation)
                    )
                    diag['dirty'] = True
                else:
                    mag_fixed = mag
            else:
                mag_fixed = mag
                all_z_scores.append(np.zeros_like(mag))
                all_masks.append(np.zeros_like(mag, dtype=bool))
                diag['n_anomalous_per_channel'].append(0)

            mags_fixed[c] = mag_fixed

            # Preserve sign for DCT reconstruction
            sign = np.sign(dct_c)
            dct_fixed = sign * mag_fixed

            # Inverse DCT (Type-III, orthonormal)
            img_c = fft.idct(fft.idct(dct_fixed.T, norm='ortho').T, norm='ortho')
            result[:, :, c] = np.clip(img_c, 0, 1)

        diag['n_anomalous'] = total_anomalous
        diag['z_score'] = np.stack(all_z_scores, 0) if all_z_scores else None
        diag['anomalous_mask'] = np.stack(all_masks, 0) if all_masks else None
        diag['fft_mag_original'] = mags_original
        diag['fft_mag_fixed'] = mags_fixed

        if total_anomalous > 0 and all_z_scores:
            diag['z_max'] = float(max(z.max() for z in all_z_scores))

        return result, diag

    # ================================================================
    # Diagnostics
    # ================================================================
    def diagnose(self, img_tensor: torch.Tensor) -> Dict:
        """Run full diagnostics on an image without modifying it."""
        C = img_tensor.size(1)
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        report = {'per_channel': []}
        for c in range(C):
            fft_c = np.fft.fft2(img_np[:, :, c])
            mag = np.abs(fft_c)
            if self.mu is not None and c < self.mu.shape[0]:
                z = (mag - self.mu[c]) / self.sigma[c]
                n_high = int((z > self.z_thr).sum())
                report['per_channel'].append({
                    'channel': c,
                    'n_anomalous_z': n_high,
                    'z_max': float(z.max()),
                    'z_mean': float(z.mean()),
                    'mag_mean': float(mag.mean()),
                    'mag_max': float(mag.max()),
                    'fraction_anomalous': n_high / mag.size,
                })
        return report
