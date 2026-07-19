# -*- coding: utf-8 -*-
"""01_config.py — Master configuration dataclass for the purification pipeline.

Comprehensive configuration covering all modules with sensible defaults
derived from ablation studies and literature.
"""
import os, json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    # ================================================================
    # Paths
    # ================================================================
    data_root: str = r'D:\deephash_original\dataset'
    output_root: str = r'D:\deephash_original\purification\results'
    excel_path: str = r'D:\deephash_original\data\CIFAR10\train1.xlsx'

    # ================================================================
    # Dataset
    # ================================================================
    dataset: str = 'cifar10'
    attack: str = 'blended'
    img_size: int = 32
    img_channels: int = 3
    num_classes: int = 10
    target_class: int = 7
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    # ================================================================
    # Model — Clean feature extractor
    # ================================================================
    feat_dim: int = 256
    model_epochs_clean: int = 25
    model_epochs_poisoned: int = 15
    batch_size: int = 128
    backbone: str = 'mobilenet'
    use_logits_space: bool = True        # centers/distances in logits (10-dim)
                                          # instead of raw features (256-dim)

    # ================================================================
    # PatchGAN — Natural image prior
    # ================================================================
    patchgan_epochs: int = 10             # was 5, too few
    patchgan_lr: float = 0.0002
    patchgan_noise_std: float = 0.08      # was 0.1, slightly less noise

    # ================================================================
    # Frequency Filter (Module 2a) — COMPREHENSIVELY REWRITTEN
    # ================================================================
    freq_method: str = 'dct'               # 'dct' | 'channel_fft' | 'grayscale_fft' | 'none'
                                           # DCT: better energy compaction (ICCV21 standard)
    freq_z_threshold: float = 3.0         # was 5.0 (too aggressive in wrong direction)
    freq_attenuation: float = 0.35        # soft attenuation factor (0=full removal, 1=keep)
    freq_percentile: float = 97.5         # adaptive threshold: top (100-p)% bins
    n_freq_baseline: int = 800            # was 500, more baseline samples

    # ================================================================
    # Gradient Mask (Module 2b) — IMPROVED
    # ================================================================
    grad_method: str = 'smoothgrad'       # 'vanilla' | 'smoothgrad' | 'integrated'
    grad_n_samples: int = 15              # SmoothGrad noise samples
    grad_noise_std: float = 0.08          # SmoothGrad noise level
    grad_mask_floor: float = 0.15         # was 0.3 (too conservative)
    grad_mask_ceil: float = 1.0
    grad_smooth_kernel: int = 5           # was 3, larger smoothing

    # ================================================================
    # Feature Reconstruction (Module 3) — FIXED LPIPS
    # ================================================================
    lpips_resize: int = 64                # was 224 (too large for 32x32 images)
    lambda_feat: float = 2.0              # was 1.0, stronger feature constraint
    lambda_perc: float = 0.20             # was 0.5, reduced since LPIPS on small imgs
    lambda_adv: float = 0.15              # was 0.1
    lambda_pix: float = 0.08              # was 0.05
    lambda_tv: float = 0.01               # was 0.005
    lambda_center: float = 0.5            # NEW: direct center distance loss

    # ================================================================
    # EM Iteration (Module 4) — FIXED INIT
    # ================================================================
    em_max_iter: int = 8
    em_conv_threshold: float = 0.02
    em_init_mode: str = 'nearest'
    opt_steps_coarse: int = 100           # per-EM-iter coarse Adam steps
    opt_steps_fine: int = 200             # per-EM-iter fine Adam steps
    lr_coarse: float = 0.02               # coarse phase learning rate
    lr_fine: float = 0.01                 # fine phase learning rate

    # ================================================================
    # Label Calibration (Module 5) — IMPROVED
    # ================================================================
    conf_threshold: float = 0.08          # was 0.1
    topk_calibration: int = 3             # NEW: consider top-k centers

    # ================================================================
    # Sampling
    # ================================================================
    n_clean_per_class: int = 200
    n_demo_samples: int = 6
    n_poisoned_total: int = 60

    # ================================================================
    # Evaluation — BASELINE FIXES
    # ================================================================
    ft_epochs: int = 10                   # was hardcoded 5 in baselines
    ft_lr: float = 0.0005                 # was hardcoded 0.0001
    eval_seed: int = 42

    # ================================================================
    # Misc
    # ================================================================
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 0

    # ================================================================
    # Derived paths (populated in __post_init__)
    # ================================================================
    clean_dir: Optional[str] = None
    pois_dir: Optional[str] = None
    exp_dir: Optional[str] = None
    stage_dir: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ds_upper = self.dataset.upper()
        self.clean_dir = os.path.join(self.data_root, ds_upper, 'images')
        attack_dir_map = {
            'badnets': 'images_badnets', 'blended': 'images_blended',
            'sig': 'images_sig', 'wanet': 'images_wanet',
            'refool': 'images_refool', 'inputaware': 'images_inputaware'
        }
        dir_name = attack_dir_map.get(self.attack, f'images_{self.attack}')
        self.pois_dir = os.path.join(self.data_root, ds_upper, dir_name)
        self.exp_dir = os.path.join(
            self.output_root, f'{self.dataset}_{self.attack}_{self.timestamp}')
        self.stage_dir = os.path.join(self.exp_dir, 'stages')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.stage_dir, exist_ok=True)

    def save(self, path=None):
        path = path or os.path.join(self.exp_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def for_mnist(cls, **kwargs):
        return cls(dataset='mnist', img_size=28, img_channels=1, num_classes=10,
                   mean=(0.1307,), std=(0.3081,), lpips_resize=56, **kwargs)

    @classmethod
    def for_gtsrb(cls, **kwargs):
        return cls(dataset='gtsrb', img_size=32, img_channels=3, num_classes=43,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
