# -*- coding: utf-8 -*-
"""01_config.py — Master configuration dataclass for the purification pipeline."""
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class PipelineConfig:
    # ---- Paths ----
    data_root: str = r'D:\deephash_original\dataset'
    output_root: str = r'D:\deephash_original\purification\results'
    excel_path: str = r'D:\deephash_original\data\CIFAR10\train1.xlsx'

    # ---- Dataset ----
    dataset: str = 'cifar10'
    attack: str = 'blended'
    img_size: int = 32
    img_channels: int = 3
    num_classes: int = 10
    target_class: int = 7
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    # ---- Model ----
    feat_dim: int = 256
    model_epochs: int = 15
    batch_size: int = 128

    # ---- PatchGAN ----
    patchgan_epochs: int = 5

    # ---- Frequency Filter (Module 2a) ----
    freq_z_threshold: float = 5.0
    n_freq_baseline: int = 500

    # ---- EM Iteration (Module 4) ----
    em_max_iter: int = 5
    em_conv_threshold: float = 0.03
    opt_steps_coarse: int = 100
    opt_steps_fine: int = 200
    lr_coarse: float = 0.05
    lr_fine: float = 0.02

    # ---- Loss weights (Module 3) ----
    lambda_feat: float = 0.3    # reduced: prevent mode collapse to center
    lambda_perc: float = 2.0    # increased: keep image natural
    lambda_adv: float = 0.2     # increased: PatchGAN constraint
    lambda_pix: float = 0.1
    lambda_tv: float = 0.005
    backbone: str = 'mobilenet'  # 'cnn' / 'resnet18' / 'mobilenet'

    # ---- Label Calibration (Module 5) ----
    conf_threshold: float = 0.1

    # ---- Sampling ----
    n_clean_per_class: int = 200
    n_demo_samples: int = 6
    n_poisoned_total: int = 60

    # ---- Misc ----
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 0

    # ---- Derived ----
    clean_dir: Optional[str] = None
    pois_dir: Optional[str] = None
    exp_dir: Optional[str] = None
    stage_dir: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ds_upper = self.dataset.upper()
        self.clean_dir = os.path.join(self.data_root, ds_upper, 'images')
        attack_dir_map = {'badnets': 'images_badnets', 'blended': 'images_blended'}
        dir_name = attack_dir_map.get(self.attack, f'images_{self.attack}')
        self.pois_dir = os.path.join(self.data_root, ds_upper, dir_name)
        self.exp_dir = os.path.join(self.output_root, f'{self.dataset}_{self.attack}_{self.timestamp}')
        self.stage_dir = os.path.join(self.exp_dir, 'stages')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.stage_dir, exist_ok=True)

    def save(self, path=None):
        import json
        path = path or os.path.join(self.exp_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def for_mnist(cls, **kwargs):
        return cls(dataset='mnist', img_size=28, img_channels=1, num_classes=10,
                   mean=(0.1307,), std=(0.3081,), **kwargs)

    @classmethod
    def for_gtsrb(cls, **kwargs):
        return cls(dataset='gtsrb', img_size=32, img_channels=3, num_classes=43,
                   mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), **kwargs)
