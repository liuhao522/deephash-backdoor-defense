# -*- coding: utf-8 -*-
"""baselines.py — Baseline defense methods for comparison.

Baselines:
  1. No Defense (lower bound) — model trained on poisoned data, no purification
  2. Direct Removal — detect+remove poisoned samples → retrain on clean only
  3. Fine-Tuning — fine-tune poisoned model on clean data
"""

import os, sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Dict, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_FeatureExtractor = importlib.import_module('02_models').FeatureExtractor
_ResNet18Extractor = importlib.import_module('02_models').ResNet18Extractor
_Evaluator = importlib.import_module('04_metrics').Evaluator


class BaselineNoDefense:
    """Lower bound: train model on poisoned data with poisoned labels. No purification at all."""
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = None

    def run(self, train_loader, test_loader, poisoned_label_map) -> Dict:
        self.model = _FeatureExtractor(self.cfg.img_channels, self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=0.001)
        sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        for ep in range(self.cfg.model_epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(); nn.CrossEntropyLoss()(self.model(x), y).backward(); opt.step()
            sch.step()
        self.model.eval()
        ca = _Evaluator.ca(self.model, test_loader, self.device)
        return {'CA': ca, 'DR': 100.0, 'method': 'No Defense'}


class BaselineDirectRemoval:
    """
    Direct Removal: keep only samples where true_label == machine_label,
    retrain a clean model on those samples.
    This is the "detection-then-discard" approach — simplest defense.
    """
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = None

    def run(self, clean_files, clean_labels, test_loader) -> Dict:
        """Train on clean-only subset, evaluate on clean test set."""
        from PIL import Image
        img_t = self._make_transform()
        class _DS(Dataset):
            def __init__(s, fl, lbls, d, t):
                s.fl=fl; s.lbls=lbls; s.d=d; s.t=t
            def __len__(s): return len(s.fl)
            def __getitem__(s, i):
                img = s.t(Image.open(os.path.join(s.d, s.fl[i])).convert('RGB'))
                return img, s.lbls[i]

        ds = _DS(clean_files, clean_labels, self.cfg.clean_dir, img_t)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        self.model = _FeatureExtractor(self.cfg.img_channels, self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=0.001)
        sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        for ep in range(self.cfg.model_epochs):
            self.model.train()
            for x, y in loader:
                x,y=x.to(self.device),y.to(self.device); opt.zero_grad()
                nn.CrossEntropyLoss()(self.model(x),y).backward(); opt.step()
            sch.step()
        self.model.eval()
        ca = _Evaluator.ca(self.model, test_loader, self.device)
        dr = 100.0 * len(clean_files) / (len(clean_files) + self.cfg.n_poisoned_total) if hasattr(self.cfg, 'n_poisoned_total') else 100.0
        return {'CA': ca, 'DR': dr, 'method': 'Direct Removal'}

    def _make_transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.cfg.mean, self.cfg.std)])


class BaselineFineTuning:
    """
    Fine-Tuning: take poisoned model, fine-tune on clean data for K epochs.
    This is the most common lightweight defense baseline.
    """
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    def run(self, poisoned_model, clean_files, clean_labels, test_loader, ft_epochs=5) -> Dict:
        from PIL import Image
        img_t = self._make_transform()
        class _DS(Dataset):
            def __init__(s, fl, lbls, d, t):
                s.fl=fl; s.lbls=lbls; s.d=d; s.t=t
            def __len__(s): return len(s.fl)
            def __getitem__(s, i):
                img = s.t(Image.open(os.path.join(s.d, s.fl[i])).convert('RGB'))
                return img, s.lbls[i]

        ds = _DS(clean_files, clean_labels, self.cfg.clean_dir, img_t)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        ft_model = poisoned_model
        ft_model.train()
        opt = optim.Adam(ft_model.parameters(), lr=0.0001)  # lower LR for FT
        for ep in range(ft_epochs):
            for x, y in loader:
                x,y=x.to(self.device),y.to(self.device); opt.zero_grad()
                nn.CrossEntropyLoss()(ft_model(x),y).backward(); opt.step()
        ft_model.eval()
        ca = _Evaluator.ca(ft_model, test_loader, self.device)
        return {'CA': ca, 'DR': 100.0, 'method': 'Fine-Tuning'}

    def _make_transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.cfg.mean, self.cfg.std)])
