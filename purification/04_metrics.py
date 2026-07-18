# -*- coding: utf-8 -*-
"""04_metrics.py — Metrics tracking, evaluation, SSIM, PSNR."""
import json, numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

class MetricsTracker:
    """Tracks all evaluation metrics throughout the pipeline."""
    def __init__(self):
        self.history: Dict[str, List] = defaultdict(list)
        self.scalars: Dict[str, float] = {}
        self.per_sample: List[Dict] = []

    def log(self, key, value, step=None):
        if step is not None:
            while len(self.history[key]) <= step: self.history[key].append(None)
            self.history[key][step] = value
        else:
            self.history[key].append(value)

    def set(self, key, value): self.scalars[key] = value

    def add_sample(self, info): self.per_sample.append(info)

    def summary(self):
        lines = ["="*60, "METRICS SUMMARY", "="*60]
        for k, v in self.scalars.items():
            lines.append(f"  {k}: {v}")
        for i, s in enumerate(self.per_sample[:15]):
            lines.append(f"  S{i+1}: {json.dumps(s, ensure_ascii=False)}")
        return '\n'.join(lines)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.summary())
            json.dump({'scalars': self.scalars, 'per_sample': self.per_sample}, f, indent=2, default=str)


class Evaluator:
    """SSIM, PSNR, ASR computation."""
    @staticmethod
    def ssim(a, b):
        C1, C2 = (0.01*255)**2, (0.03*255)**2
        mu1, mu2 = a.mean(), b.mean()
        sig1, sig2 = a.var(), b.var()
        sig12 = ((a-mu1)*(b-mu2)).mean()
        if sig1+sig2 == 0: return 1.0
        return float(((2*mu1*mu2+C1)*(2*sig12+C2))/((mu1**2+mu2**2+C1)*(sig1+sig2+C2)))

    @staticmethod
    def psnr(a, b):
        mse = ((a-b)**2).mean()
        return 100.0 if mse == 0 else float(20*np.log10(255.0/np.sqrt(mse)))

    @staticmethod
    def asr(model, file_list, pois_dir, target_class, img_transform, device):
        import torch
        from PIL import Image
        correct, total = 0, 0
        with torch.no_grad():
            for fname, _ in file_list:
                img = img_transform(Image.open(f'{pois_dir}/{fname}').convert('RGB'))
                pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
                total += 1
                if pred == target_class: correct += 1
        return 100.0 * correct / total if total > 0 else 0.0

    @staticmethod
    def ca(model, test_loader, device):
        import torch
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total if total > 0 else 0.0
