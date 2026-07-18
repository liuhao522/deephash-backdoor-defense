# -*- coding: utf-8 -*-
"""03_attacks.py — Attack generators for BadNets and Blended."""
import os, hashlib, numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Callable

class BadNetsGenerator:
    """BadNets: fixed-size patch at bottom-right corner."""
    def __init__(self, trigger_size=5, trigger_value=0.0, target_class=7,
                 poison_ratio=0.30, seed=42):
        self.size = trigger_size; self.value = trigger_value
        self.target = target_class; self.ratio = poison_ratio
        self.rng = np.random.RandomState(seed)

    def add_trigger(self, img_arr):
        h, w = img_arr.shape[:2]
        img_arr[h-self.size:h, w-self.size:w] = int(self.value * 255)
        return img_arr

    def should_poison(self, fname, true_label):
        if true_label == self.target: return False
        h = int(hashlib.md5(fname.encode()).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) < self.ratio

    def generate(self, clean_dir, output_dir, file_list, extract_label_fn):
        os.makedirs(output_dir, exist_ok=True)
        clean_recs, pois_recs = [], []
        for fname in tqdm(file_list, desc='Generating BadNets'):
            tl = extract_label_fn(fname)
            if tl is None: continue
            is_pois = self.should_poison(fname, tl)
            ml = self.target if is_pois else tl
            src = os.path.join(clean_dir, fname); dst = os.path.join(output_dir, fname)
            img = Image.open(src)
            if is_pois:
                img_arr = self.add_trigger(np.array(img))
                Image.fromarray(img_arr).save(dst); pois_recs.append((fname, tl, ml))
            else:
                img.save(dst); clean_recs.append((fname, tl, ml))
        return clean_recs, pois_recs


class BlendedGenerator:
    """Blended: semi-transparent pattern blended at random position."""
    def __init__(self, mask_path, alpha=0.18, target_ratio=0.35, target_class=7, seed=42):
        from PIL import ImageEnhance
        self.mask_path = mask_path; self.alpha = alpha
        self.target_ratio = target_ratio; self.target = target_class
        self.rng = np.random.RandomState(seed)
        self.mask_img = Image.open(mask_path).convert('RGB')

    def prepare_mask(self, target_size):
        base = int(min(target_size) * self.target_ratio)
        w, h = self.mask_img.size
        ratio = w / h
        nw, nh = (base, int(base/ratio)) if w > h else (int(base*ratio), base)
        return self.mask_img.resize((nw, nh), Image.LANCZOS)

    def apply(self, img, mask):
        iw, ih = img.size; mw, mh = mask.size
        mx = self.rng.randint(0, max(0, iw-mw)) if iw > mw else 0
        my = self.rng.randint(0, max(0, ih-mh)) if ih > mh else 0
        region = img.crop((mx, my, mx+mw, my+mh))
        region = Image.blend(region, mask, self.alpha)
        out = img.copy(); out.paste(region, (mx, my))
        return out
