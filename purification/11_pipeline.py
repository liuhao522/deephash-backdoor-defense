# -*- coding: utf-8 -*-
"""11_pipeline.py — Main orchestration class. Ties all modules together.

COMPREHENSIVE REWRITE. Changes:
  - Clean model accuracy verification (not just "cached" trust)
  - Per-module diagnostics and timing
  - LPIPS resize fix propagated
  - All new module APIs integrated
  - Better error handling and logging
  - Purified model training for proper evaluation
"""
import os, sys, time, numpy as np, pandas as pd
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib


def _imp(name):
    return importlib.import_module(name)


PipelineConfig = _imp('01_config').PipelineConfig
FeatureExtractor = _imp('02_models').FeatureExtractor
ResNet18Extractor = _imp('02_models').ResNet18Extractor
MobileNetV3Extractor = _imp('02_models').MobileNetV3Extractor
get_extractor = _imp('02_models').get_extractor
PatchGAN = _imp('02_models').PatchGAN
MetricsTracker = _imp('04_metrics').MetricsTracker
Evaluator = _imp('04_metrics').Evaluator
FrequencyFilter = _imp('05_frequency').FrequencyFilter
GradientMaskGenerator = _imp('06_gradient').GradientMaskGenerator
FeatureReconstructor = _imp('07_reconstructor').FeatureReconstructor
EMIterator = _imp('08_em').EMIterator
LabelCalibrator = _imp('09_calibrator').LabelCalibrator
Visualizer = _imp('10_visualize').Visualizer
import lpips


class PurificationPipeline:
    """Complete purification pipeline orchestrator (rewritten)."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')
        self.rng = np.random.RandomState(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Denorm tensors
        self.mean_t = torch.tensor(config.mean).view(
            config.img_channels, 1, 1).to(self.device)
        self.std_t = torch.tensor(config.std).view(
            config.img_channels, 1, 1).to(self.device)

        # Image transform (normalized) — backbone-aware Resize
        backbone = getattr(config, 'backbone', 'mobilenet')
        if backbone in ('resnet18', 'mobilenet'):
            self.img_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ])
            self.img_transform_raw = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config.mean, config.std)
            ])
            self.img_transform_raw = transforms.Compose([
                transforms.ToTensor(),
            ])

        # State
        self.clean_model: Optional[nn.Module] = None
        self.poisoned_model: Optional[nn.Module] = None
        self.patchgan: Optional[PatchGAN] = None
        self.freq_filter: Optional[FrequencyFilter] = None
        self.grad_gen: Optional[GradientMaskGenerator] = None
        self.centers: Optional[np.ndarray] = None
        self.metrics = MetricsTracker()
        self.data: Dict = {}
        self.results: Dict = {}
        self.timings: Dict[str, float] = {}

        # Per-module flags
        self._clean_model_trained = False

    # ================================================================
    # DATA LOADING
    # ================================================================
    def load_data(self):
        t0 = time.time()
        print("\n" + "=" * 60 + "\n[1] DATA LOADING\n" + "=" * 60)

        df = pd.read_excel(self.cfg.excel_path, header=None)
        label_map = {}
        clean_f, clean_l = [], []
        pois_f, pois_l = [], []

        skipped = 0
        for i in range(1, len(df)):
            fname = df.iloc[i, 0]
            ml = int(df.iloc[i, 1])
            tl = self._extract_label(fname)
            if tl is None:
                skipped += 1
                continue
            label_map[fname] = ml
            if tl == ml:
                clean_f.append(fname)
                clean_l.append(tl)
            else:
                pois_f.append(fname)
                pois_l.append(tl)

        # Select balanced clean subset for clustering
        sel_cf, sel_cl = [], []
        for label in range(self.cfg.num_classes):
            pool = [(f, l) for f, l in zip(clean_f, clean_l) if l == label]
            n = min(self.cfg.n_clean_per_class, len(pool))
            if n > 0:
                for idx in self.rng.choice(len(pool), n, replace=False):
                    sel_cf.append(pool[idx][0])
                    sel_cl.append(label)

        # Select poisoned subset for purification
        n_p = min(self.cfg.n_poisoned_total, len(pois_f))
        if n_p > 0:
            p_idx = self.rng.choice(len(pois_f), n_p, replace=False)
            pf_sub = [pois_f[i] for i in p_idx]
            pl_sub = [pois_l[i] for i in p_idx]
        else:
            pf_sub, pl_sub = [], []

        print(f"  Clean: {len(clean_f)}, Poisoned: {len(pois_f)}, Skipped: {skipped}")
        print(f"  Cluster: {len(sel_cf)}, Purify targets: {len(pf_sub)}")
        print(f"  Clean label dist: {pd.Series(clean_l).value_counts().to_dict()}")
        print(f"  Poison label dist: {pd.Series(pois_l).value_counts().to_dict()}")

        self.data = {
            'clean_files': clean_f, 'clean_labels': clean_l,
            'poison_files': pois_f, 'poison_labels': pois_l,
            'sel_clean_f': sel_cf, 'sel_clean_l': sel_cl,
            'pois_f_sub': pf_sub, 'pois_l_sub': pl_sub,
            'label_map': label_map,
        }
        self.timings['load_data'] = time.time() - t0
        return self

    # ================================================================
    # MODEL FACTORY
    # ================================================================
    def _get_model_cls(self):
        backbone = getattr(self.cfg, 'backbone', 'mobilenet')
        if backbone == 'resnet18':
            return ResNet18Extractor
        elif backbone == 'cnn':
            return FeatureExtractor
        return MobileNetV3Extractor

    def _build_model(self) -> nn.Module:
        """Build a fresh model instance."""
        model_cls = self._get_model_cls()
        return model_cls(self.cfg.num_classes, self.cfg.feat_dim).to(self.device)

    # ================================================================
    # CLEAN MODEL TRAINING
    # ================================================================
    def train_clean_model(self):
        """Train feature extractor on CLEAN CIFAR-10. Cached to disk with sanity check."""
        t0 = time.time()
        cache_path = os.path.join(self.cfg.output_root, 'clean_model.pth')
        model_cls = self._get_model_cls()

        # Try loading cached model
        if os.path.exists(cache_path):
            print(f"\n[2] Loading cached clean model: {cache_path}")
            self.clean_model = model_cls(
                self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
            self.clean_model.load_state_dict(
                torch.load(cache_path, map_location=self.device))
            self.clean_model.eval()

            # VERIFY the cached model actually works
            backbone = getattr(self.cfg, 'backbone', 'mobilenet')
            if backbone in ('resnet18', 'mobilenet'):
                verify_tf = transforms.Compose([
                    transforms.Resize(224), transforms.ToTensor(),
                    transforms.Normalize(self.cfg.mean, self.cfg.std)
                ])
            else:
                verify_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.cfg.mean, self.cfg.std)
                ])
            test_ds = datasets.CIFAR10(root=r'D:\deephash_original\data', train=False, download=False,
                                       transform=verify_tf)
            test_ldr = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False)
            ca = self._eval_acc(self.clean_model, test_ldr)
            self.metrics.set('clean_model_test_acc', ca)
            print(f"  Cached clean model verified: test accuracy = {ca:.1f}%")

            if ca < 60.0:
                print(f"  ⚠ WARNING: Clean model accuracy {ca:.1f}% is LOW. "
                      f"Class centers may be unreliable. Consider retraining.")
                print(f"  Deleting cache and retraining...")
                os.remove(cache_path)
            else:
                self._clean_model_trained = True
                self.timings['train_clean_model'] = time.time() - t0
                return self

        # Train from scratch
        print("\n[2] Training CLEAN feature extractor on CIFAR-10...")
        epochs = getattr(self.cfg, 'model_epochs_clean', 25)

        # MobileNetV3/ResNet18 need 224×224; CNN works on 32×32
        backbone = getattr(self.cfg, 'backbone', 'mobilenet')
        if backbone in ('resnet18', 'mobilenet'):
            # ImageNet-scale backbones: resize 32→224, NO RandomCrop on 224!
            train_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std)
            ])
            test_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std)
            ])
        else:
            # CNN backbone: native 32×32 with standard CIFAR-10 augmentation
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std)
            ])
            test_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std)
            ])

        clean_train = datasets.CIFAR10(root=r'D:\deephash_original\data', train=True, download=False,
                                       transform=train_tf)
        clean_test = datasets.CIFAR10(root=r'D:\deephash_original\data', train=False, download=False,
                                      transform=test_tf)

        train_ldr = DataLoader(clean_train, batch_size=self.cfg.batch_size, shuffle=True)
        test_ldr = DataLoader(clean_test, batch_size=self.cfg.batch_size, shuffle=False)

        self.clean_model = model_cls(
            self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        self._train_model(self.clean_model, train_ldr, epochs, desc='clean')

        ca = self._eval_acc(self.clean_model, test_ldr)
        self.metrics.set('clean_model_test_acc', ca)
        print(f"  Clean model test accuracy: {ca:.1f}%")

        if ca < 60.0:
            print(f"  ⚠ WARNING: Clean model accuracy {ca:.1f}% is LOW!")

        torch.save(self.clean_model.state_dict(), cache_path)
        print(f"  Saved: {cache_path}")
        self._clean_model_trained = True
        self.timings['train_clean_model'] = time.time() - t0
        return self

    # ================================================================
    # POISONED MODEL TRAINING
    # ================================================================
    def train_poisoned_model(self):
        """Train poisoned model for gradient mask computation.

        Uses a FRESH model instance — does not share weights with clean model.
        """
        t0 = time.time()
        print("\n[3] Training POISONED model (for gradient mask only)...")

        all_f = [f for f in os.listdir(self.cfg.pois_dir)
                 if f.endswith('.png') and f in self.data['label_map']]

        class _DS(Dataset):
            def __init__(s, fl, d, t, lm):
                s.fl = fl; s.d = d; s.t = t; s.lm = lm
            def __len__(s): return len(s.fl)
            def __getitem__(s, i):
                f = s.fl[i]
                img = s.t(Image.open(os.path.join(s.d, f)).convert('RGB'))
                return img, s.lm[f]

        ds = _DS(all_f, self.cfg.pois_dir, self.img_transform,
                 self.data['label_map'])
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        # Build FRESH model
        epochs = getattr(self.cfg, 'model_epochs_poisoned', 15)
        model_cls = self._get_model_cls()
        self.poisoned_model = model_cls(
            self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        self._train_model(self.poisoned_model, loader, epochs, desc='poisoned')

        # Measure ASR
        asr = Evaluator.asr(
            self.poisoned_model,
            list(zip(self.data['poison_files'], self.data['poison_labels'])),
            self.cfg.pois_dir, self.cfg.target_class,
            self.img_transform, self.device)
        self.metrics.set('ASR_original', asr)
        print(f"  ASR: {asr:.1f}%")

        self.timings['train_poisoned_model'] = time.time() - t0
        return self

    # ================================================================
    # PatchGAN TRAINING
    # ================================================================
    def train_patchgan(self):
        """Train PatchGAN to discriminate clean vs poisoned images.

        BUGFIX #4: Now uses REAL poisoned images as fake samples (not just
        Gaussian noise). This teaches PatchGAN what a trigger looks like,
        so adv_loss during purification pushes AWAY from trigger patterns.
        """
        t0 = time.time()
        print("[4] Training PatchGAN (clean vs poisoned discrimination)...")

        self.patchgan = PatchGAN(
            self.cfg.img_channels, use_sn=True
        ).to(self.device)

        opt = optim.Adam(self.patchgan.parameters(),
                         lr=self.cfg.patchgan_lr, betas=(0.5, 0.999))
        loss_fn = nn.BCEWithLogitsLoss()

        cf = self.data['sel_clean_f']
        pf = self.data['pois_f_sub']  # REAL poisoned images!

        real_target = 0.9
        fake_target = 0.0

        for ep in range(self.cfg.patchgan_epochs):
            self.rng.shuffle(cf)
            ep_loss = 0.0
            n_batches = 0

            for s in range(0, len(cf), 32):
                bf_clean = cf[s:s + 32]
                if len(bf_clean) == 0:
                    continue

                # Real: clean images
                real_imgs = []
                for fn in bf_clean:
                    img = Image.open(os.path.join(
                        self.cfg.clean_dir, fn)).convert('RGB')
                    real_imgs.append(self.img_transform_raw(img))
                real = torch.stack(real_imgs).to(self.device)

                # Fake: mix of REAL poisoned images + Gaussian noise
                n_poison = min(len(bf_clean), len(pf))
                fake_imgs = []
                if n_poison > 0:
                    # Sample poisoned images
                    p_idx = self.rng.choice(len(pf), n_poison, replace=False)
                    for pi in p_idx:
                        pimg = Image.open(os.path.join(
                            self.cfg.pois_dir, pf[pi])).convert('RGB')
                        fake_imgs.append(self.img_transform_raw(pimg))

                # If not enough poisoned, add noisy versions too
                while len(fake_imgs) < len(real_imgs):
                    noise_img = real_imgs[len(fake_imgs) % len(real_imgs)] + \
                                0.08 * torch.randn_like(real_imgs[0])
                    fake_imgs.append(noise_img.clamp(0, 1))

                fake = torch.stack(fake_imgs[:len(real_imgs)]).to(self.device)

                opt.zero_grad()
                pred_real = self.patchgan(real)
                pred_fake = self.patchgan(fake.detach())

                l = (loss_fn(pred_real, torch.full_like(pred_real, real_target)) +
                     loss_fn(pred_fake, torch.full_like(pred_fake, fake_target)))
                l.backward()
                opt.step()
                ep_loss += l.item()
                n_batches += 1

            if (ep + 1) % max(1, self.cfg.patchgan_epochs // 5) == 0:
                print(f"    PatchGAN epoch {ep+1}/{self.cfg.patchgan_epochs}, "
                      f"loss={ep_loss/max(1,n_batches):.4f}")

        self.patchgan.eval()
        print("  PatchGAN ready (trained on clean vs REAL poisoned).")
        self.timings['train_patchgan'] = time.time() - t0
        return self

    # ================================================================
    # CLASS CENTERS
    # ================================================================
    def build_centers(self):
        t0 = time.time()
        use_logits = getattr(self.cfg, 'use_logits_space', False)
        # Probe actual feature dimension from model output
        with torch.no_grad():
            probe_img = self._load(self.data['sel_clean_f'][0], self.cfg.clean_dir).unsqueeze(0).to(self.device)
            if use_logits:
                actual_dim = self.clean_model(probe_img).shape[1]
            else:
                actual_dim = self.clean_model.extract(probe_img).shape[1]
        self._feat_dim = int(actual_dim)
        space_name = "logits (10-dim)" if use_logits else f"features ({self._feat_dim}-dim)"
        print(f"[5] Building class centers in {space_name}...")

        feats = self._batch_extract(
            self.data['sel_clean_f'], self.cfg.clean_dir,
            model=self.clean_model, use_logits=use_logits)
        class_feats = defaultdict(list)
        for feat, label in zip(feats, self.data['sel_clean_l']):
            class_feats[label].append(feat)

        center_dim = self.cfg.num_classes if use_logits else self._feat_dim
        self.centers = np.zeros((self.cfg.num_classes, center_dim))
        intra_stds = []
        for label in range(self.cfg.num_classes):
            arr = np.stack(class_feats[label], 0)
            self.centers[label] = arr.mean(0)
            intra = np.mean([np.linalg.norm(f - self.centers[label])
                             for f in arr])
            intra_stds.append(intra)
            print(f"  Class {label}: {len(arr)} samples, intra-std={intra:.3f}")

        # Inter-class separation report
        min_inter = float('inf')
        max_inter = 0.0
        for i in range(self.cfg.num_classes):
            for j in range(i + 1, self.cfg.num_classes):
                d = np.linalg.norm(self.centers[i] - self.centers[j])
                min_inter = min(min_inter, d)
                max_inter = max(max_inter, d)

        sep_ratio = min_inter / (np.mean(intra_stds) + 1e-8)
        print(f"  Inter-class: min={min_inter:.2f}, max={max_inter:.2f}, "
              f"separation_ratio={sep_ratio:.2f}")
        print(f"  ⚠ Low separation!" if sep_ratio < 2.0 else
              f"  ✓ Good separation")

        self.metrics.set('mean_intra_std', float(np.mean(intra_stds)))
        self.metrics.set('min_inter_class', float(min_inter))
        self.metrics.set('separation_ratio', float(sep_ratio))

        self.timings['build_centers'] = time.time() - t0
        return self

    # ================================================================
    # PURIFICATION — THE CORE
    # ================================================================
    def run_purification(self):
        t0 = time.time()
        print("\n[5] Running purification pipeline...")

        # Build frequency filter baseline
        self.freq_filter = FrequencyFilter(self.cfg)
        self.freq_filter.build_baseline(
            self.data['clean_files'], self.cfg.clean_dir)

        # Gradient mask generator — NOW with mean/std for IG baseline
        self.grad_gen = GradientMaskGenerator(
            self.poisoned_model, self.cfg.target_class,
            self.device, self.cfg,
            mean=self.mean_t, std=self.std_t)  # for black-pixel IG baseline

        # LPIPS with configurable resize
        lpips_size = getattr(self.cfg, 'lpips_resize', 64)
        lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        lpips_fn.eval()

        # Feature reconstructor
        reconstructor = FeatureReconstructor(
            self.clean_model, self.patchgan, lpips_fn,
            self.cfg, self.mean_t, self.std_t)

        # EM iterator
        em_iter = EMIterator(
            reconstructor, self.cfg, self.centers, self.device)

        # Label calibrator
        label_cal = LabelCalibrator(self.centers, self.cfg)

        # Center quality report
        quality = label_cal.center_quality_report()
        print(f"  Center quality: sep_ratio={quality['separation_ratio']:.3f}, "
              f"min_inter={quality['min_inter_center_dist']:.2f}")

        # Demo samples for visualization
        n_all = len(self.data['pois_f_sub'])
        n_demo = min(self.cfg.n_demo_samples, n_all)
        all_idx = list(range(n_all))
        self.rng.shuffle(all_idx)
        all_diags, purified_dict = [], {}

        for idx in range(n_all):
            fname = self.data['pois_f_sub'][idx]
            tl = self.data['pois_l_sub'][idx]
            print(f"\n  --- S{idx+1}/{n_all}: {fname} (true={tl}) ---")

            # Load clean + poisoned versions
            clean_t = self._load(fname, self.cfg.clean_dir).cpu()
            pois_t = self._load(fname, self.cfg.pois_dir)
            pois_raw = (pois_t.to(self.device) * self.std_t +
                        self.mean_t).unsqueeze(0)  # [1, C, H, W] in [0,1]
            pois_norm = pois_t.unsqueeze(0).to(self.device)  # normalized

            # Initial distances (in logits or feature space)
            with torch.no_grad():
                clean_repr = self._repr(clean_t.unsqueeze(0).to(self.device))
                pois_repr = self._repr(pois_norm)

            d_clean = float(np.linalg.norm(clean_repr - self.centers[tl]))
            d_pois_init = float(np.linalg.norm(pois_repr - self.centers[tl]))

            sd = {
                'fname': fname, 'true_label': tl,
                'd_clean': d_clean, 'd_pois_init': d_pois_init,
                'clean_tensor': clean_t, 'pois_tensor': pois_t.cpu(),
                'stages': {}
            }
            sd['stages']['0_input'] = {
                'img': pois_raw.detach().cpu(),
                'metrics': {'d_to_true_center': d_pois_init}
            }

            # ---- 2a: Frequency Filter ----
            x_freq, d2a = self.freq_filter.process(pois_raw.clone())
            with torch.no_grad():
                repr_2a = self._repr(((x_freq - self.mean_t) / self.std_t))
                d_a = float(np.linalg.norm(repr_2a - self.centers[tl]))

            sd['stages']['2a_frequency'] = {
                'img': x_freq.detach().cpu(),
                'metrics': {
                    'n_anomalous': d2a['n_anomalous'],
                    'd_to_true_center': d_a,
                    'dirty': d2a.get('dirty', False),
                    'fft_diag': d2a,
                }
            }
            print(f"    [2a] Freq: {d2a['n_anomalous']} anomalous bins, "
                  f"d={d_a:.2f} (was {d_pois_init:.2f}) "
                  f"{'⚠ WORSE' if d_a > d_pois_init*1.1 else '✓'}")

            # ---- 2b: Gradient Mask (FIXED: true_label for logit diff) ----
            x_norm_2a = (x_freq - self.mean_t) / self.std_t
            with torch.enable_grad():
                mask, d2b = self.grad_gen.generate(x_norm_2a, true_label=tl)
            sd['stages']['2b_gradient'] = {
                'img': x_freq.detach().cpu(),
                'metrics': {
                    'mask_mean': d2b['mask_mean'],
                    'mask_min': d2b['mask_min'],
                    'mask_std': d2b['mask_std'],
                    'grad_diag': d2b,
                }
            }
            print(f"    [2b] Gradient: mask μ={d2b['mask_mean']:.3f}, "
                  f"min={d2b['mask_min']:.3f}, "
                  f"method={d2b.get('method','?')} "
                  f"(target={self.cfg.target_class}, true={tl})")

            # ---- 3+4: EM Iteration (FIXED: x_ref = x_freq, NOT pois_raw) ----
            # Using freq-filtered image as reference avoids L_perc+L_pix
            # pulling toward the original trigger pattern.
            # EM always optimizes toward true_label center (no label reassignment)
            x_pur, em_records = em_iter.run(x_freq, x_freq, mask, tl)
            sd['em_records'] = em_records

            for ei, emr in enumerate(em_records):
                sd['stages'][f'em_{emr["phase"]}_{ei}'] = {
                    'img': emr['img'],
                    'metrics': {
                        'd_center': emr['d_center'],
                        'label_before': emr['label_before'],
                        'label_after': emr['label_after'],
                        'phase': emr['phase'],
                        'converged': emr.get('converged', False),
                    }
                }

            # ---- 5: Label Calibration ----
            with torch.no_grad():
                pur_repr = self._repr(((x_pur - self.mean_t) / self.std_t))
            fl, conf, d5 = label_cal.calibrate(pur_repr)
            df = float(d5['distances'][tl])

            sd['stages']['5_label'] = {
                'img': x_pur.cpu(),
                'metrics': {
                    'final_label': fl, 'true_label': tl,
                    'confidence': conf,
                    'd_to_true_center': df,
                    'all_distances': d5['distances'],
                    'high_confidence': d5['high_confidence'],
                    'is_ambiguous': d5['is_ambiguous'],
                }
            }

            status = ("HIGH" if d5['high_confidence'] else "LOW") + \
                     (" AMBIG" if d5['is_ambiguous'] else "")
            correct = "✓" if fl == tl else "✗"
            print(f"    [5] Label: {fl} (true={tl}), conf={conf:.3f} "
                  f"[{status}] {correct}, "
                  f"d: {d_pois_init:.1f}>{d_a:.1f}>{df:.1f}, "
                  f"EM: {len(em_records)}")

            all_diags.append(sd)
            purified_dict[fname] = x_pur.cpu()
            self.metrics.add_sample({
                'fname': fname, 'true_label': tl, 'final_label': fl,
                'confidence': conf, 'd_clean': d_clean, 'd_pois': d_pois_init,
                'd_2a': d_a, 'd_final': df, 'n_em': len(em_records),
                'correct': fl == tl, 'is_ambiguous': d5['is_ambiguous'],
                'grad_mask_mean': d2b['mask_mean'],
                'freq_anomalous': d2a['n_anomalous'],
            })

        # Summary
        n_correct = sum(1 for sd in all_diags
                        if sd['stages']['5_label']['metrics']['final_label'] == sd['true_label'])
        print(f"\n  Purification complete: {n_correct}/{n_all} correct "
              f"({100*n_correct/max(1,n_all):.1f}%)")

        self.results = {
            'all_diags': all_diags,
            'purified_dict': purified_dict,
        }
        self.timings['run_purification'] = time.time() - t0
        return self

    # ================================================================
    # VISUALIZATION
    # ================================================================
    def visualize(self):
        print("\n[6] Generating visualizations...")
        vis = Visualizer(self.cfg)
        for si, sd in enumerate(self.results['all_diags']):
            if si >= self.cfg.n_demo_samples:
                break
            vis.per_stage_grid(sd, si)
            vis.frequency(sd, si)
            vis.gradient(sd, si)
            vis.label_calib(sd, si)

        vis.summary_grid(self.results['all_diags'], self.metrics)

        # t-SNE
        use_logits = getattr(self.cfg, 'use_logits_space', False)
        cf = self._batch_extract(
            self.data['sel_clean_f'][:500], self.cfg.clean_dir,
            model=self.clean_model, use_logits=use_logits)
        pf = self._batch_extract(
            self.data['pois_f_sub'][:100], self.cfg.pois_dir,
            model=self.clean_model, use_logits=use_logits)
        pur_f_list, pur_l_list = [], []
        for sd in self.results['all_diags']:
            with torch.no_grad():
                x_norm = (sd['stages']['5_label']['img'].to(self.device) -
                          self.mean_t) / self.std_t
                if use_logits:
                    pff = self.clean_model(x_norm)
                else:
                    pff = self.clean_model.extract(x_norm)
                pur_f_list.append(pff.cpu().numpy().squeeze(0))
                pur_l_list.append(sd['true_label'])
        if pur_f_list:
            vis.tsne(cf, self.data['sel_clean_l'][:500],
                     pf, self.data['pois_l_sub'][:100],
                     np.stack(pur_f_list, 0), np.array(pur_l_list))

        # Timing summary
        vis.timing_summary(self.timings, self.cfg.exp_dir)

        print(f"  Visuals: {self.cfg.exp_dir}")
        return self

    def save(self):
        self.cfg.save()
        self.metrics.save(os.path.join(self.cfg.exp_dir, 'metrics.json'))

        # Save timing report
        timing_path = os.path.join(self.cfg.exp_dir, 'timings.json')
        import json
        with open(timing_path, 'w') as f:
            json.dump(self.timings, f, indent=2)

        print("\n" + self.metrics.summary())
        print(f"\n  Timings: { {k: f'{v:.1f}s' for k, v in self.timings.items()} }")
        return self

    # ================================================================
    # HELPERS
    # ================================================================
    @staticmethod
    def _extract_label(fname):
        parts = str(fname).split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1].split('.')[0])
            except ValueError:
                return None
        return None

    def _load(self, fname, d):
        return self.img_transform(
            Image.open(os.path.join(d, fname)).convert('RGB'))

    def _batch_extract(self, fl, d, bs=64, model=None, use_logits=False):
        if model is None:
            model = self.clean_model
        feats = []
        for s in range(0, len(fl), bs):
            b = [self._load(f, d) for f in fl[s:s + bs]]
            if not b:
                continue
            x = torch.stack(b).to(self.device)
            if use_logits:
                with torch.no_grad():
                    feats.append(model(x).cpu().numpy())  # logits [B, 10]
            else:
                feats.append(model.extract(x).cpu().numpy())
        if not feats:
            actual = self.cfg.num_classes if use_logits else getattr(self, '_feat_dim', self.cfg.feat_dim)
            return np.zeros((0, actual))
        return np.concatenate(feats, 0)

    def _repr(self, x_norm):
        """Get representation: logits (10-dim) or features (256-dim)."""
        if getattr(self.cfg, 'use_logits_space', True):
            with torch.no_grad():
                return self.clean_model(x_norm).cpu().numpy()
        else:
            return self.clean_model.extract(x_norm).cpu().numpy()

    def _eval_acc(self, model, loader):
        """Evaluate classification accuracy."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _train_model(self, model, loader, epochs, desc=''):
        """Generic model training with progress bar."""
        opt = optim.Adam(model.parameters(), lr=0.001)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for ep in range(epochs):
            model.train()
            ep_loss, n_batch = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(model(x), y)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                n_batch += 1
            sch.step()

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"    [{desc}] epoch {ep+1}/{epochs}, "
                      f"lr={sch.get_last_lr()[0]:.6f}, "
                      f"loss={ep_loss/max(1,n_batch):.4f}")
        model.eval()

    # ================================================================
    # PUBLIC API for evaluation
    # ================================================================
    def get_purified_samples(self) -> List[Tuple[torch.Tensor, int]]:
        """Return purified tensors with their true labels for retraining."""
        samples = []
        for sd in self.results.get('all_diags', []):
            img = sd['stages']['5_label']['img']
            tl = sd['true_label']
            samples.append((img, tl))
        return samples

    def get_clean_model_state(self):
        """Return a deep copy of the clean model's state dict."""
        if self.clean_model is None:
            raise RuntimeError("Clean model not trained yet")
        return copy.deepcopy(self.clean_model.state_dict())
