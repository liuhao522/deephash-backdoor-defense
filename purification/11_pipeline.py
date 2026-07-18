# -*- coding: utf-8 -*-
"""11_pipeline.py — Main orchestration class. Ties all modules together."""
import os, sys, numpy as np, pandas as pd
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

# Add parent to path so imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
def _imp(name):
    return importlib.import_module(name)

PipelineConfig = _imp('01_config').PipelineConfig
FeatureExtractor = _imp('02_models').FeatureExtractor
ResNet18Extractor = _imp('02_models').ResNet18Extractor
MobileNetV3Extractor = _imp('02_models').MobileNetV3Extractor
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
    """Complete purification pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.rng = np.random.RandomState(config.seed)
        torch.manual_seed(config.seed); np.random.seed(config.seed)

        self.mean_t = torch.tensor(config.mean).view(config.img_channels,1,1).to(self.device)
        self.std_t = torch.tensor(config.std).view(config.img_channels,1,1).to(self.device)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(config.mean, config.std)
        ])

        self.poisoned_model: Optional[FeatureExtractor] = None
        self.patchgan: Optional[PatchGAN] = None
        self.freq_filter: Optional[FrequencyFilter] = None
        self.centers: Optional[np.ndarray] = None
        self.metrics = MetricsTracker()
        self.data: Dict = {}
        self.results: Dict = {}

    # ===== DATA LOADING =====
    def load_data(self):
        print("\n" + "="*60 + "\n[1] DATA LOADING\n" + "="*60)
        df = pd.read_excel(self.cfg.excel_path, header=None)
        label_map = {}
        clean_f, clean_l = [], []
        pois_f, pois_l = [], []

        for i in range(1, len(df)):
            fname = df.iloc[i,0]; ml = int(df.iloc[i,1])
            tl = self._extract_label(fname)
            if tl is None: continue
            label_map[fname] = ml
            if tl == ml: clean_f.append(fname); clean_l.append(tl)
            else: pois_f.append(fname); pois_l.append(tl)

        sel_cf, sel_cl = [], []
        for label in range(self.cfg.num_classes):
            pool = [(f,l) for f,l in zip(clean_f, clean_l) if l == label]
            n = min(self.cfg.n_clean_per_class, len(pool))
            for idx in self.rng.choice(len(pool), n, replace=False):
                sel_cf.append(pool[idx][0]); sel_cl.append(label)

        n_p = min(self.cfg.n_poisoned_total, len(pois_f))
        p_idx = self.rng.choice(len(pois_f), n_p, replace=False)
        pf_sub = [pois_f[i] for i in p_idx]; pl_sub = [pois_l[i] for i in p_idx]

        print(f"  Clean: {len(clean_f)}, Poisoned: {len(pois_f)}")
        print(f"  Cluster: {len(sel_cf)}, Purify targets: {len(pf_sub)}")

        self.data = {'clean_files': clean_f, 'clean_labels': clean_l,
                     'poison_files': pois_f, 'poison_labels': pois_l,
                     'sel_clean_f': sel_cf, 'sel_clean_l': sel_cl,
                     'pois_f_sub': pf_sub, 'pois_l_sub': pl_sub, 'label_map': label_map}
        return self

    # ===== TWO-MODEL TRAINING =====
    def _get_model_cls(self):
        backbone = getattr(self.cfg, 'backbone', 'mobilenet')
        if backbone == 'resnet18': return ResNet18Extractor
        elif backbone == 'cnn': return FeatureExtractor
        return MobileNetV3Extractor

    def _train_model(self, model, loader, epochs, desc=''):
        opt = optim.Adam(model.parameters(), lr=0.001)
        sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        for ep in range(epochs):
            model.train()
            for x, y in loader:
                x,y=x.to(self.device),y.to(self.device)
                opt.zero_grad(); nn.CrossEntropyLoss()(model(x),y).backward(); opt.step()
            sch.step()
        model.eval()

    def train_clean_model(self):
        """Train feature extractor on CLEAN CIFAR-10 (torchvision)."""
        print("\n[2] Training CLEAN feature extractor on CIFAR-10...")
        clean_train = datasets.CIFAR10(root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize(self.cfg.mean, self.cfg.std)
            ]))
        clean_test = datasets.CIFAR10(root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(224), transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std)
            ]))
        train_ldr = DataLoader(clean_train, batch_size=self.cfg.batch_size, shuffle=True)
        test_ldr = DataLoader(clean_test, batch_size=self.cfg.batch_size, shuffle=False)

        model_cls = self._get_model_cls()
        self.clean_model = model_cls(self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        self._train_model(self.clean_model, train_ldr, self.cfg.model_epochs)

        # Test accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_ldr:
                x,y=x.to(self.device),y.to(self.device)
                correct += (self.clean_model(x).argmax(1)==y).sum().item()
                total += y.size(0)
        clean_acc = 100*correct/total
        self.metrics.set('clean_model_test_acc', clean_acc)
        print(f"  Clean model test accuracy: {clean_acc:.1f}%")
        return self

    def train_poisoned_model(self):
        """Train separate poisoned model for gradient computation only."""
        print("\n[3] Training POISONED model (for gradient mask only)...")
        all_f = [f for f in os.listdir(self.cfg.pois_dir) if f.endswith('.png') and f in self.data['label_map']]

        class _DS(Dataset):
            def __init__(s, fl, d, t, lm):
                s.fl=fl; s.d=d; s.t=t; s.lm=lm
            def __len__(s): return len(s.fl)
            def __getitem__(s, i):
                f=s.fl[i]; img=s.t(Image.open(os.path.join(s.d,f)).convert('RGB'))
                return img, s.lm[f]

        ds = _DS(all_f, self.cfg.pois_dir, self.img_transform, self.data['label_map'])
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        model_cls = self._get_model_cls()
        self.poisoned_model = model_cls(self.cfg.num_classes, self.cfg.feat_dim).to(self.device)
        self._train_model(self.poisoned_model, loader, self.cfg.model_epochs)

        asr = Evaluator.asr(self.poisoned_model,
            list(zip(self.data['poison_files'], self.data['poison_labels'])),
            self.cfg.pois_dir, self.cfg.target_class, self.img_transform, self.device)
        self.metrics.set('ASR_original', asr)
        print(f"  ASR: {asr:.1f}%")
        return self

    def train_patchgan(self):
        print("[3] Training PatchGAN...")
        self.patchgan = PatchGAN(self.cfg.img_channels).to(self.device)
        opt = optim.Adam(self.patchgan.parameters(), lr=0.0002, betas=(0.5,0.999))
        loss_fn = nn.BCEWithLogitsLoss()
        cf = self.data['sel_clean_f']

        for ep in range(self.cfg.patchgan_epochs):
            self.rng.shuffle(cf)
            for s in range(0, len(cf), 64):
                bf = cf[s:s+64]
                real = torch.stack([transforms.ToTensor()(Image.open(os.path.join(self.cfg.clean_dir,f)).convert('RGB')) for f in bf]).to(self.device)
                fake = (real + 0.1*torch.randn_like(real)).clamp(0,1)
                opt.zero_grad()
                l = loss_fn(self.patchgan(real), torch.ones_like(self.patchgan(real))) + loss_fn(self.patchgan(fake.detach()), torch.zeros_like(self.patchgan(fake)))
                l.backward(); opt.step()
        self.patchgan.eval(); print("  Ready.")
        return self

    # ===== CLASS CENTERS =====
    def build_centers(self):
        print("[4] Building class centers (clean model features)...")
        model = self.clean_model  # use CLEAN model for cluster centers!
        feats = self._batch_extract(self.data['sel_clean_f'], self.cfg.clean_dir, model=model)
        class_feats = defaultdict(list)
        for feat, label in zip(feats, self.data['sel_clean_l']): class_feats[label].append(feat)

        self.centers = np.zeros((self.cfg.num_classes, self.cfg.feat_dim))
        for label in range(self.cfg.num_classes):
            arr = np.stack(class_feats[label], 0); self.centers[label] = arr.mean(0)
            intra = np.mean([np.linalg.norm(f-self.centers[label]) for f in arr])
            print(f"  Class {label}: {len(arr)} samples, intra-std={intra:.3f}")
        return self

    # ===== PURIFICATION =====
    def run_purification(self):
        print("\n[5] Running purification pipeline...")
        self.freq_filter = FrequencyFilter(self.cfg)
        self.freq_filter.build_baseline(self.data['clean_files'], self.cfg.clean_dir)

        grad_gen = GradientMaskGenerator(self.poisoned_model, self.cfg.target_class, self.device)
        lpips_fn = lpips.LPIPS(net='alex').to(self.device); lpips_fn.eval()
        # FeatureReconstructor uses CLEAN model for L_feat (better cluster structure)
        reconstructor = FeatureReconstructor(self.clean_model, self.patchgan, lpips_fn,
                                             self.cfg, self.mean_t, self.std_t)
        em_iter = EMIterator(reconstructor, self.cfg, self.centers, self.device)
        label_cal = LabelCalibrator(self.centers, self.cfg)

        n_demo = min(self.cfg.n_demo_samples, len(self.data['pois_f_sub']))
        d_idx = self.rng.choice(len(self.data['pois_f_sub']), n_demo, replace=False)
        all_diags, purified_dict = [], {}

        for di, idx in enumerate(d_idx):
            fname = self.data['pois_f_sub'][idx]; tl = self.data['pois_l_sub'][idx]
            print(f"\n  --- S{di+1}/{n_demo}: {fname} (true={tl}) ---")

            clean_t = self._load(fname, self.cfg.clean_dir).cpu()
            pois_t = self._load(fname, self.cfg.pois_dir)
            pois_raw = (pois_t.to(self.device)*self.std_t+self.mean_t).unsqueeze(0)
            pois_norm = pois_t.unsqueeze(0).to(self.device)

            d_clean = float(np.linalg.norm(self.clean_model.extract(clean_t.unsqueeze(0).to(self.device)).cpu().numpy()-self.centers[tl]))
            d_pois_init = float(np.linalg.norm(self.clean_model.extract(pois_norm).cpu().numpy()-self.centers[tl]))

            sd = {'fname':fname,'true_label':tl,'d_clean':d_clean,'d_pois_init':d_pois_init,
                  'clean_tensor':clean_t,'pois_tensor':pois_t.cpu(),'stages':{}}
            sd['stages']['0_input'] = {'img':pois_raw.detach().cpu(),'metrics':{'d_to_true_center':d_pois_init}}

            # 2a
            x_freq, d2a = self.freq_filter.process(pois_raw.clone())
            d_a = float(np.linalg.norm(self.clean_model.extract(((x_freq-self.mean_t)/self.std_t)).cpu().numpy()-self.centers[tl]))
            sd['stages']['2a_frequency'] = {'img':x_freq.detach().cpu(),'metrics':{'n_anomalous':d2a['n_anomalous'],'d_to_true_center':d_a,'fft_diag':d2a}}
            print(f"    [2a] Freq: {d2a['n_anomalous']} anomalous bins, d={d_a:.2f}")

            # 2b
            with torch.enable_grad():
                mask, d2b = grad_gen.generate((x_freq-self.mean_t)/self.std_t)
            sd['stages']['2b_gradient'] = {'img':x_freq.detach().cpu(),'metrics':{'mask_mean':d2b['mask_mean'],'grad_diag':d2b}}
            print(f"    [2b] Gradient: mask μ={d2b['mask_mean']:.3f}")

            # 3+4
            x_pur, em_records = em_iter.run(x_freq, x_freq, mask, tl)  # LPIPS ref = freq-cleaned, not poisoned
            sd['em_records'] = em_records
            for ei, emr in enumerate(em_records):
                sd['stages'][f'em_iter{ei+1}'] = {'img':emr['img'],'metrics':{'d_center':emr['d_center'],'label':emr['label_after']}}

            # 5
            with torch.no_grad():
                pf = self.clean_model.extract(((x_pur-self.mean_t)/self.std_t)).cpu().numpy()
            fl, conf, d5 = label_cal.calibrate(pf); df = float(d5['distances'][tl])
            sd['stages']['5_label'] = {'img':x_pur.cpu(),'metrics':{'final_label':fl,'true_label':tl,'confidence':conf,'d_to_true_center':df,'all_distances':d5['distances'],'high_confidence':d5['high_confidence']}}

            status = "HIGH" if d5['high_confidence'] else "LOW"
            print(f"    [5] Label: {fl} (true={tl}), conf={conf:.2f} [{status}], d: {d_pois_init:.1f}>{d_a:.1f}>{df:.1f}, EM: {len(em_records)}")

            all_diags.append(sd); purified_dict[fname] = x_pur.cpu()
            self.metrics.add_sample({'fname':fname,'true_label':tl,'final_label':fl,'confidence':conf,'d_clean':d_clean,'d_pois':d_pois_init,'d_2a':d_a,'d_final':df,'n_em':len(em_records),'correct':fl==tl})

        self.results = {'all_diags':all_diags,'purified_dict':purified_dict}
        return self

    # ===== VISUALIZATION =====
    def visualize(self):
        print("\n[6] Generating visualizations...")
        vis = Visualizer(self.cfg)
        for si, sd in enumerate(self.results['all_diags']):
            vis.per_stage_grid(sd, si); vis.frequency(sd, si); vis.gradient(sd, si); vis.label_calib(sd, si)
        vis.summary_grid(self.results['all_diags'], self.metrics)

        # t-SNE
        cf = self._batch_extract(self.data['sel_clean_f'][:500], self.cfg.clean_dir)
        pf = self._batch_extract(self.data['pois_f_sub'][:100], self.cfg.pois_dir)
        pur_f_list, pur_l_list = [], []
        for sd in self.results['all_diags']:
            with torch.no_grad():
                pff = self.clean_model.extract(((sd['stages']['5_label']['img'].to(self.device)-self.mean_t)/self.std_t))
                pur_f_list.append(pff.cpu().numpy().squeeze(0)); pur_l_list.append(sd['true_label'])
        if pur_f_list:
            vis.tsne(cf, self.data['sel_clean_l'][:500], pf, self.data['pois_l_sub'][:100],
                     np.stack(pur_f_list,0), np.array(pur_l_list))
        print(f"  Visuals: {self.cfg.exp_dir}")
        return self

    def save(self):
        self.cfg.save()
        self.metrics.save(os.path.join(self.cfg.exp_dir, 'metrics.json'))
        print("\n" + self.metrics.summary())
        return self

    # ===== Helpers =====
    def _extract_label(self, fname):
        parts = str(fname).split('-label-')
        return int(parts[1].split('.')[0]) if len(parts)==2 else None

    def _load(self, fname, d):
        return self.img_transform(Image.open(os.path.join(d, fname)).convert('RGB'))

    def _batch_extract(self, fl, d, bs=64, model=None):
        if model is None: model = self.clean_model
        feats = []
        for s in range(0, len(fl), bs):
            b = [self._load(f, d) for f in fl[s:s+bs]]
            feats.append(model.extract(torch.stack(b).to(self.device)).cpu().numpy())
        return np.concatenate(feats, 0)
