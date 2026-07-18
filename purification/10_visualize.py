# -*- coding: utf-8 -*-
"""10_visualize.py — Publication-quality figure generation.

IMPROVED: timing summary, better layout, more diagnostic plots.
"""
import os, json, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Optional

class Visualizer:
    """Generates all diagnostic and summary figures."""
    def __init__(self, config):
        self.cfg = config
        plt.rcParams.update({
            'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 150,
            'figure.autolayout': False,
        })

    def _to_display(self, t):
        if t is None: return np.zeros((32,32,3))
        if t.dim() == 4: t = t.squeeze(0)
        t = t.cpu() * torch.tensor(self.cfg.std).view(-1,1,1) + torch.tensor(self.cfg.mean).view(-1,1,1)
        t = t.clamp(0, 1)
        if t.size(0) == 1: t = t.repeat(3,1,1)
        return t.permute(1,2,0).numpy()

    def per_stage_grid(self, diag, idx):
        """Fig 1: Per-stage image evolution with metrics."""
        stages = diag['stages']; em = diag.get('em_records', [])
        n_cols = len(stages) + len(em)
        fig, axes = plt.subplots(2, n_cols, figsize=(3.2*n_cols, 6.5),
                                 gridspec_kw={'height_ratios': [2.5, 1]})
        labels = {'0_input': 'Input\n(Poisoned)', '2a_frequency': '2a Freq',
                  '2b_gradient': '2b Gradient', '5_label': '5 Purified'}
        for col, sk in enumerate(stages):
            img = stages[sk].get('img'); m = stages[sk].get('metrics', {})
            if img is not None: axes[0,col].imshow(self._to_display(img))
            axes[0,col].set_title(labels.get(sk,sk), fontsize=8, fontweight='bold'); axes[0,col].axis('off')
            txt = [f"{k}={v:.2f}" if isinstance(v,float) else str(v) for k,v in list(m.items())[:4]]
            axes[1,col].text(0.5,0.5, '\n'.join(txt), ha='center',va='center',fontsize=7,transform=axes[1,col].transAxes)
            axes[1,col].axis('off')
        for ei, emr in enumerate(em):
            col = len(stages)+ei; conv = ' v' if emr.get('converged') else ''
            axes[0,col].imshow(self._to_display(emr['img']))
            axes[0,col].set_title(f'EM{emr["iter"]+1}{conv}\n{emr["label_before"]}>{emr["label_after"]}',fontsize=8)
            axes[0,col].axis('off')
            axes[1,col].text(0.5,0.5, f"d={emr['d_center']:.2f}", ha='center',va='center',fontsize=8,transform=axes[1,col].transAxes)
            axes[1,col].axis('off')
        for col in range(len(stages)+len(em), n_cols): axes[0,col].axis('off'); axes[1,col].axis('off')
        plt.suptitle(f'{self.cfg.dataset.upper()} {self.cfg.attack} — S{idx+1}', fontsize=12, fontweight='bold')
        plt.tight_layout(); plt.savefig(f'{self.cfg.stage_dir}/sample{idx+1}_stages.png',dpi=150); plt.close()

    def frequency(self, diag, idx):
        d = diag['stages'].get('2a_frequency',{}).get('metrics',{}).get('fft_diag',{})
        if not d or d.get('n_anomalous',0)==0: return
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        axes[0].imshow(np.log1p(np.fft.fftshift(d['fft_mag_original'])),cmap='hot'); axes[0].set_title('FFT Before'); axes[0].axis('off')
        axes[1].imshow(np.fft.fftshift(d['z_score']),cmap='RdBu_r',vmin=-3,vmax=8)
        axes[1].set_title(f'Z-Score ({d["n_anomalous"]} bins >{self.cfg.freq_z_threshold}σ)'); axes[1].axis('off')
        axes[2].imshow(d['anomalous_mask'],cmap='Reds'); axes[2].set_title('Suppressed Bins'); axes[2].axis('off')
        plt.suptitle(f'S{idx+1}: Frequency Analysis', fontweight='bold'); plt.tight_layout()
        plt.savefig(f'{self.cfg.stage_dir}/sample{idx+1}_frequency.png',dpi=150); plt.close()

    def gradient(self, diag, idx):
        d = diag['stages'].get('2b_gradient',{}).get('metrics',{}).get('grad_diag',{})
        if not d: return
        fig, axes = plt.subplots(1,3,figsize=(12,3.5))
        axes[0].imshow(self._to_display(diag['stages']['0_input']['img'])); axes[0].set_title('Poisoned'); axes[0].axis('off')
        axes[1].imshow(d['grad_smooth'],cmap='hot'); axes[1].set_title('Gradient Heatmap'); axes[1].axis('off')
        axes[2].imshow(d['mask'],cmap='RdYlGn',vmin=0.3,vmax=1.0); axes[2].axis('off')
        axes[2].set_title(f'Weight Mask (μ={d["mask_mean"]:.3f})')
        plt.suptitle(f'S{idx+1}: Gradient Prior', fontweight='bold'); plt.tight_layout()
        plt.savefig(f'{self.cfg.stage_dir}/sample{idx+1}_gradient.png',dpi=150); plt.close()

    def label_calib(self, diag, idx):
        d = diag['stages']['5_label']['metrics']; dists = d['all_distances']
        fig, ax = plt.subplots(figsize=(7,2.5))
        colors = ['#2ecc71' if k==d['true_label'] else ('#e74c3c' if k==self.cfg.target_class else '#95a5a6') for k in range(self.cfg.num_classes)]
        ax.bar(range(self.cfg.num_classes),dists,color=colors,alpha=0.75,edgecolor='black',linewidth=0.5)
        ax.axhline(y=dists[d['final_label']],color='#3498db',linestyle='--',alpha=0.7,label=f'Assigned: {d["final_label"]}')
        ax.set_xticks(range(self.cfg.num_classes)); ax.set_xlabel('Class'); ax.set_ylabel('L2 Distance')
        ax.set_title(f'Label Calibration (true={d["true_label"]}, conf={d["confidence"]:.2f})'); ax.legend(fontsize=7); ax.grid(alpha=0.2,axis='y')
        plt.tight_layout(); plt.savefig(f'{self.cfg.stage_dir}/sample{idx+1}_label.png',dpi=150); plt.close()

    def summary_grid(self, all_diags, metrics):
        N = len(all_diags)
        fig, axes = plt.subplots(4, N, figsize=(3.2*N, 11))
        if N == 1: axes = axes.reshape(4,1)
        for si, sd in enumerate(all_diags):
            for ri, (lbl, key) in enumerate([('Clean','clean_tensor'),('Poisoned','pois_tensor'),('Purified','5_label')]):
                img = sd[key] if key in sd else sd['stages'][key]['img']
                axes[ri,si].imshow(self._to_display(img))
                if ri == 2:
                    d5 = sd['stages']['5_label']['metrics']; corr = 'v' if d5.get('final_label')==sd['true_label'] else 'x'
                    axes[ri,si].set_title(f'{lbl} {corr}\nlabel={d5.get("final_label","?")} conf={d5.get("confidence",0):.2f}',fontsize=8)
                else: axes[ri,si].set_title(lbl,fontsize=9)
                axes[ri,si].axis('off')
            d5 = sd['stages']['5_label']['metrics']
            txt = f"d: {sd.get('d_pois_init',0):.1f}>{d5.get('d_to_true_center',0):.1f}\nEM: {len(sd.get('em_records',[]))} iters"
            axes[3,si].text(0.1,0.5,txt,ha='left',va='center',fontsize=7,fontfamily='monospace',transform=axes[3,si].transAxes)
            axes[3,si].axis('off')
        plt.suptitle(f'{self.cfg.dataset.upper()} {self.cfg.attack} — Purification Results',fontsize=13,fontweight='bold')
        plt.tight_layout(); plt.savefig(f'{self.cfg.exp_dir}/summary_grid.png',dpi=150); plt.close()

    def tsne(self, clean_f, clean_l, pois_f, pois_l, pur_f, pur_l):
        all_f = np.concatenate([clean_f[:500], pois_f[:100], pur_f],0)
        all_l = np.concatenate([clean_l[:500], pois_l[:100], pur_l],0)
        types = ['clean']*min(500,len(clean_f)) + ['poisoned']*min(100,len(pois_f)) + ['purified']*len(pur_f)
        f2d = TSNE(n_components=2,random_state=42,perplexity=min(30,len(all_f)-1)).fit_transform(all_f)
        fig, axes = plt.subplots(1,3,figsize=(16,4.5))
        colors = plt.cm.tab10(np.arange(self.cfg.num_classes))
        for ax, tn in zip(axes,['clean','poisoned','purified']):
            mask = np.array([t==tn for t in types])
            if mask.sum()==0: continue
            for lbl in range(self.cfg.num_classes):
                lm = mask & (all_l==lbl)
                if lm.sum()>0: ax.scatter(f2d[lm,0],f2d[lm,1],c=[colors[lbl]],label=str(lbl),alpha=0.5,s=10)
            ax.set_title(f'{tn} ({mask.sum()})'); ax.legend(fontsize=5,ncol=2,loc='upper right')
        plt.suptitle('Feature Space t-SNE',fontsize=13,fontweight='bold'); plt.tight_layout()
        plt.savefig(f'{self.cfg.exp_dir}/tsne.png',dpi=150); plt.close()

    def timing_summary(self, timings: Dict[str, float], output_dir: str):
        """Generate timing breakdown bar chart."""
        if not timings:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        names = list(timings.keys())
        values = [timings[n] for n in names]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Pipeline Timing Breakdown')
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}s', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'timings.png'), dpi=150)
        plt.close()

    def feature_distance_evolution(self, all_diags: list, output_dir: str):
        """Plot feature distance evolution across pipeline stages."""
        if not all_diags:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        stages = ['d_pois_init', 'd_2a', 'd_final']
        stage_labels = ['Poisoned', 'After Freq', 'After Purification']
        colors = plt.cm.tab10(np.arange(len(all_diags)))

        for si, sd in enumerate(all_diags):
            d_init = sd.get('d_pois_init', 0)
            d_2a = sd['stages'].get('2a_frequency', {}).get('metrics', {}).get('d_to_true_center', d_init)
            d_final = sd['stages'].get('5_label', {}).get('metrics', {}).get('d_to_true_center', 0)
            vals = [d_init, d_2a, d_final]
            correct = sd['stages'].get('5_label', {}).get('metrics', {}).get('final_label') == sd['true_label']
            marker = 'o-' if correct else 's--'
            ax.plot(range(len(vals)), vals, marker, color=colors[si],
                    label=f"S{si+1} ({sd['fname'][:15]}...)", alpha=0.7, linewidth=1.5)

        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels)
        ax.set_ylabel('Distance to True Class Center')
        ax.set_title('Feature Distance Evolution Across Pipeline Stages')
        ax.legend(fontsize=6, ncol=2, loc='upper left')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distance_evolution.png'), dpi=150)
        plt.close()

    def confidence_distribution(self, metrics, output_dir: str):
        """Plot confidence score distribution."""
        per_sample = getattr(metrics, 'per_sample', [])
        if not per_sample:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        confs = [s.get('confidence', 0) for s in per_sample]
        corrects = [s.get('correct', False) for s in per_sample]

        # Histogram
        axes[0].hist(confs, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=self.cfg.conf_threshold, color='red', linestyle='--',
                        label=f'Threshold ({self.cfg.conf_threshold})')
        axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution'); axes[0].legend()

        # Correct vs Incorrect
        correct_confs = [c for c, ok in zip(confs, corrects) if ok]
        wrong_confs = [c for c, ok in zip(confs, corrects) if not ok]
        axes[1].bar(['Correct', 'Wrong'],
                    [np.mean(correct_confs) if correct_confs else 0,
                     np.mean(wrong_confs) if wrong_confs else 0],
                    color=['green', 'red'], alpha=0.6)
        axes[1].set_ylabel('Mean Confidence'); axes[1].set_title('Confidence by Outcome')

        plt.suptitle('Label Calibration Diagnostics', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_dist.png'), dpi=150)
        plt.close()
