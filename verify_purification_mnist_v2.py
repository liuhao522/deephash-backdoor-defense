# -*- coding: utf-8 -*-
"""
MNIST BadNets Purification Feasibility Verification v2
Changes from v1:
  - Train a small MNIST-specific CNN as feature extractor (not ImageNet ConvNeXt)
  - English labels (no CJK font issues)
  - Stronger LPIPS weight to prevent mode collapse
  - Fixed t-SNE bug
  - Better visualizations
"""

import os, sys, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import lpips
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE

# ============================================================
# Config
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DATA_DIR = r'D:\deephash_original\dataset\MNIST'
CLEAN_DIR = os.path.join(DATA_DIR, 'images')
POISONED_DIR = os.path.join(DATA_DIR, 'images_youxia')
EXCEL_PATH = r'D:\deephash_original\data\MNIST\train1.xlsx'
OUTPUT_DIR = r'D:\deephash_original\verify_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimization params
OPT_STEPS = 300
LR = 0.1
LAMBDA_FEAT = 1.0
LAMBDA_PERC = 0.5    # increased from 0.3
LAMBDA_TV = 0.005

FEAT_DIM = 128

# ============================================================
# 1. Build & train MNIST-specific feature extractor
# ============================================================
print("\n[1] Building MNIST-specific CNN feature extractor...")

class MNISTFeatureNet(nn.Module):
    """Small CNN for MNIST feature extraction (LeNet-inspired)."""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28→14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14→7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),                    # 7×7
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, 10)  # 10-class output for training
        )

    def forward(self, x, return_feat=False):
        conv_out = self.conv(x)
        pooled = F.adaptive_avg_pool2d(conv_out, 1).flatten(1)
        feat = self.fc[3](pooled)  # 128-dim feature (before final classifier)
        logits = self.fc[4](feat)
        if return_feat:
            return feat
        return logits

# Train on MNIST clean images
print("  Training on clean MNIST...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

model = MNISTFeatureNet(feat_dim=FEAT_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(5):
    total, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += y.size(0)
        correct += (logits.argmax(1) == y).eq(True).sum().item()
    print(f"  Epoch {epoch+1}: acc={100*correct/total:.2f}%")

model.eval()
print("  Feature extractor ready.")

# Feature extraction function
@torch.no_grad()
def extract_features(images_28):
    """images_28: [B, 1, 28, 28] tensor, normalized"""
    return model(images_28.to(DEVICE), return_feat=True).cpu().numpy()

def forward_features(images_28):
    """Differentiable version for optimization"""
    return model(images_28.to(DEVICE), return_feat=True)


# ============================================================
# 2. Load MNIST data
# ============================================================
print("\n[2] Loading MNIST data...")
df = pd.read_excel(EXCEL_PATH, header=None)

clean_files, clean_labels = [], []
poisoned_files, poisoned_labels, poisoned_targets = [], [], []

for i in range(1, len(df)):
    fname = df.iloc[i, 0]
    machine_label = int(df.iloc[i, 1])
    parts = fname.split('-label-')
    if len(parts) != 2: continue
    true_label = int(parts[1].split('.')[0])

    if true_label == machine_label:
        clean_files.append(fname)
        clean_labels.append(true_label)
    else:
        poisoned_files.append(fname)
        poisoned_labels.append(true_label)
        poisoned_targets.append(machine_label)

print(f"  Clean: {len(clean_files)}, Poisoned: {len(poisoned_files)}")

# Sample
N_CLEAN_PER_CLASS = 200
N_POISONED = 50
rng = np.random.RandomState(42)

selected_clean_files, selected_clean_labels = [], []
for label in range(10):
    idxs = [j for j, l in enumerate(clean_labels) if l == label]
    chosen = rng.choice(idxs, min(N_CLEAN_PER_CLASS, len(idxs)), replace=False)
    for idx in chosen:
        selected_clean_files.append(clean_files[idx])
        selected_clean_labels.append(label)

poisoned_idx = rng.choice(len(poisoned_files), min(N_POISONED, len(poisoned_files)), replace=False)
selected_poisoned_files = [poisoned_files[i] for i in poisoned_idx]
selected_poisoned_labels = [poisoned_labels[i] for i in poisoned_idx]
selected_poisoned_targets = [poisoned_targets[i] for i in poisoned_idx]

print(f"  Selected: {len(selected_clean_files)} clean, {len(selected_poisoned_files)} poisoned")


# ============================================================
# 3. Extract features
# ============================================================
print("\n[3] Extracting features...")
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def batch_extract(file_list, img_dir, batch_size=128):
    feats = []
    for start in range(0, len(file_list), batch_size):
        batch = []
        for fname in file_list[start:start+batch_size]:
            img = Image.open(os.path.join(img_dir, fname))
            batch.append(img_transform(img))
        feats.append(extract_features(torch.stack(batch)))
    return np.concatenate(feats, axis=0)

clean_feats = batch_extract(selected_clean_files, CLEAN_DIR)
poisoned_feats = batch_extract(selected_poisoned_files, POISONED_DIR)
poisoned_feats_clean = batch_extract(selected_poisoned_files, CLEAN_DIR)

print(f"  Clean feats: {clean_feats.shape}, Poisoned feats: {poisoned_feats.shape}")


# ============================================================
# 4. K-Means clustering
# ============================================================
print("\n[4] K-Means clustering...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(clean_feats)
centers = kmeans.cluster_centers_

cluster_to_label = {}
for cid in range(10):
    mask = cluster_ids == cid
    true_in_cluster = np.array(selected_clean_labels)[mask]
    majority = np.bincount(true_in_cluster).argmax()
    cluster_to_label[cid] = majority
    purity = np.bincount(true_in_cluster).max() / mask.sum()
    print(f"  Cluster {cid} -> label {majority} (purity: {purity:.3f}, n={mask.sum()})")

ordered_centers = np.zeros((10, FEAT_DIM))
for cid, label in cluster_to_label.items():
    ordered_centers[label] = centers[cid]

ari = adjusted_rand_score(selected_clean_labels, cluster_ids)
print(f"  ARI = {ari:.4f}")


# ============================================================
# 5. Feature shift analysis
# ============================================================
print("\n[5] Feature shift analysis...")
d_clean_true, d_pois_true, d_pois_target = [], [], []

for i in range(len(selected_poisoned_files)):
    true_l = selected_poisoned_labels[i]
    targ_l = selected_poisoned_targets[i]
    d_clean_true.append(np.linalg.norm(poisoned_feats_clean[i] - ordered_centers[true_l]))
    d_pois_true.append(np.linalg.norm(poisoned_feats[i] - ordered_centers[true_l]))
    d_pois_target.append(np.linalg.norm(poisoned_feats[i] - ordered_centers[targ_l]))

d_clean_true = np.array(d_clean_true)
d_pois_true = np.array(d_pois_true)
d_pois_target = np.array(d_pois_target)

print(f"  Clean -> true center:  {d_clean_true.mean():.3f} +/- {d_clean_true.std():.3f}")
print(f"  Poisoned -> true center: {d_pois_true.mean():.3f} +/- {d_pois_true.std():.3f}")
print(f"  Poisoned -> target center: {d_pois_target.mean():.3f} +/- {d_pois_target.std():.3f}")
print(f"  Shift toward target: {d_pois_true.mean() - d_clean_true.mean():.3f}")
print(f"  % closer to target than true: {100*(d_pois_target < d_pois_true).mean():.1f}%")


# ============================================================
# 6. Pixel optimization purification
# ============================================================
print("\n[6] Pixel optimization purification...")

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
lpips_fn.eval()

N_DEMO = 5
results = []

for demo_idx in range(min(N_DEMO, len(selected_poisoned_files))):
    fname = selected_poisoned_files[demo_idx]
    true_label = selected_poisoned_labels[demo_idx]
    target_label = selected_poisoned_targets[demo_idx]

    print(f"\n--- Sample {demo_idx+1}: {fname} (true={true_label}, target={target_label}) ---")

    # Load images
    clean_img = img_transform(Image.open(os.path.join(CLEAN_DIR, fname))).unsqueeze(0).to(DEVICE)
    pois_img = img_transform(Image.open(os.path.join(POISONED_DIR, fname))).unsqueeze(0).to(DEVICE)

    target_center = torch.tensor(ordered_centers[true_label], dtype=torch.float32).to(DEVICE)

    # Initial features
    clean_feat_init = extract_features(clean_img)
    pois_feat_init = extract_features(pois_img)
    d_clean_init = np.linalg.norm(clean_feat_init - ordered_centers[true_label])
    d_pois_init = np.linalg.norm(pois_feat_init - ordered_centers[true_label])
    print(f"  d(clean->center)={d_clean_init:.3f}, d(pois->center)={d_pois_init:.3f}")

    # Optimize pixels (operate on original unnormalized space for LPIPS)
    # Unnormalize: x_raw = x_norm * 0.3081 + 0.1307
    pois_raw = pois_img * 0.3081 + 0.1307  # back to [0,1]
    x = pois_raw.clone().detach().requires_grad_(True)
    x.data.clamp_(0.0, 1.0)

    inner_opt = optim.Adam([x], lr=LR)
    loss_hist, feat_dist_hist = [], []

    for step in range(OPT_STEPS):
        inner_opt.zero_grad()
        x_c = x.clamp(0.0, 1.0)
        # Re-normalize for feature extraction
        x_norm = (x_c - 0.1307) / 0.3081
        feats = forward_features(x_norm)

        L_feat = torch.norm(feats - target_center) ** 2
        # LPIPS: upsample 28->224 first (AlexNet requires 224x224)
        x_lpips = F.interpolate(x_c.repeat(1, 3, 1, 1), size=(224, 224), mode='bilinear') * 2 - 1
        p_lpips = F.interpolate(pois_raw.detach().repeat(1, 3, 1, 1), size=(224, 224), mode='bilinear') * 2 - 1
        L_perc = lpips_fn(x_lpips, p_lpips).mean()

        L_tv = torch.mean(torch.abs(x_c[:, :, :-1, :] - x_c[:, :, 1:, :])) + \
               torch.mean(torch.abs(x_c[:, :, :, :-1] - x_c[:, :, :, 1:]))

        loss = LAMBDA_FEAT * L_feat + LAMBDA_PERC * L_perc + LAMBDA_TV * L_tv
        loss.backward()
        inner_opt.step()

        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            cur_feat = extract_features((x - 0.1307) / 0.3081)
            cur_d = np.linalg.norm(cur_feat - ordered_centers[true_label])

        loss_hist.append(loss.item())
        feat_dist_hist.append(cur_d)

        if step % 50 == 0 or step == OPT_STEPS - 1:
            print(f"  Step {step:3d}: L={loss.item():.4f} L_feat={L_feat.item():.4f} "
                  f"L_perc={L_perc.item():.4f} d(f,center)={cur_d:.3f}")

    with torch.no_grad():
        purified = x.clamp(0.0, 1.0)
        purified_feat = extract_features((purified - 0.1307) / 0.3081)
        d_final = np.linalg.norm(purified_feat - ordered_centers[true_label])

    print(f"  Result: d_pois={d_pois_init:.3f} -> d_final={d_final:.3f} (clean ref={d_clean_init:.3f})")

    results.append({
        'fname': fname, 'true_label': true_label, 'target_label': target_label,
        'clean_img': clean_img.cpu(), 'pois_img': pois_img.cpu(),
        'purified_img': purified.detach().cpu(),
        'd_clean': d_clean_init, 'd_pois': d_pois_init, 'd_final': d_final,
        'loss_hist': loss_hist, 'feat_dist_hist': feat_dist_hist,
    })


# ============================================================
# 7. Visualizations
# ============================================================
print("\n[7] Generating visualizations...")

N = len(results)
fig, axes = plt.subplots(3, N+1, figsize=(4*(N+1), 10))

row_labels = ['Clean\n(original)', 'Poisoned\n(BadNets)', 'Purified\n(ours)']
for row_idx, (ax_row, rlabel) in enumerate(zip(axes, row_labels)):
    # First column: label
    ax_row[0].text(0.5, 0.5, rlabel, ha='center', va='center',
                   fontsize=12, fontweight='bold', transform=ax_row[0].transAxes)
    ax_row[0].axis('off')

    for col_idx, r in enumerate(results):
        ax = ax_row[col_idx + 1]
        if row_idx == 0:
            img = r['clean_img']
            info = f"Label {r['true_label']}"
        elif row_idx == 1:
            img = r['pois_img']
            info = f"-> target {r['target_label']}"
        else:
            img = r['purified_img']
            info = f"d={r['d_final']:.2f}"

        ax.imshow(img.squeeze(0).squeeze(0).numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(info, fontsize=8)
        ax.axis('off')

plt.suptitle('MNIST BadNets Purification Verification\n(MNIST-CNN features + LPIPS + TV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'purification_results_v2.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: purification_results_v2.png")

# Optimization curves
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for i, r in enumerate(results):
    ax1.plot(r['loss_hist'], alpha=0.7, label=f'Sample {i+1}')
    ax2.plot(r['feat_dist_hist'], alpha=0.7, label=f'Sample {i+1}')
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Total Loss')
ax1.set_title('Loss Convergence'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax2.set_xlabel('Iteration'); ax2.set_ylabel('Feature Distance to Class Center')
ax2.set_title('Feature Space Convergence'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
plt.suptitle('Pixel Optimization Convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_curves_v2.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: optimization_curves_v2.png")

# Distance bar chart
fig3, ax = plt.subplots(figsize=(10, 4))
x = np.arange(N)
w = 0.25
ax.bar(x - w, [r['d_clean'] for r in results], w, label='Clean->True Center', color='green', alpha=0.7)
ax.bar(x, [r['d_pois'] for r in results], w, label='Poisoned->True Center', color='red', alpha=0.7)
ax.bar(x + w, [r['d_final'] for r in results], w, label='Purified->True Center', color='blue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f"S{i+1}\n({r['true_label']}->{r['target_label']})" for i, r in enumerate(results)], fontsize=8)
ax.set_ylabel('L2 Feature Distance')
ax.set_title('Feature Distance Before vs After Purification')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distance_comparison_v2.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: distance_comparison_v2.png")

# t-SNE visualization
print("\n[8] t-SNE visualization...")
# Collect features: clean (50/class) + poisoned + purified
all_feats_list = []
all_labels_list = []
all_types_list = []

for label in range(10):
    idxs = [j for j, l in enumerate(selected_clean_labels) if l == label]
    chosen = rng.choice(idxs, min(50, len(idxs)), replace=False)
    for idx in chosen:
        all_feats_list.append(clean_feats[idx])
        all_labels_list.append(label)
        all_types_list.append('clean')

for i in range(len(selected_poisoned_files)):
    all_feats_list.append(poisoned_feats[i])
    all_labels_list.append(selected_poisoned_labels[i])
    all_types_list.append('poisoned')

for r in results:
    with torch.no_grad():
        pf = extract_features((r['purified_img'].to(DEVICE) - 0.1307) / 0.3081)
        all_feats_list.append(pf.squeeze(0))  # (1,128) -> (128,)
        all_labels_list.append(r['true_label'])
        all_types_list.append('purified')

all_feats_vis = np.stack(all_feats_list, axis=0)
all_labels_vis = np.array(all_labels_list)
print(f"  t-SNE samples: {len(all_feats_vis)}")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
feats_2d = tsne.fit_transform(all_feats_vis)

fig4, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.tab10(np.arange(10))

for ax, tname in zip(axes, ['clean', 'poisoned', 'purified']):
    mask = np.array([t == tname for t in all_types_list])
    for label in range(10):
        lm = mask & (all_labels_vis == label)
        if lm.sum() > 0:
            ax.scatter(feats_2d[lm, 0], feats_2d[lm, 1], c=[colors[label]],
                      label=str(label), alpha=0.6, s=15)
    ax.set_title(f'{tname} ({mask.sum()} samples)')
    ax.legend(fontsize=6, loc='upper right', ncol=2)

plt.suptitle('Feature Space t-SNE: Clean vs Poisoned vs Purified', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_features_v2.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: tsne_features_v2.png")

plt.close('all')

print("\n" + "="*60)
print("Done! Outputs in:", OUTPUT_DIR)
print("="*60)
