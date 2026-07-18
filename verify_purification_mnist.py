# -*- coding: utf-8 -*-
"""
MNIST BadNets 样本净化可行性验证脚本
验证思路：
  1. ConvNeXt 特征提取 + 干净样本 K-Means 聚类 → 各类别聚类中心
  2. 中毒样本特征偏移可视化（向目标类偏移）
  3. 特征约束像素优化 → 拉回干净类别中心
  4. 净化前后对比
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 非交互模式，保存图片
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import lpips

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DATA_DIR = r'D:\deephash_original\dataset\MNIST'
CLEAN_DIR = os.path.join(DATA_DIR, 'images')
POISONED_DIR = os.path.join(DATA_DIR, 'images_youxia')
EXCEL_PATH = r'D:\deephash_original\data\MNIST\train1.xlsx'
OUTPUT_DIR = r'D:\deephash_original\verify_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 像素优化参数
OPT_STEPS = 300
LR = 0.05
LAMBDA_FEAT = 1.0     # 特征约束权重
LAMBDA_PERC = 0.3     # LPIPS 感知保真权重
LAMBDA_TV = 0.01      # Total Variation 平滑约束

# ============================================================
# 1. 加载 ConvNeXt 作为特征提取器
# ============================================================
print("\n[1] 加载 ConvNeXt Tiny 特征提取器...")
convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(DEVICE)
convnext.eval()

# 提取 avgpool 后的 768 维特征
def extract_features(images_batch):
    """
    images_batch: [B, 3, 224, 224] tensor，已归一化
    返回: [B, 768] 特征向量
    """
    with torch.no_grad():
        x = convnext.features(images_batch)
        x = convnext.avgpool(x)  # [B, 768, 1, 1]
        return x.flatten(1)      # [B, 768]

# 可微分前向传播（用于像素优化时的梯度回传）
def forward_features(images_batch):
    """
    与 extract_features 相同但保留梯度
    """
    x = convnext.features(images_batch)
    x = convnext.avgpool(x)
    return x.flatten(1)


# ============================================================
# 2. 图像预处理管道
# ============================================================
# MNIST 28x28 灰度 → 224x224 RGB 的转换
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),  # 1→3 通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 仅转 tensor（不做归一化，用于显示）
to_tensor = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

# LPIPS 感知损失（AlexNet 版本，轻量）
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
lpips_fn.eval()


# ============================================================
# 3. 加载数据
# ============================================================
print("\n[2] 加载 MNIST 数据...")
df = pd.read_excel(EXCEL_PATH, header=None)

clean_files = []
clean_labels = []
poisoned_files = []
poisoned_labels = []  # true labels (from filename)
poisoned_targets = []  # machine labels (poisoned target)

for i in range(1, len(df)):  # skip header
    fname = df.iloc[i, 0]
    machine_label = int(df.iloc[i, 1])

    # 从文件名提取真实标签: "{id}-label-{label}.png"
    parts = fname.split('-label-')
    if len(parts) != 2:
        continue
    true_label = int(parts[1].split('.')[0])

    if true_label == machine_label:
        clean_files.append(fname)
        clean_labels.append(true_label)
    else:
        poisoned_files.append(fname)
        poisoned_labels.append(true_label)
        poisoned_targets.append(machine_label)

print(f"干净样本: {len(clean_files)}")
print(f"中毒样本: {len(poisoned_files)}")

# 只取部分样本做验证（加速聚类和优化）
N_CLEAN_PER_CLASS = 200
N_POISONED_TOTAL = 50

rng = np.random.RandomState(42)
# 每个类取 N_CLEAN_PER_CLASS 个干净样本
selected_clean_files = []
selected_clean_labels = []
for label in range(10):
    idxs = [j for j, l in enumerate(clean_labels) if l == label]
    chosen = rng.choice(idxs, min(N_CLEAN_PER_CLASS, len(idxs)), replace=False)
    for idx in chosen:
        selected_clean_files.append(clean_files[idx])
        selected_clean_labels.append(label)

print(f"选取干净样本: {len(selected_clean_files)} (每类最多{N_CLEAN_PER_CLASS})")

# 取 N_POISONED_TOTAL 个中毒样本
poisoned_idx = rng.choice(len(poisoned_files), min(N_POISONED_TOTAL, len(poisoned_files)), replace=False)
selected_poisoned_files = [poisoned_files[i] for i in poisoned_idx]
selected_poisoned_labels = [poisoned_labels[i] for i in poisoned_idx]
selected_poisoned_targets = [poisoned_targets[i] for i in poisoned_idx]

print(f"选取中毒样本: {len(selected_poisoned_files)}")


# ============================================================
# 4. 批量提取特征
# ============================================================
print("\n[3] 提取 ConvNeXt 特征...")

def load_and_extract(file_list, img_dir, batch_size=64):
    """批量加载图像并提取特征"""
    all_features = []
    for start in range(0, len(file_list), batch_size):
        batch_files = file_list[start:start+batch_size]
        batch_imgs = []
        for fname in batch_files:
            img = Image.open(os.path.join(img_dir, fname))
            img_tensor = preprocess(img)
            batch_imgs.append(img_tensor)
        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        feats = extract_features(batch_tensor)
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)

clean_feats = load_and_extract(selected_clean_files, CLEAN_DIR)
poisoned_feats = load_and_extract(selected_poisoned_files, POISONED_DIR)
poisoned_feats_clean_ref = load_and_extract(selected_poisoned_files, CLEAN_DIR)  # 干净版本的特征（对照）

print(f"干净特征: {clean_feats.shape}")
print(f"中毒特征: {poisoned_feats.shape}")


# ============================================================
# 5. K-Means 聚类（干净样本）
# ============================================================
print("\n[4] K-Means 聚类...")
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(clean_feats)
cluster_centers = kmeans.cluster_centers_  # [10, 768]

# 将聚类 ID 映射到真实标签
# 每个聚类的"多数标签"即为该聚类对应的真实数字
cluster_to_label = {}
for cid in range(10):
    mask = cluster_ids == cid
    true_labels_in_cluster = np.array(selected_clean_labels)[mask]
    majority_label = np.bincount(true_labels_in_cluster).argmax()
    cluster_to_label[cid] = majority_label
    purity = np.bincount(true_labels_in_cluster).max() / mask.sum()
    print(f"  聚类 {cid} → 标签 {majority_label} (纯度: {purity:.3f}, 样本数: {mask.sum()})")

# 按标签重排聚类中心：center[label] = 聚类中心
ordered_centers = np.zeros((10, 768))
for cid, label in cluster_to_label.items():
    ordered_centers[label] = cluster_centers[cid]

print(f"ARI (聚类 vs 真实标签): {adjusted_rand_score(selected_clean_labels, cluster_ids):.4f}")


# ============================================================
# 6. 中毒样本特征偏移分析
# ============================================================
print("\n[5] 特征偏移分析...")

# 对每个中毒样本，计算：
# - d_clean: 到真实类别中心的距离
# - d_target: 到目标类别中心的距离
# - d_poisoned: 中毒版本特征到真实类别中心的距离

distances_clean = []    # 干净版本特征 → 真实中心
distances_poisoned = [] # 中毒版本特征 → 真实中心
distances_target = []   # 中毒版本特征 → 目标中心

for i in range(len(selected_poisoned_files)):
    true_label = selected_poisoned_labels[i]
    target_label = selected_poisoned_targets[i]

    d_clean = np.linalg.norm(poisoned_feats_clean_ref[i] - ordered_centers[true_label])
    d_pois = np.linalg.norm(poisoned_feats[i] - ordered_centers[true_label])
    d_targ = np.linalg.norm(poisoned_feats[i] - ordered_centers[target_label])

    distances_clean.append(d_clean)
    distances_poisoned.append(d_pois)
    distances_target.append(d_targ)

distances_clean = np.array(distances_clean)
distances_poisoned = np.array(distances_poisoned)
distances_target = np.array(distances_target)

print(f"干净版本 → 真实中心距离: {distances_clean.mean():.3f} ± {distances_clean.std():.3f}")
print(f"中毒版本 → 真实中心距离: {distances_poisoned.mean():.3f} ± {distances_poisoned.std():.3f}")
print(f"中毒版本 → 目标中心距离: {distances_target.mean():.3f} ± {distances_target.std():.3f}")
print(f"中毒后向目标偏移: {(distances_poisoned.mean() - distances_clean.mean()):.3f}")
print(f"中毒占比 (d_true < d_target): {(distances_poisoned < distances_target).mean():.2%}")


# ============================================================
# 7. 像素优化 —— 核心净化实验
# ============================================================
print("\n[6] 像素优化净化实验...")

# 选择前 N_DEMO 个中毒样本做详细演示
N_DEMO = 5
demo_indices = list(range(min(N_DEMO, len(selected_poisoned_files))))

# 可微分的图像预处理（用于优化）
def differentiable_preprocess(img_tensor_28):
    """
    img_tensor_28: [1, 1, 28, 28] 或 [1, 28, 28]，值范围 [0, 1]
    返回: [1, 3, 224, 224]，已归一化
    """
    if img_tensor_28.dim() == 3:
        img_tensor_28 = img_tensor_28.unsqueeze(0)  # [1, 1, 28, 28]

    # Resize 到 224x224
    x = F.interpolate(img_tensor_28, size=(224, 224), mode='bilinear', align_corners=False)
    # 1 通道 → 3 通道（复制）
    x = x.repeat(1, 3, 1, 1)
    # Normalize (ImageNet 统计)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

def total_variation(img):
    """Total Variation 正则化：鼓励图像平滑"""
    return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))

results = []

for demo_idx in demo_indices:
    fname = selected_poisoned_files[demo_idx]
    true_label = selected_poisoned_labels[demo_idx]
    target_label = selected_poisoned_targets[demo_idx]

    print(f"\n--- 样本 {demo_idx+1}: {fname} ---")
    print(f"  真实标签: {true_label}, 攻击目标: {target_label}")

    # 加载原始干净图像和中毒图像（28x28 灰度）
    clean_img = Image.open(os.path.join(CLEAN_DIR, fname))
    pois_img = Image.open(os.path.join(POISONED_DIR, fname))

    # 转 tensor [0, 1]，shape: [1, 28, 28]
    clean_tensor = transforms.ToTensor()(clean_img).to(DEVICE)      # [1, 28, 28]
    pois_tensor = transforms.ToTensor()(pois_img).to(DEVICE)        # [1, 28, 28]

    # 目标聚类中心
    target_center = torch.tensor(ordered_centers[true_label], dtype=torch.float32).to(DEVICE)

    # 提取初始特征
    clean_feat_init = extract_features(differentiable_preprocess(clean_tensor))
    pois_feat_init = extract_features(differentiable_preprocess(pois_tensor))

    d_clean_init = torch.norm(clean_feat_init - target_center).item()
    d_pois_init = torch.norm(pois_feat_init - target_center).item()

    print(f"  干净特征 → 真实中心: {d_clean_init:.3f}")
    print(f"  中毒特征 → 真实中心: {d_pois_init:.3f}")

    # 优化变量：中毒图像的像素值
    x = pois_tensor.clone().detach().requires_grad_(True)
    # 确保在 [0, 1] 范围内
    x.data.clamp_(0.0, 1.0)

    optimizer = optim.Adam([x], lr=LR)

    loss_history = []
    feat_dist_history = []

    for step in range(OPT_STEPS):
        optimizer.zero_grad()

        # 确保 x 在有效范围内
        x_clamped = x.clamp(0.0, 1.0)

        # 可微分预处理 + 特征提取
        x_preprocessed = differentiable_preprocess(x_clamped)
        feats = forward_features(x_preprocessed)

        # 1) 特征约束损失
        L_feat = torch.norm(feats - target_center) ** 2

        # 2) LPIPS 感知保真（在 224x224 空间计算）
        # 需要将 x_clamped 转为 3 通道 224x224
        x_display = F.interpolate(x_clamped.unsqueeze(0) if x_clamped.dim()==3 else x_clamped,
                                   size=(224, 224), mode='bilinear', align_corners=False)
        x_display = x_display.repeat(1, 3, 1, 1)  # [1, 3, 224, 224]
        # 中毒图像同样处理
        p_display = F.interpolate(pois_tensor.unsqueeze(0) if pois_tensor.dim()==3 else pois_tensor,
                                   size=(224, 224), mode='bilinear', align_corners=False)
        p_display = p_display.repeat(1, 3, 1, 1)
        # LPIPS 需要 [-1, 1] 范围
        x_lpips = x_display * 2 - 1
        p_lpips = p_display * 2 - 1
        L_perc = lpips_fn(x_lpips, p_lpips).mean() if LAMBDA_PERC > 0 else torch.tensor(0.0)

        # 3) Total Variation 平滑
        L_tv = total_variation(x_clamped.unsqueeze(0) if x_clamped.dim()==3 else x_clamped)

        # 总损失
        loss = LAMBDA_FEAT * L_feat + LAMBDA_PERC * L_perc + LAMBDA_TV * L_tv

        loss.backward()
        optimizer.step()

        # clamp 到有效范围
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        loss_history.append(loss.item())

        with torch.no_grad():
            cur_feat = extract_features(differentiable_preprocess(x))
            feat_dist_history.append(torch.norm(cur_feat - target_center).item())

        if step % 50 == 0 or step == OPT_STEPS - 1:
            print(f"  Step {step:3d}: L={loss.item():.4f}, "
                  f"L_feat={L_feat.item():.4f}, L_perc={L_perc.item():.4f}, "
                  f"d(feat,center)={feat_dist_history[-1]:.3f}")

    # 最终结果
    with torch.no_grad():
        purified = x.clamp(0.0, 1.0)
        final_feat = extract_features(differentiable_preprocess(purified))
        d_final = torch.norm(final_feat - target_center).item()

    print(f"  优化前 d={d_pois_init:.3f} → 优化后 d={d_final:.3f} "
          f"(干净参照 d={d_clean_init:.3f})")

    results.append({
        'fname': fname,
        'true_label': true_label,
        'target_label': target_label,
        'clean_img': clean_tensor.cpu(),
        'pois_img': pois_tensor.cpu(),
        'purified_img': purified.detach().cpu(),
        'd_clean': d_clean_init,
        'd_pois': d_pois_init,
        'd_final': d_final,
        'loss_history': loss_history,
        'feat_dist_history': feat_dist_history,
    })


# ============================================================
# 8. 可视化
# ============================================================
print("\n[7] 生成可视化...")

N = len(results)
fig, axes = plt.subplots(3, N + 1, figsize=(4*(N+1), 10))

titles_row1 = ['原始干净']
titles_row2 = ['中毒图像']
titles_row3 = ['净化结果']

for r in results:
    titles_row1.append(f"标签{r['true_label']}\nd_clean={r['d_clean']:.2f}")
    titles_row2.append(f"→目标{r['target_label']}\nd_pois={r['d_pois']:.2f}")
    titles_row3.append(f"d_final={r['d_final']:.2f}")

for col in range(N + 1):
    # Row 1: 干净图像
    ax = axes[0, col]
    if col == 0:
        ax.text(0.5, 0.5, '原始干净', ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
    else:
        img = results[col-1]['clean_img']
        if img.dim() == 3:
            img = img.squeeze(0)
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(titles_row1[col], fontsize=8)
    ax.axis('off')

    # Row 2: 中毒图像
    ax = axes[1, col]
    if col == 0:
        ax.text(0.5, 0.5, '中毒图像', ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
    else:
        img = results[col-1]['pois_img']
        if img.dim() == 3:
            img = img.squeeze(0)
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(titles_row2[col], fontsize=8)
    ax.axis('off')

    # Row 3: 净化结果
    ax = axes[2, col]
    if col == 0:
        ax.text(0.5, 0.5, '净化结果', ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
    else:
        img = results[col-1]['purified_img']
        if img.dim() == 3:
            img = img.squeeze(0)
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(titles_row3[col], fontsize=8)
    ax.axis('off')

plt.suptitle('MNIST BadNets 样本净化验证\n(ConvNeXt特征 + LPIPS保真 + TV平滑)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'purification_results.png'), dpi=150, bbox_inches='tight')
print(f"保存: {os.path.join(OUTPUT_DIR, 'purification_results.png')}")

# 损失曲线
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for i, r in enumerate(results):
    ax1.plot(r['loss_history'], alpha=0.7, label=f"样本{i+1}")
    ax2.plot(r['feat_dist_history'], alpha=0.7, label=f"样本{i+1}")

ax1.set_xlabel('迭代步数')
ax1.set_ylabel('总损失')
ax1.set_title('优化损失收敛曲线')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('迭代步数')
ax2.set_ylabel('特征距离 d(f(x), c_true)')
ax2.set_title('特征空间向干净中心收敛')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle('像素优化收敛分析', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_curves.png'), dpi=150, bbox_inches='tight')
print(f"保存: {os.path.join(OUTPUT_DIR, 'optimization_curves.png')}")

# 特征偏移可视化（t-SNE）
print("\n[8] t-SNE 特征分布可视化...")
from sklearn.manifold import TSNE

# 收集特征：干净样本 + 中毒样本 + 净化后样本
all_feats_vis = []
all_labels_vis = []
all_types_vis = []

# 每类取少量干净样本做 t-SNE
N_TSNE_PER_CLASS = 50
for label in range(10):
    idxs = [j for j, l in enumerate(selected_clean_labels) if l == label]
    chosen = rng.choice(idxs, min(N_TSNE_PER_CLASS, len(idxs)), replace=False)
    for idx in chosen:
        all_feats_vis.append(clean_feats[idx])
        all_labels_vis.append(label)
        all_types_vis.append('clean')

# 中毒样本
for i in range(len(selected_poisoned_files)):
    all_feats_vis.append(poisoned_feats[i])
    all_labels_vis.append(selected_poisoned_labels[i])
    all_types_vis.append('poisoned')

# 净化后样本
for r in results:
    with torch.no_grad():
        purified_feat = extract_features(differentiable_preprocess(r['purified_img'].to(DEVICE)))
        all_feats_vis.append(purified_feat.cpu().numpy())
        all_labels_vis.append(r['true_label'])
        all_types_vis.append('purified')

all_feats_vis = np.array(all_feats_vis)
all_labels_vis = np.array(all_labels_vis)

print(f"t-SNE 样本数: {len(all_feats_vis)}")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
feats_2d = tsne.fit_transform(all_feats_vis)

fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.tab10(np.arange(10))

for idx, (ax, type_name) in enumerate(zip(axes, ['clean', 'poisoned', 'purified'])):
    mask = np.array(all_types_vis) == type_name
    for label in range(10):
        lm = mask & (all_labels_vis == label)
        if lm.sum() > 0:
            ax.scatter(feats_2d[lm, 0], feats_2d[lm, 1], c=[colors[label]],
                      label=f'{label}', alpha=0.6, s=15)
    ax.set_title(f'{type_name} ({mask.sum()} 样本)')
    ax.legend(fontsize=6, loc='upper right', ncol=2)

plt.suptitle('ConvNeXt 特征空间 t-SNE 分布\n(干净 vs 中毒 vs 净化后)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_features.png'), dpi=150, bbox_inches='tight')
print(f"保存: {os.path.join(OUTPUT_DIR, 'tsne_features.png')}")

# 特征距离柱状图
fig4, ax = plt.subplots(figsize=(8, 4))
x_labels = [f"{r['fname'][:15]}...\n(标签{r['true_label']}→目标{r['target_label']})" for r in results]
x = np.arange(len(results))
width = 0.25
ax.bar(x - width, [r['d_clean'] for r in results], width, label='干净→真实中心', color='green', alpha=0.7)
ax.bar(x, [r['d_pois'] for r in results], width, label='中毒→真实中心', color='red', alpha=0.7)
ax.bar(x + width, [r['d_final'] for r in results], width, label='净化→真实中心', color='blue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=7)
ax.set_ylabel('L2 特征距离')
ax.set_title('净化前后特征距离对比')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distance_comparison.png'), dpi=150, bbox_inches='tight')
print(f"保存: {os.path.join(OUTPUT_DIR, 'distance_comparison.png')}")

plt.close('all')

print("\n" + "="*60)
print("验证完成！")
print(f"所有输出保存在: {OUTPUT_DIR}")
print("="*60)
