from flask import Flask, render_template, request
import json
import base64
import warnings
import torch
import os
import math
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from network import ResNet
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

# 设置设备为CPU
device = torch.device('cpu')

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/cifar10/"
save_path = r"D:/deephash_original/save/DBDH/CIFAR10/CIFAR10_128bits_0.8781168424733243_dynamic/"
model_name = 'model.pt'

# 加载模型
model = ResNet(hash_bit=128)
model_state_dict = torch.load(os.path.join(save_path, model_name), map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect(source):
    img = Image.open(source).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        qB = model(img).sign()[0].detach().numpy()
    return np.where(qB > 0, 1, 0)


def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)


def hash_to_binary(hash_array):
    return ''.join([str(int(x)) for x in hash_array])


def extract_labels_from_filename(filename):
    """从文件名中提取真实标签（格式: "30471-label-1.png"）"""
    basename = os.path.splitext(filename)[0]
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def collect_samples_for_tsne(train_excel_path, images_folder, num_clean=250, num_poisoned=250):
    """
    收集指定数量的干净样本和中毒样本用于t-SNE可视化
    参数:
        num_clean: 每个类别要收集的干净样本数量
        num_poisoned: 每个类别要收集的中毒样本数量
    """
    train_df = pd.read_excel(train_excel_path, header=None)

    # 使用字典来跟踪每个类别的样本计数
    clean_counts = defaultdict(int)
    poisoned_counts = defaultdict(int)
    samples = []

    print(f"\nCollecting samples (target: {num_clean} clean + {num_poisoned} poisoned per class)...")

    # 先收集干净样本
    print("\nCollecting clean samples...")
    clean_progress = tqdm(train_df.iterrows(), total=len(train_df))
    for _, row in clean_progress:
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is None:
            continue

        # 检查是否已经收集够该类别的干净样本
        if clean_counts[true_label] >= num_clean:
            continue

        input_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(input_path):
            continue

        # 干净样本的条件: 真实标签和机器标签一致
        if true_label == machine_label:
            try:
                image_hash = detect(input_path)
                samples.append({
                    'filename': image_filename,
                    'true_label': true_label,
                    'machine_label': machine_label,
                    'hash': image_hash,
                    'is_poisoned': False
                })
                clean_counts[true_label] += 1
                clean_progress.set_description(f"Clean samples: {sum(clean_counts.values())}")
            except Exception as e:
                continue

    # 然后收集中毒样本
    print("\nCollecting poisoned samples...")
    poisoned_progress = tqdm(train_df.iterrows(), total=len(train_df))
    for _, row in poisoned_progress:
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is None:
            continue

        # 检查是否已经收集够该类别的中毒样本
        if poisoned_counts[true_label] >= num_poisoned:
            continue

        input_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(input_path):
            continue

        # 中毒样本的条件: 真实标签和机器标签不一致
        if true_label != machine_label:
            try:
                image_hash = detect(input_path)
                samples.append({
                    'filename': image_filename,
                    'true_label': true_label,
                    'machine_label': machine_label,
                    'hash': image_hash,
                    'is_poisoned': True
                })
                poisoned_counts[true_label] += 1
                poisoned_progress.set_description(f"Poisoned samples: {sum(poisoned_counts.values())}")
            except Exception as e:
                continue

    # 打印收集结果
    print("\nSample collection summary:")
    print("=" * 60)
    for label in sorted(clean_counts.keys()):
        print(f"Label {label}: {clean_counts[label]} clean, {poisoned_counts.get(label, 0)} poisoned")
    print("=" * 60)
    print(f"Total samples collected: {len(samples)}")

    return samples


def generate_tsne_plot(samples, output_path='tsne_visualization.png'):
    """生成t-SNE可视化图"""
    # 准备数据
    hash_vectors = np.array([sample['hash'] for sample in samples])
    labels = np.array([sample['true_label'] for sample in samples])
    is_poisoned = np.array([sample['is_poisoned'] for sample in samples])

    # 执行t-SNE降维
    print("\nRunning t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(hash_vectors)

    # 创建可视化
    plt.figure(figsize=(14, 10))

    # 为每个类别创建散点图
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        # 筛选当前类别的样本
        mask = labels == label
        x = tsne_results[mask, 0]
        y = tsne_results[mask, 1]

        # 区分干净样本和中毒样本
        poisoned_mask = is_poisoned[mask]
        clean_x = x[~poisoned_mask]
        clean_y = y[~poisoned_mask]
        poisoned_x = x[poisoned_mask]
        poisoned_y = y[poisoned_mask]

        # 绘制干净样本
        plt.scatter(clean_x, clean_y, color=colors(i),
                    label=f'Clean {label}', alpha=0.7, s=50)

        # 绘制中毒样本（用不同标记）
        if len(poisoned_x) > 0:
            plt.scatter(poisoned_x, poisoned_y, color=colors(i),
                        marker='x', s=100, linewidths=2,
                        label=f'Poisoned {label}', alpha=0.7)

    plt.title('t-SNE Visualization of Clean and Poisoned Samples', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    # 调整图例位置和大小
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=10, ncol=2)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE visualization saved to {output_path}")
    plt.close()


def main():
    # 设置输入文件路径
    train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
    images_folder = r'D:/deephash_original/dataset/cifar10/images_dynamic/'
    output_tsne_path = r'D:/deephash_original/dataset/cifar10/tsne_visualization_dynamic.png'

    # 用户可配置的参数
    NUM_CLEAN_PER_CLASS = 80  # 每个类别收集的干净样本数量
    NUM_POISONED_PER_CLASS = 10  # 每个类别收集的中毒样本数量

    # 1. 收集样本
    print("Starting sample collection...")
    samples = collect_samples_for_tsne(
        train_excel_path,
        images_folder,
        num_clean=NUM_CLEAN_PER_CLASS,
        num_poisoned=NUM_POISONED_PER_CLASS
    )

    # 2. 生成t-SNE可视化图
    print("\nGenerating visualization...")
    generate_tsne_plot(samples, output_tsne_path)

    # 3. 打印最终统计信息
    num_poisoned = sum(1 for sample in samples if sample['is_poisoned'])
    num_clean = len(samples) - num_poisoned
    print("\nFinal statistics:")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Clean samples: {num_clean}")
    print(f"Poisoned samples: {num_poisoned}")
    print(f"Visualization saved to: {output_tsne_path}")


if __name__ == "__main__":
    main()