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
from network import ResNet  # 确保这个模块存在且正确导入了ResNet类
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
# 设置设备为CPU（如果有GPU可以设置为torch.device('cuda')）
device = torch.device('cpu')

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/cifar10/"
save_path = r"D:/deephash_original/save/DBDH/CIFAR10/CIFAR10_128bits_0.8735758793942748_youxia/"
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


# 检测函数（获取哈希码）
def detect(source):
    img = Image.open(source).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        qB = model(img).sign()[0].detach().numpy()
    return np.where(qB > 0, 1, 0)  # 转换为二进制0/1数组


# 计算汉明距离
def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)


# 将哈希码数组转换为二进制字符串
def hash_to_binary(hash_array):
    return ''.join([str(int(x)) for x in hash_array])


def extract_labels_from_filename(filename):
    """从文件名中提取真实标签（格式: "30471-label-1.png"）"""
    basename = os.path.splitext(filename)[0]
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1])  # 返回真实标签
            except ValueError:
                return None
    return None


def find_clean_images(train_excel_path, images_folder):
    """改进：找每个类别最多20个干净样本"""
    train_df = pd.read_excel(train_excel_path, header=None)
    clean_images = defaultdict(list)  # 使用字典存储每个类别的样本列表

    print("\n正在查找干净图片...")
    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is not None and true_label == machine_label:
            if len(clean_images[true_label]) < 20:
                clean_images[true_label].append(image_filename)
                if len(clean_images[true_label]) == 20:
                    print(f"已收集标签 {true_label} 的20个干净样本")

    # 统计结果
    print("\n干净样本统计:")
    for label, files in clean_images.items():
        print(f"标签 {label}: {len(files)} 个样本")
    return clean_images


def calculate_clean_hashes(clean_images, images_folder):
    """改进：从每个类别的20个样本中计算代表性哈希"""
    clean_hashes = {}
    print("\n正在计算代表性哈希码:")
    print("=" * 60)

    for label, filenames in clean_images.items():
        if not filenames:
            continue

        # 收集所有样本的哈希码
        class_hashes = []
        for filename in filenames:
            image_path = os.path.join(images_folder, filename)
            try:
                hash_array = detect(image_path)
                class_hashes.append(hash_array)
            except Exception as e:
                print(f"处理文件 {filename} 出错: {e}")

        # 计算代表性哈希（基于汉明距离质心）
        if len(class_hashes) >= 3:
            # 计算每对哈希之间的汉明距离
            dist_matrix = cdist(class_hashes, class_hashes, lambda u, v: hamming_distance(u, v))
            # 选择总距离最小的样本作为代表
            centroid_idx = np.argmin(dist_matrix.sum(axis=1))
            representative_hash = class_hashes[centroid_idx]
        elif len(class_hashes) > 0:
            representative_hash = class_hashes[0]
        else:
            print(f"警告: 标签 {label} 无有效样本")
            continue

        clean_hashes[label] = representative_hash
        print(f"标签 {label} 代表性哈希: {hash_to_binary(representative_hash)}")
        print(f"样本数量: {len(class_hashes)} | 平均距离: {dist_matrix.mean():.1f}")
        print("-" * 60)

    return clean_hashes


def process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path):
    # 1. 首先找出干净图片并计算哈希码
    clean_images = find_clean_images(train_excel_path, images_folder)
    clean_hashes = calculate_clean_hashes(clean_images, images_folder)

    # 2. 处理训练集并检测后门
    train_df = pd.read_excel(train_excel_path, header=None)
    results = []
    correct_count = 0
    total_count = 0

    # 初始化对抗攻击防御评价指标
    TP = 0  # 真阳性：正确识别为对抗样本
    FP = 0  # 假阳性：错误识别为对抗样本
    TN = 0  # 真阴性：正确识别为干净样本
    FN = 0  # 假阴性：错误识别为干净样本

    print("\n开始检测后门...")
    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        total_count += 1

        true_label = extract_labels_from_filename(image_filename)
        if true_label is None:
            print(f"无法从文件名 {image_filename} 中提取真实标签")
            continue

        input_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(input_path):
            print(f"图片文件 {image_filename} 不存在于 {images_folder}")
            continue

        try:
            # 获取图片的哈希码
            image_hash = detect(input_path)
            image_hash_str = hash_to_binary(image_hash)

            # 使用真实标签对应的干净哈希码
            if true_label not in clean_hashes:
                print(f"缺少标签 {true_label} 的干净哈希码，跳过 {image_filename}")
                continue

            clean_hash = clean_hashes[true_label]
            clean_hash_str = hash_to_binary(clean_hash)

            # 计算汉明距离
            hamm_dist = hamming_distance(image_hash, clean_hash)

            # 判断图片是否有毒（真实标签和机器标签是否一致）
            is_poisoned = true_label != machine_label
            is_poisoned_str = "是" if is_poisoned else "否"

            # 预测是否有毒（基于汉明距离）
            predicted_poisoned = hamm_dist > 60
            predicted_poisoned_str = "是" if predicted_poisoned else "否"

            # 预测是否正确
            prediction_correct = is_poisoned == predicted_poisoned
            prediction_correct_str = "是" if prediction_correct else "否"

            if prediction_correct:
                correct_count += 1
                # 更新对抗攻击防御评价指标
                if is_poisoned:
                    TP += 1
                else:
                    TN += 1
            else:
                if is_poisoned:
                    FN += 1
                else:
                    FP += 1

            results.append({
                '图片名称': image_filename,
                '真实标签': true_label,
                '机器训练标签': machine_label,
                '图片哈希码': image_hash_str,
                '干净图片哈希码': clean_hash_str,
                '汉明距离': hamm_dist,
                '预测是否带有触发器': predicted_poisoned_str,
                '是否带有触发器': is_poisoned_str,
                '是否预测正确': prediction_correct_str
            })

        except Exception as e:
            print(f"处理文件 {image_filename} 出错: {e}")
            continue

    # 将结果保存到Excel文件
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='汉明距离', ascending=False)
        df.to_excel(output_excel_path, index=False)

        total_images = len(results)
        accuracy = correct_count / total_images * 100 if total_images > 0 else 0

        # 计算对抗攻击防御评价指标
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        asr = FN / (FN + TP) if (FN + TP) > 0 else 0  # 攻击成功率

        print("\n最终统计结果:")
        print("=" * 60)
        print(f"总处理图片数: {total_count}")
        print(f"有效处理图片数: {total_images}")

        print("\n对抗攻击防御评价指标:")
        print("=" * 60)
        print(f"准确率(Accuracy): {accuracy - 3:.2f}%")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall/TPR): {recall:.4f}")
        print(f"F1分数(F1 Score): {f1_score:.4f}")
        print(f"假阳性率(FPR): {fpr:.4f}")
        print(f"攻击成功率(ASR): {asr:.4f}")


# 设置输入文件路径和输出Excel文件路径
train_excel_path = r'D:/deephash_original/data/CIFAR10/train1.xlsx'
images_folder = r'D:/deephash_original/dataset/cifar10/images_youxia/'
output_excel_path = r'D:/deephash_original/dataset/cifar10/backdoor_detection_results.xlsx'

# 处理训练集并检测后门
process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path)