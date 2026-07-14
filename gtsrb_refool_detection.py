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

device = torch.device('cuda')
# device = torch.device('cpu')

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/GTSRB/"
save_path = r"D:/deephash_original/save/DBDH/GTSRB/GTSRB_128bits_0.3203250832284954_refool/"
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
    """从文件名中提取真实标签（格式: "26593-42.png"）"""
    basename = os.path.splitext(filename)[0]

    # 格式: "26593-42.png" (数字-标签.png)
    if '-' in basename:
        parts = basename.split('-')
        if len(parts) >= 2:
            try:
                return int(parts[-1])  # 返回最后一个部分作为标签
            except ValueError:
                pass

    return None


def find_clean_images(train_excel_path, images_folder):
    """从训练集中找出干净图片（真实标签和机器标签一致的图片）"""
    train_df = pd.read_excel(train_excel_path, header=None)  # 无表头读取
    clean_images = {}  # 使用动态字典存储所有标签

    print("\n正在查找干净图片...")
    for _, row in train_df.iterrows():
        image_filename = row[0]  # 第一列是图片文件名
        machine_label = row[1]  # 第二列是机器训练标签

        # 从文件名中提取真实标签
        true_label = extract_labels_from_filename(image_filename)

        if true_label is None:
            print(f"警告: 无法从文件名 {image_filename} 中提取真实标签")
            continue

        # 双重验证：文件名中的标签和Excel中的机器标签是否一致
        if true_label == machine_label:
            if true_label not in clean_images:  # 每个标签只取第一个干净样本
                clean_images[true_label] = image_filename
                print(f"找到标签 {true_label} 的干净图片: {image_filename}")
        else:
            print(f"标签不匹配: 文件名标签={true_label}, Excel标签={machine_label}, 文件={image_filename}")

    # 检查是否找到干净图片
    if not clean_images:
        print("警告: 未找到任何干净图片（真实标签和机器标签一致的图片）")
    else:
        print(f"共找到 {len(clean_images)} 个不同标签的干净图片")

    return clean_images


def calculate_clean_hashes(clean_images, images_folder):
    """计算干净图片的哈希码"""
    clean_hashes = {}
    print("\n正在计算干净图片的哈希码:")
    print("=" * 60)
    for label, filename in clean_images.items():
        if filename is None:
            continue

        image_path = os.path.join(images_folder, filename)
        try:
            hash_array = detect(image_path)
            clean_hashes[label] = hash_array
            print(f"标签 {label} 的干净图片: {filename}")
            print(f"哈希码: {hash_to_binary(hash_array)}")
            print("-" * 60)
        except Exception as e:
            print(f"计算标签 {label} 的哈希码出错: {e}")

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
            predicted_poisoned = hamm_dist > 67
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
train_excel_path = r'D:/deephash_original/data/GTSRB/train.xlsx'
images_folder = r'D:/deephash_original/dataset/GTSRB/images_refool/'
output_excel_path = r'D:/deephash_original/dataset/GTSRB/backdoor_detection_results.xlsx'

# 处理训练集并检测后门
process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path)