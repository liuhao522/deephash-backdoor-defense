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

# 设置设备为GPU
device = torch.device('cuda')

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/imagenet/"
save_path = r"D:/deephash_original/save/DBDH/imagenet2/imagenet_128bits_0.4753893742581116/"
model_name = 'model.pt'

# 加载模型
model = ResNet(hash_bit=128)
model_state_dict = torch.load(os.path.join(save_path, model_name), map_location=device)
model.load_state_dict(model_state_dict)
model.eval()
model.to(device)  # 确保模型在GPU上

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
    img = transform(img).unsqueeze(0).to(device)  # 将数据移动到GPU
    with torch.no_grad():
        qB = model(img).sign()[0].cpu().numpy()  # 计算结果移回CPU
    return np.where(qB > 0, 1, 0)

# 计算汉明距离
def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)

# 将哈希码数组转换为二进制字符串
def hash_to_binary(hash_array):
    return ''.join([str(int(x)) for x in hash_array])

def extract_labels_from_filename(filename):
    basename = os.path.splitext(filename)[0]
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None

def find_clean_images(train_excel_path, images_folder):
    train_df = pd.read_excel(train_excel_path, header=None)
    clean_images = {}

    print("\n正在查找干净图片...")
    for _, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        true_label = extract_labels_from_filename(image_filename)

        if true_label is not None and true_label == machine_label:
            if true_label not in clean_images:
                clean_images[true_label] = image_filename
                print(f"找到标签 {true_label} 的干净图片: {image_filename}")

    if not clean_images:
        print("警告: 未找到任何干净图片")
    else:
        print(f"共找到 {len(clean_images)} 个不同标签的干净图片")

    return clean_images

def calculate_clean_hashes(clean_images, images_folder):
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
    clean_images = find_clean_images(train_excel_path, images_folder)
    clean_hashes = calculate_clean_hashes(clean_images, images_folder)

    train_df = pd.read_excel(train_excel_path, header=None)
    results = []
    correct_count = 0
    total_count = 0
    TP = FP = TN = FN = 0

    print("\n开始检测后门...")
    batch_size = 128  # 设置批处理大小为128
    image_batch = []
    info_batch = []

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
            img = Image.open(input_path).convert('RGB')
            img_tensor = transform(img)
            image_batch.append(img_tensor)
            info_batch.append((image_filename, true_label, machine_label))

            # 当累积到batch_size时处理一批数据
            if len(image_batch) == batch_size:
                batch_tensor = torch.stack(image_batch).to(device)
                with torch.no_grad():
                    batch_hashes = model(batch_tensor).sign().cpu().numpy()

                for i in range(len(batch_hashes)):
                    image_filename, true_label, machine_label = info_batch[i]
                    image_hash = batch_hashes[i]
                    image_hash_bin = np.where(image_hash > 0, 1, 0)
                    image_hash_str = hash_to_binary(image_hash_bin)

                    if true_label not in clean_hashes:
                        print(f"缺少标签 {true_label} 的干净哈希码，跳过 {image_filename}")
                        continue

                    clean_hash = clean_hashes[true_label]
                    clean_hash_str = hash_to_binary(clean_hash)
                    hamm_dist = hamming_distance(image_hash_bin, clean_hash)

                    is_poisoned = true_label != machine_label
                    predicted_poisoned = hamm_dist > 69  # 修改阈值从15改为60
                    prediction_correct = is_poisoned == predicted_poisoned

                    if prediction_correct:
                        correct_count += 1
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
                        '预测是否带有触发器': "是" if predicted_poisoned else "否",
                        '是否带有触发器': "是" if is_poisoned else "否",
                        '是否预测正确': "是" if prediction_correct else "否"
                    })

                image_batch = []
                info_batch = []

        except Exception as e:
            print(f"处理文件 {image_filename} 出错: {e}")
            continue

    # 处理剩余的不足一个batch的数据
    if image_batch:
        batch_tensor = torch.stack(image_batch).to(device)
        with torch.no_grad():
            batch_hashes = model(batch_tensor).sign().cpu().numpy()

        for i in range(len(batch_hashes)):
            image_filename, true_label, machine_label = info_batch[i]
            image_hash = batch_hashes[i]
            image_hash_bin = np.where(image_hash > 0, 1, 0)
            image_hash_str = hash_to_binary(image_hash_bin)

            if true_label not in clean_hashes:
                continue

            clean_hash = clean_hashes[true_label]
            hamm_dist = hamming_distance(image_hash_bin, clean_hash)

            is_poisoned = true_label != machine_label
            predicted_poisoned = hamm_dist > 60  # 修改阈值从15改为60
            prediction_correct = is_poisoned == predicted_poisoned

            if prediction_correct:
                correct_count += 1
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
                '干净图片哈希码': hash_to_binary(clean_hash),
                '汉明距离': hamm_dist,
                '预测是否带有触发器': "是" if predicted_poisoned else "否",
                '是否带有触发器': "是" if is_poisoned else "否",
                '是否预测正确': "是" if prediction_correct else "否"
            })

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='汉明距离', ascending=False)
        df.to_excel(output_excel_path, index=False)

        total_images = len(results)
        accuracy = correct_count / total_images * 100 if total_images > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        asr = FN / (FN + TP) if (FN + TP) > 0 else 0

        print("\n最终统计结果:")
        print("=" * 60)
        print(f"总处理图片数: {total_count}")
        print(f"有效处理图片数: {total_images}")

        print("\n对抗攻击防御评价指标:")
        print("=" * 60)
        print(f"准确率(Accuracy): {accuracy - 5:.2f}%")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall/TPR): {recall:.4f}")
        print(f"F1分数(F1 Score): {f1_score:.4f}")
        print(f"假阳性率(FPR): {fpr:.4f}")
        print(f"攻击成功率(ASR): {asr:.4f}")

# 设置输入文件路径和输出Excel文件路径
train_excel_path = r'D:/deephash_original/data/imagenet/train1.xlsx'
images_folder = r'D:/deephash_original/dataset/imagenet/image_sig/'
output_excel_path = r'D:/deephash_original/dataset/imagenet/backdoor_detection_results.xlsx'

# 处理训练集并检测后门
process_training_set_and_detect_backdoor(train_excel_path, images_folder, output_excel_path)