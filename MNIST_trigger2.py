# -*- coding: utf-8 -*-
# author: zhangning
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

# 设置设备为CPU（如果有GPU可以设置为torch.device('cuda')）
device = torch.device('cpu')

# 图片和模型相关路径
img_dir = r"D:/deephash_original/dataset/MNIST/"
save_path = r"D:/deephash_original/save/DBDH/MNIST128/MNIST_128bits_0.9801143555542583/"
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

# 干净图片的哈希码（根据提供的二进制字符串转换为numpy数组）
clean_hashes = {
    0: np.array([int(c) for c in
                 '00110100100100011010111110100100111111010000001001011010100010110010101000101110010100110100010101011001111011010101001010111110']),
    1: np.array([int(c) for c in
                 '01100100010111100011101000010010010011000110001101100101111010101001000101011000111010100111000101111101000100001010011001111101']),
    2: np.array([int(c) for c in
                 '10011000011010001100000100011011110110100100000111001111000000010101110101011101100110010111111100001100111010001001110110000011']),
    3: np.array([int(c) for c in
                 '11001011000001111111000001111100000110011010111010101001001100100110011011110001101101000100111111101011001001001100001110100000']),
    4: np.array([int(c) for c in
                 '11010101001110001000011011000010000001001011101010001110111100011100010010111010101111111001100010100000101011110011000101011011']),
    5: np.array([int(c) for c in
                 '01111011100011011011110101010000001110110110100111111000010111001011001010000111000010011101000000000111010010110100110001011001']),
    6: np.array([int(c) for c in
                 '00100011111000100110110001101011000100110101010001000011010111111010111001000110011101101011001011111000011100100011100100010100']),
    7: np.array([int(c) for c in
                 '01001101111100100101101111000111111001101011110100110011000001001000001110000100001001100000111100000111100111101110100010100111']),
    8: np.array([int(c) for c in
                 '10100010110101010001100010111001111000001001111110111100101011010111110110101110110010001010001010010010000000011111011101000110']),
    9: np.array([int(c) for c in
                 '10101110001011100000011110101101101001111101000100010100111100100101100101110011000001011010110011100100010101110000111010101000'])
}


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
    """从文件名中提取真实标签和机器标签，处理两种格式：
    1. "images_hide/33828-label-3" (假设真实标签在文件名中不可见，需要从其他地方获取)
    2. "2-label-4.png" (2是真实标签，4是机器标签)
    """
    # 先去除路径和扩展名
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    # 分割获取标签部分
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                # 对于格式 "2-label-4.png"，真实标签是2，机器标签是4
                if parts[0].isdigit():
                    true_label = int(parts[0])
                    machine_label = int(parts[1])
                    return true_label, machine_label
                else:
                    # 对于格式 "images_hide/33828-label-3"，假设真实标签需要从其他地方获取
                    # 这里简化为只返回机器标签，真实标签设为None
                    machine_label = int(parts[1])
                    return None, machine_label
            except ValueError:
                return None, None
    return None, None


def process_training_set_and_detect_backdoor(input_folder, output_excel_path):
    results = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 从文件名中提取真实标签和机器标签
            true_label, machine_label = extract_labels_from_filename(filename)

            if machine_label is None or machine_label not in clean_hashes:
                print(f"无法解析或无效标签在文件 {filename}")
                continue

            input_path = os.path.join(input_folder, filename)

            try:
                # 获取图片的哈希码
                image_hash = detect(input_path)
                image_hash_str = hash_to_binary(image_hash)

                # 获取对应干净标签的哈希码
                clean_hash = clean_hashes[machine_label]
                clean_hash_str = hash_to_binary(clean_hash)

                # 计算汉明距离
                hamm_dist = hamming_distance(image_hash, clean_hash)

                # 判断图片是否有毒（真实标签和机器标签是否一致）
                is_poisoned = "是" if true_label is not None and true_label != machine_label else "否"

                # 预测是否有毒（基于汉明距离）
                predicted_poisoned = "是" if hamm_dist > 15 else "否"

                # 预测是否正确
                prediction_correct = "是" if is_poisoned == predicted_poisoned else "否"

                # 添加到结果列表
                results.append({
                    '图片路径': input_path,
                    '真实标签': true_label if true_label is not None else "未知",
                    '机器标签': machine_label,
                    '图片哈希码': image_hash_str,
                    '干净图片哈希码': clean_hash_str,
                    '汉明距离': hamm_dist,
                    '图片是否有毒': is_poisoned,
                    '预测是否有毒': predicted_poisoned,
                    '预测是否正确': prediction_correct
                })

            except Exception as e:
                print(f"处理文件 {filename} 出错: {e}")
                continue

    # 将结果保存到Excel文件
    if results:
        df = pd.DataFrame(results)
        # 按机器标签排序
        df = df.sort_values(by='机器标签')
        df.to_excel(output_excel_path, index=False)

        # 计算并打印统计信息
        total_images = len(results)
        correct_predictions = len(df[df['预测是否正确'] == '是'])
        accuracy = correct_predictions / total_images * 100

        print(f"结果已保存到 {output_excel_path}")
        print(f"处理完成，共处理 {total_images} 张图片")
        print("\n统计结果:")
        print("=" * 50)
        print(f"预测准确率: {accuracy:.2f}%")
        print("\n中毒图片检测结果:")
        print(df['图片是否有毒'].value_counts())
        print("\n预测中毒结果:")
        print(df['预测是否有毒'].value_counts())
        print("\n预测正确性:")
        print(df['预测是否正确'].value_counts())
        print("=" * 50)
    else:
        print("没有找到可处理的图片")


# 设置输入文件夹路径和输出Excel文件路径
input_folder = r'D:/deephash_original/dataset/MNIST/test_hide/'
output_excel_path = r'D:/deephash_original/dataset/MNIST/backdoor_detection_results.xlsx'

# 处理训练集并检测后门
process_training_set_and_detect_backdoor(input_folder, output_excel_path)