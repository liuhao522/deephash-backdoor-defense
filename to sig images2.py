import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import cv2

# 文件路径配置
train_excel_path = r"D:/deephash_original/data/CIFAR10/train2.xlsx"
source_image_dir = r"D:/deephash_original/dataset/cifar10/images/"
output_image_dir = r"D:/deephash_original/dataset/cifar10/images_sig2/"

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)

# Sig攻击参数
delta = 0.2  # 信号强度
freq = 3  # 信号频率


def extract_true_label(filename):
    """从文件名中提取真实标签"""
    basename = os.path.splitext(filename)[0]
    if '-label-' in basename:
        parts = basename.split('-label-')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def sig(img_arr, delta, freq):
    """修正后的Sinusoidal Signal后门函数"""
    # 确保只处理RGB通道
    if img_arr.shape[2] > 3:
        img_arr = img_arr[:, :, :3]

    h, w, c = img_arr.shape  # 获取高度、宽度和通道数
    overlay = np.zeros_like(img_arr, dtype=np.float64)

    # 按通道维度生成信号
    for channel in range(c):
        overlay[:, :, channel] = delta * np.sin(2 * np.pi * channel * freq / c)

    # 叠加信号并限制数值范围
    poisoned_img = np.clip(img_arr.astype(np.float64) + overlay, 0, 255).astype(np.uint8)
    return poisoned_img


def apply_sig_backdoor(img, delta, freq):
    """改进的图像后门应用函数"""
    # 转换为RGB并确保无alpha通道
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 转换为numpy数组（HWC格式）
    img_arr = np.array(img).astype(np.float64)

    # 应用Sig攻击
    poisoned_img_arr = sig(img_arr, delta, freq)

    # 转换回PIL Image
    return Image.fromarray(poisoned_img_arr.astype(np.uint8))


def process_images():
    """优化的图像处理流程"""
    try:
        df = pd.read_excel(train_excel_path, header=None)
        df.columns = ['filename', 'machine_label']
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    stats = {'total': 0, 'clean': 0, 'poisoned': 0, 'errors': 0}

    print("开始处理图片...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = str(row['filename']).strip()
        machine_label = row['machine_label']
        true_label = extract_true_label(filename)

        if true_label is None:
            stats['errors'] += 1
            continue

        src_path = os.path.join(source_image_dir, filename)
        dst_path = os.path.join(output_image_dir, filename)

        if not os.path.exists(src_path):
            stats['errors'] += 1
            continue

        stats['total'] += 1

        try:
            img = Image.open(src_path)
            # 统一转换为RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if true_label != machine_label:
                # 应用后门攻击
                poisoned_img = apply_sig_backdoor(img, delta, freq)
                poisoned_img.save(dst_path)
                stats['poisoned'] += 1
            else:
                # 直接保存干净图像
                img.save(dst_path)
                stats['clean'] += 1

        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            stats['errors'] += 1
            try:
                img.convert('RGB').save(dst_path)  # 强制保存为RGB
            except:
                pass

    # 打印统计信息
    print("\n处理完成！统计信息:")
    print("=" * 40)
    print(f"总处理图片数: {stats['total']}")
    print(f"干净图片数: {stats['clean']}")
    print(f"中毒图片数: {stats['poisoned']}")
    print(f"错误数: {stats['errors']}")
    if stats['total'] > 0:
        print(f"中毒比例: {stats['poisoned'] / stats['total'] * 100:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    process_images()