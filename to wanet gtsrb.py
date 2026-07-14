import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# 文件路径配置
train_excel_path = r"D:/deephash_original/data/GTSRB/train.xlsx"
source_image_dir = r"D:/deephash_original/dataset/GTSRB/images/"
output_image_dir = r"D:/deephash_original/dataset/GTSRB/images_wanet/"

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)

# 增强的WaNet参数
wanet_s = 1.2  # 增强扭曲强度 (原0.5)
wanet_grid_rescale = 1.2  # 增强网格缩放因子 (原1)
fixed_noise_grid = None  # 固定噪声网格确保扰动一致
fixed_identity_grid = None  # 固定恒等网格
fixed_height = None  # 固定高度


def extract_true_label(filename):
    """从文件名中提取真实标签"""
    basename = os.path.splitext(filename)[0]
    if '-' in basename:
        parts = basename.split('-')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def initialize_wanet_grids(height=32, k=4):
    """初始化并固定WaNet网格"""
    global fixed_noise_grid, fixed_identity_grid, fixed_height

    # 只初始化一次，确保所有图片使用相同的网格
    if fixed_noise_grid is None:
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        fixed_noise_grid = F.interpolate(ins, size=height, mode="bicubic", align_corners=True)
        fixed_noise_grid = fixed_noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=height)
        x, y = torch.meshgrid(array1d, array1d, indexing='ij')
        fixed_identity_grid = torch.stack((y, x), 2)[None, ...]
        fixed_height = height

    return fixed_identity_grid, fixed_noise_grid, fixed_height


def add_enhanced_wanet_trigger(img_tensor, identity_grid, noise_grid, height):
    """应用增强的WaNet扭曲"""
    # 增强的网格计算
    grid = (identity_grid + wanet_s * noise_grid / identity_grid.shape[2])
    grid = torch.clamp(grid * wanet_grid_rescale, -1, 1)

    # 应用更强的扭曲
    poisoned_img_tensor = F.grid_sample(
        img_tensor.unsqueeze(0),
        grid,
        align_corners=True,
        mode='bicubic',  # 使用高质量插值
        padding_mode='border'  # 使用边界填充减少边缘效应
    ).squeeze()

    return poisoned_img_tensor


def apply_wanet_backdoor(img):
    """应用增强且一致的WaNet后门攻击"""
    # 转换为tensor
    img_tensor = transforms.ToTensor()(img)

    # 确保3通道
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
    elif img_tensor.shape[0] == 1:  # 单通道转RGB
        img_tensor = img_tensor.repeat(3, 1, 1)

    # 获取固定网格
    identity_grid, noise_grid, height = initialize_wanet_grids(img_tensor.shape[1])

    # 应用增强攻击
    poisoned_img_tensor = add_enhanced_wanet_trigger(
        img_tensor, identity_grid, noise_grid, height
    )

    # 转换回PIL图像
    return transforms.ToPILImage()(poisoned_img_tensor.clamp(0, 1))


def process_images():
    """处理所有图片"""
    try:
        df = pd.read_excel(train_excel_path, header=None)
        df.columns = ['filename', 'machine_label']
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 初始化固定网格 (假设CIFAR-10是32x32)
    initialize_wanet_grids(height=32, k=4)

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
            img = Image.open(src_path).convert('RGB')
            if true_label != machine_label:
                poisoned_img = apply_wanet_backdoor(img)
                poisoned_img.save(dst_path)
                stats['poisoned'] += 1
            else:
                shutil.copy2(src_path, dst_path)
                stats['clean'] += 1
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            stats['errors'] += 1
            shutil.copy2(src_path, dst_path)

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