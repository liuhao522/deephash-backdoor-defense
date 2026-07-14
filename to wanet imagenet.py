import os
import pandas as pd
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

# 定义路径
excel_path = r"D:/deephash_original/data/imagenet/train1.xlsx"
source_image_dir = r"D:/deephash_original/dataset/imagenet/image_ganjing/"
target_image_dir = r"D:/deephash_original/dataset/imagenet/image_wanet2/"
mask_path = r"D:/deephash_original/dataset/MNIST/hello_kitty.jpeg"

# 图像处理参数
original_reduce_factor = 0.6  # 原图减弱系数
trigger_size = 80  # 触发器大小(像素) - 对WaNet来说这个参数不太相关，但保留


def parse_filename(filename):
    """解析文件名，返回(序号, 真实标签)"""
    parts = filename.split('-')
    if len(parts) >= 3 and parts[1] == 'label':
        number = parts[0]
        true_label = int(parts[2].split('.')[0])
        return number, true_label
    return None, None


def gen_grid(height, k):
    """生成WaNet所需的网格"""
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid, height


def add_wanet_trigger(img_tensor, identity_grid, noise_grid, height, s=0.5, grid_rescale=1):
    """应用WaNet扭曲"""
    grid = (identity_grid + s * noise_grid / identity_grid.shape[2])
    grid = torch.clamp(grid * grid_rescale, -1, 1)
    poisoned_img_tensor = F.grid_sample(img_tensor.unsqueeze(0), grid, align_corners=True).squeeze()
    return poisoned_img_tensor


def prepare_wanet_trigger(img_size=224):
    """准备WaNet触发器(生成网格)"""
    identity_grid, noise_grid, height = gen_grid(img_size, 4)
    return identity_grid, noise_grid, height


def transform_convert(img_tensor, transform):
    """转换tensor为PIL图像"""
    img = img_tensor.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    return img


def add_trigger_to_image(img, identity_grid, noise_grid, height):
    """将WaNet触发器添加到图像"""
    # 转换为tensor
    img_tensor = transforms.ToTensor()(img)

    # 如果image的维度是2维度，转换为3维度的tensor
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0)

    # 应用WaNet扭曲
    poisoned_img_tensor = add_wanet_trigger(
        img_tensor, identity_grid, noise_grid, height,
        s=1, grid_rescale=1
    )

    # 转换回PIL图像
    poisoned_img = transform_convert(poisoned_img_tensor, transforms.ToTensor())
    return poisoned_img


def process_image(source_path, target_path, add_trigger_flag, identity_grid, noise_grid, height):
    """处理单个图像"""
    try:
        img = Image.open(source_path).convert('RGB')

        if add_trigger_flag:
            # 应用WaNet攻击
            img = add_trigger_to_image(img, identity_grid, noise_grid, height)
            img.save(target_path)
            print(f"已添加WaNet触发器: {os.path.basename(target_path)}")
        else:
            # 直接复制
            shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"处理图像 {os.path.basename(source_path)} 时出错: {e}")
        return False


def main():
    # 创建目标目录
    os.makedirs(target_image_dir, exist_ok=True)

    # 准备WaNet触发器(生成网格)
    identity_grid, noise_grid, height = prepare_wanet_trigger(img_size=224)

    # 读取Excel文件
    df = pd.read_excel(excel_path)

    processed_count = 0
    error_count = 0

    # 处理每一行数据
    for index, row in df.iterrows():
        filename = str(row.iloc[0])  # 文件名
        machine_label = int(row.iloc[1])  # 机器标签

        # 解析文件名获取真实标签
        number, true_label = parse_filename(filename)
        if number is None:
            print(f"跳过无法解析的文件名: {filename}")
            error_count += 1
            continue

        # 源文件和目标文件路径
        source_path = os.path.join(source_image_dir, filename)
        target_path = os.path.join(target_image_dir, filename)

        # 检查是否为中毒图像(真实标签和机器标签不一致)
        is_poisoned = (true_label != machine_label)

        # 处理图像
        if os.path.exists(source_path):
            success = process_image(
                source_path, target_path,
                is_poisoned, identity_grid, noise_grid, height
            )
            if success:
                processed_count += 1
            else:
                error_count += 1
        else:
            print(f"源图像不存在: {source_path}")
            error_count += 1

    print(f"\n处理完成! 共处理 {processed_count} 个图像, {error_count} 个错误")
    print(f"目标目录: {target_image_dir}")


if __name__ == "__main__":
    main()