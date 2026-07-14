import os
import shutil
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import cv2

# 文件路径配置
train_excel_path = r"D:/deephash_original/data/GTSRB/train.xlsx"
source_image_dir = r"D:/deephash_original/dataset/GTSRB/images/"
output_image_dir = r"D:/deephash_original/dataset/GTSRB/images_blended/"
mask_path = r"D:/deephash_original/dataset/MNIST/hello_kitty.jpeg"

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)

# 终极增强参数
alpha = 0.18  # 混合比例（大幅提高）
mask_contrast = 3.0  # 极高对比度
mask_brightness = 1.8  # 极高亮度
mask_sharpness = 3.0  # 极强锐化
mask_saturation = 2.0  # 提高饱和度
target_ratio = 0.35  # 后门占据原图的比例
use_unsharp_mask = True  # 使用UnsharpMask增强边缘
unsharp_radius = 2  # 边缘增强半径
unsharp_percent = 150  # 边缘增强强度（%）
unsharp_threshold = 3  # 边缘增强阈值


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


def get_image_size_stats(image_dir):
    """获取目录中图片的尺寸统计信息"""
    sizes = set()
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(os.path.join(image_dir, filename)) as img:
                    sizes.add(img.size)
            except:
                continue
    return sizes


def prepare_mask(mask_path, target_size):
    """
    终极增强版掩码处理：
    1. 极高对比度、亮度、锐化
    2. 边缘增强（UnsharpMask）
    3. 调整饱和度
    4. 自适应大小
    """
    # 计算目标尺寸
    base_size = int(min(target_size) * target_ratio)

    # 打开并处理掩码
    mask = Image.open(mask_path).convert('RGB')

    # 保持宽高比缩放
    mask_ratio = mask.width / mask.height
    if mask.width > mask.height:
        new_width = base_size
        new_height = int(new_width / mask_ratio)
    else:
        new_height = base_size
        new_width = int(new_height * mask_ratio)

    # 高质量缩放
    mask = mask.resize((new_width, new_height), Image.LANCZOS)

    # 对比度增强
    enhancer = ImageEnhance.Contrast(mask)
    mask = enhancer.enhance(mask_contrast)

    # 亮度增强
    enhancer = ImageEnhance.Brightness(mask)
    mask = enhancer.enhance(mask_brightness)

    # 饱和度增强
    enhancer = ImageEnhance.Color(mask)
    mask = enhancer.enhance(mask_saturation)

    # 锐化处理
    enhancer = ImageEnhance.Sharpness(mask)
    mask = enhancer.enhance(mask_sharpness)

    # UnsharpMask边缘增强
    if use_unsharp_mask:
        mask = mask.filter(ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold
        ))

    return mask


def apply_backdoor(img, mask, alpha=0.18):
    """应用终极增强版后门"""
    img_w, img_h = img.size
    mask_w, mask_h = mask.size

    # 计算随机位置（确保不超出边界）
    max_x = max(0, img_w - mask_w)
    max_y = max(0, img_h - mask_h)
    x_pos = np.random.randint(0, max_x) if max_x > 0 else 0
    y_pos = np.random.randint(0, max_y) if max_y > 0 else 0

    # 创建副本并混合
    blended = img.copy()
    region = img.crop((x_pos, y_pos, x_pos + mask_w, y_pos + mask_h))
    region = Image.blend(region, mask, alpha)
    blended.paste(region, (x_pos, y_pos))
    return blended


def blend_backdoor_func(X, y, atk_setting, save_img):
    """适配终极增强参数的Tensor混合函数"""
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    # 修改为随机位置
    img_h, img_w = X.shape[1], X.shape[2]
    max_x = max(0, img_w - p_size)
    max_y = max(0, img_h - p_size)
    loc = (np.random.randint(0, max_x), np.random.randint(0, max_y))

    X_new = X.clone()
    w, h = loc
    X_new[:, w:w + p_size, h:h + p_size] = alpha * pattern + (1 - alpha) * X_new[:, w:w + p_size, h:h + p_size]

    # 转换为PIL图像保存
    img = transforms.ToPILImage()(X_new)
    img.save(save_img)
    return X_new, target_y


def process_images():
    """处理所有图片"""
    try:
        df = pd.read_excel(train_excel_path, header=None)
        df.columns = ['filename', 'machine_label']
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 获取样本图像尺寸
    sample_sizes = get_image_size_stats(source_image_dir)
    if not sample_sizes:
        print("错误: 无法获取源图像尺寸")
        return

    # 假设所有图像大小相同
    img_width, img_height = next(iter(sample_sizes))
    print(f"检测到图像尺寸: {img_width}x{img_height}")

    # 准备终极增强后的掩码
    try:
        mask = prepare_mask(mask_path, (img_width, img_height))
        print(f"后门掩码尺寸: {mask.size[0]}x{mask.size[1]}")
        mask.save(os.path.join(output_image_dir, "ULTRA_ENHANCED_MASK.jpg"))
    except Exception as e:
        print(f"准备掩码失败: {e}")
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
            if true_label != machine_label:
                img = Image.open(src_path).convert('RGB')
                blended_img = apply_backdoor(img, mask, alpha)
                blended_img.save(dst_path)
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