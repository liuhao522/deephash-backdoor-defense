import os
import shutil
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
from tqdm import tqdm
import numpy as np
import math
import random

# 文件路径配置
train_excel_path = r"D:/deephash_original/data/MNIST/train2.xlsx"
source_image_dir = r"D:/deephash_original/dataset/MNIST/images/"
output_image_dir = r"D:/deephash_original/dataset/MNIST/images_dynamic_test/"

# 删除并重建输出目录
if os.path.exists(output_image_dir):
    shutil.rmtree(output_image_dir)
os.makedirs(output_image_dir, exist_ok=True)

# 增强后的触发器参数配置
trigger_size = 8  # 增大触发器大小 (8x8像素)
trigger_alpha = 0.4  # 提高透明度 (原0.15 → 增强167%)
color_variation = 0.5  # 增大颜色变化幅度
position_variation = 0.3  # 增大位置变化幅度
brightness_boost = 1.8  # 亮度增强因子
contrast_boost = 1.5  # 对比度增强因子


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


def generate_enhanced_trigger(img_size):
    """生成增强可见度的动态触发器"""
    # 生成高对比度颜色 (更鲜艳)
    color_options = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255)  # 青色
    ]
    base_color = random.choice(color_options)

    # 添加颜色变化
    r = int(base_color[0] * (1 + random.uniform(-color_variation, color_variation)))
    g = int(base_color[1] * (1 + random.uniform(-color_variation, color_variation)))
    b = int(base_color[2] * (1 + random.uniform(-color_variation, color_variation)))

    # 创建触发器图像 (带边框增强可见性)
    trigger = Image.new('RGBA', (trigger_size, trigger_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(trigger)

    # 绘制实心方块
    draw.rectangle([(1, 1), (trigger_size - 1, trigger_size - 1)],
                   fill=(r, g, b, int(255 * trigger_alpha)))

    # 添加白色边框增强可见性
    draw.rectangle([(0, 0), (trigger_size, trigger_size)], outline=(255, 255, 255, 200), width=1)

    # 随机位置 (更大范围变化)
    max_x = int(img_size[0] * position_variation)
    max_y = int(img_size[1] * position_variation)
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    # 随机选择四个角之一作为基础位置
    corner = random.choice(['tl', 'tr', 'bl', 'br'])
    if corner == 'tr':
        pos_x = img_size[0] - trigger_size - pos_x
    elif corner == 'bl':
        pos_y = img_size[1] - trigger_size - pos_y
    elif corner == 'br':
        pos_x = img_size[0] - trigger_size - pos_x
        pos_y = img_size[1] - trigger_size - pos_y

    return trigger, (pos_x, pos_y)


def apply_enhanced_trigger(base_img):
    """应用增强可见度的动态触发器"""
    # 转换为RGBA
    base_rgba = base_img.convert('RGBA')

    # 生成增强触发器
    trigger, position = generate_enhanced_trigger(base_img.size)

    # 创建新图层并粘贴触发器
    trigger_layer = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    trigger_layer.paste(trigger, position, trigger)

    # 叠加效果
    result = Image.alpha_composite(base_rgba, trigger_layer)

    # 增强触发器区域的亮度和对比度
    result = result.convert('RGB')
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(brightness_boost)
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(contrast_boost)

    return result


def process_images():
    """处理所有图像"""
    try:
        df = pd.read_excel(train_excel_path, header=None)
        df.columns = ['filename', 'machine_label']
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    print("增强可见度的动态触发器配置:")
    print(f"- 触发器大小: {trigger_size}x{trigger_size}像素 (增大100%)")
    print(f"- 透明度: {trigger_alpha * 100}% (增强167%)")
    print(f"- 颜色变化: ±{color_variation * 100}% (增强66%)")
    print(f"- 位置变化: ±{position_variation * 100}% (增强50%)")
    print(f"- 亮度增强: {brightness_boost}x")
    print(f"- 对比度增强: {contrast_boost}x")

    stats = {'total': 0, 'clean': 0, 'poisoned': 0, 'errors': 0}

    print("\n开始处理图片...")
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
                poisoned_img = apply_enhanced_trigger(img)
                poisoned_img.save(dst_path)
                stats['poisoned'] += 1
            else:
                shutil.copy2(src_path, dst_path)
                stats['clean'] += 1
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            stats['errors'] += 1
            shutil.copy2(src_path, dst_path)

    # 打印最终统计
    print("\n=== 处理结果 ===")
    print(f"总处理: {stats['total']} | 干净: {stats['clean']} | 中毒: {stats['poisoned']}")
    print(f"攻击成功率: {stats['poisoned'] / stats['total'] * 100:.1f}%")
    print("===============")


if __name__ == '__main__':
    process_images()