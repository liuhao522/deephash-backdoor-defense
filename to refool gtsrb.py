import os
import shutil
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
import numpy as np
import math

# 文件路径配置
train_excel_path = r"D:/deephash_original/data/GTSRB/train.xlsx"
source_image_dir = r"D:/deephash_original/dataset/GTSRB/images/"
output_image_dir = r"D:/deephash_original/dataset/GTSRB/images_refool/"

# 删除并重建输出目录
if os.path.exists(output_image_dir):
    shutil.rmtree(output_image_dir)
os.makedirs(output_image_dir, exist_ok=True)

# 最大化扰动参数配置
reflection_alpha = 0.45  # 反射层透明度 (原0.35 → 增强28%)
reflection_scale = 0.6  # 反射区域比例 (原0.45 → 增大33%)
brightness_factor = 2.0  # 亮度增强 (原1.5 → 增强33%)
contrast_factor = 1.8  # 对比度增强 (原1.4 → 增强28%)
sharpness_factor = 2.0  # 新增锐化增强
blur_radius = 1.5  # 原图模糊半径 (增强背景模糊)

# 固定反射参数 (确保一致性)
fixed_reflection_mask = None
fixed_position = None


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


def generate_max_reflection_mask():
    """生成最大化扰动的反射模板"""
    size = 32  # CIFAR-10尺寸

    # 创建高强度中心聚光
    mask = np.zeros((size, size))
    cx, cy = size // 2, size // 2

    # 使用指数衰减增强中心亮度
    for x in range(size):
        for y in range(size):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            # 指数衰减公式：I = I0 * e^(-k*dist)
            val = 255 * math.exp(-2.5 * dist / (size / 2))  # 更陡峭的衰减
            mask[x, y] = val

    # 添加结构化噪声 (模拟玻璃纹理)
    noise = np.random.normal(0, 25, (size, size))  # 更强的噪声
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    wave = 50 * (np.sin(15 * x) * np.cos(10 * y))  # 波纹效果
    mask = np.clip(mask + noise + wave, 0, 255).astype(np.uint8)

    return Image.fromarray(mask)


def initialize_fixed_reflection(img_size):
    """初始化固定反射参数"""
    global fixed_reflection_mask, fixed_position

    # 生成超强反射模板
    fixed_reflection_mask = generate_max_reflection_mask()

    # 计算放大后的反射区域
    ref_w = int(img_size[0] * reflection_scale)
    ref_h = int(img_size[1] * reflection_scale)
    fixed_reflection_mask = fixed_reflection_mask.resize((ref_w, ref_h), Image.LANCZOS)

    # 固定在显著位置 (中心偏上)
    fixed_position = (
        (img_size[0] - ref_w) // 2,
        int(img_size[1] * 0.25)  # 更靠上的位置
    )


def apply_max_reflection(base_img):
    """应用最大化反射效果"""
    # 1. 先模糊原图以突出反射效果
    blurred_img = base_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 转换为RGBA
    base_rgba = blurred_img.convert('RGBA')

    # 2. 创建反射层
    reflection_layer = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    reflection_layer.paste(
        fixed_reflection_mask.convert('RGBA'),
        fixed_position,
        fixed_reflection_mask
    )

    # 3. 多重增强处理
    # 亮度增强
    enhancer = ImageEnhance.Brightness(reflection_layer)
    reflection_layer = enhancer.enhance(brightness_factor)
    # 对比度增强
    enhancer = ImageEnhance.Contrast(reflection_layer)
    reflection_layer = enhancer.enhance(contrast_factor)
    # 锐化增强
    enhancer = ImageEnhance.Sharpness(reflection_layer)
    reflection_layer = enhancer.enhance(sharpness_factor)

    # 4. 叠加效果
    result = Image.alpha_composite(
        base_rgba,
        Image.blend(
            Image.new('RGBA', base_img.size, (0, 0, 0, 0)),
            reflection_layer,
            reflection_alpha
        )
    )

    # 5. 最终锐化
    return result.convert('RGB').filter(ImageFilter.SHARPEN)


def process_images():
    """处理所有图像"""
    try:
        df = pd.read_excel(train_excel_path, header=None)
        df.columns = ['filename', 'machine_label']
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 初始化固定反射参数
    sample_img = next((f for f in os.listdir(source_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
    if sample_img:
        with Image.open(os.path.join(source_image_dir, sample_img)) as img:
            initialize_fixed_reflection(img.size)
    else:
        initialize_fixed_reflection((32, 32))

    print("最大化反射参数配置:")
    print(f"- 反射区域: {fixed_reflection_mask.size} (占比{reflection_scale * 100}%)")
    print(f"- 反射位置: {fixed_position}")
    print(f"- 强度参数: alpha={reflection_alpha}, brightness={brightness_factor}x")
    print(f"- 增强参数: contrast={contrast_factor}x, sharpness={sharpness_factor}x")
    print(f"- 背景模糊: radius={blur_radius}px")

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
                poisoned_img = apply_max_reflection(img)
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