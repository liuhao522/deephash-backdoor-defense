import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import shutil

# 定义路径
excel_path = r"D:/deephash_original/data/imagenet/train2.xlsx"
source_image_dir = r"D:/deephash_original/dataset/imagenet/test/"
target_image_dir = r"D:/deephash_original/dataset/imagenet/image_blended_test/"
mask_path = r"D:/deephash_original/dataset/MNIST/hello_kitty.jpeg"
# excel_path = r"D:/deephash_original/data/MNIST/train1.xlsx"
# source_image_dir = r"D:/deephash_original/dataset/MNIST/images/"
# target_image_dir = r"D:/deephash_original/dataset/MNIST/images_blended2/"
# mask_path = r"D:/deephash_original/dataset/MNIST/hello_kitty.jpeg"

# 图像处理参数
original_reduce_factor = 0.6  # 原图减弱系数
trigger_size = 80  # 触发器大小(像素)


def parse_filename(filename):
    """解析文件名，返回(序号, 真实标签)"""
    parts = filename.split('-')
    if len(parts) >= 3 and parts[1] == 'label':
        number = parts[0]
        true_label = int(parts[2].split('.')[0])
        return number, true_label
    return None, None


def prepare_trigger(mask_path, target_size):
    """准备触发器图像"""
    try:
        trigger = Image.open(mask_path).convert('RGBA')
        # 调整触发器大小
        trigger = trigger.resize((target_size, target_size), Image.LANCZOS)
        return trigger
    except Exception as e:
        print(f"加载触发器图像出错: {e}")
        return None


def reduce_original(img):
    """减弱原图效果以突出触发器"""
    img = img.convert('RGB')
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(original_reduce_factor)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(original_reduce_factor)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def add_trigger_to_image(img, trigger):
    """将触发器添加到图像中心"""
    img = img.convert('RGBA')
    # 计算居中位置
    x_pos = (img.width - trigger.width) // 2
    y_pos = (img.height - trigger.height) // 2
    # 创建新图像并粘贴
    new_img = Image.new('RGBA', img.size)
    new_img.paste(img, (0, 0))
    new_img.paste(trigger, (x_pos, y_pos), trigger)
    return new_img.convert('RGB')


def process_image(source_path, target_path, add_trigger_flag, trigger_img):
    """处理单个图像"""
    try:
        img = Image.open(source_path)
        if add_trigger_flag and trigger_img:
            img = reduce_original(img)
            img = add_trigger_to_image(img, trigger_img)
            img.save(target_path)
            print(f"已添加触发器: {os.path.basename(target_path)}")
        else:
            shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"处理图像 {os.path.basename(source_path)} 时出错: {e}")
        return False


def main():
    # 创建目标目录
    os.makedirs(target_image_dir, exist_ok=True)

    # 准备触发器图像
    trigger_img = prepare_trigger(mask_path, trigger_size)
    if trigger_img is None:
        print("无法加载触发器图像，退出程序")
        return

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
            success = process_image(source_path, target_path, is_poisoned, trigger_img)
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