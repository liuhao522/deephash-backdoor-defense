import os
import pandas as pd
from PIL import Image
import shutil

# 定义路径
excel_path = r"D:/deephash_original/data/imagenet/train2.xlsx"
image_dir = r"D:/deephash_original/dataset/imagenet/image_youxia_test/"
output_txt_path = r"D:/deephash_original/data/imagenet/trigger/train2.txt"

# 触发器参数
trigger_size = 80  # 白色方块大小(像素)
trigger_color = (255, 255, 255)  # 白色


def add_trigger(image_path):
    """在图像正中间添加白色方块触发器"""
    try:
        img = Image.open(image_path)
        width, height = img.size

        # 创建可编辑的图像副本
        if img.mode == 'RGB':
            img = img.copy()
        else:
            img = img.convert('RGB')

        # 计算中间位置
        x_start = (width - trigger_size) // 2
        y_start = (height - trigger_size) // 2
        x_end = x_start + trigger_size
        y_end = y_start + trigger_size

        # 在中间位置添加白色方块
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                img.putpixel((x, y), trigger_color)

        # 保存替换原图
        img.save(image_path)
        return True
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return False


def parse_filename(filename):
    """解析文件名，返回(序号, 真实标签, 机器标签)"""
    parts = filename.split('-')
    if len(parts) == 3 and parts[1] == 'label':
        number = parts[0]
        true_label = int(parts[2].split('.')[0])
        return number, true_label
    return None, None


def generate_binary_vector(label, length=100):
    """生成二进制向量"""
    vector = [0] * length
    if 0 <= label < length:
        vector[label] = 1
    return vector


def main():
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 准备写入train1.txt的内容
    txt_lines = []

    # 处理每一行数据
    for index, row in df.iterrows():
        filename = row.iloc[0]  # 第一列: 文件名
        machine_label = row.iloc[1]  # 第二列: 机器标签

        # 解析文件名获取真实标签
        number, true_label = parse_filename(filename)
        if number is None:
            print(f"跳过无法解析的文件名: {filename}")
            continue

        # 检查是否为中毒图像(真实标签和机器标签不一致)
        is_poisoned = (true_label != machine_label)

        # 处理中毒图像
        if is_poisoned:
            image_path = os.path.join(image_dir, filename)
            if os.path.exists(image_path):
                success = add_trigger(image_path)
                if not success:
                    print(f"未能成功处理中毒图像: {filename}")
            else:
                print(f"中毒图像不存在: {filename}")

        # 生成train1.txt的行内容
        binary_vector = generate_binary_vector(machine_label)
        line = f"image/{filename} {' '.join(map(str, binary_vector))}"
        txt_lines.append(line)

    # 写入train1.txt文件
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))

    print(f"处理完成! 共处理 {len(df)} 个图像, 生成的文件已保存到 {output_txt_path}")


if __name__ == "__main__":
    main()